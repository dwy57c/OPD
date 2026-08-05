from copy import deepcopy
import json
from threading import Lock

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.data_model.tasks import InitialState, RewardType, Task
from tau2.agent.llm_agent import LLMGTAgent
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.evaluator import evaluator_nl_assertions
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator

from coevo.config import InfraConfig
from coevo.models import Tau2PolicyFactory


_NL_EVALUATOR_LOCK = Lock()


def dump_messages(messages: list[Message]) -> list[dict]:
    return [message.model_dump(mode="json") for message in messages]


def load_messages(rows: list[dict]) -> list[Message]:
    classes = {
        "assistant": AssistantMessage,
        "system": SystemMessage,
        "user": UserMessage,
        "tool": ToolMessage,
    }
    messages = []
    for row in rows:
        message_class = (
            MultiToolMessage
            if row["role"] == "tool" and "tool_messages" in row
            else classes[row["role"]]
        )
        messages.append(message_class.model_validate(row))
    return messages


class Tau2Environment:
    """τ² state/tool/verifier adapter. No policy optimization lives here."""

    def __init__(self, config: InfraConfig):
        self.config = config
        self.task = get_tasks(
            config.domain,
            task_split_name=config.task_split,
            task_ids=[config.task_id],
        )[0]
        if not LLMGTAgent.check_valid_task(self.task):
            raise ValueError(
                f"Task {config.domain}/{config.task_split}/{config.task_id} has no "
                "oracle actions and "
                "cannot be used for closed-model-hinted Teacher correctability"
            )
        self.policies = Tau2PolicyFactory(config)

    def initial_history(self) -> list[Message]:
        history = []
        if self.task.initial_state and self.task.initial_state.message_history:
            history = deepcopy(self.task.initial_state.message_history)
        if not history:
            history.append(
                AssistantMessage(
                    role="assistant", content="Hi! How can I help you today?"
                )
            )
        return history

    def task_at(self, history: list[Message]) -> Task:
        task = self.task.model_copy(deep=True)
        if task.initial_state is None:
            task.initial_state = InitialState(message_history=deepcopy(history))
        else:
            task.initial_state.message_history = deepcopy(history)
        return task

    def fresh_environment(self):
        return registry.get_env_constructor(self.config.domain)()

    def orchestrator(
        self, history: list[Message], policy: str, seed: int | None = None
    ) -> Orchestrator:
        task = self.task_at(history)
        environment = self.fresh_environment()
        if policy == "teacher":
            agent = self.policies.teacher(environment, task)
        else:
            agent = self.policies.student(environment)
        buyer = self.policies.buyer_reference(environment, task)
        return Orchestrator(
            domain=self.config.domain,
            agent=agent,
            user=buyer,
            environment=environment,
            task=task,
            max_steps=self.config.branch_max_steps,
            seed=self.config.seed if seed is None else seed,
        )

    def continue_to_terminal(
        self, history: list[Message], policy: str, seed: int | None = None
    ) -> SimulationRun:
        orchestrator = self.orchestrator(history, policy, seed=seed)
        simulation = orchestrator.run()
        try:
            simulation.reward_info = self.evaluate(simulation)
        except ValueError as error:
            if "Unknown tool" not in str(error):
                raise
            simulation.reward_info = RewardInfo(
                reward=0.0,
                info={
                    "invalid_continuation": True,
                    "evaluation_error": str(error),
                },
            )
        return simulation

    def evaluate(self, simulation: SimulationRun):
        reward_basis = (
            set(self.task.evaluation_criteria.reward_basis)
            if self.task.evaluation_criteria is not None
            else set()
        )
        if RewardType.NL_ASSERTION not in reward_basis:
            return evaluate_simulation(
                simulation=simulation,
                task=self.task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=self.config.domain,
            )

        judge = self.config.nl_judge or self.config.policy
        judge_args = {
            **judge.litellm_args,
            "temperature": 0.0,
            "max_tokens": self.config.nl_judge_max_tokens,
        }
        # tau2 v1 keeps its NL judge in module globals rather than accepting it as
        # evaluate_simulation input. Hold a process-wide lock while selecting the
        # configured fixed judge so concurrent Buyer rollouts cannot cross-wire it.
        with _NL_EVALUATOR_LOCK:
            previous_model = evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS
            previous_args = evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS
            evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = judge.litellm_model
            evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = judge_args
            try:
                return evaluate_simulation(
                    simulation=simulation,
                    task=self.task,
                    evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
                    solo_mode=False,
                    domain=self.config.domain,
                )
            finally:
                evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = previous_model
                evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = previous_args

    def advance_student(self, history: list[Message]) -> list[Message]:
        """Advance through Student tool calls until Student speaks to Buyer once."""
        orchestrator = self.orchestrator(history, "student")
        orchestrator.initialize()
        while not orchestrator.done:
            orchestrator.step()
            if (
                orchestrator.from_role == Role.AGENT
                and orchestrator.to_role == Role.USER
            ):
                break
        return orchestrator.get_trajectory()

    def execute_user_tools(self, history: list[Message]) -> list[Message]:
        if not history or not isinstance(history[-1], UserMessage):
            raise ValueError("User tool execution requires a final UserMessage")
        pending_message = history[-1]
        if not pending_message.is_tool_call():
            raise ValueError("Final UserMessage does not contain a tool call")

        # τ² replays completed tool call/response pairs while restoring state. A pending
        # call cannot be included in that replay, so restore the completed prefix first
        # and execute the new user call against the reconstructed environment.
        replay = self.orchestrator(history[:-1], "student")
        replay.initialize()
        observations = [
            replay.environment.get_response(tool_call)
            for tool_call in pending_message.tool_calls
        ]
        return [*history, *observations]

    @staticmethod
    def buyer_message(content: str, tool_calls=None) -> UserMessage:
        calls = None
        if tool_calls:
            calls = [
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=json.loads(call.function.arguments),
                    requestor="user",
                )
                for call in tool_calls
            ]
        return UserMessage(role="user", content=content or None, tool_calls=calls)

    @staticmethod
    def buyer_stopped(message: UserMessage) -> bool:
        return UserSimulator.is_stop(message)
