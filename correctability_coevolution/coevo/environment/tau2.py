from copy import deepcopy
import json

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import InitialState, Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator

from coevo.config import InfraConfig
from coevo.models import Tau2PolicyFactory


def dump_messages(messages: list[Message]) -> list[dict]:
    return [message.model_dump(mode="json") for message in messages]


def load_messages(rows: list[dict]) -> list[Message]:
    classes = {
        "assistant": AssistantMessage,
        "user": UserMessage,
        "tool": ToolMessage,
    }
    return [classes[row["role"]].model_validate(row) for row in rows]


class Tau2Environment:
    """τ² state/tool/verifier adapter. No policy optimization lives here."""

    def __init__(self, config: InfraConfig):
        self.config = config
        self.task = get_tasks(config.domain, task_ids=[config.task_id])[0]
        self.policies = Tau2PolicyFactory(config)

    def initial_history(self) -> list[Message]:
        history = []
        if self.task.initial_state and self.task.initial_state.message_history:
            history = deepcopy(self.task.initial_state.message_history)
        if not history:
            history.append(
                AssistantMessage(role="assistant", content="Hi! How can I help you today?")
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
        simulation.reward_info = evaluate_simulation(
            simulation=simulation,
            task=self.task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=self.config.domain,
        )
        return simulation

    def advance_student(self, history: list[Message]) -> list[Message]:
        """Advance through Student tool calls until Student speaks to Buyer once."""
        orchestrator = self.orchestrator(history, "student")
        orchestrator.initialize()
        while not orchestrator.done:
            orchestrator.step()
            if orchestrator.from_role == Role.AGENT and orchestrator.to_role == Role.USER:
                break
        return orchestrator.get_trajectory()

    def execute_user_tools(self, history: list[Message]) -> list[Message]:
        orchestrator = self.orchestrator(history, "student")
        orchestrator.initialize()
        orchestrator.step()
        return orchestrator.get_trajectory()

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
