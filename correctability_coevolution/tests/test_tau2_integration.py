from coevo.config import InfraConfig, ModelEndpoint
from coevo.environment import Tau2Environment
from tau2.data_model.message import (
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.evaluator import evaluator_nl_assertions
from tau2.evaluator.evaluator import EvaluationType
from tau2.run import get_tasks

from coevo.environment import tau2 as tau2_environment_module
from coevo.environment.tau2 import dump_messages, load_messages


def test_tau2_environment_loads_data_and_separates_role_information():
    endpoint = ModelEndpoint("unused", "http://127.0.0.1:1")
    environment = Tau2Environment(
        InfraConfig(
            teacher=endpoint,
            student=endpoint,
            buyer_reference=endpoint,
            domain="airline",
            task_split="train",
            task_id="1",
        )
    )
    raw_environment = environment.fresh_environment()
    student = environment.policies.student(raw_environment)
    teacher = environment.policies.teacher(raw_environment, environment.task)
    buyer = environment.policies.buyer_reference(raw_environment, environment.task)

    scenario = str(environment.task.user_scenario)
    privileged_resolution = teacher.make_agent_instructions_from_actions()

    assert raw_environment.get_tools()
    assert scenario in buyer.system_prompt
    assert scenario not in student.system_prompt
    assert privileged_resolution in teacher.system_prompt
    assert privileged_resolution not in student.system_prompt


def test_tau2_user_tool_call_runs_after_replaying_completed_history():
    endpoint = ModelEndpoint("unused", "http://127.0.0.1:1")
    task_id = get_tasks("telecom", task_split_name="train")[0].id
    environment = Tau2Environment(
        InfraConfig(
            teacher=endpoint,
            student=endpoint,
            buyer_reference=endpoint,
            domain="telecom",
            task_split="train",
            task_id=task_id,
        )
    )
    history = [
        *environment.initial_history(),
        UserMessage(
            role="user",
            tool_calls=[
                ToolCall(
                    id="user-status",
                    name="check_status_bar",
                    arguments={},
                    requestor="user",
                )
            ],
        ),
    ]

    result = environment.execute_user_tools(history)

    assert result[:-1] == history
    assert isinstance(result[-1], ToolMessage)
    assert result[-1].requestor == "user"
    assert result[-1].error is False


def test_tau2_v1_official_train_and_test_splits_are_disjoint():
    expected_counts = {
        "airline": (30, 20),
        "retail": (74, 40),
        "telecom": (74, 40),
    }
    for domain, (train_count, test_count) in expected_counts.items():
        train_ids = {task.id for task in get_tasks(domain, task_split_name="train")}
        test_ids = {task.id for task in get_tasks(domain, task_split_name="test")}
        assert len(train_ids) == train_count
        assert len(test_ids) == test_count
        assert train_ids.isdisjoint(test_ids)


def test_tau2_v1_multi_tool_message_round_trips():
    message = MultiToolMessage(
        role="tool",
        tool_messages=[
            ToolMessage(
                id="call-1",
                role="tool",
                content="ok",
                requestor="assistant",
            ),
            ToolMessage(
                id="call-2",
                role="tool",
                content="failed",
                requestor="assistant",
                error=True,
            ),
        ],
    )

    restored = load_messages(dump_messages([message]))[0]

    assert isinstance(restored, MultiToolMessage)
    assert len(restored.tool_messages) == 2
    assert restored.tool_messages[1].error is True


def test_tau2_v1_nl_assertions_use_configured_fixed_judge(monkeypatch):
    endpoint = ModelEndpoint("unused", "http://127.0.0.1:1")
    judge = ModelEndpoint(
        "gemini-3.1-pro-preview",
        "http://127.0.0.1:9000",
        "runtime-secret",
    )
    environment = Tau2Environment(
        InfraConfig(
            teacher=endpoint,
            student=endpoint,
            buyer_reference=endpoint,
            nl_judge=judge,
            nl_judge_max_tokens=777,
            domain="retail",
            task_split="train",
            task_id="0",
        )
    )
    previous_model = evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS
    previous_args = evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS
    reward = object()

    def fake_evaluate_simulation(**kwargs):
        assert kwargs["evaluation_type"] == EvaluationType.ALL_WITH_NL_ASSERTIONS
        assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS == (
            "hosted_vllm/gemini-3.1-pro-preview"
        )
        assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS["api_base"] == (
            "http://127.0.0.1:9000/v1"
        )
        assert (
            evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS["max_tokens"] == 777
        )
        assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS["api_key"] == (
            "runtime-secret"
        )
        return reward

    monkeypatch.setattr(
        tau2_environment_module, "evaluate_simulation", fake_evaluate_simulation
    )

    assert environment.evaluate(object()) is reward
    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS == previous_model
    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS is previous_args


def test_unknown_tool_in_continuation_is_scored_invalid(monkeypatch):
    class Simulation:
        reward_info = None

    class Orchestrator:
        @staticmethod
        def run():
            return Simulation()

    environment = object.__new__(Tau2Environment)
    monkeypatch.setattr(
        environment,
        "orchestrator",
        lambda history, policy, seed=None: Orchestrator(),
    )
    monkeypatch.setattr(
        environment,
        "evaluate",
        lambda simulation: (_ for _ in ()).throw(
            ValueError("Unknown tool 'hallucinated_tool' encountered during replay")
        ),
    )

    simulation = environment.continue_to_terminal([], "teacher")

    assert simulation.reward_info.reward == 0.0
    assert simulation.reward_info.info["invalid_continuation"] is True
