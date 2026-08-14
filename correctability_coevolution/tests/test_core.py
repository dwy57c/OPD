from types import SimpleNamespace

from tau2.data_model.message import AssistantMessage, UserMessage

from coevo.environment.tau2 import dump_messages
from coevo.rollout.collector import NaturalDecisionCollector


def test_unlimited_collector_materializes_every_natural_teacher_target():
    history = [UserMessage(role="user", content="Start")]
    for index in range(5):
        history.extend(
            [
                AssistantMessage(role="assistant", content=f"Student action {index}."),
                UserMessage(role="user", content=f"Buyer action {index}."),
            ]
        )

    class Orchestrator:
        done = True
        step_count = 0

        def initialize(self):
            pass

        def get_trajectory(self):
            return history

    class Environment:
        config = SimpleNamespace(
            seed=42,
            branch_max_steps=24,
            domain="test",
            task_split="train",
        )
        task = SimpleNamespace(id="test")
        policies = SimpleNamespace(
            student=lambda environment: SimpleNamespace(
                system_prompt="policy",
                tools=[SimpleNamespace(openai_schema={"type": "function"})],
            )
        )

        @staticmethod
        def fresh_environment():
            return object()

        @staticmethod
        def initial_history():
            return history[:1]

        @staticmethod
        def orchestrator(initial_history, policy, seed):
            return Orchestrator()

    class Labeler:
        @staticmethod
        def score_decision(trajectory, message_index):
            action = trajectory[message_index]
            return {
                "message_index": message_index,
                "state_hash": f"state-{message_index}",
                "sample_hash": f"sample-{message_index}",
                "history_before": dump_messages(trajectory[:message_index]),
                "student_action": action.model_dump(mode="json"),
                "teacher_action": AssistantMessage(
                    role="assistant", content=f"Teacher action {message_index}."
                ).model_dump(mode="json"),
                "teacher_hint": {"hint": {"plan": "correct it"}},
            }

    class Target:
        raw_teacher_target_hash = "raw"
        teacher_target_hash = "sharp"
        teacher_action_hash = "action"
        target_loss_mask = (1,)

        @staticmethod
        def to_dict():
            return {"target": "record"}

    class Builder:
        cache_stats = {"hits": 0, "misses": 5, "scoring_failures": 0}

        @staticmethod
        def build(**kwargs):
            assert kwargs["tool_schemas"] == [{"type": "function"}]
            return Target()

    record = NaturalDecisionCollector(
        Environment(), Labeler(), max_decisions=0, target_builder=Builder()
    ).collect_one()

    assert len(record["student_decisions"]) == 5
    assert all(turn["student_eligible"] for turn in record["student_decisions"])
    assert all("teacher_quality" not in turn for turn in record["student_decisions"])


def test_invalid_target_is_preserved_for_audit_instead_of_training():
    class Builder:
        @staticmethod
        def build(**kwargs):
            raise ValueError("token mismatch")

    collector = object.__new__(NaturalDecisionCollector)
    collector._target_builder = Builder()
    collector._student_system_prompt = "policy"
    collector._student_tool_schemas = []
    collector.environment = SimpleNamespace()
    decision = {
        "state_hash": "state",
        "history_before": dump_messages([UserMessage(role="user", content="help")]),
        "teacher_action": AssistantMessage(
            role="assistant", content="answer"
        ).model_dump(mode="json"),
        "teacher_hint": None,
    }

    result = collector._materialize_target(decision)

    assert result["student_eligible"] is False
    assert "token mismatch" in result["student_rejection_reason"]
