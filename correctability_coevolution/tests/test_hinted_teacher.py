from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.run import get_tasks

from coevo.config import HintEndpoint, InfraConfig, ModelEndpoint
from coevo.models import hinted_teacher as hinted_module
from coevo.models.hinted_teacher import HintedTeacherAgent, TeacherHintResult


class FakeHinter:
    def __init__(self):
        self.payloads = []

    def hint(self, payload):
        self.payloads.append(payload)
        return TeacherHintResult(
            hint={
                "current_state": "Need the user's confirmation.",
                "latest_user_intent": "Cancel the reservation.",
                "completed_or_obsolete_steps": [],
                "next_step": "Ask for explicit confirmation.",
                "remaining_steps": ["Call the oracle cancellation tool."],
                "policy_checks": ["Do not mutate data before confirmation."],
            },
            model="gemini-3.1-pro-preview",
            latency_ms=12,
            sha256="abc123",
        )


def test_infra_has_one_policy_endpoint_for_student_and_teacher():
    policy = ModelEndpoint("Qwen3-4B", "http://policy")
    config = InfraConfig(
        policy=policy,
        buyer_reference=ModelEndpoint("buyer", "http://buyer"),
        teacher_hint_mode="none",
    )

    assert config.student is policy
    assert config.teacher is policy


def test_teacher_is_shared_policy_with_private_hint_only(monkeypatch):
    task = get_tasks("airline", task_split_name="train", task_ids=["1"])[0]
    hinter = FakeHinter()
    endpoint = HintEndpoint("gemini-3.1-pro-preview", "http://hinter/v1", "key")
    agent = HintedTeacherAgent(
        tools=[],
        domain_policy="Follow the airline policy.",
        task=task,
        llm="hosted_vllm/Qwen3-32B",
        llm_args={},
        hinter_endpoint=endpoint,
        hinter=hinter,
    )
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return AssistantMessage(role="assistant", content="Please confirm.")

    monkeypatch.setattr(hinted_module, "generate", fake_generate, raising=False)
    # LLMAgent imports generate in tau2's module, so patch the exact call site.
    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate)
    state = agent.get_init_state([])
    response, state = agent.generate_next_message(
        UserMessage(role="user", content="Please cancel reservation ABC123."), state
    )

    assert response.content == "Please confirm."
    assert hinter.payloads[0]["current_history"][-1]["content"] == (
        "Please cancel reservation ABC123."
    )
    system = captured["messages"][0].content
    assert "<private_teacher_hint>" in system
    assert "Ask for explicit confirmation." in system
    assert "<oracle_reference>" not in system
    assert "<resolution_steps>" not in system
    assert agent.hint_records[0]["latency_ms"] == 12
