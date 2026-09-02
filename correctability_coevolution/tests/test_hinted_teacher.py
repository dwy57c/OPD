from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.run import get_tasks

from coevo.config import HintEndpoint, InfraConfig, ModelEndpoint
from coevo.models import hinted_teacher as hinted_module
from coevo.models.hinted_teacher import (
    ClosedModelTeacherHinter,
    OpenModelTeacherHinter,
    HintedTeacherAgent,
    TeacherHintResult,
    _validate_natural_note,
    format_teacher_system_prompt_with_hint,
)
from types import SimpleNamespace
from coevo.hinter_prompt import build_hinter_messages


class FakeHinter:
    def __init__(self):
        self.payloads = []

    def hint(self, payload):
        self.payloads.append(payload)
        return TeacherHintResult(
            hint={"plan": "The requested cancellation is ready, but changing the "
            "reservation requires the user's explicit confirmation. Asking for that "
            "confirmation is the safest next direction; no booking data should change "
            "until the user clearly agrees."},
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
    agent.generate_next_message(
        UserMessage(role="user", content="Yes, I confirm."), state
    )

    assert response.content == "Please confirm."
    assert hinter.payloads[0]["current_history"][-1]["content"] == (
        "Please cancel reservation ABC123."
    )
    system = captured["messages"][0].content
    assert "<private_teacher_note>" in system
    assert "explicit confirmation" in system
    assert '"plan"' not in system
    assert "<oracle_reference>" not in system
    assert "<resolution_steps>" not in system
    assert agent.hint_records[0]["latency_ms"] == 12
    assert len(hinter.payloads) == 1


def test_teacher_can_refresh_private_hint_each_turn(monkeypatch):
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
        refresh_hint_each_turn=True,
    )

    def fake_generate(**_kwargs):
        return AssistantMessage(role="assistant", content="Please confirm.")

    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate)
    state = agent.get_init_state([])
    _, state = agent.generate_next_message(
        UserMessage(role="user", content="Please cancel reservation ABC123."), state
    )
    agent.generate_next_message(
        UserMessage(role="user", content="Yes, I confirm."), state
    )

    assert len(hinter.payloads) == 2
    assert len(agent.hint_records) == 2
    assert hinter.payloads[0]["current_history"][-1]["content"] == (
        "Please cancel reservation ABC123."
    )
    assert hinter.payloads[1]["current_history"][-1]["content"] == (
        "Yes, I confirm."
    )


def test_privileged_prompt_excludes_hint_audit_metadata_and_null_hint():
    prompt = format_teacher_system_prompt_with_hint(
        "policy",
        {
            "hint": {"plan": "Ask for confirmation."},
            "model": "audit-only-model",
            "latency_ms": 12,
            "sha256": "audit-only-hash",
        },
    )

    assert "Ask for confirmation." in prompt
    assert "audit-only-model" not in prompt
    assert "latency_ms" not in prompt
    assert "audit-only-hash" not in prompt
    assert "<private_teacher_note>" in prompt
    assert '"plan"' not in prompt
    assert format_teacher_system_prompt_with_hint(
        "policy", {"hint": None, "model": "audit-only-model"}
    ) == "policy"


def test_natural_note_validation_rejects_tool_names_and_truncation():
    payload = {
        "available_tools": [
            {"type": "function", "function": {"name": "cancel_reservation"}}
        ]
    }

    _validate_natural_note(
        "The reservation is eligible, but explicit confirmation is still missing. "
        "The safest direction is to ask the user to confirm before changing it.",
        payload,
    )

    for invalid in (
        "Call cancel_reservation after confirmation.",
        "The reservation is eligible, but confirmation is still",
        "NEXT ACTION\nAsk for confirmation.",
    ):
        try:
            _validate_natural_note(invalid, payload)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid natural note was accepted: {invalid}")


def test_closed_hinter_retries_validation_with_feedback_and_temperature(monkeypatch):
    endpoint = HintEndpoint(
        "teacher", "http://hinter/v1", "key", retries=2
    )
    hinter = ClosedModelTeacherHinter(endpoint)
    calls = []
    outputs = iter(
        [
            "Call get_booking_details now.",
            "Ask for the missing lookup key, retrieve the record through the "
            "authorized source, and confirm the result before proceeding.",
        ]
    )

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=next(outputs))
                )
            ]
        )

    hinter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr("coevo.models.hinted_teacher.time.sleep", lambda *_args: None)
    result = hinter.hint(
        {
            "available_tools": [
                {
                    "type": "function",
                    "function": {"name": "get_booking_details"},
                }
            ]
        },
        "L2_PROCEDURAL",
    )
    assert result.error is None
    assert calls[0]["temperature"] == 0
    assert calls[1]["temperature"] == 0.7
    assert "previous draft was rejected" in calls[1]["messages"][1]["content"]
    assert "get_booking_details" in calls[1]["messages"][1]["content"]


def test_closed_hinter_exhaustion_returns_auditable_error(monkeypatch):
    endpoint = HintEndpoint(
        "teacher", "http://hinter/v1", "key", retries=2
    )
    hinter = ClosedModelTeacherHinter(endpoint)
    hinter.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="Call get_booking_details now."
                            )
                        )
                    ]
                )
            )
        )
    )
    monkeypatch.setattr("coevo.models.hinted_teacher.time.sleep", lambda *_args: None)
    result = hinter.hint(
        {
            "available_tools": [
                {
                    "type": "function",
                    "function": {"name": "get_booking_details"},
                }
            ]
        },
        "L2_PROCEDURAL",
    )
    assert result.hint == {}
    assert result.error["attempts"] == 2


def test_open_hinter_uses_the_same_prompt_builder_as_grpo():
    endpoint = HintEndpoint("open-hinter", "http://hinter/v1", "key")
    hinter = OpenModelTeacherHinter(endpoint)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Ask for the missing booking number before looking up the record."
                    )
                )
            ]
        )

    hinter.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    payload = {
        "task_id": "1",
        "domain_policy": "policy",
        "available_tools": [],
        "authoritative_oracle_steps": "lookup ABC123",
        "current_history": [{"role": "user", "content": "help"}],
    }
    result = hinter.hint(payload, "L3_ORACLE")
    expected_privileged = {
        **{key: value for key, value in payload.items() if key != "current_history"},
        "hint_level": "L3_ORACLE",
    }
    assert calls[0]["messages"] == build_hinter_messages(
        payload["current_history"], expected_privileged
    )
    assert result.error is None
