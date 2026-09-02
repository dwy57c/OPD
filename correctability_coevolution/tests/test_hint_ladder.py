import pytest
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.run import get_tasks

from coevo.config import HintEndpoint
from coevo.hints import (
    HintLevel,
    hint_fact_leaks,
    prepare_hint_payload,
    validate_hint_note,
)
from coevo.models.hinted_teacher import HintedTeacherAgent


def payload():
    return {
        "available_tools": [
            {"type": "function", "function": {"name": "get_booking_details"}}
        ],
        "authoritative_oracle_steps": "[Step 1] get_booking_details ABC123 on 2027-05-03 for $75",
    }


def test_l2_rejects_instance_values_and_accepts_fact_to_procedure():
    valid = (
        "The applicable record should be retrieved from the authorized source before "
        "stating any charge. If the customer has not supplied the lookup key, ask for "
        "it, then confirm the result before proposing the next action."
    )
    validate_hint_note(valid, payload(), HintLevel.L2_PROCEDURAL)
    for invalid in (
        "Retrieve booking ABC123 and state that the fee is $75.",
        "The relevant travel date is 2027-05-03, so proceed after confirmation.",
        "Call get_booking_details before answering the customer.",
    ):
        with pytest.raises(ValueError):
            validate_hint_note(invalid, payload(), HintLevel.L2_PROCEDURAL)


def test_public_fact_audit_is_reusable_by_grpo_reward():
    findings = hint_fact_leaks(
        "Use get_booking_details for ABC123 on 2027-05-03 and quote $75.",
        payload(),
    )
    assert "date" in findings
    assert "amount" in findings
    assert "identifier" in findings
    assert "tool_name:get_booking_details" in findings
    assert any(value.startswith("oracle_literal:") for value in findings)
    assert hint_fact_leaks(
        "Ask for the lookup key, retrieve the record, and confirm the result.",
        payload(),
    ) == ()


@pytest.mark.parametrize(
    "text",
    [
        "Ask for the order number before searching.",
        "Request the booking ID and confirm it with the customer.",
        "The account number must be obtained from the user.",
    ],
)
def test_identifier_audit_allows_slot_names_without_values(text):
    assert "identifier" not in hint_fact_leaks(text, payload())


@pytest.mark.parametrize(
    "text",
    [
        "Use order number ABC123 before searching.",
        "The booking ID is ZX-99881.",
        "Use confirmation number 123456.",
    ],
)
def test_identifier_audit_still_rejects_concrete_values(text):
    assert "identifier" in hint_fact_leaks(text, payload())


def test_l1_has_strict_length_and_never_receives_oracle_steps():
    prepared = prepare_hint_payload(payload(), HintLevel.L1_POLICY)
    assert "authoritative_oracle_steps" not in prepared
    validate_hint_note(
        "Verify customer-specific facts through public evidence before making a claim, "
        "and request any missing lookup key rather than guessing the answer.",
        prepared,
        HintLevel.L1_POLICY,
    )
    with pytest.raises(ValueError, match="15-40"):
        validate_hint_note("Ask before acting.", prepared, HintLevel.L1_POLICY)


def test_hint_levels_enforce_channel_capacity():
    with pytest.raises(ValueError, match="100 words"):
        validate_hint_note(
            " ".join(["procedure"] * 101) + ".",
            {"available_tools": []},
            HintLevel.L2_PROCEDURAL,
        )
    with pytest.raises(ValueError, match="140 words"):
        validate_hint_note(
            " ".join(["oracle"] * 141) + ".",
            {"available_tools": []},
            HintLevel.L3_ORACLE,
        )


def test_l0_agent_does_not_call_hinter(monkeypatch):
    task = get_tasks("airline", task_split_name="train", task_ids=["1"])[0]

    class ExplodingHinter:
        def hint(self, *_args, **_kwargs):
            raise AssertionError("L0 called the hinter")

    agent = HintedTeacherAgent(
        tools=[],
        domain_policy="Follow policy.",
        task=task,
        llm="hosted_vllm/Qwen3-4B",
        llm_args={},
        hinter_endpoint=HintEndpoint("unused", "http://unused/v1", "key"),
        hinter=ExplodingHinter(),
        hint_level=HintLevel.L0_NONE,
    )

    def fake_generate(**_kwargs):
        return AssistantMessage(role="assistant", content="How can I help?")

    monkeypatch.setattr("tau2.agent.llm_agent.generate", fake_generate)
    state = agent.get_init_state([])
    response, _ = agent.generate_next_message(
        UserMessage(role="user", content="I need help."), state
    )
    assert response.content == "How can I help?"
    assert agent.hint_records == []
