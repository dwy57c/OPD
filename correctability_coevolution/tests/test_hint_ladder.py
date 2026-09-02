import pytest
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.run import get_tasks

from coevo.config import HintEndpoint
from coevo.hints import (
    HintLevel,
    hint_fact_leaks,
    hint_instruction,
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


def alfworld_payload():
    return {
        "domain": "alfworld",
        "available_tools": [],
        "authoritative_oracle_steps": "Take mug from coffeemachine 1 and use cabinet 4.",
        "goal_object_locations": {"mug": "coffeemachine 1"},
        "destination_receptacle": "cabinet 4",
        "unobserved_states": {"mug": "cold"},
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


def test_alfworld_l2_rejects_direct_and_hedged_structured_facts():
    leaked = (
        "The agent has only the room overview and no inventory yet. The next step "
        "is to head toward the coffee machine area and observe whether a mug is "
        "present there, since that is the most plausible spot; if not, inspect "
        "other receptacles one at a time and confirm by observation before acting."
    )
    wrapped = (
        "Mugs are most plausibly at the coffee machine, so begin there, verify the "
        "object by observation, keep track of inventory, and then perform the "
        "required state-changing step before placing it at the destination."
    )
    state_leak = (
        "The mug has not been observed yet, but it is cold. Search generic storage "
        "locations without prioritizing one, confirm the object by observation, "
        "track inventory, and only then perform the required state-changing step."
    )
    for value in (leaked, wrapped):
        with pytest.raises(ValueError, match="structured_fact:coffee machine"):
            validate_hint_note(value, alfworld_payload(), HintLevel.L2_PROCEDURAL)
    with pytest.raises(ValueError, match="structured_fact:mug is cold"):
        validate_hint_note(
            state_leak,
            alfworld_payload(),
            HintLevel.L2_PROCEDURAL,
        )


def test_alfworld_l2_accepts_uninformed_exploration_procedure():
    compliant = (
        "Inspect the generic kinds of receptacles where the target category may be "
        "kept, in no privileged order, checking them one at a time until the object "
        "is visibly confirmed. Track what is held, then perform the task-required "
        "state change with the appropriate appliance. Finally inspect destination "
        "instances without prioritizing one, and place the object only after the "
        "required state and destination are confirmed."
    )
    validate_hint_note(compliant, alfworld_payload(), HintLevel.L2_PROCEDURAL)


def test_alfworld_l3_must_state_every_structured_fact():
    omitted = (
        "Use the privileged plan to move efficiently through the room, retrieve the "
        "goal object, change its state with the appropriate appliance, and place it "
        "in the correct destination. The next move should begin the efficient route "
        "rather than exploring unrelated receptacles, while keeping inventory and "
        "object state current throughout the task."
    )
    with pytest.raises(ValueError, match="omitted required structured fact"):
        validate_hint_note(omitted, alfworld_payload(), HintLevel.L3_ORACLE)

    explicit = (
        "The mug is currently at coffee machine 1, and the mug is cold even though "
        "the goal requires it to be hot. The correct destination is cabinet 4. The "
        "efficient next move is to approach the coffee machine because that directly "
        "reaches the known goal object instead of spending turns searching unrelated "
        "receptacles. After taking it, use the appliance required for heating, verify "
        "that it is hot, and then place it in cabinet 4 while tracking inventory and "
        "the observed state after every action."
    )
    validate_hint_note(explicit, alfworld_payload(), HintLevel.L3_ORACLE)

    pronoun = (
        "The mug currently sits on coffee machine 1, and it is still cold, which "
        "has not been observed yet. The correct destination is cabinet 4. The "
        "efficient next move is to approach the coffee machine and retrieve the "
        "mug because its location is already known. Heat it with the appropriate "
        "appliance, confirm the new state, and place it in cabinet 4."
    )
    validate_hint_note(pronoun, alfworld_payload(), HintLevel.L3_ORACLE)


def test_l1_payload_removes_all_structured_privilege():
    prepared = prepare_hint_payload(alfworld_payload(), HintLevel.L1_POLICY)
    for key in (
        "authoritative_oracle_steps",
        "goal_object_locations",
        "destination_receptacle",
        "unobserved_states",
    ):
        assert key not in prepared


def test_domain_slot_changes_the_prompt_and_l3_requires_disclosure():
    l2 = hint_instruction(HintLevel.L2_PROCEDURAL, "alfworld")
    l3 = hint_instruction(HintLevel.L3_ORACLE, "alfworld")
    normalized_l2 = " ".join(l2.split())
    normalized_l3 = " ".join(l3.split())
    assert "which receptacle currently holds" in normalized_l2
    assert "most plausible spot" in normalized_l2
    assert "MUST use it explicitly" in normalized_l3
    assert "Do not withhold a fact" in normalized_l3


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
