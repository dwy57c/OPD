from coevo.audit import audit_alfworld_behavior, privilege_from_agentgym_eto_record


def test_agentgym_eto_privilege_combines_expert_trace_and_hidden_state():
    result = privilege_from_agentgym_eto_record(
        {
            "expert_actions": ["go to coffeemachine 1", "take mug 1"],
            "hidden_state": {
                "goal_object_locations": {"mug": "coffeemachine 1"},
                "destination_receptacle": "cabinet 4",
                "unobserved_states": {"mug": "cold"},
            },
        }
    )
    payload = result.hint_payload()
    assert payload["goal_object_locations"] == {"mug": "coffeemachine 1"}
    assert payload["destination_receptacle"] == "cabinet 4"
    assert "go to coffeemachine 1" in payload["authoritative_oracle_steps"]
    assert result.oracle_reference_actions()[0] == {
        "role": "assistant",
        "content": "go to coffeemachine 1",
    }


def test_alfworld_behavior_audit_measures_direct_hit_and_query_before_pickup():
    leaked = audit_alfworld_behavior(
        ["go to coffeemachine 1", "take mug 1"],
        true_goal_locations=["coffeemachine 1"],
    )
    exploring = audit_alfworld_behavior(
        ["look", "open cabinet 1", "go to coffeemachine 1", "take mug 1"],
        true_goal_locations=["coffeemachine 1"],
    )
    assert leaked.direct_location_hit
    assert not leaked.queried_before_pickup
    assert exploring.direct_location_hit
    assert exploring.queried_before_pickup
