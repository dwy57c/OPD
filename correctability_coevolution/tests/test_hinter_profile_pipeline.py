from scripts.filter_hinter_grpo_dataset import filter_and_reprofile


def test_grpo_rows_are_reprofiled_from_current_dosage_manifest():
    rows = [
        {
            "scenario_id": "task",
            "public_state": [{"role": "user", "content": "help"}],
            "privileged_context": {
                "domain_policy": "policy",
                "authoritative_oracle_steps": "oracle",
            },
            "student_profile": {
                "checkpoint": "/stale/path",
                "round_index": 2,
            },
            "messages": [],
        }
    ]
    result = filter_and_reprofile(
        rows,
        {"task"},
        {
            "task": {
                "no_hint_score": 0.26,
                "band": "frontier",
            }
        },
    )
    assert result[0]["student_profile"] == {
        "unhinted_success": 0.3,
        "curriculum_band": "frontier",
    }
    assert "/stale/path" not in result[0]["messages"][1]["content"]
