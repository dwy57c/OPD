from scripts.audit_hint_ladder import summarize_level_rows


def row(error=None, clarification=0.5):
    return {
        "hint_error": error,
        "hint_words": 20,
        "behavior": {
            "clarification_rate": clarification,
            "lookup_rate": 0.25,
            "ungrounded_assertion_rate": 0.1,
        },
    }


def test_hint_errors_are_reported_but_excluded_from_behavior_means():
    summary = summarize_level_rows(
        [row(), row({"message": "invalid"}, clarification=1.0)]
    )
    assert summary["rows"] == 2
    assert summary["valid_rows"] == 1
    assert summary["hint_error_rows"] == 1
    assert summary["contract_violation_rate"] == 0.5
    assert summary["clarification_rate"] == 0.5
