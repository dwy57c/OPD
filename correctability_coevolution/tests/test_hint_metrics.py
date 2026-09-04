from coevo.audit import aggregate_session_signals, counterfactual_invariance
from scripts.generate_hint_counterfactuals import validate_same_task_counterfactual


def row(level, hint):
    return {
        "state_id": "s",
        "hint_level": level,
        "hint": {"hint": {"plan": hint}},
        "hint_error": None,
    }


def test_counterfactual_invariance_pairs_same_state_and_level():
    result = counterfactual_invariance(
        [
            row("L2_PROCEDURAL", "Check every available receptacle."),
            row("L3_ORACLE", "The mug is at the coffee machine."),
        ],
        [
            row("L2_PROCEDURAL", "Check every available receptacle."),
            row("L3_ORACLE", "The mug is in cabinet four."),
        ],
    )
    assert result["levels"]["L2_PROCEDURAL"]["mean_similarity"] == 1.0
    assert result["levels"]["L3_ORACLE"]["mean_similarity"] < 0.8


def test_counterfactual_must_be_plausible_and_same_task():
    source = {
        "state_id": "task:0",
        "task_id": "task",
        "fact_audit_context": {"answer": "a"},
    }
    validate_same_task_counterfactual(
        source,
        {
            "state_id": "task:0",
            "task_id": "task",
            "plausible_alternative": True,
            "fact_audit_context": {"answer": "b"},
        },
    )


def test_session_signals_equal_weight_turns_then_sessions():
    rows = [
        {
            "session_id": "short",
            "analytical_signals": {"mean_lift": 1.0, "mean_copy": 1.0},
        },
        {
            "session_id": "long",
            "analytical_signals": {"mean_lift": 0.0, "mean_copy": 0.0},
        },
        {
            "session_id": "long",
            "analytical_signals": {"mean_lift": 0.0, "mean_copy": 0.0},
        },
    ]
    result = aggregate_session_signals(rows)
    assert result["sessions"] == 2
    assert result["scored_turns"] == 3
    assert result["mean_copy"] == 0.5
