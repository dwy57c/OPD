from coevo.audit import counterfactual_invariance


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
