import pytest

from coevo.curriculum import ScenarioBand, curriculum_weights, minimal_sufficient_level
from coevo.hints import HintLevel


def scores(l0, l1, l2, l3):
    return {
        HintLevel.L0_NONE: l0,
        HintLevel.L1_POLICY: l1,
        HintLevel.L2_PROCEDURAL: l2,
        HintLevel.L3_ORACLE: l3,
    }


def test_hstar_selects_smallest_sufficient_level_and_quadrant():
    frontier = minimal_sufficient_level(scores(0.25, 0.4, 0.75, 0.9), sufficient=0.7)
    assert frontier.level is HintLevel.L2_PROCEDURAL
    assert frontier.band is ScenarioBand.FRONTIER
    assert frontier.monotone

    scaffolded = minimal_sufficient_level(scores(0.0, 0.1, 0.2, 0.8), sufficient=0.7)
    assert scaffolded.level is HintLevel.L3_ORACLE
    assert scaffolded.band is ScenarioBand.SCAFFOLDED

    mastered = minimal_sufficient_level(scores(0.8, 0.8, 0.9, 1.0), sufficient=0.7)
    assert mastered.level is HintLevel.L0_NONE
    assert mastered.band is ScenarioBand.MASTERED


def test_hstar_marks_nonmonotonic_measurements_and_scheduler_normalizes():
    decision = minimal_sufficient_level(scores(0.1, 0.8, 0.4, 0.9), sufficient=0.7)
    assert decision.level is HintLevel.L1_POLICY
    assert not decision.monotone
    weights = curriculum_weights({"task": decision})
    assert weights == {"task": pytest.approx(1.0)}
