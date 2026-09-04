import pytest
from dataclasses import dataclass
from types import SimpleNamespace

from coevo.curriculum import (
    ScenarioBand,
    classify_hinter_reachability,
    curriculum_weights,
    minimal_sufficient_level,
    probe_scenario,
)
from coevo.hints import HintLevel
from coevo.curriculum.hstar import ProbeResult
from scripts.run_dosage_experiment import collect_teacher_probe_results


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


def test_emergent_hinter_sensor_classifies_without_assigning_fixed_dose():
    decision = classify_hinter_reachability(0.2, 0.8, sufficient=0.7)
    assert decision.level is HintLevel.HINTER
    assert decision.band is ScenarioBand.FRONTIER
    mastered = classify_hinter_reachability(0.8, 0.9, sufficient=0.7)
    assert mastered.level is HintLevel.L0_NONE


def test_hstar_marks_nonmonotonic_measurements_and_scheduler_normalizes():
    decision = minimal_sufficient_level(scores(0.1, 0.8, 0.4, 0.9), sufficient=0.7)
    assert decision.level is HintLevel.L1_POLICY
    assert not decision.monotone
    weights = curriculum_weights({"task": decision})
    assert weights == {"task": pytest.approx(1.0)}


def test_hinted_probe_reuses_one_task_hint(monkeypatch):
    observed = []

    @dataclass(frozen=True)
    class Config:
        task_id: str = "1"
        hint_level: HintLevel = HintLevel.L0_NONE
        seed: int = 42

    class Orchestrator:
        def __init__(self):
            self.agent = SimpleNamespace(refresh_hint_each_turn=False)

        def run(self):
            observed.append(self.agent.refresh_hint_each_turn)
            return SimpleNamespace(reward_info=None)

    class Environment:
        def __init__(self, config):
            self.config = config

        @staticmethod
        def initial_history():
            return []

        @staticmethod
        def orchestrator(_history, _policy, seed):
            assert seed in {42, 43}
            return Orchestrator()

        @staticmethod
        def evaluate(_simulation):
            return SimpleNamespace(reward=1.0)

    monkeypatch.setattr("coevo.curriculum.hstar.Tau2Environment", Environment)
    result = probe_scenario(Config(), "1", HintLevel.L2_PROCEDURAL, k=2)
    assert result.successes == 2
    assert observed == [False, False]


def test_e2_records_teacher_success_for_every_hint_level():
    def fake_probe(_config, task_id, level, k):
        successes = {"L0_NONE": 1, "L1_POLICY": 2, "L2_PROCEDURAL": 3}[level.value]
        return ProbeResult(
            task_id,
            level,
            k,
            successes,
            tuple([1.0] * successes + [0.0] * (k - successes)),
        )

    levels = [
        HintLevel.L0_NONE,
        HintLevel.L1_POLICY,
        HintLevel.L2_PROCEDURAL,
    ]
    results, summary = collect_teacher_probe_results(
        object(), ["a", "b"], levels, 4, probe_fn=fake_probe
    )
    assert results["a"]["L2_PROCEDURAL"]["success_rate"] == 0.75
    assert summary["L1_POLICY"]["mean_success_rate"] == 0.5
    assert summary["L2_PROCEDURAL"]["tasks"] == 2


def test_hinted_probe_excludes_trials_with_hinter_errors(monkeypatch):
    @dataclass(frozen=True)
    class Config:
        task_id: str = "1"
        hint_level: HintLevel = HintLevel.L0_NONE
        seed: int = 42

    class Orchestrator:
        agent = SimpleNamespace(
            refresh_hint_each_turn=False,
            hint_records=[{"error": {"message": "invalid hint"}}],
        )

        @staticmethod
        def run():
            return SimpleNamespace(reward_info=None)

    class Environment:
        def __init__(self, config):
            self.config = config

        @staticmethod
        def initial_history():
            return []

        @staticmethod
        def orchestrator(_history, _policy, seed):
            return Orchestrator()

        @staticmethod
        def evaluate(_simulation):
            return SimpleNamespace(reward=1.0)

    monkeypatch.setattr("coevo.curriculum.hstar.Tau2Environment", Environment)
    result = probe_scenario(Config(), "1", HintLevel.L2_PROCEDURAL, k=1)
    assert result.hint_error_trials == 1
    assert result.valid_trials == 0
    assert result.successes == 0
    assert result.success_rate is None
    assert result.pass_at_k is None


def test_hstar_skips_unmeasured_hint_level_instead_of_treating_it_as_zero():
    unmeasured = ProbeResult(
        "task",
        HintLevel.L1_POLICY,
        8,
        0,
        (0.0,) * 8,
        hint_error_trials=5,
    )
    result = minimal_sufficient_level(
        {
            HintLevel.L0_NONE: 0.1,
            HintLevel.L1_POLICY: unmeasured,
            HintLevel.L2_PROCEDURAL: 0.8,
            HintLevel.L3_ORACLE: 0.9,
        },
        sufficient=0.7,
    )
    assert result.level is HintLevel.L2_PROCEDURAL

    unresolved = minimal_sufficient_level(
        {
            HintLevel.L0_NONE: 0.1,
            HintLevel.L1_POLICY: unmeasured,
            HintLevel.L2_PROCEDURAL: 0.2,
            HintLevel.L3_ORACLE: 0.3,
        },
        sufficient=0.7,
    )
    assert unresolved.level is None
    assert unresolved.band is ScenarioBand.UNMEASURED
