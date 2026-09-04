from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class PassKSnapshot:
    scores: Mapping[str, float]
    k: int

    def __post_init__(self) -> None:
        if self.k < 1 or not self.scores:
            raise ValueError("pass@k snapshot requires k > 0 and at least one scenario")
        if any(not 0 <= float(value) <= 1 for value in self.scores.values()):
            raise ValueError("pass@k scores must be in [0, 1]")

    @property
    def mean(self) -> float:
        return sum(float(value) for value in self.scores.values()) / len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {"k": self.k, "scores": dict(self.scores), "mean": self.mean}


@dataclass(frozen=True)
class AcceptanceRule:
    """Statistically calibrated rollback for the fixed pass@k panel."""

    mean_tolerance: float | None = None
    mean_std_multiplier: float = 1.65
    per_scenario_drop: float = 0.25
    max_regressed_fraction: float = 0.35

    def __post_init__(self) -> None:
        if self.mean_tolerance is not None and self.mean_tolerance < 0:
            raise ValueError("mean tolerance must be non-negative")
        if self.mean_std_multiplier <= 0:
            raise ValueError("mean_std_multiplier must be positive")
        if not 0 < self.per_scenario_drop <= 1:
            raise ValueError("per_scenario_drop must be in (0, 1]")
        if not 0 < self.max_regressed_fraction <= 1:
            raise ValueError("max_regressed_fraction must be in (0, 1]")

    def _mean_tolerance(self, baseline: PassKSnapshot) -> float:
        if self.mean_tolerance is not None:
            return self.mean_tolerance
        rate = min(max(baseline.mean, 1e-6), 1.0 - 1e-6)
        trials = len(baseline.scores) * baseline.k
        return self.mean_std_multiplier * (rate * (1.0 - rate) / trials) ** 0.5

    def regressions(
        self, baseline: PassKSnapshot, measured: PassKSnapshot
    ) -> tuple[str, ...]:
        if baseline.k != measured.k or set(baseline.scores) != set(measured.scores):
            raise ValueError("hinter acceptance requires the identical pass@k panel")
        failures = []
        tolerance = self._mean_tolerance(baseline)
        if measured.mean < baseline.mean - tolerance:
            failures.append(f"mean_pass_at_k:-{baseline.mean - measured.mean:.4f}")
        regressed = [
            scenario_id
            for scenario_id in baseline.scores
            if float(measured.scores[scenario_id])
            <= float(baseline.scores[scenario_id]) - self.per_scenario_drop
        ]
        fraction = len(regressed) / len(baseline.scores)
        if fraction > self.max_regressed_fraction:
            failures.append(
                f"scenario_fraction:{fraction:.2f}>{self.max_regressed_fraction:.2f}"
            )
        return tuple(failures)


@dataclass(frozen=True)
class AlternatingRoundResult:
    round_index: int
    student_before: str
    student_candidate: str
    student_after: str
    hinter_under_test: str
    fallback_hinter: str
    accepted_hinter: str
    next_hinter_candidate: str
    student_steps: int
    hinter_grpo_steps: int
    curriculum_scenarios: tuple[str, ...]
    measured_pass_at_k: PassKSnapshot
    acceptance_baseline: PassKSnapshot | None
    measured_distillation_gain: float | None
    prior_hinter_rolled_back: bool
    rollback_reasons: tuple[str, ...]
    pass_measurements_this_round: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["curriculum_scenarios"] = list(self.curriculum_scenarios)
        value["measured_pass_at_k"] = self.measured_pass_at_k.to_dict()
        value["acceptance_baseline"] = (
            self.acceptance_baseline.to_dict()
            if self.acceptance_baseline is not None
            else None
        )
        value["rollback_reasons"] = list(self.rollback_reasons)
        return value


class AlternatingHinterLoop:
    """Student segment, one pass@k fuse, then analytical-reward hinter GRPO."""

    def __init__(
        self,
        *,
        train_student: Callable[[str, str, int], str],
        measure_pass_at_k: Callable[[str, Mapping[str, Any], int], PassKSnapshot],
        schedule_curriculum: Callable[
            [PassKSnapshot, Mapping[str, Any]], Mapping[str, Any]
        ],
        train_hinter_grpo: Callable[[str, str, Mapping[str, Any], int], str],
        rollback_student: Callable[[str, str], None],
        rollback_hinter: Callable[[str, str], None],
        acceptance: AcceptanceRule | None = None,
    ):
        self.train_student = train_student
        self.measure_pass_at_k = measure_pass_at_k
        self.schedule_curriculum = schedule_curriculum
        self.train_hinter_grpo = train_hinter_grpo
        self.rollback_student = rollback_student
        self.rollback_hinter = rollback_hinter
        self.acceptance = acceptance or AcceptanceRule()

    def run_round(
        self,
        *,
        round_index: int,
        student_checkpoint: str,
        hinter_under_test: str,
        fallback_hinter_checkpoint: str,
        scenario_pool: Mapping[str, Any],
        acceptance_baseline: PassKSnapshot | None,
        student_steps: int,
        hinter_grpo_steps: int,
        pass_k: int = 8,
    ) -> AlternatingRoundResult:
        if student_steps < 1 or hinter_grpo_steps < 1:
            raise ValueError("Student and hinter step counts must be positive")
        if not scenario_pool:
            raise ValueError("scenario pool must be non-empty")

        student_candidate = self.train_student(
            student_checkpoint, hinter_under_test, student_steps
        )
        measured = self.measure_pass_at_k(student_candidate, scenario_pool, pass_k)
        regressions = (
            self.acceptance.regressions(acceptance_baseline, measured)
            if acceptance_baseline is not None
            else ()
        )
        measured_gain = (
            measured.mean - acceptance_baseline.mean
            if acceptance_baseline is not None
            else None
        )

        rolled_back = bool(regressions)
        if rolled_back:
            self.rollback_student(student_candidate, student_checkpoint)
            self.rollback_hinter(hinter_under_test, fallback_hinter_checkpoint)
            student_after = student_checkpoint
            accepted_hinter = fallback_hinter_checkpoint
            scheduling_snapshot = acceptance_baseline
        else:
            student_after = student_candidate
            accepted_hinter = hinter_under_test
            scheduling_snapshot = measured

        curriculum = self.schedule_curriculum(scheduling_snapshot, scenario_pool)
        if not curriculum:
            raise ValueError("pass@k curriculum selected no scenarios")
        next_hinter_candidate = self.train_hinter_grpo(
            student_after,
            accepted_hinter,
            curriculum,
            hinter_grpo_steps,
        )
        if not next_hinter_candidate or next_hinter_candidate == accepted_hinter:
            raise ValueError("hinter GRPO must produce a distinct candidate checkpoint")

        return AlternatingRoundResult(
            round_index=round_index,
            student_before=student_checkpoint,
            student_candidate=student_candidate,
            student_after=student_after,
            hinter_under_test=hinter_under_test,
            fallback_hinter=fallback_hinter_checkpoint,
            accepted_hinter=accepted_hinter,
            next_hinter_candidate=next_hinter_candidate,
            student_steps=student_steps,
            hinter_grpo_steps=hinter_grpo_steps,
            curriculum_scenarios=tuple(str(value) for value in curriculum),
            measured_pass_at_k=measured,
            acceptance_baseline=acceptance_baseline,
            measured_distillation_gain=measured_gain,
            prior_hinter_rolled_back=rolled_back,
            rollback_reasons=regressions,
        )
