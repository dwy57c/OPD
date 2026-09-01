from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from coevo.artifacts import canonical_hash

from .behavior_discriminator import DiscriminatorControlReport, DiscriminatorGate
from .discriminator_data import (
    BehaviorHintSample,
    CopyingDiscriminatorPair,
    build_fresh_discriminator_pairs,
)


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
    """Statistically calibrated rollback for the pass@k panel."""

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
                f"scenario_fraction:{fraction:.2f}>"
                f"{self.max_regressed_fraction:.2f}"
            )
        return tuple(failures)


@dataclass(frozen=True)
class DiscriminatorUpdate:
    checkpoint: str
    round_index: int
    training_examples: int
    training_fingerprint: str
    converged: bool
    control_report: DiscriminatorControlReport
    initialized_from_student: bool = True
    fresh_score_head: bool = True

    def __post_init__(self) -> None:
        if (
            not self.checkpoint
            or self.training_examples < 1
            or not self.training_fingerprint
        ):
            raise ValueError("discriminator update must identify a trained checkpoint")


@dataclass(frozen=True)
class IndependentAuditResult:
    checkpoint: str
    control_report: DiscriminatorControlReport
    agreement_with_training_discriminator: float

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("independent auditor must identify its checkpoint")
        if not 0 <= self.agreement_with_training_discriminator <= 1:
            raise ValueError("independent-auditor agreement must be in [0, 1]")


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
    discriminator_after: str
    discriminator_control_report: DiscriminatorControlReport
    independent_auditor_checkpoint: str | None
    independent_auditor_agreement: float | None
    student_steps: int
    hinter_grpo_steps: int
    curriculum_scenarios: tuple[str, ...]
    measured_pass_at_k: PassKSnapshot
    acceptance_baseline: PassKSnapshot | None
    measured_distillation_gain: float | None
    prior_hinter_rolled_back: bool
    rollback_reasons: tuple[str, ...]
    fresh_discriminator_samples: int
    discriminator_training_examples: int
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
        value["discriminator_control_report"] = (
            self.discriminator_control_report.to_dict()
        )
        return value


class AlternatingHinterLoop:
    """One pass@k measurement, used for both acceptance and scheduling.

    The hinter produced at the end of round t is tested by the Student update at
    the start of round t+1. The post-Student pass@k measurement is therefore a
    real distillation outcome, not a reward proxy. GRPO receives only the local
    usefulness/copying/length reward registered in ``grpo_reward.py``.
    """

    def __init__(
        self,
        *,
        train_student: Callable[[str, str, int], str],
        measure_pass_at_k: Callable[[str, Mapping[str, Any], int], PassKSnapshot],
        schedule_curriculum: Callable[
            [PassKSnapshot, Mapping[str, Any]], Mapping[str, Any]
        ],
        collect_fresh_discriminator_samples: Callable[
            [str, str, Mapping[str, Any]], Sequence[BehaviorHintSample]
        ],
        retrain_discriminator: Callable[
            [Sequence[CopyingDiscriminatorPair], int], DiscriminatorUpdate
        ],
        train_independent_auditor: Callable[
            [Sequence[CopyingDiscriminatorPair], str, int], IndependentAuditResult
        ]
        | None,
        train_hinter_grpo: Callable[
            [str, str, str, Mapping[str, Any], int], str
        ],
        rollback_student: Callable[[str, str], None],
        rollback_hinter: Callable[[str, str], None],
        acceptance: AcceptanceRule | None = None,
        discriminator_gate: DiscriminatorGate | None = None,
        independent_audit_interval: int = 3,
        minimum_independent_agreement: float = 0.8,
    ):
        if independent_audit_interval < 1:
            raise ValueError("independent_audit_interval must be positive")
        self.train_student = train_student
        self.measure_pass_at_k = measure_pass_at_k
        self.schedule_curriculum = schedule_curriculum
        self.collect_fresh_discriminator_samples = (
            collect_fresh_discriminator_samples
        )
        self.retrain_discriminator = retrain_discriminator
        self.train_independent_auditor = train_independent_auditor
        self.train_hinter_grpo = train_hinter_grpo
        self.rollback_student = rollback_student
        self.rollback_hinter = rollback_hinter
        self.acceptance = acceptance or AcceptanceRule()
        self.discriminator_gate = discriminator_gate or DiscriminatorGate()
        self.independent_audit_interval = independent_audit_interval
        self.minimum_independent_agreement = minimum_independent_agreement

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
        measured = self.measure_pass_at_k(
            student_candidate, scenario_pool, pass_k
        )
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
            self.rollback_hinter(
                hinter_under_test, fallback_hinter_checkpoint
            )
            student_after = student_checkpoint
            accepted_hinter = fallback_hinter_checkpoint
            scheduling_snapshot = acceptance_baseline
        else:
            student_after = student_candidate
            accepted_hinter = hinter_under_test
            scheduling_snapshot = measured

        curriculum = self.schedule_curriculum(
            scheduling_snapshot, scenario_pool
        )
        if not curriculum:
            raise ValueError("pass@k curriculum selected no scenarios")

        fresh_samples = tuple(
            self.collect_fresh_discriminator_samples(
                student_after, accepted_hinter, curriculum
            )
        )
        if not fresh_samples:
            raise ValueError("each round must retrain the discriminator on fresh samples")
        discriminator_examples = tuple(
            build_fresh_discriminator_pairs(
                fresh_samples, seed=round_index
            )
        )
        discriminator_update = self.retrain_discriminator(
            discriminator_examples, round_index
        )
        expected_discriminator_fingerprint = canonical_hash(
            [example.to_dict() for example in discriminator_examples]
        )
        if (
            not discriminator_update.converged
            or discriminator_update.round_index != round_index
            or discriminator_update.training_examples != len(discriminator_examples)
            or discriminator_update.training_fingerprint
            != expected_discriminator_fingerprint
            or not discriminator_update.initialized_from_student
            or not discriminator_update.fresh_score_head
        ):
            raise ValueError(
                "discriminator must be freshly retrained to convergence on this round"
            )
        self.discriminator_gate.validate(discriminator_update.control_report)
        discriminator_after = discriminator_update.checkpoint
        independent_audit = None
        if round_index % self.independent_audit_interval == 0:
            if self.train_independent_auditor is None:
                raise ValueError(
                    "this round requires an independently initialized discriminator audit"
                )
            independent_audit = self.train_independent_auditor(
                discriminator_examples, discriminator_after, round_index
            )
            if independent_audit.checkpoint == discriminator_after:
                raise ValueError("independent auditor reused the training discriminator")
            self.discriminator_gate.validate(independent_audit.control_report)
            if (
                independent_audit.agreement_with_training_discriminator
                < self.minimum_independent_agreement
            ):
                raise ValueError("independent auditor disagrees with copying penalties")

        next_hinter_candidate = self.train_hinter_grpo(
            student_after,
            accepted_hinter,
            discriminator_after,
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
            discriminator_after=discriminator_after,
            discriminator_control_report=discriminator_update.control_report,
            independent_auditor_checkpoint=(
                independent_audit.checkpoint if independent_audit else None
            ),
            independent_auditor_agreement=(
                independent_audit.agreement_with_training_discriminator
                if independent_audit
                else None
            ),
            student_steps=student_steps,
            hinter_grpo_steps=hinter_grpo_steps,
            curriculum_scenarios=tuple(str(value) for value in curriculum),
            measured_pass_at_k=measured,
            acceptance_baseline=acceptance_baseline,
            measured_distillation_gain=measured_gain,
            prior_hinter_rolled_back=rolled_back,
            rollback_reasons=regressions,
            fresh_discriminator_samples=len(fresh_samples),
            discriminator_training_examples=len(discriminator_examples),
        )
