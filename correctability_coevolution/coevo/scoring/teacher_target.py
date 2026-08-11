from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from coevo.artifacts import assistant_action_hash, canonical_hash
from coevo.intervention.decision_state import DecisionState
from coevo.intervention.teacher_action import TeacherActionGenerator, TeacherActionResult
from coevo.rewards.tau2_soft_score import SoftScoreResult, soft_completion_score


TEACHER_TARGET_SCHEMA_VERSION = 2


def _as_tuple_rows(value: Sequence[Sequence], cast):
    return tuple(tuple(cast(item) for item in row) for row in value)


@dataclass(frozen=True)
class TeacherTargetRecord:
    schema_version: int
    state_hash: str
    teacher_action_hash: str
    raw_teacher_target_hash: str
    teacher_target_hash: str
    teacher_checkpoint: str
    teacher_hint_hash: str
    student_visible_messages: tuple[dict, ...]
    hinted_teacher_messages: tuple[dict, ...]
    teacher_action: dict
    target_token_ids: tuple[int, ...]
    target_loss_mask: tuple[int, ...]
    hinted_topk_logprobs: tuple[tuple[float, ...], ...]
    hinted_topk_token_ids: tuple[tuple[int, ...], ...]
    hinted_support_mass: tuple[float, ...]
    unhinted_reference_checkpoint: str
    unhinted_reference_topk_logprobs: tuple[tuple[float, ...], ...]
    unhinted_reference_topk_token_ids: tuple[tuple[int, ...], ...]
    unhinted_reference_support_mass: tuple[float, ...]
    skill_contrast_scores: tuple[float, ...]
    skill_gate_values: tuple[float, ...]
    sharpening_temperatures: tuple[float, ...]
    sharpened_topk_logprobs: tuple[tuple[float, ...], ...]
    sharpened_topk_token_ids: tuple[tuple[int, ...], ...]
    sharpened_support_mass: tuple[float, ...]
    raw_teacher_entropy: tuple[float, ...]
    sharpened_teacher_entropy: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.schema_version != TEACHER_TARGET_SCHEMA_VERSION:
            raise ValueError(
                f"TeacherTargetRecord schema_version must be {TEACHER_TARGET_SCHEMA_VERSION}"
            )
        if not self.state_hash or not self.teacher_checkpoint:
            raise ValueError("TeacherTargetRecord must identify state and checkpoint")
        if not self.teacher_hint_hash:
            raise ValueError("TeacherTargetRecord must contain a private-hint hash")
        if assistant_action_hash(self.teacher_action) != self.teacher_action_hash:
            raise ValueError("teacher_action_hash does not match teacher_action")
        if not self.student_visible_messages or not self.hinted_teacher_messages:
            raise ValueError("TeacherTargetRecord requires both policy views")
        visible_action = self.student_visible_messages[-1]
        hinted_action = self.hinted_teacher_messages[-1]
        if assistant_action_hash(visible_action) != self.teacher_action_hash:
            raise ValueError("student-visible view ends in a different Teacher action")
        if assistant_action_hash(hinted_action) != self.teacher_action_hash:
            raise ValueError("hinted view ends in a different Teacher action")
        count = len(self.target_token_ids)
        if count == 0:
            raise ValueError("Teacher target has no tokens")
        aligned = (
            self.target_loss_mask,
            self.hinted_topk_logprobs,
            self.hinted_topk_token_ids,
            self.hinted_support_mass,
            self.unhinted_reference_topk_logprobs,
            self.unhinted_reference_topk_token_ids,
            self.unhinted_reference_support_mass,
            self.skill_contrast_scores,
            self.skill_gate_values,
            self.sharpening_temperatures,
            self.sharpened_topk_logprobs,
            self.sharpened_topk_token_ids,
            self.sharpened_support_mass,
            self.raw_teacher_entropy,
            self.sharpened_teacher_entropy,
        )
        if any(len(value) != count for value in aligned):
            raise ValueError("TeacherTargetRecord token-level fields are misaligned")
        if any(value not in {0, 1} for value in self.target_loss_mask):
            raise ValueError("target_loss_mask must contain only zero or one")
        if not any(self.target_loss_mask):
            raise ValueError("TeacherTargetRecord has no active target tokens")
        for logs, ids in zip(self.hinted_topk_logprobs, self.hinted_topk_token_ids):
            if len(logs) != len(ids) or not logs:
                raise ValueError("hinted sparse rows must be non-empty and aligned")
        for logs, ids in zip(
            self.unhinted_reference_topk_logprobs,
            self.unhinted_reference_topk_token_ids,
        ):
            if len(logs) != len(ids) or not logs:
                raise ValueError("unhinted sparse rows must be non-empty and aligned")
        for logs, ids in zip(
            self.sharpened_topk_logprobs, self.sharpened_topk_token_ids
        ):
            if len(logs) != len(ids) or not logs:
                raise ValueError("sharpened sparse rows must be non-empty and aligned")
        if self.teacher_checkpoint != self.unhinted_reference_checkpoint:
            raise ValueError(
                "skill contrast must use the same checkpoint for hinted and unhinted views"
            )

    @staticmethod
    def raw_hash_payload(
        *,
        teacher_checkpoint: str,
        teacher_hint_hash: str,
        state_hash: str,
        teacher_action_hash: str,
        target_token_ids: Sequence[int],
        hinted_topk_logprobs: Sequence[Sequence[float]],
        hinted_topk_token_ids: Sequence[Sequence[int]],
    ) -> dict:
        return {
            "schema_version": TEACHER_TARGET_SCHEMA_VERSION,
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_hint_hash": teacher_hint_hash,
            "state_hash": state_hash,
            "teacher_action_hash": teacher_action_hash,
            "target_token_ids": list(target_token_ids),
            "hinted_topk_logprobs": [list(row) for row in hinted_topk_logprobs],
            "hinted_topk_token_ids": [list(row) for row in hinted_topk_token_ids],
        }

    @staticmethod
    def target_hash_payload(
        *,
        raw_teacher_target_hash: str,
        sharpened_topk_logprobs: Sequence[Sequence[float]],
        sharpened_topk_token_ids: Sequence[Sequence[int]],
        sharpening_temperatures: Sequence[float],
    ) -> dict:
        return {
            "schema_version": TEACHER_TARGET_SCHEMA_VERSION,
            "raw_teacher_target_hash": raw_teacher_target_hash,
            "sharpened_topk_logprobs": [
                list(row) for row in sharpened_topk_logprobs
            ],
            "sharpened_topk_token_ids": [
                list(row) for row in sharpened_topk_token_ids
            ],
            "sharpening_temperatures": list(sharpening_temperatures),
        }

    def assert_hashes(self) -> None:
        raw_hash = canonical_hash(
            self.raw_hash_payload(
                teacher_checkpoint=self.teacher_checkpoint,
                teacher_hint_hash=self.teacher_hint_hash,
                state_hash=self.state_hash,
                teacher_action_hash=self.teacher_action_hash,
                target_token_ids=self.target_token_ids,
                hinted_topk_logprobs=self.hinted_topk_logprobs,
                hinted_topk_token_ids=self.hinted_topk_token_ids,
            )
        )
        if raw_hash != self.raw_teacher_target_hash:
            raise ValueError("raw_teacher_target_hash does not match its distribution")
        target_hash = canonical_hash(
            self.target_hash_payload(
                raw_teacher_target_hash=raw_hash,
                sharpened_topk_logprobs=self.sharpened_topk_logprobs,
                sharpened_topk_token_ids=self.sharpened_topk_token_ids,
                sharpening_temperatures=self.sharpening_temperatures,
            )
        )
        if target_hash != self.teacher_target_hash:
            raise ValueError("teacher_target_hash does not match sharpened target")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return deepcopy(value)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherTargetRecord":
        if value.get("schema_version") != TEACHER_TARGET_SCHEMA_VERSION:
            raise ValueError(
                "Teacher target schema_version=2 is required; recollect or migrate "
                f"the row, got {value.get('schema_version')!r}"
            )
        scalar_tuple_fields = {
            "target_token_ids": int,
            "target_loss_mask": int,
            "hinted_support_mass": float,
            "unhinted_reference_support_mass": float,
            "skill_contrast_scores": float,
            "skill_gate_values": float,
            "sharpening_temperatures": float,
            "sharpened_support_mass": float,
            "raw_teacher_entropy": float,
            "sharpened_teacher_entropy": float,
        }
        row_tuple_fields = {
            "hinted_topk_logprobs": float,
            "hinted_topk_token_ids": int,
            "unhinted_reference_topk_logprobs": float,
            "unhinted_reference_topk_token_ids": int,
            "sharpened_topk_logprobs": float,
            "sharpened_topk_token_ids": int,
        }
        converted = deepcopy(value)
        converted["student_visible_messages"] = tuple(
            converted["student_visible_messages"]
        )
        converted["hinted_teacher_messages"] = tuple(
            converted["hinted_teacher_messages"]
        )
        for field, cast in scalar_tuple_fields.items():
            converted[field] = tuple(cast(item) for item in converted[field])
        for field, cast in row_tuple_fields.items():
            converted[field] = _as_tuple_rows(converted[field], cast)
        record = cls(**converted)
        record.assert_hashes()
        return record


@dataclass(frozen=True)
class TeacherValidationSample:
    seed: int
    score: SoftScoreResult

    def to_dict(self) -> dict:
        return {"seed": self.seed, "score": self.score.to_dict()}


@dataclass(frozen=True)
class TeacherTargetLabel:
    """One privileged Teacher macro-action at one frozen Student state."""

    decision: DecisionState
    teacher_action: TeacherActionResult

    def to_dict(self) -> dict:
        decision = self.decision
        return {
            "message_index": decision.message_index,
            "state_hash": decision.state_hash,
            "sample_hash": decision.sample_hash,
            "history_before": decision.to_dict()["history_before"],
            "student_action": decision.student_action.model_dump(mode="json"),
            "teacher_action": self.teacher_action.action.model_dump(mode="json"),
            "teacher_hint": deepcopy(self.teacher_action.hint),
        }


class TeacherTargetLabeler:
    """Generate supervision without running a takeover continuation."""

    def __init__(
        self,
        environment,
        *,
        teacher_generator: TeacherActionGenerator | None = None,
    ):
        self.environment = environment
        self.teacher_generator = teacher_generator or TeacherActionGenerator(environment)

    def run(self, decision: DecisionState) -> TeacherTargetLabel:
        teacher_action = self.teacher_generator.generate(
            decision, self.environment.config.seed
        )
        return TeacherTargetLabel(decision, teacher_action)

    def score_decision(self, history, message_index: int) -> dict:
        return self.run(DecisionState.from_history(history, message_index)).to_dict()


@dataclass(frozen=True)
class TeacherValidationResult:
    decision: DecisionState
    teacher_action: TeacherActionResult
    samples: tuple[TeacherValidationSample, ...]

    @property
    def teacher_quality(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.score.score for sample in self.samples) / len(self.samples)

    def to_dict(self) -> dict:
        decision = self.decision
        return {
            "message_index": decision.message_index,
            "state_hash": decision.state_hash,
            "sample_hash": decision.sample_hash,
            "history_before": decision.to_dict()["history_before"],
            "student_action": decision.student_action.model_dump(mode="json"),
            "teacher_action": self.teacher_action.action.model_dump(mode="json"),
            "teacher_hint": deepcopy(self.teacher_action.hint),
            "teacher_validation": [sample.to_dict() for sample in self.samples],
            "teacher_quality": self.teacher_quality,
        }


class TeacherTargetValidator:
    """Generate one privileged action and validate only its absolute quality."""

    def __init__(
        self,
        environment,
        *,
        continuations: int = 1,
        teacher_generator: TeacherActionGenerator | None = None,
        score_fn=soft_completion_score,
    ):
        if continuations < 1:
            raise ValueError("Teacher validation continuations must be positive")
        self.environment = environment
        self.continuations = continuations
        self.teacher_generator = teacher_generator or TeacherActionGenerator(environment)
        self.score_fn = score_fn

    def run(self, decision: DecisionState) -> TeacherValidationResult:
        base_seed = self.environment.config.seed
        teacher_action = self.teacher_generator.generate(decision, base_seed)
        samples = []
        for sample_index in range(self.continuations):
            seed = base_seed + sample_index
            continuation = self.environment.continue_to_terminal(
                decision.branch_history(teacher_action.action), "student", seed=seed
            )
            samples.append(
                TeacherValidationSample(
                    seed=seed,
                    score=self.score_fn(continuation.reward_info),
                )
            )
        return TeacherValidationResult(decision, teacher_action, tuple(samples))

    def score_decision(self, history, message_index: int) -> dict:
        return self.run(DecisionState.from_history(history, message_index)).to_dict()
