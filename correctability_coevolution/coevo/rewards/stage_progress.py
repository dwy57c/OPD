from dataclasses import asdict, dataclass
import math
from typing import Sequence


REWARD_NAME = "tau2_stage_learning_progress"
REWARD_FORMULA_VERSION = "previous-skill-anchor-progress-v3"


@dataclass(frozen=True)
class StageProgressResult:
    previous_gap: float
    current_gap: float
    learning_progress: float
    positive_learning_progress: float
    decision_reward: float

    def to_dict(self) -> dict:
        return asdict(self)


def _deduplicated_lookup(
    logprobs: Sequence[float], token_ids: Sequence[int]
) -> dict[int, float]:
    if len(logprobs) != len(token_ids):
        raise ValueError("sparse policy token IDs and log-probabilities must align")
    result: dict[int, float] = {}
    for token_id, logprob in zip(token_ids, logprobs):
        value = float(logprob)
        if not math.isfinite(value):
            continue
        token_id = int(token_id)
        result[token_id] = max(value, result.get(token_id, float("-inf")))
    if not result:
        raise ValueError("sparse policy row has no finite entries")
    return result


def token_forward_kl(
    *,
    teacher_logprobs: Sequence[float],
    teacher_token_ids: Sequence[int],
    student_logprobs: Sequence[float],
    student_token_ids: Sequence[int],
    actual_target_token_id: int,
    epsilon: float = 1e-8,
) -> float:
    """Forward KL on fixed Teacher support plus one explicit tail bucket."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    teacher = _deduplicated_lookup(teacher_logprobs, teacher_token_ids)
    student = _deduplicated_lookup(student_logprobs, student_token_ids)
    actual_target_token_id = int(actual_target_token_id)
    if actual_target_token_id not in teacher:
        raise ValueError("actual target token is absent from sharpened Teacher support")
    if actual_target_token_id not in student:
        raise ValueError("actual target token is absent from Student scoring support")

    support = tuple(teacher)
    q_explicit = [math.exp(teacher[token_id]) for token_id in support]
    q_mass = sum(q_explicit)
    if q_mass > 1 + 1e-5:
        raise ValueError("sharpened Teacher support mass exceeds one")
    q_tail = max(0.0, 1.0 - q_mass)
    q = q_explicit + [q_tail]

    p_explicit = [
        math.exp(student[token_id]) if token_id in student else epsilon
        for token_id in support
    ]
    p_mass = sum(p_explicit)
    if p_mass > 1 + 1e-5:
        raise ValueError("Student probabilities on Teacher support exceed one")
    p_tail = max(0.0, 1.0 - p_mass)
    p_normalizer = p_mass + p_tail
    p = [value / p_normalizer for value in p_explicit] + [p_tail / p_normalizer]

    return sum(
        q_value
        * (
            math.log(max(q_value, epsilon))
            - math.log(max(p_value, epsilon))
        )
        for q_value, p_value in zip(q, p)
        if q_value > 0
    )


def mean_forward_kl(
    *,
    teacher_logprobs: Sequence[Sequence[float]],
    teacher_token_ids: Sequence[Sequence[int]],
    student_logprobs: Sequence[Sequence[float]],
    student_token_ids: Sequence[Sequence[int]],
    target_token_ids: Sequence[int],
    target_loss_mask: Sequence[int] | None = None,
    epsilon: float = 1e-8,
) -> float:
    count = len(target_token_ids)
    inputs = (
        teacher_logprobs,
        teacher_token_ids,
        student_logprobs,
        student_token_ids,
    )
    if any(len(value) != count for value in inputs):
        raise ValueError("Teacher, Student, and target token positions must align")
    mask = list(target_loss_mask) if target_loss_mask is not None else [1] * count
    if len(mask) != count:
        raise ValueError("target_loss_mask must align with target_token_ids")
    active = [index for index, value in enumerate(mask) if int(value) == 1]
    if any(int(value) not in {0, 1} for value in mask):
        raise ValueError("target_loss_mask must contain only zero or one")
    if not active:
        raise ValueError("Teacher target has no active tokens")
    values = [
        token_forward_kl(
            teacher_logprobs=teacher_logprobs[index],
            teacher_token_ids=teacher_token_ids[index],
            student_logprobs=student_logprobs[index],
            student_token_ids=student_token_ids[index],
            actual_target_token_id=target_token_ids[index],
            epsilon=epsilon,
        )
        for index in active
    ]
    return sum(values) / len(values)


def hard_target_nll(
    *,
    student_logprobs: Sequence[Sequence[float]],
    student_token_ids: Sequence[Sequence[int]],
    target_token_ids: Sequence[int],
) -> float:
    """Diagnostic only; the curriculum baseline uses forward Teacher KL."""
    if not (
        len(student_logprobs) == len(student_token_ids) == len(target_token_ids)
    ):
        raise ValueError("hard-target diagnostic rows must align")
    losses = []
    for logs, ids, target in zip(
        student_logprobs, student_token_ids, target_token_ids
    ):
        lookup = _deduplicated_lookup(logs, ids)
        if int(target) not in lookup:
            raise ValueError("actual target token is missing from Student support")
        losses.append(-lookup[int(target)])
    if not losses:
        raise ValueError("hard-target diagnostic has no tokens")
    return sum(losses) / len(losses)


def score_stage_progress(
    *,
    previous_gap: float,
    current_gap: float,
) -> StageProgressResult:
    previous_gap = float(previous_gap)
    current_gap = float(current_gap)
    values = (previous_gap, current_gap)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("stage-progress inputs must be finite")
    if previous_gap < 0 or current_gap < 0:
        raise ValueError("forward KL gaps must be non-negative")
    learning_progress = previous_gap - current_gap
    positive = max(0.0, learning_progress)
    return StageProgressResult(
        previous_gap=previous_gap,
        current_gap=current_gap,
        learning_progress=learning_progress,
        positive_learning_progress=positive,
        decision_reward=positive,
    )
