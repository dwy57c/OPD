from dataclasses import asdict, dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class SkillContrastConfig:
    low: float = 0.0
    high: float = 0.05
    minimum_temperature: float = 0.7
    minimum_support_mass: float = 0.95
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.low < 0 or self.high <= self.low:
            raise ValueError("skill thresholds must satisfy 0 <= low < high")
        if not 0 < self.minimum_temperature < 1:
            raise ValueError("minimum_temperature must be in (0, 1)")
        if not 0 < self.minimum_support_mass <= 1:
            raise ValueError("minimum_support_mass must be in (0, 1]")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class SkillContrastResult:
    skill_contrast_scores: tuple[float, ...]
    skill_gate_values: tuple[float, ...]
    sharpening_temperatures: tuple[float, ...]
    sharpened_topk_logprobs: tuple[tuple[float, ...], ...]
    sharpened_topk_token_ids: tuple[tuple[int, ...], ...]
    hinted_support_mass: tuple[float, ...]
    unhinted_support_mass: tuple[float, ...]
    sharpened_support_mass: tuple[float, ...]
    raw_teacher_entropy: tuple[float, ...]
    sharpened_teacher_entropy: tuple[float, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        return {key: _lists(item) for key, item in value.items()}


def _lists(value):
    if isinstance(value, tuple):
        return [_lists(item) for item in value]
    return value


def _unique_distribution(
    logprobs: Sequence[float], token_ids: Sequence[int]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if len(logprobs) != len(token_ids):
        raise ValueError("sparse token IDs and log-probabilities must align")
    values: dict[int, float] = {}
    order: list[int] = []
    for token_id, logprob in zip(token_ids, logprobs):
        logprob = float(logprob)
        if not math.isfinite(logprob):
            continue
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError("sparse token IDs must be non-negative")
        if token_id not in values:
            order.append(token_id)
            values[token_id] = logprob
        else:
            values[token_id] = max(values[token_id], logprob)
    if not order:
        raise ValueError("sparse distribution has no finite entries")
    return tuple(order), tuple(values[token_id] for token_id in order)


def _coarse_distribution(
    support_ids: tuple[int, ...],
    support_logprobs: tuple[float, ...],
    other_ids: tuple[int, ...] | None,
    other_logprobs: tuple[float, ...] | None,
    epsilon: float,
) -> tuple[list[float], float]:
    """Project a full-vocabulary view onto fixed support plus one tail bucket."""
    if other_ids is None or other_logprobs is None:
        probabilities = [math.exp(value) for value in support_logprobs]
    else:
        lookup = {
            token_id: math.exp(logprob)
            for token_id, logprob in zip(other_ids, other_logprobs)
        }
        probabilities = [lookup.get(token_id, epsilon) for token_id in support_ids]
    explicit_mass = sum(probabilities)
    if explicit_mass > 1 + 1e-5:
        raise ValueError("reported sparse probabilities exceed one")
    tail = max(0.0, 1.0 - explicit_mass)
    normalizer = explicit_mass + tail
    return [value / normalizer for value in probabilities] + [tail / normalizer], explicit_mass


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def construct_skill_contrast_target(
    *,
    hinted_topk_logprobs: Sequence[Sequence[float]],
    hinted_topk_token_ids: Sequence[Sequence[int]],
    unhinted_topk_logprobs: Sequence[Sequence[float]],
    unhinted_topk_token_ids: Sequence[Sequence[int]],
    target_token_ids: Sequence[int],
    config: SkillContrastConfig | None = None,
) -> SkillContrastResult:
    """Construct one detached sharpened target on Teacher support plus tail.

    The fixed support is the raw hinted Teacher top-k augmented with the actual
    target token. The final element of each internal probability vector is an
    explicit aggregate tail bucket; it is intentionally not serialized as a
    vocabulary token.
    """
    config = config or SkillContrastConfig()
    row_count = len(target_token_ids)
    inputs = (
        hinted_topk_logprobs,
        hinted_topk_token_ids,
        unhinted_topk_logprobs,
        unhinted_topk_token_ids,
    )
    if any(len(value) != row_count for value in inputs):
        raise ValueError("all skill-contrast views must align with target tokens")

    contrasts: list[float] = []
    gates: list[float] = []
    temperatures: list[float] = []
    sharpened_logs: list[tuple[float, ...]] = []
    sharpened_ids: list[tuple[int, ...]] = []
    hinted_masses: list[float] = []
    unhinted_masses: list[float] = []
    sharpened_masses: list[float] = []
    raw_entropies: list[float] = []
    sharpened_entropies: list[float] = []

    for row_index, target_token_id in enumerate(target_token_ids):
        q_ids, q_logs = _unique_distribution(
            hinted_topk_logprobs[row_index], hinted_topk_token_ids[row_index]
        )
        p_ids, p_logs = _unique_distribution(
            unhinted_topk_logprobs[row_index], unhinted_topk_token_ids[row_index]
        )
        if int(target_token_id) not in q_ids:
            raise ValueError(
                f"actual target token is absent from hinted support at row {row_index}"
            )
        if int(target_token_id) not in p_ids:
            raise ValueError(
                f"actual target token is absent from unhinted support at row {row_index}"
            )

        q, hinted_mass = _coarse_distribution(
            q_ids, q_logs, None, None, config.epsilon
        )
        p, unhinted_mass = _coarse_distribution(
            q_ids, q_logs, p_ids, p_logs, config.epsilon
        )
        if hinted_mass + 1e-8 < config.minimum_support_mass:
            raise ValueError(
                "hinted Teacher support mass below configured minimum: "
                f"row={row_index}, mass={hinted_mass:.8f}, "
                f"minimum={config.minimum_support_mass:.8f}"
            )

        contrast = sum(
            q_value * (math.log(max(q_value, config.epsilon)) - math.log(max(p_value, config.epsilon)))
            for q_value, p_value in zip(q, p)
        )
        contrast = max(0.0, contrast)
        gate = min(1.0, max(0.0, (contrast - config.low) / (config.high - config.low)))
        temperature = 1.0 - gate * (1.0 - config.minimum_temperature)
        alpha = 1.0 / temperature
        powered = [value**alpha for value in q]
        normalizer = sum(powered)
        sharpened = [value / normalizer for value in powered]

        raw_order = sorted(range(len(q_ids)), key=lambda index: (-q[index], q_ids[index]))
        sharp_order = sorted(
            range(len(q_ids)), key=lambda index: (-sharpened[index], q_ids[index])
        )
        if raw_order != sharp_order:
            raise RuntimeError("temperature sharpening changed Teacher support ordering")

        contrasts.append(contrast)
        gates.append(gate)
        temperatures.append(temperature)
        sharpened_logs.append(
            tuple(math.log(max(value, config.epsilon)) for value in sharpened[:-1])
        )
        sharpened_ids.append(q_ids)
        hinted_masses.append(hinted_mass)
        unhinted_masses.append(unhinted_mass)
        sharpened_masses.append(sum(sharpened[:-1]))
        raw_entropies.append(_entropy(q))
        sharpened_entropies.append(_entropy(sharpened))

    return SkillContrastResult(
        tuple(contrasts),
        tuple(gates),
        tuple(temperatures),
        tuple(sharpened_logs),
        tuple(sharpened_ids),
        tuple(hinted_masses),
        tuple(unhinted_masses),
        tuple(sharpened_masses),
        tuple(raw_entropies),
        tuple(sharpened_entropies),
    )
