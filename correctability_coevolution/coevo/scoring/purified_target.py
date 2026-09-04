from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coevo.scoring.stage_gap import SparseTargetView


@dataclass(frozen=True)
class PurifiedTargetResult:
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


def _lookup(view: "SparseTargetView", index: int) -> dict[int, float]:
    return {
        int(token_id): math.exp(float(logprob))
        for token_id, logprob in zip(
            view.topk_token_ids[index], view.topk_logprobs[index]
        )
        if math.isfinite(float(logprob))
    }


def _project(lookup: dict[int, float], support: list[int]) -> list[float]:
    values = [lookup[token_id] for token_id in support]
    values.append(max(0.0, 1.0 - sum(values)))
    return values


def _entropy(values: list[float]) -> float:
    return -sum(value * math.log(value) for value in values if value > 0)


def construct_purified_target(
    *,
    unhinted: "SparseTargetView",
    hinted: "SparseTargetView",
    hint_only: "SparseTargetView",
    beta: float = 1.0,
) -> PurifiedTargetResult:
    """Purified OPSD target P0 * exp((log q_h - log p_h) / beta)."""

    if beta <= 0:
        raise ValueError("purified target beta must be positive")
    if not (
        unhinted.target_input_ids
        == hinted.target_input_ids
        == hint_only.target_input_ids
    ):
        raise ValueError("purified target views must have aligned target tokens")

    scores = []
    output_logs = []
    output_ids = []
    output_mass = []
    raw_entropies = []
    target_entropies = []
    for index, actual in enumerate(hinted.target_input_ids):
        p0_lookup = _lookup(unhinted, index)
        q_lookup = _lookup(hinted, index)
        reference_lookup = _lookup(hint_only, index)
        support = sorted(set(p0_lookup) & set(q_lookup) & set(reference_lookup))
        if int(actual) not in support:
            raise ValueError("actual target token is absent from shared purified support")
        p0 = _project(p0_lookup, support)
        q = _project(q_lookup, support)
        reference = _project(reference_lookup, support)
        logits = [
            math.log(max(base, 1e-12))
            + (math.log(max(teacher, 1e-12)) - math.log(max(ref, 1e-12)))
            / beta
            for base, teacher, ref in zip(p0, q, reference)
        ]
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        normalizer = sum(weights)
        target = [value / normalizer for value in weights]
        correction = sum(
            teacher
            * abs(math.log(max(teacher, 1e-12)) - math.log(max(ref, 1e-12)))
            for teacher, ref in zip(q, reference)
        )
        scores.append(correction)
        output_ids.append(tuple(support))
        output_logs.append(tuple(math.log(max(value, 1e-12)) for value in target[:-1]))
        output_mass.append(sum(target[:-1]))
        raw_entropies.append(_entropy(q))
        target_entropies.append(_entropy(target))

    count = len(scores)
    return PurifiedTargetResult(
        skill_contrast_scores=tuple(scores),
        skill_gate_values=tuple(1.0 for _ in range(count)),
        sharpening_temperatures=tuple(beta for _ in range(count)),
        sharpened_topk_logprobs=tuple(output_logs),
        sharpened_topk_token_ids=tuple(output_ids),
        hinted_support_mass=hinted.support_mass,
        unhinted_support_mass=unhinted.support_mass,
        sharpened_support_mass=tuple(output_mass),
        raw_teacher_entropy=tuple(raw_entropies),
        sharpened_teacher_entropy=tuple(target_entropies),
    )
