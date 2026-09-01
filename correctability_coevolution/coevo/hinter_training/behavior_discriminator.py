from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from .discriminator_data import (
    CopyingDiscriminatorPair,
    format_discriminator_input,
)


def pairwise_ranking_loss(
    positive_scores: torch.Tensor, negative_scores: torch.Tensor
) -> torch.Tensor:
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive and negative discriminator scores must align")
    return F.softplus(-(positive_scores - negative_scores)).mean()


def pairwise_copy_probability(positive_score: float, negative_score: float) -> float:
    margin = float(positive_score) - float(negative_score)
    if margin >= 0:
        return 1.0 / (1.0 + math.exp(-margin))
    exp_margin = math.exp(margin)
    return exp_margin / (1.0 + exp_margin)


@dataclass(frozen=True)
class DiscriminatorControlReport:
    ordinary_pair_accuracy: float
    explicit_copy_accuracy: float
    useless_mean_distance_from_chance: float
    ordinary_pairs: int
    explicit_copy_pairs: int
    useless_pairs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscriminatorGate:
    minimum_explicit_copy_accuracy: float = 0.9
    maximum_useless_distance_from_chance: float = 0.1

    def validate(self, report: DiscriminatorControlReport) -> None:
        if report.explicit_copy_pairs < 1 or report.useless_pairs < 1:
            raise ValueError(
                "discriminator validation requires explicit-copy and useless controls"
            )
        failures = []
        if report.explicit_copy_accuracy < self.minimum_explicit_copy_accuracy:
            failures.append("explicit-copy control not detected")
        if (
            report.useless_mean_distance_from_chance
            > self.maximum_useless_distance_from_chance
        ):
            failures.append("useless-hint control is distinguishable")
        if failures:
            raise ValueError("discriminator control gate failed: " + "; ".join(failures))


def evaluate_pair_scores(
    pairs: Sequence[CopyingDiscriminatorPair],
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> DiscriminatorControlReport:
    if not (
        len(pairs) == len(positive_scores) == len(negative_scores)
    ):
        raise ValueError("pair rows and discriminator scores must align")
    buckets: dict[str, list[float]] = {
        "ordinary": [],
        "explicit_copy": [],
        "useless": [],
    }
    for pair, positive, negative in zip(pairs, positive_scores, negative_scores):
        buckets[pair.control_type].append(
            pairwise_copy_probability(positive, negative)
        )

    def accuracy(values: list[float]) -> float:
        return sum(value > 0.5 for value in values) / len(values) if values else 0.0

    useless = buckets["useless"]
    return DiscriminatorControlReport(
        ordinary_pair_accuracy=accuracy(buckets["ordinary"]),
        explicit_copy_accuracy=accuracy(buckets["explicit_copy"]),
        useless_mean_distance_from_chance=(
            sum(abs(value - 0.5) for value in useless) / len(useless)
            if useless
            else 1.0
        ),
        ordinary_pairs=len(buckets["ordinary"]),
        explicit_copy_pairs=len(buckets["explicit_copy"]),
        useless_pairs=len(useless),
    )


class PairwiseDiscriminatorCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        positive = [row["positive_text"] for row in rows]
        negative = [row["negative_text"] for row in rows]
        positive_batch = self.tokenizer(
            positive,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        negative_batch = self.tokenizer(
            negative,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            **{f"positive_{key}": value for key, value in positive_batch.items()},
            **{f"negative_{key}": value for key, value in negative_batch.items()},
        }


def pair_to_training_row(pair: CopyingDiscriminatorPair) -> dict[str, str]:
    common = {
        "public_state": pair.public_state,
        "student_behavior": pair.student_behavior,
    }
    return {
        "positive_text": format_discriminator_input(
            public_state=common["public_state"],
            student_behavior=common["student_behavior"],
            candidate_hint=pair.positive_hint,
        ),
        "negative_text": format_discriminator_input(
            public_state=common["public_state"],
            student_behavior=common["student_behavior"],
            candidate_hint=pair.negative_hint,
        ),
    }


def score_texts(model, tokenizer, texts: Iterable[str], *, max_length: int) -> list[float]:
    values = list(texts)
    if not values:
        return []
    device = next(model.parameters()).device
    batch = tokenizer(
        values,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits.reshape(-1)
    return [float(value) for value in logits.detach().cpu().tolist()]
