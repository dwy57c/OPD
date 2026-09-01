from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class AgreementReport:
    count: int
    agreement: float
    cohen_kappa: float
    human_positive_rate: float
    judge_positive_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def binary_agreement(
    human: Sequence[bool | int], judge: Sequence[bool | int]
) -> AgreementReport:
    if len(human) != len(judge) or not human:
        raise ValueError("human and judge labels must be non-empty and aligned")
    human_values = [bool(value) for value in human]
    judge_values = [bool(value) for value in judge]
    count = len(human_values)
    observed = sum(left == right for left, right in zip(human_values, judge_values)) / count
    human_positive = sum(human_values) / count
    judge_positive = sum(judge_values) / count
    expected = (
        human_positive * judge_positive
        + (1 - human_positive) * (1 - judge_positive)
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float(observed == 1)
    return AgreementReport(
        count=count,
        agreement=observed,
        cohen_kappa=kappa,
        human_positive_rate=human_positive,
        judge_positive_rate=judge_positive,
    )


def validate_annotation_rows(
    rows: Iterable[Mapping],
    fields: Sequence[str] = ("clarifying", "lookup", "ungrounded"),
) -> dict[str, AgreementReport]:
    values = list(rows)
    reports = {}
    for field in fields:
        human_key = f"human_{field}"
        judge_key = f"judge_{field}"
        if any(human_key not in row or judge_key not in row for row in values):
            raise ValueError(f"annotation rows must contain {human_key} and {judge_key}")
        reports[field] = binary_agreement(
            [row[human_key] for row in values],
            [row[judge_key] for row in values],
        )
    return reports
