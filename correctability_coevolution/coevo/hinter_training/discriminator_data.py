from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Any, Iterable


@dataclass(frozen=True)
class BehaviorHintSample:
    """One actual frozen-Student macro-action produced with one sampled hint."""

    state_hash: str
    public_state: Any
    hint: str
    student_behavior: Any
    control_type: str = "ordinary"

    def __post_init__(self) -> None:
        if self.control_type not in {"ordinary", "explicit_copy", "useless"}:
            raise ValueError("unknown discriminator control_type")


@dataclass(frozen=True)
class CopyingDiscriminatorPair:
    state_hash: str
    public_state: Any
    student_behavior: Any
    positive_hint: str
    negative_hint: str
    control_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_discriminator_input(
    *, public_state: Any, student_behavior: Any, candidate_hint: str
) -> str:
    """One scalar-head input; task and behavior stay fixed across a pair."""

    return json.dumps(
        {
            "public_state": public_state,
            "student_operation_record": student_behavior,
            "candidate_hint": candidate_hint,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_fresh_discriminator_pairs(
    samples: Iterable[BehaviorHintSample], *, seed: int = 42
) -> list[CopyingDiscriminatorPair]:
    """Pair the true hint with a same-state unused hint."""

    grouped: dict[str, list[BehaviorHintSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.state_hash, []).append(sample)
    rng = random.Random(seed)
    pairs = []
    for state_hash, rows in grouped.items():
        if len(rows) < 2 or len({row.hint for row in rows}) < 2:
            raise ValueError(
                f"state {state_hash!r} needs at least two distinct sampled hints"
            )
        for sample in rows:
            alternatives = [row.hint for row in rows if row.hint != sample.hint]
            pairs.append(
                CopyingDiscriminatorPair(
                    state_hash=state_hash,
                    public_state=sample.public_state,
                    student_behavior=sample.student_behavior,
                    positive_hint=sample.hint,
                    negative_hint=rng.choice(alternatives),
                    control_type=sample.control_type,
                )
            )
    if not pairs:
        raise ValueError("fresh discriminator pair set is empty")
    return pairs
