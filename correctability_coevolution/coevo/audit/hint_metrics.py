from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping


def _hint_text(row: Mapping[str, Any]) -> str:
    value = row.get("hint")
    if isinstance(value, Mapping) and "hint" in value:
        value = value.get("hint")
    if isinstance(value, Mapping):
        value = value.get("plan")
    lines = [
        line.strip().casefold()
        for line in str(value or "").splitlines()
        if line.strip() and not line.casefold().startswith("level:")
    ]
    return " ".join(" ".join(lines).split())


def counterfactual_invariance(
    original_rows: Iterable[Mapping[str, Any]],
    counterfactual_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare hints after swapping only hidden instance facts."""

    def index(rows):
        return {
            (str(row["state_id"]), str(row["hint_level"])): row
            for row in rows
            if not row.get("hint_error")
        }

    original = index(original_rows)
    changed = index(counterfactual_rows)
    grouped: dict[str, list[float]] = {}
    for key in sorted(set(original) & set(changed)):
        left = _hint_text(original[key])
        right = _hint_text(changed[key])
        if not left or not right:
            continue
        grouped.setdefault(key[1], []).append(SequenceMatcher(None, left, right).ratio())
    return {
        "levels": {
            level: {
                "pairs": len(values),
                "mean_similarity": sum(values) / len(values),
            }
            for level, values in sorted(grouped.items())
        }
    }
