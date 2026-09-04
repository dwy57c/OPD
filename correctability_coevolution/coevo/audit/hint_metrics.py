from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping


def aggregate_session_signals(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Average tokens within turns, turns within sessions, then sessions equally."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        signals = row.get("analytical_signals")
        if not isinstance(signals, Mapping):
            continue
        session_id = str(row.get("session_id") or row.get("state_id") or "session")
        grouped.setdefault(session_id, []).append(signals)
    session_rows = []
    for session_id, values in grouped.items():
        session_rows.append(
            {
                "session_id": session_id,
                "turns": len(values),
                "mean_lift": sum(float(value["mean_lift"]) for value in values)
                / len(values),
                "mean_copy": sum(float(value["mean_copy"]) for value in values)
                / len(values),
                "mean_hint_only_vs_empty": sum(
                    float(value["mean_hint_only_vs_empty"])
                    for value in values
                    if "mean_hint_only_vs_empty" in value
                )
                / sum("mean_hint_only_vs_empty" in value for value in values)
                if any("mean_hint_only_vs_empty" in value for value in values)
                else None,
            }
        )
    return {
        "sessions": len(session_rows),
        "scored_turns": sum(row["turns"] for row in session_rows),
        "mean_lift": (
            sum(row["mean_lift"] for row in session_rows) / len(session_rows)
            if session_rows
            else None
        ),
        "mean_copy": (
            sum(row["mean_copy"] for row in session_rows) / len(session_rows)
            if session_rows
            else None
        ),
        "mean_hint_only_vs_empty": (
            sum(
                row["mean_hint_only_vs_empty"]
                for row in session_rows
                if row["mean_hint_only_vs_empty"] is not None
            )
            / sum(
                row["mean_hint_only_vs_empty"] is not None for row in session_rows
            )
            if any(
                row["mean_hint_only_vs_empty"] is not None for row in session_rows
            )
            else None
        ),
        "per_session": session_rows,
    }


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
