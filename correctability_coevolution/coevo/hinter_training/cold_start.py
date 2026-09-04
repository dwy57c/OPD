from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from coevo.hinter_prompt import (
    build_hinter_messages,
    narrow_privileged_context,
    student_profile_from_decision,
)
from coevo.hints import HintLevel


@dataclass(frozen=True)
class ColdStartSource:
    student_checkpoint: str
    audit_rows: tuple[Mapping[str, Any], ...]
    hstar_manifest: Mapping[str, Any]


def _hint_text(row: Mapping[str, Any]) -> str:
    hint = row.get("hint")
    if isinstance(hint, Mapping) and "hint" in hint:
        hint = hint.get("hint")
    if isinstance(hint, Mapping):
        return str(hint.get("plan") or "").strip()
    return str(hint or "").strip()


def build_hinter_cold_start_dataset(
    sources: Iterable[ColdStartSource],
    *,
    max_mean_copy: float = 0.1,
) -> list[dict[str, Any]]:
    """Build a fail-closed, dose-diverse SFT seed over multiple Students."""

    sources = tuple(sources)
    checkpoints = {source.student_checkpoint for source in sources}
    if len(checkpoints) < 2:
        raise ValueError("cold-start SFT requires at least two Student checkpoints")
    if max_mean_copy < 0:
        raise ValueError("max_mean_copy must be non-negative")

    result = []
    selected_levels: set[HintLevel] = set()
    selected_checkpoints: set[str] = set()
    selected_sessions: set[tuple[str, str, HintLevel]] = set()
    for source in sources:
        decisions = source.hstar_manifest.get("decisions") or {}
        for row in source.audit_rows:
            decision = decisions.get(str(row.get("task_id"))) or decisions.get(
                str(row.get("state_id"))
            )
            if not decision:
                continue
            level_value = decision.get("level")
            if level_value in {None, HintLevel.L0_NONE.value, HintLevel.HINTER.value}:
                continue
            level = HintLevel.parse(level_value)
            if row.get("hint_level") != level.value or row.get("hint_error"):
                continue
            session_id = str(
                row.get("session_id")
                or f"{row.get('task_id', 'task')}:{row.get('seed', 0)}"
            )
            session_key = (source.student_checkpoint, session_id, level)
            if session_key in selected_sessions:
                continue
            hint = _hint_text(row)
            if not hint:
                continue
            signals = row.get("analytical_signals") or {}
            if "mean_copy" not in signals:
                continue
            if float(signals["mean_copy"]) > max_mean_copy:
                continue
            privileged = narrow_privileged_context(row["privileged_context"])
            student_profile = student_profile_from_decision(decision)
            messages = build_hinter_messages(
                row.get("task_public_state", row["public_state"]),
                privileged,
                student_profile,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"level: {level.value.split('_', 1)[0]}\n{hint}",
                }
            )
            result.append(
                {
                    "messages": messages,
                    "student_checkpoint": source.student_checkpoint,
                    "student_profile": student_profile,
                    "state_hash": str(row["state_hash"]),
                    "session_id": session_id,
                    "minimal_sufficient_level": level.value,
                }
            )
            selected_levels.add(level)
            selected_checkpoints.add(source.student_checkpoint)
            selected_sessions.add(session_key)

    if len(selected_checkpoints) < 2:
        raise ValueError(
            "cold-start SFT must retain low-copy rows from at least two Student checkpoints"
        )
    if len(selected_levels) < 2:
        raise ValueError(
            "cold-start SFT requires at least two non-zero minimal hint levels"
        )
    if not result:
        raise ValueError("cold-start SFT selection produced no rows")
    return result
