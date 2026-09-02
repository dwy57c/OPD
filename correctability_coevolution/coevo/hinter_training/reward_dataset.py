from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from coevo.hints import HintLevel
from coevo.hinter_prompt import HINTER_SYSTEM_PROMPT, build_hinter_messages

from .grpo_reward import validate_hinter_reward_row


HINTER_GRPO_SYSTEM_PROMPT = HINTER_SYSTEM_PROMPT


@dataclass(frozen=True)
class HinterGRPORow:
    messages: tuple[dict[str, Any], ...]
    state_hash: str
    public_state: Any
    privileged_context: Any
    student_visible_messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    standard_source_level: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("messages", "student_visible_messages", "tools"):
            value[key] = list(value[key])
        return value


def build_hinter_grpo_dataset(
    audit_rows: Iterable[Mapping[str, Any]],
    *,
    standard_source_level: HintLevel | str = HintLevel.L3_ORACLE,
) -> list[HinterGRPORow]:
    """Build one GRPO prompt per state with one fixed standard action.

    Candidate hints never define their own target. The standard action is chosen
    once from the requested audited level and is reused for every candidate in
    the GRPO group.
    """

    standard_level = HintLevel.parse(standard_source_level)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in audit_rows:
        grouped.setdefault(str(row["state_hash"]), []).append(row)
    result = []
    for state_hash, rows in grouped.items():
        candidates = [
            row
            for row in rows
            if HintLevel.parse(row.get("hint_level", "")) is standard_level
            and bool(row.get("standard_action_eligible", True))
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"state {state_hash!r} requires exactly one eligible "
                f"{standard_level.value} standard action; got {len(candidates)}"
            )
        source = candidates[0]
        required = (
            "public_state",
            "privileged_context",
            "student_visible_messages",
        )
        missing = [field for field in required if field not in source]
        if missing:
            raise ValueError(
                f"audited standard row is missing fields: {', '.join(missing)}"
            )
        messages = build_hinter_messages(
            source["public_state"], source["privileged_context"]
        )
        row = HinterGRPORow(
            messages=tuple(messages),
            state_hash=state_hash,
            public_state=source["public_state"],
            privileged_context=source["privileged_context"],
            student_visible_messages=tuple(source["student_visible_messages"]),
            tools=tuple(source.get("tools") or []),
            standard_source_level=standard_level.value,
        )
        validate_hinter_reward_row(row.to_dict())
        result.append(row)
    if not result:
        raise ValueError("no hinter GRPO rows were built")
    return result
