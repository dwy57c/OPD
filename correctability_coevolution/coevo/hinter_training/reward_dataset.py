from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from coevo.artifacts import canonical_hash
from coevo.hinter_prompt import (
    HINTER_SYSTEM_PROMPT,
    build_hinter_messages,
    narrow_privileged_context,
)
from coevo.hints import HintLevel

from .grpo_reward import validate_hinter_reward_row


HINTER_GRPO_SYSTEM_PROMPT = HINTER_SYSTEM_PROMPT


@dataclass(frozen=True)
class HinterGRPORow:
    messages: tuple[dict[str, Any], ...]
    state_hash: str
    state_hashes: tuple[str, ...]
    session_id: str
    public_state: Any
    student_profile: Any
    privileged_context: Any
    fact_audit_context: Any
    student_visible_session: tuple[tuple[dict[str, Any], ...], ...]
    tools: tuple[dict[str, Any], ...]
    standard_source_level: str
    scenario_id: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["messages"] = list(value["messages"])
        value["state_hashes"] = list(value["state_hashes"])
        value["student_visible_session"] = [
            list(turn) for turn in value["student_visible_session"]
        ]
        value["tools"] = list(value["tools"])
        return value


def _session_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("session_id")
        or f"{row.get('task_id', 'task')}:{row.get('seed', 0)}"
    )


def build_hinter_grpo_dataset(
    audit_rows: Iterable[Mapping[str, Any]],
    *,
    standard_source_level: HintLevel | str = HintLevel.L3_ORACLE,
) -> list[HinterGRPORow]:
    """Build one candidate hint and one session-average reward per task run."""

    standard_level = HintLevel.parse(standard_source_level)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in audit_rows:
        grouped.setdefault(_session_id(row), []).append(row)

    result = []
    for session_id, rows in grouped.items():
        standards = [
            row
            for row in rows
            if HintLevel.parse(row.get("hint_level", "")) is standard_level
            and bool(row.get("standard_action_eligible", True))
        ]
        if not standards:
            raise ValueError(
                f"session {session_id!r} has no eligible {standard_level.value} turns"
            )
        standards.sort(
            key=lambda row: int(
                row.get("state_order", row.get("message_index", row.get("turn", 0)))
            )
        )
        source = standards[0]
        required = (
            "public_state",
            "student_profile",
            "privileged_context",
            "student_visible_messages",
        )
        missing = [field for field in required if field not in source]
        if missing:
            raise ValueError(
                f"audited standard row is missing fields: {', '.join(missing)}"
            )
        profile_hashes = {
            canonical_hash(row["student_profile"]) for row in standards
        }
        privilege_hashes = {
            canonical_hash(narrow_privileged_context(row["privileged_context"]))
            for row in standards
        }
        if len(profile_hashes) != 1 or len(privilege_hashes) != 1:
            raise ValueError("all turns in a task session must share hint inputs")

        privileged_context = narrow_privileged_context(source["privileged_context"])
        student_profile = dict(source["student_profile"])
        task_public_state = source.get("task_public_state", source["public_state"])
        messages = build_hinter_messages(
            task_public_state, privileged_context, student_profile
        )
        fact_audit_context = dict(source.get("fact_audit_context") or {})
        fact_audit_context.setdefault("available_tools", source.get("tools") or [])
        fact_audit_context.setdefault(
            "authoritative_oracle_steps",
            privileged_context["authoritative_oracle_steps"],
        )
        for key in (
            "domain",
            "goal_object_locations",
            "destination_receptacle",
            "unobserved_states",
        ):
            if key in source:
                fact_audit_context.setdefault(key, source[key])

        state_hashes = tuple(str(row["state_hash"]) for row in standards)
        session_hash = canonical_hash(
            {"session_id": session_id, "state_hashes": state_hashes}
        )
        row = HinterGRPORow(
            messages=tuple(messages),
            state_hash=session_hash,
            state_hashes=state_hashes,
            session_id=session_id,
            public_state=task_public_state,
            student_profile=student_profile,
            privileged_context=privileged_context,
            fact_audit_context=fact_audit_context,
            student_visible_session=tuple(
                tuple(value["student_visible_messages"]) for value in standards
            ),
            tools=tuple(source.get("tools") or []),
            standard_source_level=standard_level.value,
            scenario_id=str(source.get("task_id") or session_id),
        )
        validate_hinter_reward_row(row.to_dict())
        result.append(row)
    if not result:
        raise ValueError("no hinter GRPO session rows were built")
    return result
