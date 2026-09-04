from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from coevo.artifacts import canonical_hash
from coevo.hinter_prompt import (
    HINTER_SYSTEM_PROMPT,
    build_hinter_messages,
    narrow_privileged_context,
)

from .grpo_reward import validate_hinter_reward_row


HINTER_GRPO_SYSTEM_PROMPT = HINTER_SYSTEM_PROMPT
REFERENCE_POOL_SOURCES = (
    "oracle",
    "oracle+validated_student",
    "validated_student",
)


@dataclass(frozen=True)
class HinterReferenceTrajectory:
    source: str
    session_id: str
    student_visible_session: tuple[tuple[dict[str, Any], ...], ...]
    state_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "session_id": self.session_id,
            "student_visible_session": [
                [deepcopy(message) for message in turn]
                for turn in self.student_visible_session
            ],
            "state_hashes": list(self.state_hashes),
        }


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
    reference_pool: tuple[HinterReferenceTrajectory, ...]
    tools: tuple[dict[str, Any], ...]
    reference_pool_source: str
    scenario_id: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["messages"] = list(value["messages"])
        value["state_hashes"] = list(value["state_hashes"])
        value["student_visible_session"] = [
            list(turn) for turn in value["student_visible_session"]
        ]
        value["reference_pool"] = [
            reference.to_dict() for reference in self.reference_pool
        ]
        value["tools"] = list(value["tools"])
        return value


def _session_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("session_id")
        or f"{row.get('task_id', 'task')}:{row.get('seed', 0)}"
    )


def _row_order(row: Mapping[str, Any]) -> int:
    return int(row.get("state_order", row.get("message_index", row.get("turn", 0))))


def _canonical_session_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_state: dict[str, Mapping[str, Any]] = {}
    for row in sorted(rows, key=_row_order):
        state_key = str(row.get("state_hash") or _row_order(row))
        current = by_state.get(state_key)
        if current is None or (current.get("hint_error") and not row.get("hint_error")):
            by_state[state_key] = row
    return sorted(by_state.values(), key=_row_order)


def _messages_with_action(
    source: Mapping[str, Any], action: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    messages = deepcopy(list(source["student_visible_messages"]))
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("reference source must end in an assistant action")
    if action.get("role") != "assistant":
        raise ValueError("reference action must have role=assistant")
    messages[-1] = deepcopy(dict(action))
    return tuple(messages)


def _oracle_reference(
    session_id: str, rows: Sequence[Mapping[str, Any]]
) -> HinterReferenceTrajectory | None:
    actions = next(
        (
            list(row["oracle_reference_actions"])
            for row in rows
            if row.get("oracle_reference_actions")
        ),
        [],
    )
    if not actions:
        return None
    turns = []
    hashes = []
    for index, action in enumerate(actions):
        anchor = rows[min(index, len(rows) - 1)]
        turns.append(_messages_with_action(anchor, action))
        hashes.append(
            canonical_hash(
                {
                    "reference_source": "oracle",
                    "session_id": session_id,
                    "action_index": index,
                    "public_state_hash": anchor["state_hash"],
                    "action": action,
                }
            )
        )
    return HinterReferenceTrajectory(
        source="oracle",
        session_id=session_id,
        student_visible_session=tuple(turns),
        state_hashes=tuple(hashes),
    )


def _validated_student_reference(
    session_id: str, rows: Sequence[Mapping[str, Any]]
) -> HinterReferenceTrajectory | None:
    if not rows or not all(bool(row.get("student_trajectory_verified")) for row in rows):
        return None
    if any(not row.get("student_action") for row in rows):
        return None
    turns = tuple(_messages_with_action(row, row["student_action"]) for row in rows)
    return HinterReferenceTrajectory(
        source="validated_student",
        session_id=session_id,
        student_visible_session=turns,
        state_hashes=tuple(
            canonical_hash(
                {
                    "reference_source": "validated_student",
                    "session_id": session_id,
                    "public_state_hash": row["state_hash"],
                    "action": row["student_action"],
                }
            )
            for row in rows
        ),
    )


def _reference_source(value: str) -> str:
    source = str(value).strip().lower()
    if source not in REFERENCE_POOL_SOURCES:
        raise ValueError(
            f"unknown reference pool source {value!r}; expected one of "
            f"{REFERENCE_POOL_SOURCES}"
        )
    return source


def build_hinter_grpo_dataset(
    audit_rows: Iterable[Mapping[str, Any]],
    *,
    standard_source_level: str = "oracle",
    skip_missing_references: bool = False,
) -> list[HinterGRPORow]:
    """Build one hint row per task run with an oracle-first reference pool."""

    reference_source = _reference_source(standard_source_level)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in audit_rows:
        grouped.setdefault(_session_id(row), []).append(row)

    canonical = {
        session_id: _canonical_session_rows(rows)
        for session_id, rows in grouped.items()
    }
    validated_by_task: dict[str, list[HinterReferenceTrajectory]] = {}
    for session_id, rows in canonical.items():
        reference = _validated_student_reference(session_id, rows)
        if reference is None:
            continue
        task_id = str(rows[0].get("task_id") or session_id)
        validated_by_task.setdefault(task_id, []).append(reference)

    result = []
    for session_id, standards in canonical.items():
        if not standards:
            continue
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
                f"audited reference row is missing fields: {', '.join(missing)}"
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

        task_id = str(source.get("task_id") or session_id)
        reference_pool: list[HinterReferenceTrajectory] = []
        if reference_source in {"oracle", "oracle+validated_student"}:
            oracle = _oracle_reference(session_id, standards)
            if oracle is not None:
                reference_pool.append(oracle)
        if reference_source in {"validated_student", "oracle+validated_student"}:
            reference_pool.extend(validated_by_task.get(task_id, []))
        if not reference_pool:
            if skip_missing_references:
                continue
            raise ValueError(
                f"session {session_id!r} has no {reference_source} reference trajectory"
            )

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

        primary = reference_pool[0]
        session_hash = canonical_hash(
            {
                "session_id": session_id,
                "reference_pool_source": reference_source,
                "references": [
                    {
                        "source": reference.source,
                        "session_id": reference.session_id,
                        "state_hashes": reference.state_hashes,
                    }
                    for reference in reference_pool
                ],
            }
        )
        row = HinterGRPORow(
            messages=tuple(messages),
            state_hash=session_hash,
            state_hashes=primary.state_hashes,
            session_id=session_id,
            public_state=task_public_state,
            student_profile=student_profile,
            privileged_context=privileged_context,
            fact_audit_context=fact_audit_context,
            student_visible_session=primary.student_visible_session,
            reference_pool=tuple(reference_pool),
            tools=tuple(source.get("tools") or []),
            reference_pool_source=reference_source,
            scenario_id=task_id,
        )
        validate_hinter_reward_row(row.to_dict())
        result.append(row)
    if not result:
        raise ValueError("no hinter GRPO session rows were built")
    return result
