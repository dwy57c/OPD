from __future__ import annotations

from enum import Enum
import re
from typing import Any, Mapping


class HintLevel(str, Enum):
    """Ordered privileged-context doses used by every experiment."""

    L0_NONE = "L0_NONE"
    L1_POLICY = "L1_POLICY"
    L2_PROCEDURAL = "L2_PROCEDURAL"
    L3_ORACLE = "L3_ORACLE"

    @property
    def dose(self) -> int:
        return HINT_LEVELS.index(self)

    @classmethod
    def parse(cls, value: "HintLevel | str") -> "HintLevel":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper()
        aliases = {
            "L0": cls.L0_NONE,
            "NONE": cls.L0_NONE,
            "L1": cls.L1_POLICY,
            "POLICY": cls.L1_POLICY,
            "L2": cls.L2_PROCEDURAL,
            "PROCEDURAL": cls.L2_PROCEDURAL,
            "L3": cls.L3_ORACLE,
            "ORACLE": cls.L3_ORACLE,
        }
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


HINT_LEVELS = (
    HintLevel.L0_NONE,
    HintLevel.L1_POLICY,
    HintLevel.L2_PROCEDURAL,
    HintLevel.L3_ORACLE,
)


_SHARED_CONTRACT = """
The note is private and will never be shown to the customer. Write ordinary
prose, not a controller output. Do not emit exact function or tool names,
argument keys, JSON, code, schemas, quoted calls, headings, numbered fields,
bullet lists, or the public reply. Use complete sentences and finish cleanly.
""".strip()


_INSTRUCTIONS = {
    HintLevel.L1_POLICY: f"""
Write a 15-40 word private policy reminder for the customer-service policy
model's next turn. State only the generally applicable policy principle,
information-gathering norm, or safety guardrail. Do not include task-specific
facts, a solution path, or oracle steps.

{_SHARED_CONTRACT}
""".strip(),
    HintLevel.L2_PROCEDURAL: f"""
Write a concise private procedural note that helps a customer-service policy
model choose its next turn. Use the dialogue, tool results, domain policy, and
privileged resolution information to identify what must be learned, checked,
or confirmed next.

Every task-specific fact must be converted into the public procedure that an
unprivileged agent can execute to obtain or verify it. Never state a particular
identifier, date, route, flight number, order number, fee, amount, account
value, database result, or other hidden instance value. If correct behavior
depends on one, say to ask the user, inspect the visible history, or query the
appropriate source without naming a concrete API. End with a tentative
semantic direction and any confirmation or safety guardrail. Keep it usually
between 40 and 100 words.

{_SHARED_CONTRACT}
""".strip(),
    HintLevel.L3_ORACLE: f"""
Write a short private decision note that helps a customer-service policy model
choose its next turn. Use the dialogue, tool results, domain policy, and
privileged resolution facts as evidence. Establish what is known, what remains
unresolved, and what policy constraint matters. When multiple moves are
plausible, compare their tradeoffs. You may mention task-specific identifiers,
dates, routes, or amounts as natural-language facts when necessary. Do not
mechanically restate the oracle steps. Keep it usually between 60 and 140 words.

{_SHARED_CONTRACT}
""".strip(),
}


_RIGID_HEADING = re.compile(
    r"^(?:#{1,6}\s*)?(?:state|user intent|next action|remaining plan|policy checks)\s*:?$",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r"^(?:[-*]\s+|\d+[.)]\s+)")
_DATE = re.compile(
    r"(?:\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|"
    r"\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|"
    r"july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b)",
    re.IGNORECASE,
)
_AMOUNT = re.compile(
    r"(?:[$€£¥]\s*\d|\b\d+(?:\.\d{1,2})?\s*(?:usd|eur|gbp|cny|rmb|dollars?|"
    r"euros?|pounds?|yuan|元)\b)",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(
    r"(?:\b[A-Z]{1,3}\d{2,8}\b|\b\d{6,}\b|"
    r"\b(?:order|booking|reservation|ticket|account|confirmation)\s*(?:id|number|#)?"
    r"\s*[:#-]?\s*[A-Z0-9-]{4,}\b)",
    re.IGNORECASE,
)


def hint_instruction(level: HintLevel | str) -> str:
    parsed = HintLevel.parse(level)
    if parsed is HintLevel.L0_NONE:
        raise ValueError("L0 has no hinter instruction because it must not call an API")
    return _INSTRUCTIONS[parsed]


def prepare_hint_payload(
    payload: Mapping[str, Any], level: HintLevel | str
) -> dict[str, Any]:
    """Apply the information contract before the request reaches the hinter."""

    parsed = HintLevel.parse(level)
    if parsed is HintLevel.L0_NONE:
        return {}
    result = dict(payload)
    result["hint_level"] = parsed.value
    if parsed is HintLevel.L1_POLICY:
        result.pop("authoritative_oracle_steps", None)
    return result


def _oracle_literals(payload: Mapping[str, Any]) -> tuple[str, ...]:
    oracle = str(payload.get("authoritative_oracle_steps") or "")
    values: set[str] = set()
    candidates = re.findall(r"[\"']([^\"']{4,})[\"']", oracle)
    candidates.extend(
        value
        for value in re.findall(r"[A-Za-z0-9][A-Za-z0-9./:-]{3,}", oracle)
        if any(character.isdigit() for character in value)
    )
    for value in candidates:
        normalized = value.strip(".,:;()[]{}\"'")
        if (
            len(normalized) >= 4
            and not normalized.lower().startswith("step")
            and not normalized.isspace()
        ):
            values.add(normalized)
    return tuple(sorted(values, key=len, reverse=True))


def _payload_tool_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    names = []
    for schema in payload.get("available_tools") or []:
        function = schema.get("function") if isinstance(schema, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(names)


def hint_fact_leaks(plan: str, payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return machine-checkable instance facts leaked by a hint."""

    findings: list[str] = []
    if _DATE.search(plan):
        findings.append("date")
    if _AMOUNT.search(plan):
        findings.append("amount")
    if _IDENTIFIER.search(plan):
        findings.append("identifier")
    for name in _payload_tool_names(payload):
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", plan):
            findings.append(f"tool_name:{name}")
            break
    lowered = plan.casefold()
    copied = [
        value for value in _oracle_literals(payload) if value.casefold() in lowered
    ]
    if copied:
        findings.append(f"oracle_literal:{copied[0]}")
    return tuple(findings)


def validate_hint_note(
    plan: str,
    payload: Mapping[str, Any],
    level: HintLevel | str,
) -> None:
    """Fail closed when a generated note violates its dose contract."""

    parsed = HintLevel.parse(level)
    if parsed is HintLevel.L0_NONE:
        raise ValueError("L0 must not contain a hint note")
    if not isinstance(plan, str) or not plan.strip():
        raise ValueError("closed-model hinter returned an empty note")
    plan = plan.strip()
    if plan[-1] not in ".!?。！？":
        raise ValueError("closed-model hinter returned an incomplete note")
    words = plan.split()
    if len(words) > 220:
        raise ValueError("closed-model hinter returned an overlong note")
    if parsed is HintLevel.L1_POLICY and not 15 <= len(words) <= 40:
        raise ValueError("L1 policy hint must contain 15-40 words")
    if parsed is HintLevel.L2_PROCEDURAL and len(words) > 100:
        raise ValueError("L2 procedural hint must not exceed 100 words")
    if parsed is HintLevel.L3_ORACLE and len(words) > 140:
        raise ValueError("L3 oracle hint must not exceed 140 words")
    if "```" in plan or plan.lstrip().startswith(("{", "[")):
        raise ValueError("closed-model hinter returned structured output")
    lines = [line.strip() for line in plan.splitlines() if line.strip()]
    if any(_RIGID_HEADING.match(line) or _LIST_ITEM.match(line) for line in lines):
        raise ValueError("closed-model hinter returned a rigid template")

    for name in _payload_tool_names(payload):
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", plan):
            raise ValueError("closed-model hinter copied an exact tool name")

    if parsed in {HintLevel.L1_POLICY, HintLevel.L2_PROCEDURAL}:
        findings = tuple(
            finding
            for finding in hint_fact_leaks(plan, payload)
            if not finding.startswith("tool_name:")
        )
        if findings:
            raise ValueError(
                f"{parsed.value} hint failed the fact audit: {findings[0]}"
            )
