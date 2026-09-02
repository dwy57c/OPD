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
The note is private and will never be shown to the user. Write ordinary prose,
not a controller output. Do not emit exact function or tool names, executable
command strings, argument keys, JSON, code, headings, numbered fields, bullet
lists, or the public reply. Use complete sentences and finish cleanly.
""".strip()


_DOMAIN_FACTS = {
    "alfworld": """
Instance facts for this task are: (a) which receptacle currently holds any goal
object, whether named as a class ("the coffee machine") or an instance
("coffeemachine 1"); (b) which specific receptacle instance is the destination
when the goal only names a class; (c) any object state or content that has not
yet been observed in the transcript; (d) which of several similar objects is
the correct one. General household knowledge is not an instance fact: that
heating uses a microwave, cleaning uses a sink basin, cooling uses a fridge,
and that a category of object is usually kept in certain kinds of places.
""".strip(),
    "tau2": """
Instance facts for this task are: identifiers, dates, routes, flight or order
numbers, fees, amounts, account values, and any database result that the agent
has not yet retrieved or been told in the visible history. Domain policy rules
are not instance facts.
""".strip(),
}


def _domain_key(domain: str | None) -> str:
    value = str(domain or "tau2").strip().lower()
    return "alfworld" if value == "alfworld" else "tau2"


def hint_instruction(level: HintLevel | str, domain: str | None = None) -> str:
    parsed = HintLevel.parse(level)
    if parsed is HintLevel.L0_NONE:
        raise ValueError("L0 has no hinter instruction because it must not call an API")
    domain_facts = _DOMAIN_FACTS[_domain_key(domain)]
    if parsed is HintLevel.L1_POLICY:
        task = """
Write a 15-40 word private reminder for the agent's next turn. State only a
generally applicable norm: observe before acting, confirm the state of things
before committing, keep track of what is held, and act one step at a time. Do
not mention any object, location, or route specific to this task.
""".strip()
    elif parsed is HintLevel.L2_PROCEDURAL:
        task = """
Write a concise private procedural note that helps the agent choose its next
turn. You are given privileged resolution information. Your job is to convert
it into a procedure that an agent WITHOUT that information could follow to
discover the same facts by observation.

Hard rules for instance facts:

- Never state one, in any form. Hedged forms count as stating it: "the most
  plausible spot", "likely", "usually here", and "start with X" all reveal the
  fact when X is the true answer.
- Never order candidate locations or candidate objects using the privileged
  information. If you list candidates, list the generic kinds of places such an
  object is usually kept, in no particular order, and tell the agent to check
  them one at a time and confirm by observation before acting.
- Never let the procedure be shorter than what an uninformed agent would need:
  if the true answer would be found on the first try only because you knew it,
  the note has leaked.

You may state task-derivable steps, including what must happen to the object
before the goal is met, the kind of appliance or receptacle that does it, and
the destination class if the goal names it. End with a confirmation or safety
guardrail. Usually 40-100 words.
""".strip()
    else:
        task = """
Write a short private decision note that helps the agent choose its next turn.
You are given privileged resolution information and you MUST use it explicitly:
state where each goal object currently is, which specific receptacle or
destination is correct, and any unobserved state that matters, as plain
natural-language facts. Then say what the efficient next move is and why. Do
not withhold a fact for the agent to "discover"; that is a different dose
level. Do not restate the oracle steps mechanically. Usually 60-140 words.
""".strip()
    return f"{task}\n\n{domain_facts}\n\n{_SHARED_CONTRACT}"


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
    r"(?:\b[A-Z]{1,3}-?\d{2,8}\b|\b\d{6,}\b|"
    r"\b(?:order|booking|reservation|ticket|account|confirmation)\s+"
    r"(?:id|number|#)\s*(?:is\s*)?[:#-]?\s*"
    r"(?=[A-Z0-9-]*\d[A-Z0-9-]*\b)[A-Z0-9][A-Z0-9-]{3,}\b)",
    re.IGNORECASE,
)


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
        for key in (
            "authoritative_oracle_steps",
            "goal_object_locations",
            "destination_receptacle",
            "unobserved_states",
        ):
            result.pop(key, None)
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


_ALFWORLD_ALIASES = {
    "coffeemachine": "coffee machine",
    "sinkbasin": "sink basin",
    "bathtubbasin": "bathtub basin",
    "stoveburner": "stove burner",
    "garbagecan": "garbage can",
    "sidetable": "side table",
    "countertop": "counter top",
    "desklamp": "desk lamp",
}


def _entity_variants(value: Any, *, include_class: bool) -> tuple[str, ...]:
    raw = str(value or "").strip().casefold()
    if not raw:
        return ()
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value)).casefold()
    spaced = re.sub(r"[_|-]+", " ", spaced)
    spaced = " ".join(spaced.split())
    variants = {raw, spaced}
    for source, target in _ALFWORLD_ALIASES.items():
        for item in tuple(variants):
            if source in item:
                variants.add(item.replace(source, target))
    if include_class:
        for item in tuple(variants):
            variants.add(re.sub(r"\s+\d+$", "", item).strip())
    return tuple(sorted((item for item in variants if item), key=len, reverse=True))


def _alfworld_location_fact_groups(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    locations = payload.get("goal_object_locations") or {}
    if isinstance(locations, Mapping):
        for location in locations.values():
            variants = _entity_variants(location, include_class=True)
            if variants:
                groups.append(variants)
    destination = payload.get("destination_receptacle")
    if destination:
        variants = _entity_variants(destination, include_class=False)
        if variants:
            groups.append(variants)
    return tuple(groups)


def _alfworld_state_facts(
    payload: Mapping[str, Any],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...]:
    facts: list[tuple[tuple[str, ...], tuple[str, ...], str]] = []
    states = payload.get("unobserved_states") or {}
    if isinstance(states, Mapping):
        for object_name, state in states.items():
            object_variants = _entity_variants(object_name, include_class=True)
            state_variants = _entity_variants(state, include_class=False)
            if object_variants and state_variants:
                label = f"{object_name} is {state}"
                facts.append((object_variants, state_variants, label))
    return tuple(facts)


def _mentions(plan: str, variants: tuple[str, ...]) -> bool:
    normalized = " ".join(plan.casefold().split())
    return any(
        re.search(rf"(?<!\w){re.escape(value)}(?!\w)", normalized)
        for value in variants
    )


def _discloses_state(
    plan: str,
    object_variants: tuple[str, ...],
    state_variants: tuple[str, ...],
) -> bool:
    """Allow natural coreference while keeping object and state locally grounded."""

    sentences = re.split(r"(?<=[.!?。！？])\s+", plan)
    return any(
        _mentions(sentence, object_variants) and _mentions(sentence, state_variants)
        for sentence in sentences
    )


def hint_fact_leaks(plan: str, payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return machine-checkable instance facts leaked by a hint."""

    findings: list[str] = []
    domain = _domain_key(payload.get("domain"))
    if domain == "alfworld":
        for variants in _alfworld_location_fact_groups(payload):
            leaked = next((value for value in variants if _mentions(plan, (value,))), None)
            if leaked:
                findings.append(f"structured_fact:{leaked}")
        for object_variants, state_variants, label in _alfworld_state_facts(payload):
            if _discloses_state(plan, object_variants, state_variants):
                findings.append(f"structured_fact:{label}")
        return tuple(findings)
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
            raise ValueError(
                f"closed-model hinter copied an exact tool name: {name}"
            )

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
    if (
        parsed is HintLevel.L3_ORACLE
        and _domain_key(payload.get("domain")) == "alfworld"
    ):
        missing = [
            variants[0]
            for variants in _alfworld_location_fact_groups(payload)
            if not _mentions(plan, variants)
        ]
        missing.extend(
            label
            for object_variants, state_variants, label in _alfworld_state_facts(
                payload
            )
            if not _discloses_state(plan, object_variants, state_variants)
        )
        if missing:
            raise ValueError(
                "L3_ORACLE hint omitted required structured fact: "
                f"{missing[0]}"
            )
