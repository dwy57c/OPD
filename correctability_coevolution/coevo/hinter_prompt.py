from __future__ import annotations

import json
import re
from typing import Any, Mapping


HINTER_PRIVILEGED_KEYS = frozenset(
    {"domain_policy", "authoritative_oracle_steps"}
)

HINTER_SYSTEM_PROMPT = """
Write the smallest private hint that helps the current agent choose its next
turn for the Student described by `student_profile`. The training reward
separately measures useful state-conditioned lift,
answer copying that survives without the state, distributional dose, and
length. Convert hidden instance facts into an evidence-acquisition procedure
whenever possible. Do not output exact function names, executable commands,
argument keys, JSON, code, bullet lists, or a public reply. Use no more than
140 words.

Begin with one self-report line of the form `level: L1`, `level: L2`, or
`level: L3`, then write the hint in ordinary prose. This label is diagnostic
only: it is never checked and never affects the reward.
""".strip()


def narrow_privileged_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Expose exactly the two fields shared by GRPO and open-hinter serving."""

    return {
        key: payload.get(key, "")
        for key in sorted(HINTER_PRIVILEGED_KEYS)
    }


def serialize_public_state(public_state: Any) -> str:
    return json.dumps(
        public_state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def serialize_hinter_input(
    public_state: Any,
    privileged_context: Mapping[str, Any],
    student_profile: Mapping[str, Any] | None = None,
) -> str:
    actual = set(privileged_context)
    if actual != HINTER_PRIVILEGED_KEYS:
        raise ValueError(
            "privileged_context keys must be exactly "
            f"{sorted(HINTER_PRIVILEGED_KEYS)}, got {sorted(actual)}"
        )
    return json.dumps(
        {
            "public_state": json.loads(serialize_public_state(public_state)),
            "privileged_context": dict(privileged_context),
            "student_profile": dict(student_profile or {}),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def self_reported_hint_level(text: str) -> str | None:
    """Read the optional diagnostic label without enforcing its presence."""

    match = re.search(r"(?im)^\s*level\s*:\s*(L[123])\s*$", text)
    return match.group(1).upper() if match else None


def build_hinter_messages(
    public_state: Any,
    privileged_context: Mapping[str, Any],
    student_profile: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Canonical serialization for GRPO, sampling, and Student collection."""

    return [
        {"role": "system", "content": HINTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": serialize_hinter_input(
                public_state, privileged_context, student_profile
            ),
        },
    ]
