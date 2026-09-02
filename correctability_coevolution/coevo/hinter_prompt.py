from __future__ import annotations

import json
from typing import Any, Mapping

from coevo.hints import HintLevel, hint_instruction

HINTER_SYSTEM_PROMPT = hint_instruction(HintLevel.L2_PROCEDURAL, "tau2")


def build_hinter_messages(
    public_state: Any,
    privileged_context: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Canonical prompt shared by GRPO, sampling, and Student collection."""

    level = HintLevel.parse(
        privileged_context.get("hint_level", HintLevel.L2_PROCEDURAL.value)
    )
    domain = str(privileged_context.get("domain") or "tau2")
    return [
        {"role": "system", "content": hint_instruction(level, domain)},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "public_state": public_state,
                    "privileged_context": dict(privileged_context),
                },
                ensure_ascii=False,
            ),
        },
    ]
