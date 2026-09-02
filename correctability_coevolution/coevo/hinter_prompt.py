from __future__ import annotations

import json
from typing import Any, Mapping


HINTER_SYSTEM_PROMPT = """
Write one concise private hint that helps the frozen current Student place more
probability on the supplied standard next action. The hint is private. Do not
write the public answer or an API call. Prefer the smallest useful hint: copying
hidden instance facts into observable Student behavior and unnecessary tokens
are both penalized by the training reward. If a hint_level is present, obey that
information-dose contract.
""".strip()


def build_hinter_messages(
    public_state: Any,
    privileged_context: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Canonical prompt shared by GRPO, sampling, and Student collection."""

    return [
        {"role": "system", "content": HINTER_SYSTEM_PROMPT},
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
