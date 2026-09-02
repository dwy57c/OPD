#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

from openai import OpenAI

from coevo.config import InfraConfig
from coevo.hinter_training import StudentMacroActionGenerator


def validate_hint_group(hints: list[str], state_hash: str) -> bool:
    if len(set(hints)) >= 2 and all(hints):
        return True
    print(
        json.dumps(
            {
                "event": "skip_duplicate_hint_group",
                "state_hash": state_hash,
                "sampled_hints": len(hints),
                "distinct_hints": len(set(hints)),
            }
        ),
        file=sys.stderr,
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect fresh current-hinter hints and actual Student macro-actions"
    )
    parser.add_argument("hinter_grpo_rows", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hints-per-state", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    if args.hints_per_state < 2:
        parser.error("--hints-per-state must be at least two")
    os.environ["COEVO_TEACHER_HINT_MODE"] = "none"
    base_url = os.getenv("COEVO_HINTER_URL", "").rstrip("/")
    model = os.getenv("COEVO_HINTER_MODEL", "")
    if not base_url or not model:
        parser.error("COEVO_HINTER_URL and COEVO_HINTER_MODEL are required")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    hinter = OpenAI(
        base_url=base_url,
        api_key=os.getenv("COEVO_HINTER_API_KEY", "EMPTY"),
    )
    student = StudentMacroActionGenerator(InfraConfig.from_env())
    rows = []
    skipped_duplicate_groups = []
    for line in args.hinter_grpo_rows.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        response = hinter.chat.completions.create(
            model=model,
            messages=source["messages"],
            n=args.hints_per_state,
            temperature=args.temperature,
            max_tokens=int(os.getenv("COEVO_HINTER_MAX_HINT_TOKENS", "192")),
        )
        hints = [
            (choice.message.content or "").strip() for choice in response.choices
        ]
        if not validate_hint_group(hints, str(source["state_hash"])):
            skipped_duplicate_groups.append(str(source["state_hash"]))
            continue
        for hint in hints:
            behavior = student.generate(
                student_visible_messages=source["student_visible_messages"],
                hint=hint,
                tool_schemas=source.get("tools") or [],
            )
            rows.append(
                {
                    "state_hash": source["state_hash"],
                    "public_state": source["public_state"],
                    "hint": hint,
                    "student_behavior": behavior,
                }
            )
    if not rows:
        raise ValueError("fresh behavior/hint collection produced no rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fresh_behavior_hint_samples": len(rows),
                "skipped_duplicate_groups": skipped_duplicate_groups,
            }
        )
    )


if __name__ == "__main__":
    main()
