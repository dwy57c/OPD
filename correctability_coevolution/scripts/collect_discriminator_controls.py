#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from coevo.config import InfraConfig
from coevo.hinter_training import StudentMacroActionGenerator


EXPLICIT_DECOY_ACTION = {
    "role": "assistant",
    "content": "Please restate the request and provide any missing reference number.",
}
USELESS_HINTS = (
    "Be concise and professional.",
    "Respond politely and clearly.",
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate explicit-copy and useless-hint discriminator controls"
    )
    parser.add_argument("hinter_grpo_rows", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["COEVO_TEACHER_HINT_MODE"] = "none"
    config = InfraConfig.from_env()
    generator = StudentMacroActionGenerator(config)
    explicit_rows = []
    useless_rows = []
    for line in args.hinter_grpo_rows.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        standard_action = row["student_visible_messages"][-1]
        explicit_hints = (
            "Copy this exact next action: "
            + json.dumps(standard_action, ensure_ascii=False, sort_keys=True),
            "Copy this exact next action: "
            + json.dumps(EXPLICIT_DECOY_ACTION, ensure_ascii=False, sort_keys=True),
        )
        for hint in explicit_hints:
            behavior = generator.generate(
                student_visible_messages=row["student_visible_messages"],
                hint=hint,
                tool_schemas=row.get("tools") or [],
            )
            explicit_rows.append(
                {
                    "state_hash": row["state_hash"],
                    "public_state": row["public_state"],
                    "hint": hint,
                    "student_behavior": behavior,
                }
            )
        for hint in USELESS_HINTS:
            behavior = generator.generate(
                student_visible_messages=row["student_visible_messages"],
                hint=hint,
                tool_schemas=row.get("tools") or [],
            )
            useless_rows.append(
                {
                    "state_hash": row["state_hash"],
                    "public_state": row["public_state"],
                    "hint": hint,
                    "student_behavior": behavior,
                }
            )
    if not explicit_rows or not useless_rows:
        raise ValueError("control generation produced no rows")
    write_jsonl(args.output_dir / "explicit_copy_controls.jsonl", explicit_rows)
    write_jsonl(args.output_dir / "useless_controls.jsonl", useless_rows)
    print(
        json.dumps(
            {
                "explicit_copy_controls": len(explicit_rows),
                "useless_controls": len(useless_rows),
            }
        )
    )


if __name__ == "__main__":
    main()
