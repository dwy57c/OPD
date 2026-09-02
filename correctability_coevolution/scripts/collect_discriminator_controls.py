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
    parser.add_argument("--audit-rows", type=Path, required=True)
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
    audit_by_state = {}
    for line in args.audit_rows.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        payload = row.get("hint") or {}
        note = str((payload.get("hint") or {}).get("plan") or "").strip()
        if note:
            audit_by_state.setdefault(str(row["state_hash"]), []).append(
                (row, note)
            )
    natural_pairs = []
    for state_hash, rows in audit_by_state.items():
        l3_rows = [
            value for value in rows if value[0].get("hint_level") == "L3_ORACLE"
        ]
        decoys = [
            note for row, note in rows if row.get("hint_level") != "L3_ORACLE"
        ]
        if not l3_rows or not decoys:
            continue
        for row, natural_hint in l3_rows:
            behavior = generator.generate(
                student_visible_messages=row["student_visible_messages"],
                hint=natural_hint,
                tool_schemas=row.get("tools") or [],
            )
            natural_pairs.append(
                {
                    "state_hash": state_hash,
                    "public_state": row["public_state"],
                    "student_behavior": behavior,
                    "positive_hint": natural_hint,
                    "negative_hint": decoys[0],
                    "control_type": "explicit_copy_natural",
                }
            )
    if not explicit_rows or not useless_rows or not natural_pairs:
        raise ValueError("control generation produced no rows")
    write_jsonl(args.output_dir / "explicit_copy_controls.jsonl", explicit_rows)
    write_jsonl(args.output_dir / "useless_controls.jsonl", useless_rows)
    write_jsonl(
        args.output_dir / "explicit_copy_natural_pairs.jsonl", natural_pairs
    )
    print(
        json.dumps(
            {
                "explicit_copy_controls": len(explicit_rows),
                "useless_controls": len(useless_rows),
                "explicit_copy_natural_pairs": len(natural_pairs),
            }
        )
    )


if __name__ == "__main__":
    main()
