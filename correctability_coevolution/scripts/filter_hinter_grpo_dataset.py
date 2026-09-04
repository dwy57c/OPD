#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.hinter_prompt import build_hinter_messages, student_profile_from_decision


def filter_and_reprofile(rows, selected, decisions):
    filtered = [row for row in rows if str(row.get("scenario_id")) in selected]
    if not filtered:
        raise ValueError("curriculum selected no hinter GRPO rows")
    for row in filtered:
        scenario_id = str(row["scenario_id"])
        profile = student_profile_from_decision(decisions[scenario_id])
        row["student_profile"] = profile
        row["messages"] = build_hinter_messages(
            row["public_state"], row["privileged_context"], profile
        )
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter hinter GRPO rows by curriculum")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("curriculum", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    args = parser.parse_args()
    selected = set(
        str(value)
        for value in json.loads(args.curriculum.read_text(encoding="utf-8"))
    )
    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = json.loads(
        args.profile_manifest.read_text(encoding="utf-8")
    ).get("decisions", {})
    filtered = filter_and_reprofile(rows, selected, decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in filtered),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "selected_rows": len(filtered),
                "student_profile_manifest": str(args.profile_manifest),
            }
        )
    )


if __name__ == "__main__":
    main()
