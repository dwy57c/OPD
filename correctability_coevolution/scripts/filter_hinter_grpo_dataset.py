#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter hinter GRPO rows by curriculum")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("curriculum", type=Path)
    parser.add_argument("output", type=Path)
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
    filtered = [row for row in rows if str(row.get("scenario_id")) in selected]
    if not filtered:
        raise ValueError("curriculum selected no hinter GRPO rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in filtered),
        encoding="utf-8",
    )
    print(json.dumps({"input_rows": len(rows), "selected_rows": len(filtered)}))


if __name__ == "__main__":
    main()
