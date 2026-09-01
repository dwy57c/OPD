#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.training.gated_gkd import truncate_rows_to_active_token_budget


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic equal-active-token dosage arm"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected, used = truncate_rows_to_active_token_budget(rows, args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "selected_rows": len(selected),
                "active_token_budget": args.budget,
                "active_tokens_selected": used,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
