#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.hinter_training import build_hinter_grpo_dataset
from coevo.hinter_training.reward_dataset import REFERENCE_POOL_SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed oracle/reference-pool rows for hinter GRPO"
    )
    parser.add_argument("audit_rows", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--standard-source-level",
        choices=REFERENCE_POOL_SOURCES,
        default="oracle",
        help="Reference-pool source; retained under the old flag name for CLI compatibility.",
    )
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.audit_rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = build_hinter_grpo_dataset(
        rows, standard_source_level=args.standard_source_level
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in result),
        encoding="utf-8",
    )
    print(json.dumps({"input_rows": len(rows), "grpo_rows": len(result)}))


if __name__ == "__main__":
    main()
