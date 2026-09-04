#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.audit import counterfactual_invariance


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E1 counterfactual invariance after swapping hidden facts"
    )
    parser.add_argument("original", type=Path)
    parser.add_argument("counterfactual", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = counterfactual_invariance(_rows(args.original), _rows(args.counterfactual))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
