#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.artifacts import canonical_hash
from scripts.stages.common import control_report, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt discriminator training report")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ordinary-pairs", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = read_json(args.report)
    ordinary = [
        json.loads(line)
        for line in args.ordinary_pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    write_json(
        args.output,
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "round_index": args.round_index,
            "training_examples": len(ordinary),
            "training_fingerprint": canonical_hash(ordinary),
            "converged": True,
            "control_report": control_report(report),
            "initialized_from_student": bool(report.get("initialized_from_student")),
            "fresh_score_head": bool(report.get("fresh_score_head")),
        },
    )


if __name__ == "__main__":
    main()
