#!/usr/bin/env python3
import argparse
from pathlib import Path

from scripts.stages.common import control_report, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt independent-auditor reports")
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = read_json(args.training_report)
    comparison = read_json(args.comparison_report)
    write_json(
        args.output,
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "control_report": control_report(training),
            "agreement": float(comparison["binary_agreement"]),
        },
    )


if __name__ == "__main__":
    main()
