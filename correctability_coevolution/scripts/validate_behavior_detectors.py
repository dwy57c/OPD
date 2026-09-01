#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.audit import validate_annotation_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Report judge-human agreement for behavior detectors")
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=200)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.annotations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < args.minimum_rows:
        raise ValueError(
            f"detector validation requires at least {args.minimum_rows} rows; got {len(rows)}"
        )
    reports = {
        name: value.to_dict() for name, value in validate_annotation_rows(rows).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
