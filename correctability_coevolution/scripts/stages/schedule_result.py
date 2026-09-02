#!/usr/bin/env python3
import argparse
from pathlib import Path
import random

from scripts.stages.common import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt h* weights to scenario IDs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    eligible = [
        (str(task_id), float(weight))
        for task_id, weight in manifest["sampling_weights"].items()
        if float(weight) > 0
        and (manifest["decisions"].get(task_id) or {}).get("level") is not None
    ]
    if not eligible:
        parser.error("h* manifest contains no trainable scenarios")
    count = args.samples or len(eligible)
    rng = random.Random(args.seed)
    selected = rng.choices(
        [task_id for task_id, _ in eligible],
        weights=[weight for _, weight in eligible],
        k=count,
    )
    write_json(args.output, {"scenario_ids": list(dict.fromkeys(selected))})


if __name__ == "__main__":
    main()
