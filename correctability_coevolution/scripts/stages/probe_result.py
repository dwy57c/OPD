#!/usr/bin/env python3
import argparse
from pathlib import Path

from scripts.stages.common import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt dosage probes to pass@k")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    scores = {
        str(task_id): float(levels["L0_NONE"]["success_rate"])
        for task_id, levels in manifest["probes"].items()
    }
    write_json(args.output, {"k": int(manifest["k"]), "scores": scores})


if __name__ == "__main__":
    main()
