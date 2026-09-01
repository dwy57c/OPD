#!/usr/bin/env python3
import argparse
from dataclasses import replace
import json
from pathlib import Path

from coevo.config import InfraConfig
from coevo.hints import HintLevel
from coevo.orchestration import relabel_dataset_from_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relabel one immutable trajectory pool at a new hint level"
    )
    parser.add_argument("--source-trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hint-level",
        choices=[level.value for level in HintLevel],
        required=True,
    )
    parser.add_argument(
        "--sharpen-enabled", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    config = replace(
        InfraConfig.from_env(),
        hint_level=HintLevel.parse(args.hint_level),
        sharpen_enabled=args.sharpen_enabled,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = relabel_dataset_from_trajectories(
        config, args.source_trajectories, args.output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
