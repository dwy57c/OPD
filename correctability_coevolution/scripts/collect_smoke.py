#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.config import InfraConfig
from coevo.orchestration import collect_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--trajectories-per-task", type=int, default=1)
    args = parser.parse_args()

    config = InfraConfig.from_env()
    summary = collect_dataset(
        config,
        args.output_dir,
        args.task_ids or [config.task_id],
        args.trajectories_per_task,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
