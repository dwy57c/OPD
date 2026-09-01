#!/usr/bin/env python3
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from loguru import logger
from tau2.run import get_tasks

from coevo.config import InfraConfig
from coevo.hints import HintLevel
from coevo.orchestration import collect_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--all-compatible", action="store_true")
    parser.add_argument("--trajectories-per-task", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--hint-level",
        choices=[level.value for level in HintLevel],
        help="Privileged-context dose; overrides COEVO_HINT_LEVEL",
    )
    parser.add_argument(
        "--sharpen-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable legacy skill-contrast temperature sharpening",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if level := os.getenv("COEVO_TAU2_LOG_LEVEL"):
        logger.remove()
        logger.add(sys.stderr, level=level)

    config = InfraConfig.from_env()
    if args.hint_level is not None or args.sharpen_enabled is not None:
        config = replace(
            config,
            hint_level=(
                HintLevel.parse(args.hint_level)
                if args.hint_level is not None
                else config.hint_level
            ),
            sharpen_enabled=(
                args.sharpen_enabled
                if args.sharpen_enabled is not None
                else config.sharpen_enabled
            ),
        )
    if args.all_compatible and args.task_ids:
        parser.error("--all-compatible and --task-ids are mutually exclusive")
    if args.all_compatible:
        tasks = get_tasks(config.domain, task_split_name=config.task_split)
        all_task_ids = [str(task.id) for task in tasks]
        task_ids = all_task_ids[args.shard_index :: args.num_shards]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "selection.json").write_text(
            json.dumps(
                {
                    "domain": config.domain,
                    "task_split": config.task_split,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                    "task_ids": task_ids,
                    "all_task_ids": all_task_ids,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(
            f"[collect] shard={args.shard_index}/{args.num_shards} "
            f"selected={len(task_ids)} total={len(all_task_ids)}",
            flush=True,
        )
    else:
        task_ids = args.task_ids or [config.task_id]
    summary = collect_dataset(
        config,
        args.output_dir,
        task_ids,
        args.trajectories_per_task,
        continue_on_error=args.continue_on_error,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
