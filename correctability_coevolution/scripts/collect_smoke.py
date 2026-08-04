#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

from loguru import logger
from tau2.agent.llm_agent import LLMGTAgent
from tau2.run import get_tasks

from coevo.config import InfraConfig
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
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if level := os.getenv("COEVO_TAU2_LOG_LEVEL"):
        logger.remove()
        logger.add(sys.stderr, level=level)

    config = InfraConfig.from_env()
    if args.all_compatible and args.task_ids:
        parser.error("--all-compatible and --task-ids are mutually exclusive")
    if args.all_compatible:
        tasks = get_tasks(config.domain, task_split_name=config.task_split)
        compatible = [
            str(task.id) for task in tasks if LLMGTAgent.check_valid_task(task)
        ]
        skipped = [
            str(task.id) for task in tasks if not LLMGTAgent.check_valid_task(task)
        ]
        task_ids = compatible[args.shard_index :: args.num_shards]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "selection.json").write_text(
            json.dumps(
                {
                    "domain": config.domain,
                    "task_split": config.task_split,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                    "compatible_task_ids": task_ids,
                    "all_compatible_task_ids": compatible,
                    "incompatible_task_ids": skipped,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(
            f"[collect] shard={args.shard_index}/{args.num_shards} "
            f"selected={len(task_ids)} incompatible={len(skipped)}",
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
