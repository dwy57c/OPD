#!/usr/bin/env python3
"""Create a tiny real τ² Buyer dataset without running model rollouts."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from coevo.config import InfraConfig, ModelEndpoint
    from coevo.environment import Tau2Environment
    from coevo.rollout import NaturalDecisionCollector
    from tau2.runner.helpers import get_tasks

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-path", required=True)
    parser.add_argument("--previous-path", required=True)
    parser.add_argument("--current-url", required=True)
    parser.add_argument("--previous-url", required=True)
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--task-split", default="train")
    parser.add_argument("--task-id", default="1")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=2,
        help="Number of split-local task prompts to materialize. GRPO smoke needs two.",
    )
    parser.add_argument("--plan-mode", choices=("structured", "legacy"), default="legacy")
    args = parser.parse_args()

    if args.num_prompts < 1:
        parser.error("--num-prompts must be positive")
    available_task_ids = [
        str(task.id)
        for task in get_tasks(
            task_set_name=args.domain,
            task_split_name=args.task_split,
        )
    ]
    if args.task_id not in available_task_ids:
        raise SystemExit(
            f"task {args.task_id!r} is not in {args.domain}/{args.task_split}"
        )
    first_task_index = available_task_ids.index(args.task_id)
    selected_task_ids = available_task_ids[
        first_task_index : first_task_index + args.num_prompts
    ]
    if len(selected_task_ids) != args.num_prompts:
        raise SystemExit(
            f"only {len(selected_task_ids)} tasks remain at {args.task_id!r}; "
            f"requested {args.num_prompts}"
        )

    rows = []
    for task_id in selected_task_ids:
        config = InfraConfig(
            policy=ModelEndpoint("Qwen3-4B", args.current_url),
            previous_policy=ModelEndpoint("Qwen3-4B", args.previous_url),
            buyer_reference=ModelEndpoint("Qwen3-4B", "http://127.0.0.1:8002"),
            teacher_hint_mode="none",
            current_policy_checkpoint=str(Path(args.current_path).resolve()),
            previous_policy_checkpoint=str(Path(args.previous_path).resolve()),
            buyer_checkpoint=str(Path(args.current_path).resolve()),
            buyer_plan_mode=args.plan_mode,
            domain=args.domain,
            task_split=args.task_split,
            task_id=task_id,
        )
        environment = Tau2Environment(config)
        rows.append(
            NaturalDecisionCollector(environment, labeler=object(), max_decisions=0).buyer_row()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "task_ids": [row["task_id"] for row in rows],
                "num_prompts": len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
