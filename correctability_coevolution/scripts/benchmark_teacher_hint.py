#!/usr/bin/env python3
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

from loguru import logger
from tau2.data_model.message import AssistantMessage, ToolMessage
from tau2.run import get_tasks

from coevo.config import HintEndpoint, InfraConfig
from coevo.environment import Tau2Environment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired tau2 benchmark for one policy without/with private hints"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--task-split", default="test")
    parser.add_argument("--task-ids", nargs="*")
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--hint-refresh",
        choices=["session", "turn"],
        default="session",
        help=(
            "Refresh the closed-model plan once per dialogue or before every "
            "Agent turn. 'session' is the active one-hint-per-task contract."
        ),
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["none", "closed_model"],
        default=["none", "closed_model"]
    )
    return parser.parse_args()


def evaluate_run(environment, orchestrator):
    simulation = orchestrator.run()
    try:
        simulation.reward_info = environment.evaluate(simulation)
    except ValueError as error:
        if "Unknown tool" not in str(error):
            raise
        simulation.reward_info = None
        return simulation, f"evaluation error: {error}"
    return simulation, None


def run_one(base_config, hinter, args, task_id, mode):
    config = replace(
        base_config,
        domain=args.domain,
        task_split=args.task_split,
        task_id=str(task_id),
        branch_max_steps=args.max_steps,
        seed=args.seed,
        teacher_hint_mode=mode,
        teacher_hinter=hinter if mode == "closed_model" else None,
    )
    environment = Tau2Environment(config)
    orchestrator = environment.orchestrator(
        environment.initial_history(), "teacher", seed=args.seed
    )
    if mode == "closed_model":
        orchestrator.agent.refresh_hint_each_turn = args.hint_refresh == "turn"
    started = time.monotonic()
    error = None
    try:
        simulation, error = evaluate_run(environment, orchestrator)
    except Exception as exc:
        return {
            "mode": mode,
            "task_id": str(task_id),
            "seed": args.seed,
            "hint_refresh": args.hint_refresh if mode == "closed_model" else None,
            "reward": None,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_time_seconds": round(time.monotonic() - started, 3),
            "hint_records": getattr(orchestrator.agent, "hint_records", []),
        }

    messages = simulation.get_messages()
    reward = simulation.reward_info.reward if simulation.reward_info else None
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    return {
        "mode": mode,
        "task_id": str(task_id),
        "seed": args.seed,
        "hint_refresh": args.hint_refresh if mode == "closed_model" else None,
        "reward": reward,
        "success": bool(reward is not None and reward > 0),
        "termination_reason": simulation.termination_reason.value,
        "message_count": len(messages),
        "teacher_turns": sum(isinstance(m, AssistantMessage) for m in messages),
        "tool_results": len(tool_messages),
        "tool_errors": sum(bool(message.error) for message in tool_messages),
        "simulation_duration_seconds": simulation.duration,
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "error": error,
        "hint_records": getattr(orchestrator.agent, "hint_records", []),
        "simulation": simulation.model_dump(mode="json"),
    }


def summarize(rows, modes):
    summary = {}
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        completed = [row for row in selected if row["reward"] is not None]
        hint_records = [
            record for row in completed for record in row.get("hint_records", [])
        ]
        summary[mode] = {
            "attempted": len(selected),
            "completed": len(completed),
            "successes": sum(row["success"] for row in completed),
            "success_rate": (
                sum(row["success"] for row in completed) / len(completed)
                if completed else None
            ),
            "mean_reward": (
                sum(row["reward"] for row in completed) / len(completed)
                if completed else None
            ),
            "mean_teacher_turns": (
                sum(row["teacher_turns"] for row in completed) / len(completed)
                if completed else None
            ),
            "mean_wall_time_seconds": (
                sum(row["wall_time_seconds"] for row in completed) / len(completed)
                if completed else None
            ),
            "tool_errors": sum(row.get("tool_errors", 0) for row in completed),
            "termination_reasons": dict(
                Counter(row.get("termination_reason") for row in completed)
            ),
            "hint_calls": len(hint_records),
            "mean_hint_latency_ms": (
                sum(record["latency_ms"] for record in hint_records)
                / len(hint_records)
                if hint_records else None
            ),
            "errors": sum(bool(row.get("error")) for row in selected),
        }
    paired = {}
    if set(modes) >= {"none", "closed_model"}:
        by_key = {(row["task_id"], row["mode"]): row for row in rows}
        comparable = [
            task_id for task_id in {row["task_id"] for row in rows}
            if (task_id, "none") in by_key
            and (task_id, "closed_model") in by_key
            and by_key[(task_id, "none")]["reward"] is not None
            and by_key[(task_id, "closed_model")]["reward"] is not None
        ]
        paired = {
            "count": len(comparable),
            "improved": sum(
                by_key[(task, "closed_model")]["reward"]
                > by_key[(task, "none")]["reward"] for task in comparable
            ),
            "regressed": sum(
                by_key[(task, "closed_model")]["reward"]
                < by_key[(task, "none")]["reward"] for task in comparable
            ),
            "tied": sum(
                by_key[(task, "closed_model")]["reward"]
                == by_key[(task, "none")]["reward"] for task in comparable
            ),
        }
    return {"modes": summary, "paired": paired}


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="CRITICAL")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "runs.jsonl"
    base_config = InfraConfig.from_env()
    # Paired runs use the exact same local policy endpoint and differ only by the
    # private closed-model hint. Greedy decoding removes sampling noise.
    base_config = replace(
        base_config,
        policy=replace(base_config.policy, temperature=0.0, seed=args.seed),
        buyer_reference=replace(
            base_config.buyer_reference, temperature=0.0, seed=args.seed
        ),
    )
    hinter = HintEndpoint.from_env(required="closed_model" in args.modes)
    tasks = get_tasks(args.domain, task_split_name=args.task_split)
    by_id = {str(task.id): task for task in tasks}
    if args.task_ids:
        missing = [task_id for task_id in args.task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"unknown task IDs: {missing}")
        selected = [by_id[task_id] for task_id in args.task_ids]
    else:
        selected = tasks
    if args.num_tasks is not None:
        selected = selected[: args.num_tasks]

    jobs = [(task.id, mode) for task in selected for mode in args.modes]
    rows = []
    output_jsonl.write_text("", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = {
            executor.submit(run_one, base_config, hinter, args, task_id, mode): (
                task_id,
                mode,
            )
            for task_id, mode in jobs
        }
        for future in as_completed(futures):
            task_id, mode = futures[future]
            print(f"[{mode}] task={task_id} finished", flush=True)
            row = future.result()
            rows.append(row)
            with output_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"  reward={row['reward']} turns={row.get('teacher_turns')} "
                f"error={row.get('error')}",
                flush=True,
            )

    order = {(str(task_id), mode): index for index, (task_id, mode) in enumerate(jobs)}
    rows.sort(key=lambda row: order[(row["task_id"], row["mode"])])

    summary = {
        "domain": args.domain,
        "task_split": args.task_split,
        "task_ids": [str(task.id) for task in selected],
        "seed": args.seed,
        "hint_refresh": args.hint_refresh,
        **summarize(rows, args.modes),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
