#!/usr/bin/env python3
"""Synchronous on-policy training over the complete tau2 v1 train split.

Each micro-round collects complete dialogues with one frozen Policy version,
trains on that batch, refreshes the Policy service, and only then collects the
next batch. This preserves on-policy provenance while updating throughout the
178-task run instead of after one large offline collection.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from tau2.run import get_tasks


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("airline", "retail", "telecom")


@dataclass(frozen=True)
class TaskRef:
    domain: str
    task_id: str

    def to_dict(self) -> dict[str, str]:
        return {"domain": self.domain, "task_id": self.task_id}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_logged(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[command] {' '.join(command)}\n")
        log.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = list(output_dir.glob("v*/checkpoint-*")) + list(
        output_dir.glob("checkpoint-*")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {output_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def wait_for_inference_services(env: dict[str, str]) -> None:
    endpoints = (
        (
            env.get("COEVO_POLICY_URL", "http://127.0.0.1:8000"),
            env.get("COEVO_POLICY_MODEL", "Qwen3-4B"),
        ),
        (
            env.get("COEVO_BUYER_URL", "http://127.0.0.1:8002"),
            env.get("COEVO_BUYER_MODEL", "Qwen3-4B"),
        ),
    )
    for url, model in endpoints:
        run(
            [
                sys.executable,
                "scripts/wait_for_servers.py",
                url,
                "--timeout",
                "30",
                "--model",
                model,
            ],
            env,
        )


def official_task_schedule() -> list[TaskRef]:
    """Round-robin domains so every micro-batch is approximately balanced."""
    by_domain = {
        domain: [str(task.id) for task in get_tasks(domain, task_split_name="train")]
        for domain in DOMAINS
    }
    schedule = []
    offset = 0
    while any(offset < len(by_domain[domain]) for domain in DOMAINS):
        for domain in DOMAINS:
            if offset < len(by_domain[domain]):
                schedule.append(TaskRef(domain, by_domain[domain][offset]))
        offset += 1
    return schedule


def batches(values: list[TaskRef], batch_size: int) -> list[list[TaskRef]]:
    return [
        values[index : index + batch_size]
        for index in range(0, len(values), batch_size)
    ]


def collect_domain(
    domain: str,
    task_ids: list[str],
    output_dir: Path,
    log_path: Path,
    base_env: dict[str, str],
) -> None:
    env = dict(
        base_env,
        COEVO_DOMAIN=domain,
        COEVO_TASK_SPLIT="train",
        COEVO_MAX_CUTOFF_TURNS="0",
    )
    run_logged(
        [
            sys.executable,
            "scripts/collect_round.py",
            "--output-dir",
            str(output_dir),
            "--task-ids",
            *task_ids,
            "--continue-on-error",
            "--resume",
        ],
        env,
        log_path,
    )


def merge_collection(round_dir: Path, batch: list[TaskRef], *, persist: bool) -> dict:
    expected = {(task.domain, task.task_id) for task in batch}
    records = []
    student_rows = []
    buyer_rows = []
    errors = []
    for domain in DOMAINS:
        domain_dir = round_dir / "domains" / domain
        records.extend(read_jsonl(domain_dir / "trajectories.jsonl"))
        student_rows.extend(read_jsonl(domain_dir / "student_gkd.jsonl"))
        buyer_rows.extend(read_jsonl(domain_dir / "buyer_grpo.jsonl"))
        summary_path = domain_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            errors.extend(
                {"domain": domain, **item} for item in summary.get("errors", [])
            )

    completed = {(str(row["domain"]), str(row["task_id"])) for row in records}
    duplicates = len(records) - len(completed)
    missing = sorted(expected - completed)
    unexpected = sorted(completed - expected)
    valid = not errors and not missing and not unexpected and duplicates == 0
    summary = {
        "valid": valid,
        "expected_tasks": len(expected),
        "trajectories": len(records),
        "student_rows": len(student_rows),
        "buyer_rows": len(buyer_rows),
        "decision_interventions": sum(
            len(record.get("student_decisions", [])) for record in records
        ),
        "duplicates": duplicates,
        "missing": missing,
        "unexpected": unexpected,
        "errors": errors,
    }
    if persist and valid:
        data_dir = round_dir / "data"
        write_jsonl(data_dir / "trajectories.jsonl", records)
        write_jsonl(data_dir / "student_gkd.jsonl", student_rows)
        write_jsonl(data_dir / "buyer_grpo.jsonl", buyer_rows)
        write_json(data_dir / "summary.json", summary)
    return summary


def collect_batch(
    round_dir: Path,
    batch: list[TaskRef],
    env: dict[str, str],
    workers: int,
    passes: int,
) -> dict:
    grouped: dict[str, list[str]] = defaultdict(list)
    for task in batch:
        grouped[task.domain].append(task.task_id)

    for pass_index in range(1, passes + 1):
        print(f"[online] collection pass {pass_index}/{passes}", flush=True)
        with ThreadPoolExecutor(max_workers=min(workers, len(grouped))) as executor:
            futures = {
                executor.submit(
                    collect_domain,
                    domain,
                    task_ids,
                    round_dir / "domains" / domain,
                    round_dir / "logs" / f"collect_{domain}.log",
                    env,
                ): domain
                for domain, task_ids in grouped.items()
            }
            for future in as_completed(futures):
                future.result()
                print(f"[online] collected domain={futures[future]}", flush=True)
        summary = merge_collection(round_dir, batch, persist=False)
        if summary["valid"]:
            return merge_collection(round_dir, batch, persist=True)
        print(
            "[online] incomplete batch; retrying only missing/failed tasks: "
            f"missing={summary['missing']} errors={len(summary['errors'])}",
            flush=True,
        )
    raise RuntimeError(f"batch collection incomplete after {passes} passes: {summary}")


def refresh_policy(
    round_dir: Path,
    policy_model: str,
    buyer_model: str,
    steps: int,
    env: dict[str, str],
) -> str:
    buyer_stopped = env.get("COEVO_STOP_BUYER_FOR_POLICY_TRAINING") == "1"
    if buyer_stopped:
        run(["bash", "scripts/stop_role.sh", "buyer"], env)

    train_env = dict(env, COEVO_POLICY_BASE_MODEL=policy_model)
    run_logged(
        [
            "bash",
            "scripts/train_student_full.sh",
            str(round_dir / "data/student_gkd.jsonl"),
            str(round_dir / "policy"),
            str(steps),
        ],
        train_env,
        round_dir / "logs/policy_training.log",
    )
    checkpoint = str(latest_checkpoint(round_dir / "policy"))

    run(["bash", "scripts/stop_role.sh", "policy"], env)
    try:
        run(["bash", "scripts/start_role.sh", "policy", checkpoint], env)
    except Exception:
        run(["bash", "scripts/start_role.sh", "policy", policy_model], env)
        if buyer_stopped:
            run(["bash", "scripts/start_role.sh", "buyer", buyer_model], env)
        raise
    if buyer_stopped:
        run(["bash", "scripts/start_role.sh", "buyer", buyer_model], env)
    return checkpoint


def merge_all_rounds(output_dir: Path, round_count: int) -> None:
    all_records = []
    all_student_rows = []
    all_buyer_rows = []
    for round_index in range(round_count):
        data_dir = output_dir / f"round_{round_index:04d}" / "data"
        all_records.extend(read_jsonl(data_dir / "trajectories.jsonl"))
        all_student_rows.extend(read_jsonl(data_dir / "student_gkd.jsonl"))
        all_buyer_rows.extend(read_jsonl(data_dir / "buyer_grpo.jsonl"))
    merged_dir = output_dir / "data"
    write_jsonl(merged_dir / "trajectories.jsonl", all_records)
    write_jsonl(merged_dir / "student_gkd.jsonl", all_student_rows)
    write_jsonl(merged_dir / "buyer_grpo.jsonl", all_buyer_rows)
    write_json(
        merged_dir / "summary.json",
        {
            "trajectories": len(all_records),
            "student_rows": len(all_student_rows),
            "buyer_rows": len(all_buyer_rows),
            "rounds": round_count,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--collection-workers", type=int, default=3)
    parser.add_argument("--collection-passes", type=int, default=3)
    parser.add_argument(
        "--student-steps",
        type=int,
        default=0,
        help="0 trains one effective epoch over each freshly collected batch",
    )
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--skip-policy-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.collection_workers, args.collection_passes) < 1:
        parser.error("batch size, workers, and passes must be positive")
    if args.student_steps < 0:
        parser.error("--student-steps must be non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, COEVO_MAX_CUTOFF_TURNS="0")
    policy_model = env.get("COEVO_POLICY_PATH", "/models/student")
    buyer_model = env.get("COEVO_BUYER_PATH", "/models/teacher")
    wait_for_inference_services(env)

    schedule = official_task_schedule()
    if args.task_limit is not None:
        schedule = schedule[: args.task_limit]
    micro_batches = batches(schedule, args.batch_size)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "status": "running",
        "phase": "collection",
        "mode": "synchronous_microbatch_on_policy",
        "complete_dialogue": True,
        "batch_size": args.batch_size,
        "tasks": len(schedule),
        "rounds": len(micro_batches),
        "completed_rounds": 0,
        "started_at": time.time(),
    }
    write_json(manifest_path, manifest)

    try:
        for round_index, batch in enumerate(micro_batches):
            round_dir = output_dir / f"round_{round_index:04d}"
            round_manifest_path = round_dir / "manifest.json"
            if args.resume and round_manifest_path.is_file():
                previous = json.loads(round_manifest_path.read_text())
                if previous.get("status") == "complete":
                    policy_model = previous.get("policy_checkpoint", policy_model)
                    manifest["completed_rounds"] = round_index + 1
                    write_json(manifest_path, manifest)
                    continue

            round_manifest = {
                "round": round_index,
                "status": "running",
                "phase": "collection",
                "policy_version": policy_model,
                "tasks": [task.to_dict() for task in batch],
                "complete_dialogue": True,
            }
            write_json(round_manifest_path, round_manifest)
            summary = collect_batch(
                round_dir,
                batch,
                env,
                args.collection_workers,
                args.collection_passes,
            )
            round_manifest["collection"] = summary
            world_size = len(env.get("COEVO_POLICY_TRAIN_GPUS", "3").split(","))
            accumulation = int(env.get("COEVO_POLICY_GRADIENT_ACCUMULATION_STEPS", "1"))
            steps = args.student_steps or max(
                1, math.ceil(summary["student_rows"] / (world_size * accumulation))
            )
            round_manifest["student_steps"] = steps

            if not args.skip_policy_training:
                manifest["phase"] = round_manifest["phase"] = "policy_training"
                write_json(manifest_path, manifest)
                write_json(round_manifest_path, round_manifest)
                policy_model = refresh_policy(
                    round_dir, policy_model, buyer_model, steps, env
                )
                round_manifest["policy_checkpoint"] = policy_model
            round_manifest["status"] = "complete"
            round_manifest["phase"] = "complete"
            write_json(round_manifest_path, round_manifest)
            manifest["completed_rounds"] = round_index + 1
            manifest["phase"] = "collection"
            manifest["current_policy"] = policy_model
            write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(manifest_path, manifest)
        raise

    merge_all_rounds(output_dir, len(micro_batches))
    manifest["status"] = "complete"
    manifest["phase"] = "complete"
    manifest["finished_at"] = time.time()
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
