#!/usr/bin/env python3
import argparse
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

from coevo.artifacts import (
    SCHEMA_VERSION,
    dataset_fingerprint,
    validate_compatible_artifacts,
)
from coevo.rewards import REWARD_FORMULA_VERSION, REWARD_NAME


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("airline", "retail", "telecom")


@dataclass(frozen=True)
class CollectionJob:
    domain: str
    shard_index: int
    num_shards: int
    output_dir: Path
    log_path: Path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


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


def collection_job(job: CollectionJob, base_env: dict[str, str]) -> None:
    env = dict(base_env, COEVO_DOMAIN=job.domain, COEVO_TASK_SPLIT="train")
    run_logged(
        [
            sys.executable,
            "scripts/collect_round.py",
            "--output-dir",
            str(job.output_dir),
            "--all-compatible",
            "--num-shards",
            str(job.num_shards),
            "--shard-index",
            str(job.shard_index),
            "--continue-on-error",
            "--resume",
        ],
        env,
        job.log_path,
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def merge_collection(output_dir: Path, jobs: list[CollectionJob]) -> dict:
    records = []
    student_rows = []
    buyer_rows = []
    errors = []
    contract_values = []
    for job in sorted(jobs, key=lambda item: (DOMAINS.index(item.domain), item.shard_index)):
        records.extend(read_jsonl(job.output_dir / "trajectories.jsonl"))
        student_rows.extend(read_jsonl(job.output_dir / "student_gkd.jsonl"))
        buyer_rows.extend(read_jsonl(job.output_dir / "buyer_grpo.jsonl"))
        summary_path = job.output_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            contract_values.append((str(summary_path), summary))
            errors.extend(
                {"domain": job.domain, **error}
                for error in summary.get("errors", [])
            )
    for filename, rows in (
        ("trajectories.jsonl", records),
        ("student_gkd.jsonl", student_rows),
        ("buyer_grpo.jsonl", buyer_rows),
    ):
        contract_values.extend(
            (f"{filename}:{index}", row)
            for index, row in enumerate(rows, start=1)
        )
    contract = validate_compatible_artifacts(contract_values)

    expected = {
        (domain, str(task.id))
        for domain in DOMAINS
        for task in get_tasks(domain, task_split_name="train")
    }
    completed = {(str(row["domain"]), str(row["task_id"])) for row in records}
    duplicates = len(records) - len(completed)
    missing = sorted(expected - completed)
    unexpected = sorted(completed - expected)
    if errors or missing or unexpected or duplicates:
        raise RuntimeError(
            "full train collection is incomplete: "
            f"records={len(records)} unique={len(completed)} expected={len(expected)} "
            f"duplicates={duplicates} missing={missing} unexpected={unexpected} "
            f"errors={errors}"
        )

    merged_dir = output_dir / "data"
    merged_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("trajectories.jsonl", records),
        ("student_gkd.jsonl", student_rows),
        ("buyer_grpo.jsonl", buyer_rows),
    ):
        (merged_dir / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    summary = {
        **contract.to_dict(),
        "expected_tasks": len(expected),
        "trajectories": len(records),
        "student_rows": len(student_rows),
        "buyer_rows": len(buyer_rows),
        "domains": {
            domain: sum(row["domain"] == domain for row in records)
            for domain in DOMAINS
        },
        "dataset_fingerprint": dataset_fingerprint(
            records,
            student_rows,
            buyer_rows,
        ),
    }
    write_json(merged_dir / "summary.json", summary)
    return summary


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = list(output_dir.glob("v*/checkpoint-*")) + list(
        output_dir.glob("checkpoint-*")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {output_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and train on all 178 tau2 v1 official train tasks"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards-per-domain", type=int, default=2)
    parser.add_argument("--collection-workers", type=int, default=4)
    parser.add_argument("--collection-passes", type=int, default=3)
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-policy-training", action="store_true")
    parser.add_argument("--skip-buyer-training", action="store_true")
    args = parser.parse_args()
    if (
        args.shards_per_domain < 1
        or args.collection_workers < 1
        or args.collection_passes < 1
    ):
        parser.error("shard and worker counts must be positive")
    if args.skip_policy_training and not args.skip_buyer_training:
        parser.error("Buyer stage training requires the preceding Student update")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target_schema_version": 2,
        "reward_name": REWARD_NAME,
        "reward_formula_version": REWARD_FORMULA_VERSION,
        "reward_scaling": "group",
        "status": "running",
        "phase": "collection",
        "domains": list(DOMAINS),
        "expected_tasks": sum(
            len(get_tasks(domain, task_split_name="train")) for domain in DOMAINS
        ),
        "started_at": time.time(),
    }
    write_json(manifest_path, manifest)
    env = dict(os.environ)
    initial_policy = env.get("COEVO_POLICY_PATH", "/models/policy")
    initial_buyer = env.get("COEVO_BUYER_PATH", initial_policy)
    env.update(
        COEVO_CURRENT_POLICY_CHECKPOINT=initial_policy,
        COEVO_PREVIOUS_POLICY_CHECKPOINT="",
        COEVO_BUYER_CHECKPOINT=initial_buyer,
        COEVO_ROUND_INDEX="0",
    )
    jobs = [
        CollectionJob(
            domain=domain,
            shard_index=shard_index,
            num_shards=args.shards_per_domain,
            output_dir=output_dir / "shards" / domain / f"shard_{shard_index}",
            log_path=output_dir / "logs" / f"collect_{domain}_{shard_index}.log",
        )
        for domain in DOMAINS
        for shard_index in range(args.shards_per_domain)
    ]

    try:
        if not args.skip_collection:
            for collection_pass in range(1, args.collection_passes + 1):
                print(
                    f"[full-trainset] collection pass "
                    f"{collection_pass}/{args.collection_passes}",
                    flush=True,
                )
                with ThreadPoolExecutor(
                    max_workers=args.collection_workers
                ) as executor:
                    futures = {
                        executor.submit(collection_job, job, env): job for job in jobs
                    }
                    for future in as_completed(futures):
                        job = futures[future]
                        future.result()
                        print(
                            f"[full-trainset] collected {job.domain} "
                            f"shard {job.shard_index + 1}/{job.num_shards}",
                            flush=True,
                        )
                try:
                    summary = merge_collection(output_dir, jobs)
                except RuntimeError:
                    if collection_pass == args.collection_passes:
                        raise
                    print(
                        "[full-trainset] collection incomplete; retrying only "
                        "missing/failed tasks",
                        flush=True,
                    )
                else:
                    break
        else:
            summary = merge_collection(output_dir, jobs)
        manifest["collection"] = summary
        policy_world_size = len(env.get("COEVO_POLICY_TRAIN_GPUS", "3").split(","))
        policy_accumulation = int(
            env.get("COEVO_POLICY_GRADIENT_ACCUMULATION_STEPS", "1")
        )
        buyer_world_size = len(env.get("COEVO_BUYER_TRAIN_GPUS", "3").split(","))
        buyer_accumulation = int(
            env.get("COEVO_BUYER_GRADIENT_ACCUMULATION_STEPS", "4")
        )
        policy_steps = math.ceil(
            summary["student_rows"] / (policy_world_size * policy_accumulation)
        )
        buyer_steps = math.ceil(
            summary["buyer_rows"] / (buyer_world_size * buyer_accumulation)
        )
        manifest["policy_steps"] = policy_steps
        manifest["buyer_steps"] = buyer_steps

        if not args.skip_policy_training:
            manifest["phase"] = "policy_training"
            write_json(manifest_path, manifest)
            buyer_stopped = env.get("COEVO_STOP_BUYER_FOR_POLICY_TRAINING") == "1"
            if buyer_stopped:
                subprocess.run(
                    ["bash", "scripts/stop_role.sh", "buyer"],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )
            run_logged(
                [
                    "bash",
                    "scripts/train_student_full.sh",
                    str(output_dir / "data/student_gkd.jsonl"),
                    str(output_dir / "policy"),
                    str(policy_steps),
                ],
                env,
                output_dir / "logs/policy_training.log",
            )
            policy_checkpoint = latest_checkpoint(output_dir / "policy")
            if str(policy_checkpoint) == str(initial_policy):
                raise ValueError("previous and current policy checkpoints are identical")
            manifest["policy_checkpoint"] = str(policy_checkpoint)
            manifest["phase"] = "policy_refresh"
            write_json(manifest_path, manifest)
            env = dict(
                env,
                COEVO_PREVIOUS_POLICY_PATH=initial_policy,
                COEVO_PREVIOUS_POLICY_CHECKPOINT=initial_policy,
                COEVO_CURRENT_POLICY_CHECKPOINT=str(policy_checkpoint),
            )
            subprocess.run(
                ["bash", "scripts/start_role.sh", "policy_previous", initial_policy],
                cwd=ROOT,
                env=env,
                check=True,
            )
            subprocess.run(
                ["bash", "scripts/stop_role.sh", "policy"],
                cwd=ROOT,
                env=env,
                check=True,
            )
            subprocess.run(
                ["bash", "scripts/start_role.sh", "policy", str(policy_checkpoint)],
                cwd=ROOT,
                env=env,
                check=True,
            )
            subprocess.run(
                ["bash", "scripts/stop_role.sh", "rollout"],
                cwd=ROOT,
                env=env,
                check=True,
            )
            subprocess.run(
                ["bash", "scripts/start_role.sh", "rollout", initial_buyer],
                cwd=ROOT,
                env=env,
                check=True,
            )
            subprocess.run(
                [sys.executable, "scripts/preflight.py", "services"],
                cwd=ROOT,
                env=env,
                check=True,
            )
            if buyer_stopped and not args.skip_buyer_training:
                subprocess.run(
                    [
                        "bash",
                        "scripts/start_role.sh",
                        "buyer",
                        env["COEVO_BUYER_PATH"],
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )

        if not args.skip_buyer_training:
            manifest["phase"] = "buyer_training"
            write_json(manifest_path, manifest)
            run_logged(
                [
                    "bash",
                    "scripts/train_buyer_full.sh",
                    str(output_dir / "data/buyer_grpo.jsonl"),
                    str(output_dir / "buyer"),
                    str(buyer_steps),
                ],
                env,
                output_dir / "logs/buyer_training.log",
            )
            manifest["buyer_checkpoint"] = str(latest_checkpoint(output_dir / "buyer"))
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["phase"] = "complete"
    manifest["finished_at"] = time.time()
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
