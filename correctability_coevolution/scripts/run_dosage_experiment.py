#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from coevo.hints import HintLevel
from coevo.config import InfraConfig
from coevo.curriculum import probe_scenario


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def token_count(path: Path) -> int:
    return sum(
        int(json.loads(line).get("target_token_count", 0))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def latest_checkpoint(output_dir: Path) -> str:
    checkpoints = list(output_dir.glob("v*/checkpoint-*")) + list(
        output_dir.glob("checkpoint-*")
    )
    if not checkpoints:
        raise FileNotFoundError(f"no Student checkpoint under {output_dir}")
    return str(max(checkpoints, key=lambda path: path.stat().st_mtime))


def collect_teacher_probe_results(config, task_ids, levels, k, probe_fn=probe_scenario):
    results = {}
    summary = {}
    for task_id in task_ids:
        results[str(task_id)] = {
            level.value: probe_fn(config, str(task_id), level, k).to_dict()
            for level in levels
        }
    for level in levels:
        rates = [
            results[str(task_id)][level.value]["success_rate"]
            for task_id in task_ids
            if results[str(task_id)][level.value]["success_rate"] is not None
        ]
        hint_errors = sum(
            results[str(task_id)][level.value]["hint_error_trials"]
            for task_id in task_ids
        )
        total_trials = len(task_ids) * k
        summary[level.value] = {
            "tasks": len(rates),
            "unmeasured_tasks": len(task_ids) - len(rates),
            "mean_success_rate": (
                sum(rates) / len(rates) if rates else None
            ),
            "k_per_task": k,
            "hint_error_trials": hint_errors,
            "contract_violation_rate": hint_errors / total_trials,
        }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 equal-budget hint dose-response experiment")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=[
            HintLevel.L1_POLICY.value,
            HintLevel.L2_PROCEDURAL.value,
            HintLevel.L3_ORACLE.value,
        ],
        default=[
            HintLevel.L1_POLICY.value,
            HintLevel.L2_PROCEDURAL.value,
            HintLevel.L3_ORACLE.value,
        ],
    )
    parser.add_argument("--active-token-budget", type=int)
    parser.add_argument("--student-steps", type=int, default=100)
    parser.add_argument("--teacher-probe-k", type=int, default=8)
    parser.add_argument("--skip-teacher-probes", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if args.active_token_budget is not None and args.active_token_budget < 1:
        parser.error("--active-token-budget must be positive")
    if args.teacher_probe_k < 1:
        parser.error("--teacher-probe-k must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ)
    base_env["COEVO_SHARPEN_ENABLED"] = "0"
    state_pools = {}
    for seed in args.seeds:
        pool_dir = output_dir / "public_state_pools" / f"seed_{seed}"
        env = dict(
            base_env,
            COEVO_HINT_LEVEL=HintLevel.L0_NONE.value,
            COEVO_SEED=str(seed),
        )
        run(
            [
                sys.executable,
                "scripts/collect_round.py",
                "--output-dir",
                str(pool_dir),
                "--task-ids",
                *args.task_ids,
                "--hint-level",
                HintLevel.L0_NONE.value,
                "--no-sharpen-enabled",
            ],
            env,
        )
        state_pools[seed] = pool_dir / "trajectories.jsonl"

    teacher_probe_results = {}
    teacher_probe_summary = {}
    if not args.skip_teacher_probes:
        probe_config = InfraConfig.from_env()
        probe_levels = [HintLevel.L0_NONE, *[HintLevel.parse(value) for value in args.levels]]
        teacher_probe_results, teacher_probe_summary = collect_teacher_probe_results(
            probe_config,
            args.task_ids,
            probe_levels,
            args.teacher_probe_k,
        )

    arms = []
    for level in args.levels:
        for seed in args.seeds:
            arm_dir = output_dir / level / f"seed_{seed}"
            data_dir = arm_dir / "data"
            env = dict(
                base_env,
                COEVO_HINT_LEVEL=level,
                COEVO_SEED=str(seed),
                COEVO_WANDB_RUN_NAME=f"dosage-{level}-seed-{seed}",
            )
            run(
                [
                    sys.executable,
                    "scripts/relabel_hint_level.py",
                    "--source-trajectories",
                    str(state_pools[seed]),
                    "--output-dir",
                    str(data_dir),
                    "--hint-level",
                    level,
                    "--no-sharpen-enabled",
                ],
                env,
            )
            dataset = data_dir / "student_gkd.jsonl"
            arms.append(
                {
                    "level": level,
                    "seed": seed,
                    "arm_dir": arm_dir,
                    "dataset": dataset,
                    "available_active_tokens": token_count(dataset),
                    "env": env,
                }
            )

    budget = args.active_token_budget or min(
        arm["available_active_tokens"] for arm in arms
    )
    if budget < 1:
        raise ValueError("at least one dosage arm has no active target tokens")
    if not args.collect_only:
        for arm in arms:
            env = dict(arm["env"], COEVO_ACTIVE_TOKEN_BUDGET=str(budget))
            student_dir = arm["arm_dir"] / "student"
            run(
                [
                    "bash",
                    "scripts/train_student_full.sh",
                    str(arm["dataset"]),
                    str(student_dir),
                    str(args.student_steps),
                ],
                env,
            )
            arm["student_checkpoint"] = latest_checkpoint(student_dir)

    manifest = {
        "experiment": "hint_dose_response_e2",
        "l0_control": {
            "training": False,
            "checkpoint": base_env.get("COEVO_POLICY_PATH"),
            "reason": "unhinted self-teacher is the base-checkpoint control",
        },
        "sharpen_enabled": False,
        "task_ids": args.task_ids,
        "seeds": args.seeds,
        "public_state_pools": {
            str(seed): str(path) for seed, path in state_pools.items()
        },
        "same_public_states_across_levels": True,
        "equal_active_token_budget": budget,
        "teacher_probe_results": teacher_probe_results,
        "teacher_probe_summary": teacher_probe_summary,
        "teacher_probes_skipped": args.skip_teacher_probes,
        "arms": [
            {key: value for key, value in arm.items() if key not in {"env", "arm_dir", "dataset"}}
            | {"dataset": str(arm["dataset"])}
            for arm in arms
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
