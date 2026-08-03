#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/workspace/OPD/correctability_coevolution")


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = list(output_dir.glob("v*/checkpoint-*")) + list(
        output_dir.glob("checkpoint-*")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {output_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/full_run")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--student-steps", type=int, default=2)
    parser.add_argument("--buyer-steps", type=int, default=2)
    parser.add_argument("--task-ids", nargs="+", default=["1"])
    parser.add_argument("--start-services", action="store_true")
    args = parser.parse_args()

    env = dict(os.environ)
    python_paths = [str(ROOT), "/workspace/OPD/tau2-bench/src"]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(python_paths)
    student_model = Path(env.get("COEVO_STUDENT_PATH", "/models/Qwen3-4B"))
    buyer_model = Path(env.get("COEVO_BUYER_PATH", "/models/Qwen3-4B"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.start_services:
        start_env = dict(
            env,
            COEVO_STUDENT_PATH=str(student_model),
            COEVO_BUYER_PATH=str(buyer_model),
        )
        run(["bash", "scripts/start_servers.sh"], start_env)

    for round_index in range(args.rounds):
        round_dir = args.output_dir / f"round_{round_index:04d}"
        data_dir = round_dir / "data"
        student_out = round_dir / "student"
        buyer_out = round_dir / "buyer"

        run(
            [
                "python",
                "scripts/collect_round.py",
                "--output-dir",
                str(data_dir),
                "--task-ids",
                *args.task_ids,
            ],
            env,
        )

        student_env = dict(env, COEVO_STUDENT_BASE_MODEL=str(student_model))
        run(
            [
                "bash",
                "scripts/train_student_full.sh",
                str(data_dir / "student_gkd.jsonl"),
                str(student_out),
                str(args.student_steps),
            ],
            student_env,
        )
        student_model = latest_checkpoint(student_out)
        run(["bash", "scripts/stop_role.sh", "student"], env)
        run(["bash", "scripts/start_role.sh", "student", str(student_model)], env)

        buyer_env = dict(env, COEVO_BUYER_BASE_MODEL=str(buyer_model))
        run(
            [
                "bash",
                "scripts/train_buyer_full.sh",
                str(data_dir / "buyer_grpo.jsonl"),
                str(buyer_out),
                str(args.buyer_steps),
            ],
            buyer_env,
        )
        buyer_model = latest_checkpoint(buyer_out)
        run(["bash", "scripts/stop_role.sh", "buyer"], env)
        run(["bash", "scripts/stop_role.sh", "rollout"], env)
        run(["bash", "scripts/start_role.sh", "buyer", str(buyer_model)], env)
        run(["bash", "scripts/start_role.sh", "rollout", str(buyer_model)], env)

        manifest = {
            "round": round_index,
            "task_ids": args.task_ids,
            "student_steps": args.student_steps,
            "buyer_steps": args.buyer_steps,
            "student_checkpoint": str(student_model),
            "buyer_checkpoint": str(buyer_model),
        }
        (round_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
