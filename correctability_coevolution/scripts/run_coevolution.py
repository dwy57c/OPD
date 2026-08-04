#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = list(output_dir.glob("v*/checkpoint-*")) + list(
        output_dir.glob("checkpoint-*")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {output_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def resolve_tau2_src(env: dict[str, str]) -> Path:
    if env.get("COEVO_TAU2_SRC"):
        return Path(env["COEVO_TAU2_SRC"]).expanduser().resolve()
    candidates = (
        REPO_ROOT / "tau2-bench/src",
        REPO_ROOT / "third_party/tau2-bench/src",
        ROOT / "third_party/tau2-bench/src",
    )
    return next(
        (path for path in candidates if (path / "tau2").is_dir()), candidates[1]
    )


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


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
    tau2_src = resolve_tau2_src(env)
    env["COEVO_ROOT"] = str(ROOT)
    env["COEVO_TAU2_SRC"] = str(tau2_src)
    env.setdefault("TAU2_DATA_DIR", str(tau2_src.parent / "data"))
    python_paths = [str(ROOT), str(tau2_src)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(python_paths)
    model_root = env.get("COEVO_MODEL_ROOT", "/models")
    student_model = env.get("COEVO_STUDENT_PATH", f"{model_root}/Qwen3-4B")
    buyer_model = env.get("COEVO_BUYER_PATH", f"{model_root}/Qwen3-4B")
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.start_services:
        start_env = dict(
            env,
            COEVO_STUDENT_PATH=str(student_model),
            COEVO_BUYER_PATH=str(buyer_model),
        )
        run(["bash", "scripts/start_servers.sh"], start_env)
    else:
        run([sys.executable, "scripts/preflight.py", "services"], env)

    for round_index in range(args.rounds):
        round_dir = output_dir / f"round_{round_index:04d}"
        data_dir = round_dir / "data"
        student_out = round_dir / "student"
        buyer_out = round_dir / "buyer"
        manifest = {
            "round": round_index,
            "status": "running",
            "phase": "collection",
            "task_split": env.get("COEVO_TASK_SPLIT", "train"),
            "task_ids": args.task_ids,
            "student_steps": args.student_steps,
            "buyer_steps": args.buyer_steps,
            "student_base_model": str(student_model),
            "buyer_base_model": str(buyer_model),
        }
        manifest_path = round_dir / "manifest.json"
        write_manifest(manifest_path, manifest)
        try:
            run(
                [
                    sys.executable,
                    "scripts/collect_round.py",
                    "--output-dir",
                    str(data_dir),
                    "--task-ids",
                    *args.task_ids,
                ],
                env,
            )

            manifest["phase"] = "student_training"
            write_manifest(manifest_path, manifest)
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
            next_student_model = str(latest_checkpoint(student_out))
            manifest["student_checkpoint"] = next_student_model

            manifest["phase"] = "student_refresh"
            write_manifest(manifest_path, manifest)
            run(["bash", "scripts/stop_role.sh", "student"], env)
            try:
                run(
                    ["bash", "scripts/start_role.sh", "student", next_student_model],
                    env,
                )
            except Exception:
                run(["bash", "scripts/start_role.sh", "student", student_model], env)
                raise
            student_model = next_student_model

            manifest["phase"] = "buyer_training"
            write_manifest(manifest_path, manifest)
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
            next_buyer_model = str(latest_checkpoint(buyer_out))
            manifest["buyer_checkpoint"] = next_buyer_model

            manifest["phase"] = "buyer_refresh"
            write_manifest(manifest_path, manifest)
            run(["bash", "scripts/stop_role.sh", "buyer"], env)
            run(["bash", "scripts/stop_role.sh", "rollout"], env)
            try:
                run(["bash", "scripts/start_role.sh", "buyer", next_buyer_model], env)
                run(
                    ["bash", "scripts/start_role.sh", "rollout", next_buyer_model],
                    env,
                )
            except Exception:
                run(["bash", "scripts/stop_role.sh", "buyer"], env)
                run(["bash", "scripts/stop_role.sh", "rollout"], env)
                run(["bash", "scripts/start_role.sh", "buyer", buyer_model], env)
                run(["bash", "scripts/start_role.sh", "rollout", buyer_model], env)
                raise
            buyer_model = next_buyer_model
        except Exception as error:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
            write_manifest(manifest_path, manifest)
            raise

        manifest["status"] = "complete"
        manifest["phase"] = "complete"
        write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
