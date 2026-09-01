#!/usr/bin/env python3
"""Archived Buyer/LP reproduction controller.

The active contingent-tutoring experiments use audit_hint_ladder.py,
run_dosage_experiment.py, and run_dosage_curriculum.py. This controller remains
only to reproduce historical checkpoints.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def model_revision(reference: str) -> str:
    from coevo.artifacts import model_manifest_revision

    return model_manifest_revision(reference)


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run_best_effort(
    command: list[str], env: dict[str, str], errors: list[str]
) -> None:
    try:
        run(command, env)
    except Exception as error:
        errors.append(f"{' '.join(command)}: {type(error).__name__}: {error}")


def rollback_services(
    env: dict[str, str], policy_model: str, buyer_model: str
) -> list[str]:
    """Restore the last committed checkpoint pair after a failed round."""
    errors: list[str] = []
    rollback_env = dict(
        env,
        COEVO_POLICY_PATH=policy_model,
        COEVO_BUYER_PATH=buyer_model,
        COEVO_CURRENT_POLICY_CHECKPOINT=policy_model,
        COEVO_BUYER_CHECKPOINT=buyer_model,
    )
    rollback_env.pop("COEVO_PREVIOUS_POLICY_PATH", None)
    rollback_env.pop("COEVO_PREVIOUS_POLICY_CHECKPOINT", None)
    for role in ("rollout", "policy_previous", "policy", "buyer"):
        run_best_effort(["bash", "scripts/stop_role.sh", role], rollback_env, errors)
    run_best_effort(
        ["bash", "scripts/start_role.sh", "policy", policy_model],
        rollback_env,
        errors,
    )
    run_best_effort(
        ["bash", "scripts/start_role.sh", "buyer", buyer_model],
        rollback_env,
        errors,
    )
    run_best_effort(
        ["bash", "scripts/start_role.sh", "rollout", buyer_model],
        rollback_env,
        errors,
    )
    return errors


def validate_resume_manifest(
    manifest: dict,
    *,
    path: Path,
    round_index: int,
    policy_model: str,
    buyer_model: str,
    task_ids: list[str],
    env: dict[str, str],
) -> None:
    expected = {
        "schema_version": 4,
        "round": round_index,
        "student_checkpoint_before": str(policy_model),
        "buyer_checkpoint_before": str(buyer_model),
        "task_ids": task_ids,
        "reward_name": "tau2_stage_learning_progress",
        "reward_formula_version": "previous-skill-anchor-progress-v3",
        "buyer_teacher_anchor": "previous",
        "teacher_target_version": env.get(
            "COEVO_TEACHER_TARGET_VERSION", "skill-contrast-natural-note-v3"
        ),
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"refusing to resume {path} with a different checkpoint/data contract: "
            f"{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}"
        )


def needs_previous_policy(env: dict[str, str]) -> bool:
    return bool(
        env.get("COEVO_PREVIOUS_POLICY_PATH")
        or env.get("COEVO_PREVIOUS_POLICY_CHECKPOINT")
    )


def validate_checkpoint_order(previous: str, current: str) -> None:
    if not previous or not current:
        raise ValueError("Buyer training requires both previous and current checkpoints")
    if previous == current:
        raise ValueError("Buyer training refuses an identical checkpoint pair")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/full_run")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--student-steps", type=int, default=2)
    parser.add_argument("--buyer-steps", type=int, default=2)
    parser.add_argument("--task-ids", nargs="+", default=["1"])
    parser.add_argument("--start-services", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-deprecated-buyer-lp",
        action="store_true",
        help="Explicitly acknowledge that this runs the archived Buyer/LP method",
    )
    args = parser.parse_args()
    if not args.allow_deprecated_buyer_lp:
        parser.error(
            "Buyer/LP co-evolution is archived; pass --allow-deprecated-buyer-lp "
            "only for historical reproduction"
        )

    env = dict(os.environ)
    # Pin the historical contract even when the active .env defaults to L2/raw.
    env["COEVO_HINT_LEVEL"] = "L3_ORACLE"
    env["COEVO_SHARPEN_ENABLED"] = "1"
    env["COEVO_TEACHER_TARGET_VERSION"] = "skill-contrast-natural-note-v3"
    tau2_src = resolve_tau2_src(env)
    env["COEVO_ROOT"] = str(ROOT)
    env["COEVO_TAU2_SRC"] = str(tau2_src)
    env.setdefault("TAU2_DATA_DIR", str(tau2_src.parent / "data"))
    python_paths = [str(ROOT), str(tau2_src)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(python_paths)
    model_root = env.get("COEVO_MODEL_ROOT", "/models")
    policy_model = env.get("COEVO_POLICY_PATH", f"{model_root}/policy")
    buyer_model = env.get("COEVO_BUYER_PATH", str(policy_model))
    previous_policy_model = ""
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_rounds: set[int] = set()
    found_existing = False
    for round_index in range(args.rounds):
        manifest_path = output_dir / f"round_{round_index:04d}" / "manifest.json"
        if not manifest_path.is_file():
            break
        found_existing = True
        if not args.resume:
            raise FileExistsError(
                f"{manifest_path} already exists; pass --resume to validate and reuse it"
            )
        existing = json.loads(manifest_path.read_text())
        validate_resume_manifest(
            existing,
            path=manifest_path,
            round_index=round_index,
            policy_model=str(policy_model),
            buyer_model=str(buyer_model),
            task_ids=args.task_ids,
            env=env,
        )
        if existing.get("status") != "complete":
            break
        next_policy = existing.get("student_checkpoint_after")
        next_buyer = existing.get("buyer_checkpoint_after")
        if not next_policy or not next_buyer:
            raise ValueError(
                f"complete manifest {manifest_path} is missing committed checkpoints"
            )
        previous_policy_model = str(policy_model)
        policy_model = str(next_policy)
        buyer_model = str(next_buyer)
        completed_rounds.add(round_index)
    if args.resume and found_existing and not args.start_services:
        raise ValueError(
            "--resume requires --start-services so checkpoint roles can be restored "
            "and verified by this controller"
        )

    if args.start_services:
        start_env = dict(
            env,
            # Collection needs the policy and fixed User/Buyer endpoint only.
            # Starting the GRPO rollout copy here wastes an entire model replica
            # before the Student checkpoint pair exists.
            COEVO_START_ROLLOUT="false",
            COEVO_POLICY_PATH=str(policy_model),
            COEVO_BUYER_PATH=str(buyer_model),
        )
        if needs_previous_policy(start_env):
            start_env["COEVO_PREVIOUS_POLICY_PATH"] = str(policy_model)
            start_env["COEVO_PREVIOUS_POLICY_CHECKPOINT"] = str(policy_model)
        # Controller-managed startup first removes only roles recorded in this
        # project's PID registry.  This makes a retry deterministic and avoids
        # mistaking an old listener for a newly restored checkpoint.
        run(["bash", "scripts/stop_servers.sh"], start_env)
        run(["bash", "scripts/start_servers.sh"], start_env)
    else:
        run([sys.executable, "scripts/preflight.py", "services"], env)

    for round_index in range(args.rounds):
        if round_index in completed_rounds:
            continue
        round_env = dict(
            env,
            COEVO_TEACHER_ANCHOR="current",
            COEVO_ROUND_INDEX=str(round_index),
            COEVO_CURRENT_POLICY_CHECKPOINT=str(policy_model),
            COEVO_PREVIOUS_POLICY_CHECKPOINT=str(previous_policy_model),
            COEVO_BUYER_CHECKPOINT=str(buyer_model),
            COEVO_CURRENT_POLICY_REVISION=model_revision(str(policy_model)),
            COEVO_PREVIOUS_POLICY_REVISION=model_revision(
                str(previous_policy_model)
            ),
            COEVO_BUYER_REVISION=model_revision(str(buyer_model)),
        )
        round_dir = output_dir / f"round_{round_index:04d}"
        data_dir = round_dir / "data"
        student_out = round_dir / "student"
        buyer_out = round_dir / "buyer"
        previous_manifest = None
        manifest_path = round_dir / "manifest.json"
        if manifest_path.is_file():
            previous_manifest = json.loads(manifest_path.read_text())
            validate_resume_manifest(
                previous_manifest,
                path=manifest_path,
                round_index=round_index,
                policy_model=str(policy_model),
                buyer_model=str(buyer_model),
                task_ids=args.task_ids,
                env=env,
            )
        manifest = {
            "schema_version": 4,
            "round": round_index,
            "status": "running",
            "phase": "collection",
            "task_split": env.get("COEVO_TASK_SPLIT", "train"),
            "task_ids": args.task_ids,
            "student_steps": args.student_steps,
            "buyer_steps": args.buyer_steps,
            "student_checkpoint_before": str(policy_model),
            "student_checkpoint_after": None,
            "student_revision_before": model_revision(str(policy_model)),
            "student_revision_after": None,
            "buyer_checkpoint_before": str(buyer_model),
            "buyer_checkpoint_after": None,
            "buyer_revision_before": model_revision(str(buyer_model)),
            "buyer_revision_after": None,
            "previous_policy_endpoint": round_env.get(
                "COEVO_PREVIOUS_POLICY_URL", "http://127.0.0.1:8001"
            ),
            "current_policy_endpoint": round_env.get(
                "COEVO_POLICY_URL", "http://127.0.0.1:8000"
            ),
            "reward_name": "tau2_stage_learning_progress",
            "reward_formula_version": "previous-skill-anchor-progress-v3",
            "reward_scaling": "group",
            "buyer_teacher_anchor": "previous",
            "dataset_schema_version": 4,
            "target_schema_version": 2,
            "teacher_target_version": round_env.get(
                "COEVO_TEACHER_TARGET_VERSION", "skill-contrast-natural-note-v3"
            ),
            "tokenizer_id": round_env.get(
                "COEVO_TOKENIZER_ID",
                f"{env.get('COEVO_POLICY_PATH', policy_model)}@"
                f"{model_revision(env.get('COEVO_POLICY_PATH', policy_model))}",
            ),
            "tokenizer_hash": None,
            "teacher_checkpoint": None,
            "checkpoint_tuple": None,
            "wandb": {
                "report_to": round_env.get("COEVO_REPORT_TO", "wandb"),
                "project": round_env.get(
                    "COEVO_WANDB_PROJECT", "opd-stage-curriculum"
                ),
            },
        }
        if previous_manifest is not None:
            attempts = list(previous_manifest.get("attempts", []))
            attempts.append(
                {
                    "status": previous_manifest.get("status"),
                    "phase": previous_manifest.get("phase"),
                    "error": previous_manifest.get("error"),
                    "student_checkpoint_after": previous_manifest.get(
                        "student_checkpoint_after"
                    ),
                    "buyer_checkpoint_after": previous_manifest.get(
                        "buyer_checkpoint_after"
                    ),
                }
            )
            manifest["attempts"] = attempts
        write_manifest(manifest_path, manifest)
        services_changed = False
        try:
            collect_command = [
                    sys.executable,
                    "scripts/collect_round.py",
                    "--output-dir",
                    str(data_dir),
                    "--task-ids",
                    *args.task_ids,
                ]
            if args.resume and data_dir.exists():
                collect_command.append("--resume")
            run(collect_command, round_env)
            summary_path = data_dir / "summary.json"
            if summary_path.is_file():
                collection_summary = json.loads(summary_path.read_text())
                manifest["collection"] = {
                    "dataset_fingerprint": collection_summary.get(
                        "dataset_fingerprint"
                    ),
                    "trajectories": collection_summary.get("trajectories"),
                    "student_training_rows": collection_summary.get(
                        "student_training_rows"
                    ),
                    "rejected_teacher_targets": collection_summary.get(
                        "rejected_teacher_targets"
                    ),
                }

            manifest["phase"] = "policy_training"
            write_manifest(manifest_path, manifest)
            student_env = dict(
                round_env,
                COEVO_POLICY_BASE_MODEL=str(policy_model),
            )
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
            next_policy_model = str(latest_checkpoint(student_out))
            validate_checkpoint_order(str(policy_model), next_policy_model)
            manifest["student_checkpoint_after"] = next_policy_model
            manifest["student_revision_after"] = model_revision(
                next_policy_model
            )
            from coevo.artifacts import canonical_hash

            # The latest Student update was supervised by S_k+skill. Buyer
            # progress must keep that pre-update Teacher demonstration fixed
            # instead of moving the target to S_(k+1)+skill.
            manifest["teacher_checkpoint"] = str(policy_model)
            manifest["tokenizer_hash"] = canonical_hash(manifest["tokenizer_id"])
            manifest["checkpoint_tuple"] = {
                "previous_checkpoint": str(policy_model),
                "current_checkpoint": next_policy_model,
                "buyer_checkpoint": str(buyer_model),
            }

            manifest["phase"] = "policy_refresh"
            write_manifest(manifest_path, manifest)
            buyer_phase_env = dict(
                round_env,
                COEVO_TEACHER_ANCHOR="previous",
                COEVO_PREVIOUS_POLICY_PATH=str(policy_model),
                COEVO_PREVIOUS_POLICY_CHECKPOINT=str(policy_model),
                COEVO_PREVIOUS_POLICY_REVISION=model_revision(
                    str(policy_model)
                ),
                COEVO_CURRENT_POLICY_CHECKPOINT=next_policy_model,
                COEVO_CURRENT_POLICY_REVISION=model_revision(
                    next_policy_model
                ),
            )
            services_changed = True
            if needs_previous_policy(buyer_phase_env):
                run(
                    ["bash", "scripts/stop_role.sh", "policy_previous"],
                    buyer_phase_env,
                )
                run(
                    [
                        "bash",
                        "scripts/start_role.sh",
                        "policy_previous",
                        str(policy_model),
                    ],
                    buyer_phase_env,
                )
            run(["bash", "scripts/stop_role.sh", "policy"], buyer_phase_env)
            try:
                run(
                    ["bash", "scripts/start_role.sh", "policy", next_policy_model],
                    buyer_phase_env,
                )
            except Exception:
                run(
                    ["bash", "scripts/start_role.sh", "policy", policy_model],
                    buyer_phase_env,
                )
                if needs_previous_policy(buyer_phase_env):
                    run(
                        ["bash", "scripts/stop_role.sh", "policy_previous"],
                        buyer_phase_env,
                    )
                raise

            # The rollout server reads checkpoint identities once
            # at startup. Restart it only after both frozen Student endpoints are
            # ready, so Buyer reward cannot compare the wrong checkpoint pair.
            run(["bash", "scripts/stop_role.sh", "rollout"], buyer_phase_env)
            run(
                ["bash", "scripts/start_role.sh", "rollout", str(buyer_model)],
                buyer_phase_env,
            )
            run([sys.executable, "scripts/preflight.py", "services"], buyer_phase_env)

            manifest["phase"] = "buyer_training"
            write_manifest(manifest_path, manifest)
            buyer_env = dict(
                buyer_phase_env,
                COEVO_BUYER_BASE_MODEL=str(buyer_model),
            )
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
            manifest["buyer_checkpoint_after"] = next_buyer_model
            manifest["buyer_revision_after"] = model_revision(next_buyer_model)

            manifest["phase"] = "buyer_refresh"
            write_manifest(manifest_path, manifest)
            refresh_env = dict(
                buyer_phase_env,
                COEVO_BUYER_CHECKPOINT=next_buyer_model,
            )
            run(["bash", "scripts/stop_role.sh", "buyer"], refresh_env)
            run(["bash", "scripts/stop_role.sh", "rollout"], refresh_env)
            try:
                run(
                    ["bash", "scripts/start_role.sh", "buyer", next_buyer_model],
                    refresh_env,
                )
                run(
                    ["bash", "scripts/start_role.sh", "rollout", next_buyer_model],
                    refresh_env,
                )
            except Exception:
                run(["bash", "scripts/stop_role.sh", "buyer"], refresh_env)
                run(["bash", "scripts/stop_role.sh", "rollout"], refresh_env)
                run(
                    ["bash", "scripts/start_role.sh", "buyer", buyer_model],
                    buyer_phase_env,
                )
                run(
                    ["bash", "scripts/start_role.sh", "rollout", buyer_model],
                    buyer_phase_env,
                )
                raise
            policy_model = next_policy_model
            buyer_model = next_buyer_model
            manifest["stage_progress_summary"] = {
                "reward_name": manifest["reward_name"],
                "teacher_anchor_checkpoint": manifest["student_checkpoint_before"],
                "checkpoint_previous": manifest["student_checkpoint_before"],
                "checkpoint_current": manifest["student_checkpoint_after"],
                "aggregation": "mean_over_valid_natural_decisions",
                "reward_scaling": "group",
                "dataset_fingerprint": manifest.get("collection", {}).get(
                    "dataset_fingerprint"
                ),
            }
            previous_policy_model = manifest["student_checkpoint_before"]
        except Exception as error:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
            if services_changed:
                rollback_errors = rollback_services(
                    round_env,
                    manifest["student_checkpoint_before"],
                    manifest["buyer_checkpoint_before"],
                )
                manifest["rollback"] = {
                    "status": "complete" if not rollback_errors else "failed",
                    "errors": rollback_errors,
                }
            write_manifest(manifest_path, manifest)
            raise

        manifest["status"] = "complete"
        manifest["phase"] = "complete"
        write_manifest(manifest_path, manifest)
        if needs_previous_policy(refresh_env):
            run(["bash", "scripts/stop_role.sh", "rollout"], refresh_env)
            run(
                ["bash", "scripts/stop_role.sh", "policy_previous"],
                refresh_env,
            )


if __name__ == "__main__":
    main()
