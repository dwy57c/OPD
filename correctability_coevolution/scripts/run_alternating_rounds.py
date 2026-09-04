#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from coevo.hinter_training import (
    AlternatingHinterLoop,
    PassKSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class SubprocessRoundBackend:
    """Concrete callback backend driven by auditable argv templates."""

    REQUIRED_STAGES = {
        "train_student",
        "measure_pass_at_k",
        "schedule_curriculum",
        "train_hinter_grpo",
        "rollback_student",
        "rollback_hinter",
    }

    def __init__(self, commands: dict, round_dir: Path):
        missing = sorted(self.REQUIRED_STAGES - set(commands))
        if missing:
            raise ValueError(f"alternating command config is missing: {missing}")
        self.commands = commands
        self.round_dir = round_dir
        self.context: dict[str, Any] = {}

    def _run(
        self,
        stage: str,
        values: dict[str, Any],
        output: Path | None = None,
        *,
        parse_json: bool = True,
    ):
        context = {**self.context, **values}
        if output is not None:
            context["output"] = str(output)
        command = [str(value).format_map(context) for value in self.commands[stage]]
        log_dir = self.round_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "COEVO_STAGE_OUTPUT": str(output or "")},
            text=True,
            capture_output=True,
        )
        (log_dir / f"{stage}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (log_dir / f"{stage}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"stage {stage} failed with exit code {completed.returncode}"
            )
        if output is None:
            return None
        if not output.is_file():
            raise FileNotFoundError(f"stage {stage} did not write {output}")
        return (
            json.loads(output.read_text(encoding="utf-8"))
            if parse_json
            else None
        )

    def train_student(self, student: str, hinter: str, steps: int) -> str:
        output = self.round_dir / "student_update.json"
        result = self._run(
            "train_student",
            {"student": student, "hinter": hinter, "student_steps": steps},
            output,
        )
        return str(result["checkpoint"])

    def measure_pass_at_k(self, student: str, _pool: dict, k: int) -> PassKSnapshot:
        output = self.round_dir / "pass_at_k.json"
        result = self._run(
            "measure_pass_at_k", {"student": student, "pass_k": k}, output
        )
        return PassKSnapshot(
            {str(key): float(value) for key, value in result["scores"].items()},
            int(result.get("k", k)),
        )

    def schedule_curriculum(self, snapshot: PassKSnapshot, pool: dict) -> dict:
        input_path = self.round_dir / "pass_for_schedule.json"
        write_json(input_path, snapshot.to_dict())
        output = self.round_dir / "curriculum.json"
        result = self._run(
            "schedule_curriculum",
            {"pass_snapshot": str(input_path)},
            output,
        )
        selected = [str(value) for value in result["scenario_ids"]]
        missing = [value for value in selected if value not in pool]
        if missing:
            raise ValueError(f"scheduler selected unknown scenarios: {missing}")
        return {value: pool[value] for value in selected}

    def train_hinter_grpo(
        self, student: str, hinter: str, curriculum: dict, steps: int
    ) -> str:
        self.context["current_student"] = student
        self.context["current_hinter"] = hinter
        curriculum_path = self.round_dir / "selected_scenarios.json"
        write_json(curriculum_path, curriculum)
        output = self.round_dir / "hinter_update.json"
        result = self._run(
            "train_hinter_grpo",
            {
                "student": student,
                "hinter": hinter,
                "curriculum": str(curriculum_path),
                "hinter_steps": steps,
            },
            output,
        )
        return str(result["checkpoint"])

    def rollback_student(self, candidate: str, previous: str) -> None:
        self._run(
            "rollback_student",
            {"candidate": candidate, "previous": previous},
        )

    def rollback_hinter(self, candidate: str, previous: str) -> None:
        self._run(
            "rollback_hinter",
            {"candidate": candidate, "previous": previous},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute alternating Student/hinter rounds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--scenario-pool", type=Path, required=True)
    parser.add_argument("--source-trajectories", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-curriculum-manifest", type=Path, required=True
    )
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--hinter-checkpoint", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--student-steps", type=int, default=100)
    parser.add_argument("--hinter-steps", type=int, default=20)
    parser.add_argument("--pass-k", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.student_steps, args.hinter_steps, args.pass_k) < 1:
        parser.error("round and step counts must be positive")
    commands = json.loads(args.commands.read_text(encoding="utf-8"))
    scenario_pool = json.loads(args.scenario_pool.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "state.json"
    if state_path.exists():
        if not args.resume:
            parser.error("state.json exists; pass --resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "next_round": 0,
            "student_checkpoint": args.student_checkpoint,
            "hinter_under_test": args.hinter_checkpoint,
            "fallback_hinter": args.hinter_checkpoint,
            "acceptance_baseline": None,
            "curriculum_manifest": str(
                args.bootstrap_curriculum_manifest.resolve()
            ),
        }
        write_json(state_path, state)
    for round_index in range(int(state["next_round"]), args.rounds):
        round_dir = args.output_dir / f"round_{round_index:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = round_dir / "manifest.json"
        write_json(manifest_path, {"round": round_index, "status": "running"})
        backend = SubprocessRoundBackend(commands, round_dir)
        backend.context = {
            "round": round_index,
            "round_dir": str(round_dir),
            "output_dir": str(args.output_dir),
            "curriculum_manifest": str(state["curriculum_manifest"]),
            "source_trajectories": str(args.source_trajectories.resolve()),
        }
        loop = AlternatingHinterLoop(
            train_student=backend.train_student,
            measure_pass_at_k=backend.measure_pass_at_k,
            schedule_curriculum=backend.schedule_curriculum,
            train_hinter_grpo=backend.train_hinter_grpo,
            rollback_student=backend.rollback_student,
            rollback_hinter=backend.rollback_hinter,
        )
        baseline = (
            PassKSnapshot(**state["acceptance_baseline"])
            if state["acceptance_baseline"] is not None
            else None
        )
        try:
            result = loop.run_round(
                round_index=round_index,
                student_checkpoint=state["student_checkpoint"],
                hinter_under_test=state["hinter_under_test"],
                fallback_hinter_checkpoint=state["fallback_hinter"],
                scenario_pool=scenario_pool,
                acceptance_baseline=baseline,
                student_steps=args.student_steps,
                hinter_grpo_steps=args.hinter_steps,
                pass_k=args.pass_k,
            )
        except Exception as error:
            write_json(
                manifest_path,
                {
                    "round": round_index,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        result_dict = result.to_dict()
        write_json(
            manifest_path,
            {"round": round_index, "status": "complete", "result": result_dict},
        )
        accepted_snapshot = (
            baseline if result.prior_hinter_rolled_back else result.measured_pass_at_k
        )
        state = {
            "next_round": round_index + 1,
            "student_checkpoint": result.student_after,
            "hinter_under_test": result.next_hinter_candidate,
            "fallback_hinter": result.accepted_hinter,
            "acceptance_baseline": (
                {
                    "scores": dict(accepted_snapshot.scores),
                    "k": accepted_snapshot.k,
                }
                if accepted_snapshot is not None
                else None
            ),
            "curriculum_manifest": str(
                (round_dir / "dosage" / "dosage_manifest.json").resolve()
            ),
        }
        write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
