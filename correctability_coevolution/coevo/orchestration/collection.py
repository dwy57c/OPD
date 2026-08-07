from dataclasses import replace
import json
from pathlib import Path

from coevo.config import InfraConfig
from coevo.environment import Tau2Environment
from coevo.rollout import NaturalDecisionCollector, build_action_branch_runner


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _build_summary(
    config: InfraConfig,
    task_ids: list[str],
    records: list[dict],
    errors: list[dict],
) -> dict:
    completed = {str(record["task_id"]) for record in records}
    return {
        "domain": config.domain,
        "task_split": config.task_split,
        "task_ids": task_ids,
        "completed_task_ids": [task_id for task_id in task_ids if task_id in completed],
        "trajectories": len(records),
        "student_decisions": sum(
            len(record["student_decisions"]) for record in records
        ),
        "decision_interventions": sum(
            len(record["student_decisions"]) for record in records
        ),
        "intervention_advantages": [
            turn["intervention_advantage"]
            for record in records
            for turn in record["student_decisions"]
            if "intervention_advantage" in turn
        ],
        "errors": errors,
    }


def _persist(
    config: InfraConfig,
    output_dir: Path,
    task_ids: list[str],
    records: list[dict],
    student_rows: list[dict],
    buyer_rows: list[dict],
    errors: list[dict],
) -> dict:
    _write_jsonl(output_dir / "trajectories.jsonl", records)
    _write_jsonl(output_dir / "student_gkd.jsonl", student_rows)
    _write_jsonl(output_dir / "buyer_grpo.jsonl", buyer_rows)
    summary = _build_summary(config, task_ids, records, errors)
    _write_json(output_dir / "summary.json", summary)
    return summary


def collect_dataset(
    config: InfraConfig,
    output_dir: Path,
    task_ids: list[str],
    trajectories_per_task: int = 1,
    *,
    continue_on_error: bool = False,
    resume: bool = False,
) -> dict:
    records = _read_jsonl(output_dir / "trajectories.jsonl") if resume else []
    student_rows = _read_jsonl(output_dir / "student_gkd.jsonl") if resume else []
    buyer_rows = _read_jsonl(output_dir / "buyer_grpo.jsonl") if resume else []
    errors = []
    if resume and (output_dir / "summary.json").is_file():
        previous_summary = json.loads((output_dir / "summary.json").read_text())
        errors = list(previous_summary.get("errors", []))
    completed = {(str(record["task_id"]), int(record["seed"])) for record in records}

    for task_index, task_id in enumerate(task_ids, start=1):
        print(f"[collect] task {task_index}/{len(task_ids)}: {task_id}", flush=True)
        environment = Tau2Environment(replace(config, task_id=task_id))
        collector = NaturalDecisionCollector(
            environment,
            build_action_branch_runner(environment),
            max_decisions=environment.config.max_intervention_decisions,
        )
        for trajectory_index in range(trajectories_per_task):
            seed = config.seed + trajectory_index
            if (task_id, seed) in completed:
                print(
                    f"[collect] resume skip task={task_id} seed={seed}",
                    flush=True,
                )
                continue
            try:
                record = collector.collect_one(seed=seed)
            except Exception as error:
                error_row = {
                    "task_id": task_id,
                    "seed": seed,
                    "type": type(error).__name__,
                    "message": str(error),
                }
                errors = [
                    item
                    for item in errors
                    if (str(item.get("task_id")), item.get("seed")) != (task_id, seed)
                ]
                errors.append(error_row)
                _persist(
                    config,
                    output_dir,
                    task_ids,
                    records,
                    student_rows,
                    buyer_rows,
                    errors,
                )
                print(
                    f"[collect] failed task={task_id} seed={seed}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                if continue_on_error:
                    continue
                raise
            records.append(record)
            student_rows.extend(collector.student_rows(record))
            buyer_rows.append(collector.buyer_row())
            completed.add((task_id, seed))
            errors = [
                item
                for item in errors
                if (str(item.get("task_id")), item.get("seed")) != (task_id, seed)
            ]
            _persist(
                config,
                output_dir,
                task_ids,
                records,
                student_rows,
                buyer_rows,
                errors,
            )
            print(
                f"[collect] saved task={task_id} seed={seed} "
                f"trajectories={len(records)} student_decisions={len(student_rows)}",
                flush=True,
            )

    if len(records) == 1:
        _write_json(output_dir / "trajectory.json", records[0])
    return _persist(
        config,
        output_dir,
        task_ids,
        records,
        student_rows,
        buyer_rows,
        errors,
    )
