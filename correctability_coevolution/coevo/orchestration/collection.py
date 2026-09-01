from dataclasses import replace
import json
from pathlib import Path

from coevo.artifacts import (
    artifact_metadata,
    dataset_fingerprint,
    validate_compatible_artifacts,
)
from coevo.config import InfraConfig
from coevo.environment import Tau2Environment
from coevo.environment.tau2 import load_messages
from coevo.intervention import extract_decision_states
from coevo.rollout import NaturalDecisionCollector, build_teacher_target_labeler


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def _validate_resume_fingerprint(
    summary: dict,
    records: list[dict],
    student_rows: list[dict],
    buyer_rows: list[dict],
) -> None:
    expected = summary.get("dataset_fingerprint")
    if not expected:
        raise ValueError(
            "summary.json is missing dataset_fingerprint; recollect or migrate "
            "before resume"
        )
    actual = dataset_fingerprint(records, student_rows, buyer_rows)
    if actual != expected:
        raise ValueError(
            "collection artifacts do not match summary.json dataset_fingerprint; "
            "refusing resume after a partial or modified write"
        )


def _read_jsonl(path: Path, schema_version: int) -> list[dict]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    incompatible = sorted(
        {
            row.get("schema_version")
            for row in rows
            if row.get("schema_version") != schema_version
        },
        key=lambda value: str(value),
    )
    if incompatible:
        raise ValueError(
            f"{path} contains incompatible schema versions {incompatible}; "
            f"expected {schema_version}. Recollect or migrate before resume."
        )
    return rows


def _build_summary(
    config: InfraConfig,
    task_ids: list[str],
    records: list[dict],
    student_rows: list[dict],
    buyer_rows: list[dict],
    errors: list[dict],
) -> dict:
    completed = {str(record["task_id"]) for record in records}
    return {
        **artifact_metadata(config),
        "domain": config.domain,
        "task_split": config.task_split,
        "task_ids": task_ids,
        "completed_task_ids": [task_id for task_id in task_ids if task_id in completed],
        "trajectories": len(records),
        "student_decisions": sum(
            len(record["student_decisions"]) for record in records
        ),
        "student_training_rows": len(student_rows),
        "rejected_teacher_targets": sum(
            not bool(turn.get("student_eligible"))
            for record in records
            for turn in record["student_decisions"]
        ),
        "buyer_rows": len(buyer_rows),
        "teacher_target_hashes": [
            turn["teacher_target_hash"]
            for record in records
            for turn in record["student_decisions"]
            if turn.get("student_eligible")
        ],
        "dataset_fingerprint": dataset_fingerprint(
            records,
            student_rows,
            buyer_rows,
        ),
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
    summary = _build_summary(
        config,
        task_ids,
        records,
        student_rows,
        buyer_rows,
        errors,
    )
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
    records = (
        _read_jsonl(
            output_dir / "trajectories.jsonl", config.dataset_schema_version
        )
        if resume
        else []
    )
    student_rows = (
        _read_jsonl(
            output_dir / "student_gkd.jsonl", config.dataset_schema_version
        )
        if resume
        else []
    )
    buyer_rows = (
        _read_jsonl(
            output_dir / "buyer_grpo.jsonl", config.dataset_schema_version
        )
        if resume
        else []
    )
    errors = []
    contract_values = [("current collection config", artifact_metadata(config))]
    for filename, rows in (
        ("trajectories.jsonl", records),
        ("student_gkd.jsonl", student_rows),
        ("buyer_grpo.jsonl", buyer_rows),
    ):
        contract_values.extend(
            (f"{filename}:{index}", row)
            for index, row in enumerate(rows, start=1)
        )
    summary_path = output_dir / "summary.json"
    artifact_paths = [
        output_dir / "trajectories.jsonl",
        output_dir / "student_gkd.jsonl",
        output_dir / "buyer_grpo.jsonl",
    ]
    if (
        resume
        and any(path.is_file() for path in artifact_paths)
        and not summary_path.is_file()
    ):
        raise ValueError(
            "collection rows exist without summary.json; refusing an unverifiable "
            "resume"
        )
    if resume and summary_path.is_file():
        previous_summary = json.loads(summary_path.read_text())
        if previous_summary.get("schema_version") != config.dataset_schema_version:
            raise ValueError(
                "summary.json has an incompatible schema version; recollect or "
                "migrate before resume"
            )
        _validate_resume_fingerprint(
            previous_summary,
            records,
            student_rows,
            buyer_rows,
        )
        contract_values.append(("summary.json", previous_summary))
        errors = list(previous_summary.get("errors", []))
    if resume:
        validate_compatible_artifacts(contract_values)
    completed = {(str(record["task_id"]), int(record["seed"])) for record in records}

    for task_index, task_id in enumerate(task_ids, start=1):
        print(f"[collect] task {task_index}/{len(task_ids)}: {task_id}", flush=True)
        environment = Tau2Environment(replace(config, task_id=task_id))
        collector = NaturalDecisionCollector(
            environment,
            build_teacher_target_labeler(environment),
            max_decisions=environment.config.max_teacher_targets,
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


def relabel_dataset_from_trajectories(
    config: InfraConfig,
    source_path: Path,
    output_dir: Path,
) -> dict:
    """Regenerate Teacher labels for an immutable public-state pool.

    E1/E2 must compare hint levels on identical Student/Buyer trajectories.
    This function never rolls out either policy. It reuses each source trunk and
    decision index, then generates only the level-specific hint, Teacher action,
    and target record.
    """

    source_records = _read_jsonl(source_path, config.dataset_schema_version)
    if not source_records:
        raise ValueError("source trajectory pool is empty")
    source_fingerprint = dataset_fingerprint(source_records)
    records: list[dict] = []
    student_rows: list[dict] = []
    buyer_rows: list[dict] = []
    errors: list[dict] = []
    task_ids: list[str] = []
    for source in source_records:
        task_id = str(source["task_id"])
        if task_id not in task_ids:
            task_ids.append(task_id)
        environment = Tau2Environment(
            replace(
                config,
                domain=str(source.get("domain", config.domain)),
                task_split=str(source.get("task_split", config.task_split)),
                task_id=task_id,
            )
        )
        collector = NaturalDecisionCollector(
            environment,
            build_teacher_target_labeler(environment),
            max_decisions=environment.config.max_teacher_targets,
        )
        trunk = load_messages(source["trunk"])
        source_decisions = source.get("student_decisions") or []
        indexes = [int(row["message_index"]) for row in source_decisions]
        if not indexes:
            indexes = [state.message_index for state in extract_decision_states(trunk)]
        decisions = []
        for message_index in indexes:
            try:
                labeled = collector.labeler.score_decision(trunk, message_index)
                decisions.append(collector._materialize_target(labeled))
            except Exception as error:
                errors.append(
                    {
                        "task_id": task_id,
                        "seed": source.get("seed"),
                        "message_index": message_index,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise
        record = {
            **artifact_metadata(environment.config),
            "domain": environment.config.domain,
            "task_split": environment.config.task_split,
            "task_id": task_id,
            "seed": int(source.get("seed", config.seed)),
            "trunk": source["trunk"],
            "source_dataset_fingerprint": source_fingerprint,
            "student_decisions": decisions,
            "teacher_target_cache": collector._builder().cache_stats,
        }
        records.append(record)
        student_rows.extend(collector.student_rows(record))
        buyer_rows.append(collector.buyer_row())
    summary = _persist(
        config,
        output_dir,
        task_ids,
        records,
        student_rows,
        buyer_rows,
        errors,
    )
    summary["source_trajectories"] = str(source_path)
    summary["source_dataset_fingerprint"] = source_fingerprint
    summary["public_state_pool_reused"] = True
    _write_json(output_dir / "summary.json", summary)
    return summary
