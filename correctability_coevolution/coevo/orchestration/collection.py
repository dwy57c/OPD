from dataclasses import replace
import json
from pathlib import Path

from coevo.config import InfraConfig
from coevo.environment import Tau2Environment
from coevo.rollout import CorrectabilityCollector, build_cutoff_scorer


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def collect_dataset(
    config: InfraConfig,
    output_dir: Path,
    task_ids: list[str],
    trajectories_per_task: int = 1,
) -> dict:
    records = []
    student_rows = []
    buyer_rows = []
    for task_id in task_ids:
        environment = Tau2Environment(replace(config, task_id=task_id))
        collector = CorrectabilityCollector(
            environment,
            build_cutoff_scorer(environment),
            max_turns=environment.config.max_cutoff_turns,
        )
        for _ in range(trajectories_per_task):
            record = collector.collect_one()
            records.append(record)
            student_rows.extend(collector.student_rows(record))
            buyer_rows.append(collector.buyer_row())

    if len(records) == 1:
        _write_json(output_dir / "trajectory.json", records[0])
    _write_jsonl(output_dir / "trajectories.jsonl", records)
    _write_jsonl(output_dir / "student_gkd.jsonl", student_rows)
    _write_jsonl(output_dir / "buyer_grpo.jsonl", buyer_rows)

    summary = {
        "domain": config.domain,
        "task_ids": task_ids,
        "trajectories": len(records),
        "student_turns": sum(len(record["student_turns"]) for record in records),
        "cutoffs": sum(
            len(turn["cutoffs"])
            for record in records
            for turn in record["student_turns"]
        ),
        "turn_correctability": [
            turn["correctability"]
            for record in records
            for turn in record["student_turns"]
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
