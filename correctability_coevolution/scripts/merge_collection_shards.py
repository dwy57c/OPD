#!/usr/bin/env python3
"""Merge resumable collection shards into one training-data directory."""

import argparse
import json
from pathlib import Path

from coevo.artifacts import dataset_fingerprint, validate_compatible_artifacts


JSONL_NAMES = ("trajectories.jsonl", "student_gkd.jsonl", "buyer_grpo.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def task_sort_key(task_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(task_id))
    except ValueError:
        return (1, task_id)


def canonical(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def unique_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        key = canonical(row)
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result


def normalize_chat_row(row: dict) -> dict:
    """Use Swift-compatible empty content for content-less tool calls."""
    normalized = dict(row)
    for field in ("messages",):
        if field not in row:
            continue
        normalized[field] = [
            {
                **message,
                "content": "" if message.get("content") is None else message["content"],
            }
            for message in row[field]
        ]
    return normalized


def merge_shards(input_root: Path, output_dir: Path, require_complete: bool) -> dict:
    shard_dirs = sorted(path for path in input_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories under {input_root}")

    summaries = []
    merged = {name: [] for name in JSONL_NAMES}
    contract_values = []
    for shard_dir in shard_dirs:
        summary_path = shard_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing {summary_path}")
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)
        contract_values.append((str(summary_path), summary))
        for name in JSONL_NAMES:
            rows = read_jsonl(shard_dir / name)
            merged[name].extend(rows)
            contract_values.extend(
                (f"{shard_dir / name}:{index}", row)
                for index, row in enumerate(rows, start=1)
            )

    contract = validate_compatible_artifacts(contract_values)

    domain_values = {summary["domain"] for summary in summaries}
    split_values = {summary["task_split"] for summary in summaries}
    if len(domain_values) != 1 or len(split_values) != 1:
        raise ValueError("All shards must have the same domain and task split")

    trajectories = unique_rows(merged["trajectories.jsonl"])
    student_rows = unique_rows(
        [normalize_chat_row(row) for row in merged["student_gkd.jsonl"]]
    )
    buyer_rows = unique_rows(
        [normalize_chat_row(row) for row in merged["buyer_grpo.jsonl"]]
    )
    completed_keys = {(str(row["task_id"]), int(row["seed"])) for row in trajectories}
    if len(completed_keys) != len(trajectories):
        raise ValueError("Conflicting duplicate trajectory task/seed keys")

    expected_task_ids = sorted(
        {str(task_id) for summary in summaries for task_id in summary["task_ids"]},
        key=task_sort_key,
    )
    completed_task_ids = sorted(
        {task_id for task_id, _ in completed_keys}, key=task_sort_key
    )
    errors = [error for summary in summaries for error in summary.get("errors", [])]
    missing = sorted(
        set(expected_task_ids) - set(completed_task_ids), key=task_sort_key
    )
    if require_complete and (missing or errors):
        raise RuntimeError(
            f"Collection incomplete: missing={missing}, errors={len(errors)}"
        )

    trajectory_task_ids = set(completed_task_ids)
    buyer_task_ids = {str(row["task_id"]) for row in buyer_rows}
    student_task_ids = {str(row["task_id"]) for row in student_rows}
    if buyer_task_ids != trajectory_task_ids:
        raise ValueError("Buyer rows do not cover exactly the completed trajectories")
    if not student_task_ids.issubset(trajectory_task_ids):
        raise ValueError("Student rows contain tasks without completed trajectories")

    summary = {
        **contract.to_dict(),
        "domain": next(iter(domain_values)),
        "task_split": next(iter(split_values)),
        "task_ids": expected_task_ids,
        "completed_task_ids": completed_task_ids,
        "missing_task_ids": missing,
        "shards": [path.name for path in shard_dirs],
        "trajectories": len(trajectories),
        "student_decisions": len(student_rows),
        "buyer_rows": len(buyer_rows),
        "teacher_targets": len(student_rows),
        "teacher_target_hashes": [
            row["teacher_target_hash"] for row in student_rows
        ],
        "dataset_fingerprint": dataset_fingerprint(
            trajectories,
            student_rows,
            buyer_rows,
        ),
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "trajectories.jsonl", trajectories)
    write_jsonl(output_dir / "student_gkd.jsonl", student_rows)
    write_jsonl(output_dir / "buyer_grpo.jsonl", buyer_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = merge_shards(args.input_root, args.output_dir, args.require_complete)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
