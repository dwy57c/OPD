#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import random

from coevo.config import InfraConfig
from coevo.hints import HintLevel
from coevo.orchestration import collect_mixed_dosage_dataset


def select_curriculum_tasks(manifest: dict, sample_count: int, seed: int):
    weights = manifest.get("sampling_weights") or {}
    decisions = manifest.get("decisions") or {}
    eligible = []
    dropped = {}
    for task_id, weight in weights.items():
        level = (decisions.get(task_id) or {}).get("level")
        if level is None or float(weight) <= 0:
            dropped[task_id] = {
                "weight": float(weight),
                "reason": "no trainable hint level",
            }
            continue
        eligible.append((str(task_id), HintLevel.parse(level), float(weight)))
    if not eligible:
        raise ValueError("dosage manifest has no task with a trainable hint level")
    normalizer = sum(weight for _, _, weight in eligible)
    rng = random.Random(seed)
    chosen = rng.choices(
        eligible,
        weights=[weight / normalizer for _, _, weight in eligible],
        k=sample_count,
    )
    selections = [
        {
            "task_id": task_id,
            "hint_level": level.value,
            "sampling_weight": weight,
        }
        for task_id, level, weight in chosen
    ]
    return selections, dropped, normalizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume an h* dosage manifest into a mixed-level Student dataset"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample_count = args.samples or len(manifest.get("decisions") or {})
    if sample_count < 1:
        parser.error("--samples must be positive")
    seed = int(args.seed if args.seed is not None else manifest.get("seed", 42))
    selections, dropped, normalizer = select_curriculum_tasks(
        manifest, sample_count, seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = collect_mixed_dosage_dataset(
        InfraConfig.from_env(),
        args.source_trajectories,
        args.output_dir,
        selections,
    )
    selection_manifest = {
        "source_manifest": str(args.manifest),
        "source_trajectories": str(args.source_trajectories),
        "seed": seed,
        "samples": sample_count,
        "eligible_weight_before_normalization": normalizer,
        "dropped_tasks": dropped,
        "selections": selections,
        "dataset_fingerprint": summary["dataset_fingerprint"],
    }
    (args.output_dir / "curriculum_selection.json").write_text(
        json.dumps(selection_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "selection": selection_manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
