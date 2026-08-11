#!/usr/bin/env python3
"""Score and print one cached-target decision against two Student checkpoints."""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from coevo.artifacts import canonical_hash, model_manifest_revision
    from coevo.config import InfraConfig, ModelEndpoint
    from coevo.scoring import StageGapScorer

    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=Path, required=True)
    parser.add_argument("--previous-path", required=True)
    parser.add_argument("--current-path", required=True)
    parser.add_argument("--previous-url", required=True)
    parser.add_argument("--current-url", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    row = json.loads(args.row.read_text().splitlines()[0])
    source_target = row["teacher_target_record"]
    previous = str(Path(args.previous_path).expanduser().resolve())
    current = str(Path(args.current_path).expanduser().resolve())
    config = InfraConfig(
        policy=ModelEndpoint("Qwen3-4B", args.current_url),
        previous_policy=ModelEndpoint("Qwen3-4B", args.previous_url),
        buyer_reference=ModelEndpoint("Qwen3-4B", "http://127.0.0.1:8002"),
        teacher_hint_mode="none",
        current_policy_checkpoint=current,
        previous_policy_checkpoint=previous,
        buyer_checkpoint=current,
        tokenizer_id=(
            f"{current}@{model_manifest_revision(current)}"
        ),
        current_policy_revision=model_manifest_revision(current),
        previous_policy_revision=model_manifest_revision(previous),
        teacher_gap_topk=int(os.getenv("COEVO_TEACHER_GAP_TOPK", "20")),
        teacher_gap_min_support_mass=float(
            os.getenv("COEVO_TEACHER_GAP_MIN_SUPPORT_MASS", "0.95")
        ),
    )
    scorer = StageGapScorer(config)
    scorer.validate_checkpoint_pair()
    result = scorer.score(
        student_visible_messages=source_target["student_visible_messages"],
        hinted_teacher_messages=source_target["hinted_teacher_messages"],
        teacher_action=source_target["teacher_action"],
        state_hash=source_target["state_hash"],
        teacher_hint_hash=source_target["teacher_hint_hash"],
    )
    full = result.to_dict()
    summary = {
        "state_hash": result.teacher_target.state_hash,
        "teacher_action": result.teacher_target.teacher_action,
        "raw_teacher_target_hash": result.teacher_target.raw_teacher_target_hash,
        "teacher_target_hash": result.teacher_target.teacher_target_hash,
        "target_token_count": sum(result.teacher_target.target_loss_mask),
        "skill_contrast_mean": statistics.mean(
            result.teacher_target.skill_contrast_scores
        ),
        "skill_gate_mean": statistics.mean(result.teacher_target.skill_gate_values),
        "temperature_mean": statistics.mean(
            result.teacher_target.sharpening_temperatures
        ),
        "raw_entropy_mean": statistics.mean(
            result.teacher_target.raw_teacher_entropy
        ),
        "sharpened_entropy_mean": statistics.mean(
            result.teacher_target.sharpened_teacher_entropy
        ),
        "previous_gap": result.progress.previous_gap,
        "current_gap": result.progress.current_gap,
        "learning_progress": result.progress.learning_progress,
        "positive_learning_progress": result.progress.positive_learning_progress,
        "decision_reward": result.progress.decision_reward,
        "checkpoint_previous": result.checkpoint_previous,
        "checkpoint_current": result.checkpoint_current,
        "cache": scorer.target_builder.cache_stats,
        "record_hash": canonical_hash(full),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps({"summary": summary, "score": full}, indent=2) + "\n")
        temporary.replace(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
