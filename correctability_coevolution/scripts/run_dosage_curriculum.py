#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import random

from coevo.config import InfraConfig
from coevo.curriculum import (
    curriculum_weights,
    minimal_sufficient_level,
    probe_scenario,
)
from coevo.hints import HINT_LEVELS, HintLevel


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 state/task dosage controller")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--sufficient", type=float, default=0.5)
    parser.add_argument("--near-zero", type=float, default=0.05)
    parser.add_argument("--review-epsilon", type=float, default=0.05)
    parser.add_argument("--explore-epsilon", type=float, default=0.05)
    parser.add_argument(
        "--policy",
        choices=["hstar", "fixed:L3", "fixed:L2", "random"],
        default="hstar",
    )
    args = parser.parse_args()
    config = InfraConfig.from_env()
    probes = {}
    decisions = {}
    rng = random.Random(config.seed)
    for task_id in args.task_ids:
        task_probes = {
            level: probe_scenario(config, str(task_id), level, args.k)
            for level in HINT_LEVELS
        }
        probes[str(task_id)] = {
            level.value: result.to_dict() for level, result in task_probes.items()
        }
        decision = minimal_sufficient_level(
            task_probes, sufficient=args.sufficient, near_zero=args.near_zero
        )
        if args.policy.startswith("fixed:"):
            fixed = HintLevel.parse(args.policy.split(":", 1)[1])
            decision = type(decision)(
                fixed,
                decision.band,
                decision.no_hint_score,
                decision.best_hint_score,
                decision.monotone,
                f"fixed-dose baseline {fixed.value}",
            )
        elif args.policy == "random":
            chosen = rng.choice(list(HINT_LEVELS[1:]))
            decision = type(decision)(
                chosen,
                decision.band,
                decision.no_hint_score,
                decision.best_hint_score,
                decision.monotone,
                "random-dose baseline",
            )
        decisions[str(task_id)] = decision

    weights = curriculum_weights(
        decisions,
        review_epsilon=args.review_epsilon,
        explore_epsilon=args.explore_epsilon,
    )
    manifest = {
        "controller": args.policy,
        "checkpoint": config.current_policy_checkpoint,
        "k": args.k,
        "sufficient": args.sufficient,
        "near_zero": args.near_zero,
        "probes": probes,
        "decisions": {key: value.to_dict() for key, value in decisions.items()},
        "sampling_weights": weights,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dosage_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
