#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.hinter_training import (
    BehaviorHintSample,
    build_fresh_discriminator_pairs,
)


def load_reward_trace(path: Path) -> list[BehaviorHintSample]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append(
            BehaviorHintSample(
                state_hash=str(row["state_hash"]),
                public_state=row["public_state"],
                hint=str(row["hint"]),
                student_behavior=row["student_behavior"],
                control_type="ordinary",
            )
        )
    return samples


def load_controls(path: Path, control_type: str) -> list[BehaviorHintSample]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append(
            BehaviorHintSample(
                state_hash=str(row["state_hash"]),
                public_state=row["public_state"],
                hint=str(row["hint"]),
                student_behavior=row["student_behavior"],
                control_type=control_type,
            )
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fresh same-state pairwise behavior-copy labels"
    )
    parser.add_argument("reward_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--explicit-copy-controls", type=Path, required=True)
    parser.add_argument("--useless-controls", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    groups = {
        "ordinary": load_reward_trace(args.reward_trace),
        "explicit_copy": load_controls(
            args.explicit_copy_controls, "explicit_copy"
        ),
        "useless": load_controls(args.useless_controls, "useless"),
    }
    pairs = []
    for offset, samples in enumerate(groups.values()):
        pairs.extend(
            build_fresh_discriminator_pairs(
                samples, seed=args.seed + offset
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(pair.to_dict(), ensure_ascii=False) + "\n"
            for pair in pairs
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fresh_samples": {
                    name: len(samples) for name, samples in groups.items()
                },
                "pairs": len(pairs),
            }
        )
    )


if __name__ == "__main__":
    main()
