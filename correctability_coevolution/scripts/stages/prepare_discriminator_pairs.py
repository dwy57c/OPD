#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.hinter_training import (
    BehaviorHintSample,
    CopyingDiscriminatorPair,
    build_fresh_discriminator_pairs,
)


def load_samples(path: Path, control_type: str):
    return [
        BehaviorHintSample(**json.loads(line), control_type=control_type)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_pairs(path: Path):
    return [
        CopyingDiscriminatorPair(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ordinary and held-out pairs")
    parser.add_argument("--ordinary-pairs", type=Path, required=True)
    parser.add_argument("--explicit-copy-controls", type=Path, required=True)
    parser.add_argument("--useless-controls", type=Path, required=True)
    parser.add_argument("--natural-copy-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pairs = load_pairs(args.ordinary_pairs)
    pairs.extend(
        build_fresh_discriminator_pairs(
            load_samples(args.explicit_copy_controls, "explicit_copy"),
            seed=args.seed,
        )
    )
    pairs.extend(
        build_fresh_discriminator_pairs(
            load_samples(args.useless_controls, "useless"),
            seed=args.seed + 1,
        )
    )
    natural = load_pairs(args.natural_copy_pairs)
    if any(pair.control_type != "explicit_copy_natural" for pair in natural):
        raise ValueError("natural-copy pairs are mislabeled")
    pairs.extend(natural)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n" for pair in pairs),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
