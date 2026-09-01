#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from coevo.hinter_training.behavior_discriminator import (
    pair_to_training_row,
    pairwise_copy_probability,
    score_texts,
)
from coevo.hinter_training.discriminator_data import CopyingDiscriminatorPair


def load_model(path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def probabilities(model, tokenizer, pairs, max_length):
    positive = [pair_to_training_row(pair)["positive_text"] for pair in pairs]
    negative = [pair_to_training_row(pair)["negative_text"] for pair in pairs]
    positive_scores = score_texts(model, tokenizer, positive, max_length=max_length)
    negative_scores = score_texts(model, tokenizer, negative, max_length=max_length)
    return [
        pairwise_copy_probability(left, right)
        for left, right in zip(positive_scores, negative_scores)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare active and independently reinitialized copying discriminators"
    )
    parser.add_argument("--active", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--minimum-agreement", type=float, default=0.8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    pairs = [
        CopyingDiscriminatorPair(**json.loads(line))
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not pairs:
        parser.error("audit pair set is empty")
    active_model, active_tokenizer = load_model(args.active, args.device)
    active = probabilities(active_model, active_tokenizer, pairs, args.max_length)
    del active_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    audit_model, audit_tokenizer = load_model(args.independent, args.device)
    independent = probabilities(audit_model, audit_tokenizer, pairs, args.max_length)
    agreement = sum(
        (left > 0.5) == (right > 0.5)
        for left, right in zip(active, independent)
    ) / len(pairs)
    report = {
        "pairs": len(pairs),
        "binary_agreement": agreement,
        "mean_absolute_probability_difference": sum(
            abs(left - right) for left, right in zip(active, independent)
        )
        / len(pairs),
        "active": str(args.active),
        "independent": str(args.independent),
    }
    if agreement < args.minimum_agreement:
        raise ValueError(
            f"independent discriminator agreement {agreement:.4f} is below "
            f"{args.minimum_agreement:.4f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
