#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from coevo.hinter_training.behavior_discriminator import (
    DiscriminatorGate,
    PairwiseDiscriminatorCollator,
    evaluate_pair_scores,
    pair_to_training_row,
    pairwise_ranking_loss,
    score_texts,
)
from coevo.hinter_training.discriminator_data import CopyingDiscriminatorPair
from coevo.artifacts import canonical_hash


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PairwiseTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        positive = {
            key.removeprefix("positive_"): value
            for key, value in inputs.items()
            if key.startswith("positive_")
        }
        negative = {
            key.removeprefix("negative_"): value
            for key, value in inputs.items()
            if key.startswith("negative_")
        }
        positive_scores = model(**positive).logits.reshape(-1)
        negative_scores = model(**negative).logits.reshape(-1)
        loss = pairwise_ranking_loss(positive_scores, negative_scores)
        if not torch.isfinite(loss):
            raise FloatingPointError("behavior-discriminator loss is not finite")
        outputs = {
            "positive_scores": positive_scores,
            "negative_scores": negative_scores,
        }
        return (loss, outputs) if return_outputs else loss


def load_pairs(path: Path) -> list[CopyingDiscriminatorPair]:
    return [
        CopyingDiscriminatorPair(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fresh same-size scalar-head behavior discriminator"
    )
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed")
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--minimum-explicit-copy-accuracy", type=float, default=0.9)
    parser.add_argument("--maximum-useless-distance-from-chance", type=float, default=0.1)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(
            "output directory must be empty: each round initializes a fresh score head"
        )
    pairs = load_pairs(args.pairs)
    ordinary = [pair for pair in pairs if pair.control_type == "ordinary"]
    controls = [pair for pair in pairs if pair.control_type != "ordinary"]
    if not ordinary:
        parser.error("pair dataset has no ordinary fresh training examples")
    if not controls:
        parser.error("pair dataset has no held-out explicit-copy/useless controls")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_checkpoint,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.student_checkpoint,
        num_labels=1,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    train_rows = [pair_to_training_row(pair) for pair in ordinary]
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to=args.report_to,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=PairDataset(train_rows),
        data_collator=PairwiseDiscriminatorCollator(tokenizer, args.max_length),
    )
    trainer.train()
    model.eval()

    positive_texts = [pair_to_training_row(pair)["positive_text"] for pair in controls]
    negative_texts = [pair_to_training_row(pair)["negative_text"] for pair in controls]
    positive_scores = score_texts(
        model, tokenizer, positive_texts, max_length=args.max_length
    )
    negative_scores = score_texts(
        model, tokenizer, negative_texts, max_length=args.max_length
    )
    report = evaluate_pair_scores(controls, positive_scores, negative_scores)
    gate = DiscriminatorGate(
        minimum_explicit_copy_accuracy=args.minimum_explicit_copy_accuracy,
        maximum_useless_distance_from_chance=args.maximum_useless_distance_from_chance,
    )
    gate.validate(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "discriminator_report.json").write_text(
        json.dumps(
            {
                **report.to_dict(),
                "initialized_from_student": str(args.student_checkpoint),
                "fresh_score_head": True,
                "training_pairs": len(ordinary),
                "training_fingerprint": canonical_hash(
                    [pair.to_dict() for pair in ordinary]
                ),
                "control_pairs": len(controls),
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
