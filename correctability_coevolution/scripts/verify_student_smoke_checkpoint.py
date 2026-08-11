#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = list(output_dir.glob("v*/checkpoint-*")) + list(
        output_dir.glob("checkpoint-*")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {output_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    from coevo.artifacts import model_manifest_revision

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = latest_checkpoint(args.output_dir)
    adapter_path = checkpoint / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Missing {adapter_path}")
    tensors = load_file(str(adapter_path))
    if not tensors:
        raise RuntimeError("Student checkpoint contains no adapter tensors")
    finite = all(torch.isfinite(tensor).all().item() for tensor in tensors.values())
    nonzero = sum(torch.count_nonzero(tensor).item() for tensor in tensors.values())
    if not finite or nonzero <= 0:
        raise RuntimeError("Student adapter is non-finite or entirely zero")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    inputs = tokenizer("Confirm before acting.", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    forward_finite = bool(torch.isfinite(logits).all().item())
    if not forward_finite:
        raise RuntimeError("Reloaded Student checkpoint produced non-finite logits")

    trainer_state_paths = list(args.output_dir.rglob("trainer_state.json"))
    log_history = []
    if trainer_state_paths:
        state = json.loads(trainer_state_paths[-1].read_text())
        log_history = state.get("log_history", [])
    losses = [
        float(row["loss"])
        for row in log_history
        if row.get("loss") is not None and math.isfinite(float(row["loss"]))
    ]
    metric_names = (
        "loss",
        "grad_norm",
        "hint_gate_mean",
        "hint_gate_active_fraction",
        "hint_jsd",
        "hint_active_gate_mass",
    )
    training_metrics = next(
        (
            {
                name: float(row[name])
                for name in metric_names
                if row.get(name) is not None
            }
            for row in reversed(log_history)
            if row.get("loss") is not None
        ),
        {},
    )
    if not losses:
        raise RuntimeError("Student smoke logged no finite training loss")
    grad_norm = float(training_metrics.get("grad_norm", 0.0))
    active_fraction = float(
        training_metrics.get("hint_gate_active_fraction", 0.0)
    )
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError("Student smoke logged no finite non-zero gradient norm")
    if not math.isfinite(active_fraction) or active_fraction <= 0:
        raise RuntimeError("Student smoke had no active hint-gated target tokens")
    result = {
        "status": "passed",
        "model_path": str(Path(args.model_path).resolve()),
        "model_revision": model_manifest_revision(args.model_path),
        "checkpoint": str(checkpoint.resolve()),
        "adapter_tensor_count": len(tensors),
        "adapter_nonzero_elements": nonzero,
        "adapter_all_finite": finite,
        "reload_forward_all_finite": forward_finite,
        "finite_logged_losses": losses,
        "training_metrics": training_metrics,
        "report_to": os.getenv("COEVO_REPORT_TO", "wandb"),
        "wandb_mode": os.getenv("WANDB_MODE", ""),
        "downloads_allowed": os.getenv("COEVO_ALLOW_DOWNLOADS", "0") == "1",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
