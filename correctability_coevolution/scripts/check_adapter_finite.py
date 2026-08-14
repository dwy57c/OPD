#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coevo.training.finite_check import safetensors_finite_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if a saved LoRA adapter contains NaN or infinity."
    )
    parser.add_argument("adapter", type=Path)
    args = parser.parse_args()
    summary = safetensors_finite_summary(args.adapter)
    print(
        "adapter_finite_check "
        f"all_finite={summary['all_finite']} "
        f"tensors={summary['tensor_count']} "
        f"values={summary['value_count']} "
        f"nonfinite={summary['nonfinite_count']}"
    )
    if not summary["all_finite"]:
        preview = ", ".join(summary["bad_tensors"][:5])
        raise SystemExit(
            f"non-finite adapter checkpoint rejected; first bad tensors: {preview}"
        )


if __name__ == "__main__":
    main()
