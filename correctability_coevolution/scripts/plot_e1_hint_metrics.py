#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the two E1 hint diagnostics")
    parser.add_argument("summary", type=Path)
    parser.add_argument("counterfactual", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    counterfactual = json.loads(args.counterfactual.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    levels = [level for level in ("L1_POLICY", "L2_PROCEDURAL", "L3_ORACLE") if level in summary["levels"]]
    copy = [summary["levels"][level].get("copy_fraction", 0.0) for level in levels]
    transferable = [
        summary["levels"][level].get("transferable_fraction", 0.0)
        for level in levels
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(levels, copy, label="copy")
    axis.bar(levels, transferable, bottom=copy, label="transferable")
    axis.set_ylim(0, 1)
    axis.set_ylabel("positive signal fraction")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "copy_transferable_by_level.png", dpi=180)
    plt.close(figure)

    invariance = counterfactual["levels"]
    counterfactual_levels = list(invariance)
    values = [
        invariance[level].get("mean_similarity", 0.0)
        for level in counterfactual_levels
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(counterfactual_levels, values)
    axis.set_ylim(0, 1)
    axis.set_ylabel("counterfactual hint similarity")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "counterfactual_invariance_by_level.png", dpi=180
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
