#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.hinter_training import ColdStartSource, build_hinter_cold_start_dataset


def _read_jsonl(path: Path):
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build multi-checkpoint, minimal-dose cold-start hinter SFT rows"
    )
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        metavar=("STUDENT_CHECKPOINT", "AUDIT_ROWS", "HSTAR_MANIFEST"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mean-copy", type=float, default=0.1)
    args = parser.parse_args()
    sources = [
        ColdStartSource(
            student_checkpoint=checkpoint,
            audit_rows=_read_jsonl(Path(audit_rows)),
            hstar_manifest=json.loads(Path(manifest).read_text(encoding="utf-8")),
        )
        for checkpoint, audit_rows, manifest in args.source
    ]
    rows = build_hinter_cold_start_dataset(
        sources, max_mean_copy=args.max_mean_copy
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "student_checkpoints": len({row["student_checkpoint"] for row in rows}),
                "minimal_levels": sorted(
                    {row["minimal_sufficient_level"] for row in rows}
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
