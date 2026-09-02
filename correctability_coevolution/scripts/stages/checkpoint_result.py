#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess

from scripts.stages.common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a driver checkpoint result")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-role", choices=["policy", "hinter"])
    args = parser.parse_args()
    checkpoints = list(args.checkpoint_root.glob("v*/checkpoint-*")) + list(
        args.checkpoint_root.glob("checkpoint-*")
    )
    if not checkpoints:
        parser.error(f"no checkpoint under {args.checkpoint_root}")
    checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime).resolve()
    if args.refresh_role:
        root = Path(__file__).resolve().parents[2]
        subprocess.run(
            ["bash", "scripts/stop_role.sh", args.refresh_role],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                "bash",
                "scripts/start_role.sh",
                args.refresh_role,
                str(checkpoint),
            ],
            cwd=root,
            check=True,
        )
    write_json(args.output, {"checkpoint": str(checkpoint)})


if __name__ == "__main__":
    main()
