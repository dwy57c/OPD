#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coevo.training.trace_merge import write_merged_buyer_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge distributed Buyer reward trace shards into global G groups."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    records = write_merged_buyer_trace(args.shards, args.output)
    print(f"wrote {len(records)} global Buyer groups to {args.output}")


if __name__ == "__main__":
    main()
