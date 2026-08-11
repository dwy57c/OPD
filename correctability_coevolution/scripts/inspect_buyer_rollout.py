#!/usr/bin/env python3
"""Run one Buyer rollout and preserve its stage-progress audit payload."""

import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    if not 0 <= args.row_index < len(rows):
        parser.error(f"--row-index must be in [0, {len(rows)})")
    row = rows[args.row_index]
    request = {
        "messages": row["messages"],
        "tools": row.get("tools"),
        "data_dict": row,
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        args.url.rstrip("/") + "/infer/",
        params={"use_tqdm": "false"},
        json={
            "infer_requests": [request],
            "request_config": {
                "max_tokens": args.max_tokens,
                "temperature": 0.8,
                "seed": args.seed,
                "logprobs": True,
                "return_details": True,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)

    output = payload[0] if isinstance(payload, list) else payload
    info = output.get("rollout_infos", {})
    print(
        json.dumps(
            {
                "output": str(args.output),
                "buyer_reward": info.get("buyer_reward"),
                "trajectory_validity": info.get("trajectory_validity"),
                "decision_count": info.get("decision_count"),
                "learning_progresses": info.get("learning_progresses"),
                "decision_rewards": info.get("decision_rewards"),
                "scoring_errors": info.get("scoring_errors"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
