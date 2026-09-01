#!/usr/bin/env python3
import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from coevo.audit import BehaviorAuditor, OpenAIGroundingJudge
from coevo.config import InfraConfig


def _messages(row: dict[str, Any]) -> list:
    if isinstance(row.get("messages"), list):
        return row["messages"]
    if isinstance(row.get("tau_history"), list):
        return row["tau_history"]
    simulation = row.get("simulation")
    if isinstance(simulation, dict):
        if isinstance(simulation.get("messages"), list):
            return simulation["messages"]
        if isinstance(simulation.get("trajectory"), list):
            return simulation["trajectory"]
    raise ValueError("evaluation row has no messages, tau_history, or simulation messages")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit interactive information-seeking behavior")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--use-nl-judge", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = InfraConfig.from_env() if args.use_nl_judge else None
    judge = None
    if config is not None:
        endpoint = config.nl_judge or config.policy
        judge = OpenAIGroundingJudge(endpoint, retries=config.nl_judge_retries)
    auditor = BehaviorAuditor(judge)

    audited = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        report = auditor.analyze(_messages(row)).to_dict()
        level = str(row.get("hint_level") or row.get("mode") or "unknown")
        result = {"index": index, "hint_level": level, "behavior": report}
        audited.append(result)
        grouped[level].append(report)

    summary = {}
    for level, reports in grouped.items():
        count = len(reports)
        summary[level] = {
            "rows": count,
            "clarification_rate": sum(r["clarification_rate"] for r in reports) / count,
            "lookup_rate": sum(r["lookup_rate"] for r in reports) / count,
            "ungrounded_assertion_rate": sum(
                r["ungrounded_assertion_rate"] for r in reports
            ) / count,
            "ungrounded_assertion_count": sum(
                r["ungrounded_assertion_count"] for r in reports
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "rows": audited}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
