#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from coevo.config import HintEndpoint
from coevo.hints import HintLevel
from coevo.models.hinted_teacher import OpenModelTeacherHinter


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_same_task_counterfactual(
    source: Mapping[str, Any], alternative: Mapping[str, Any]
) -> None:
    if str(source["state_id"]) != str(alternative.get("state_id")):
        raise ValueError("counterfactual must target the same public state")
    if str(source.get("task_id")) != str(alternative.get("task_id")):
        raise ValueError("counterfactual must belong to the same task")
    if alternative.get("plausible_alternative") is not True:
        raise ValueError("counterfactual must be marked as a plausible same-task answer")
    original = source.get("fact_audit_context") or {}
    changed = alternative.get("fact_audit_context") or {}
    if not changed or changed == original:
        raise ValueError("counterfactual fact context must change the answer")


def _payload(
    source: Mapping[str, Any], facts: Mapping[str, Any]
) -> dict[str, Any]:
    privilege = source.get("privileged_context") or {}
    payload = {
        "domain": facts.get("domain", source.get("domain", "tau2")),
        "domain_policy": privilege.get("domain_policy", ""),
        "authoritative_oracle_steps": facts.get(
            "authoritative_oracle_steps", ""
        ),
        "current_history": source["public_state"],
        "available_tools": source.get("tools") or [],
        "student_profile": source.get("student_profile") or {},
    }
    for key in (
        "goal_object_locations",
        "destination_receptacle",
        "unobserved_states",
    ):
        if key in facts:
            payload[key] = facts[key]
    return payload


def _result_row(source: Mapping[str, Any], result, *, counterpart: str) -> dict:
    return {
        "state_id": source["state_id"],
        "task_id": source.get("task_id"),
        "state_hash": source["state_hash"],
        "hint_level": HintLevel.HINTER.value,
        "hint": result.to_dict(),
        "hint_error": result.error,
        "counterpart": counterpart,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate open-hinter outputs under plausible same-task facts"
    )
    parser.add_argument("audit_rows", type=Path)
    parser.add_argument("--counterfactuals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    by_state = {}
    for row in _read_jsonl(args.audit_rows):
        by_state.setdefault(str(row["state_id"]), row)
    rows = list(by_state.values())
    alternatives = {
        str(row["state_id"]): row
        for row in _read_jsonl(args.counterfactuals)
    }
    hinter = OpenModelTeacherHinter(HintEndpoint.from_env(required=True))
    original_rows = []
    changed_rows = []
    for source in rows:
        alternative = alternatives.get(str(source["state_id"]))
        if alternative is None:
            raise ValueError(f"missing counterfactual for {source['state_id']}")
        validate_same_task_counterfactual(source, alternative)
        original_facts = source.get("fact_audit_context") or {}
        changed_facts = alternative["fact_audit_context"]
        original = hinter.hint(_payload(source, original_facts), HintLevel.HINTER)
        changed = hinter.hint(_payload(source, changed_facts), HintLevel.HINTER)
        original_rows.append(_result_row(source, original, counterpart="original"))
        changed_rows.append(_result_row(source, changed, counterpart="counterfactual"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in (
        ("original.jsonl", original_rows),
        ("counterfactual.jsonl", changed_rows),
    ):
        (args.output_dir / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values),
            encoding="utf-8",
        )
    print(json.dumps({"paired_open_hinter_rows": len(original_rows)}))


if __name__ == "__main__":
    main()
