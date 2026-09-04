#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from coevo.config import HintEndpoint
from coevo.hints import HintLevel
from coevo.models.hinted_teacher import ClosedModelTeacherHinter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate E1 hints after rotating only hidden instance facts"
    )
    parser.add_argument("audit_rows", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.audit_rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hinter = ClosedModelTeacherHinter(HintEndpoint.from_env(required=True))
    output = []
    for level in (HintLevel.L1_POLICY, HintLevel.L2_PROCEDURAL, HintLevel.L3_ORACLE):
        selected = [row for row in rows if row.get("hint_level") == level.value]
        if len(selected) < 2:
            raise ValueError(f"counterfactual {level.value} requires at least two states")
        rotated = selected[1:] + selected[:1]
        for row, decoy in zip(selected, rotated):
            facts = dict(decoy.get("fact_audit_context") or {})
            privilege = dict(row.get("privileged_context") or {})
            privilege["authoritative_oracle_steps"] = facts.get(
                "authoritative_oracle_steps", ""
            )
            payload = {
                "domain": facts.get("domain", row.get("domain", "tau2")),
                "domain_policy": privilege.get("domain_policy", ""),
                "authoritative_oracle_steps": privilege[
                    "authoritative_oracle_steps"
                ],
                "current_history": row["public_state"],
                "available_tools": row.get("tools") or [],
            }
            for key in (
                "goal_object_locations",
                "destination_receptacle",
                "unobserved_states",
            ):
                if key in facts:
                    payload[key] = facts[key]
            result = hinter.hint(payload, level)
            output.append(
                {
                    **row,
                    "hint": result.to_dict(),
                    "hint_error": result.error,
                    "counterfactual_source_state_id": decoy["state_id"],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
