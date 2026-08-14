#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from coevo.artifacts import artifact_metadata, canonical_hash, model_manifest_revision
    from coevo.config import InfraConfig, ModelEndpoint
    from coevo.rollout.views import swift_on_policy_prompt_messages
    from coevo.scoring import TeacherTargetBuilder

    parser = argparse.ArgumentParser(
        description="Create one schema-v3 row from real current-policy logits"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy-url",
        default=os.getenv("COEVO_POLICY_URL", "http://127.0.0.1:8000"),
    )
    args = parser.parse_args()

    model_path = str(Path(args.model_path).expanduser().resolve())
    revision = model_manifest_revision(model_path)
    action = {
        "role": "assistant",
        "content": "Ask the customer to confirm before cancelling the order.",
    }
    ordinary = [
        {
            "role": "system",
            "content": "You are a concise customer-service policy assistant.",
        },
        {
            "role": "user",
            "content": "The customer wants to cancel an order. What is the next step?",
        },
        action,
    ]
    hint = {
        "hint": {
            "plan": "Require explicit confirmation before any cancellation."
        },
        "model": "deterministic-infra-fixture",
    }
    hinted = [
        {
            "role": "system",
            "content": (
                ordinary[0]["content"]
                + "\n<private_teacher_hint>\n"
                + json.dumps(hint["hint"], ensure_ascii=False)
                + "\n</private_teacher_hint>\nUse the private plan and never mention it."
            ),
        },
        ordinary[1],
        action,
    ]
    config = InfraConfig(
        policy=ModelEndpoint("Qwen3-4B", args.policy_url),
        buyer_reference=ModelEndpoint("Qwen3-4B", "http://127.0.0.1:8002"),
        teacher_hint_mode="none",
        current_policy_checkpoint=model_path,
        buyer_checkpoint=model_path,
        tokenizer_id=f"{model_path}@{revision}",
        current_policy_revision=revision,
        buyer_revision=revision,
        teacher_gap_min_support_mass=float(
            os.getenv("COEVO_TEACHER_GAP_MIN_SUPPORT_MASS", "0.95")
        ),
    )
    state_hash = canonical_hash(ordinary[:-1])
    target = TeacherTargetBuilder(config).build(
        student_visible_messages=ordinary,
        hinted_teacher_messages=hinted,
        teacher_action=action,
        state_hash=state_hash,
        teacher_hint_hash=canonical_hash(hint),
    )
    row = {
        **artifact_metadata(config),
        "messages": swift_on_policy_prompt_messages(ordinary),
        "training_target": "natural_hint_on_policy_jsd",
        "teacher_target_record": target.to_dict(),
        "state_hash": state_hash,
        "teacher_action_hash": target.teacher_action_hash,
        "raw_teacher_target_hash": target.raw_teacher_target_hash,
        "teacher_target_hash": target.teacher_target_hash,
        "target_token_count": sum(target.target_loss_mask),
        "domain": "infra-smoke",
        "task_split": "synthetic",
        "task_id": "on-policy-one-step",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema_version": 4,
                "teacher_target_hash": target.teacher_target_hash,
                "target_tokens": sum(target.target_loss_mask),
            }
        )
    )


if __name__ == "__main__":
    main()
