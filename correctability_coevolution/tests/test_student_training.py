from copy import deepcopy
import math

import pytest

from coevo.artifacts import assistant_action_hash, canonical_hash
from coevo.scoring.skill_contrast import (
    SkillContrastConfig,
    construct_skill_contrast_target,
)
from coevo.scoring.teacher_target import TeacherTargetRecord
from coevo.training.gated_gkd import (
    NaturalDecisionStudentTrainer,
    validate_student_training_row,
)
from coevo.rollout.views import (
    swift_on_policy_prompt_messages,
    swift_training_messages,
)


def target_record(action=None):
    action = action or {"role": "assistant", "content": "Please confirm."}
    ordinary = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Cancel it."},
        deepcopy(action),
    ]
    hinted = [
        {"role": "system", "content": "policy private_teacher_hint"},
        ordinary[1],
        deepcopy(action),
    ]
    target_ids = (11, 12)
    q_logs = tuple((math.log(0.8), math.log(0.2)) for _ in target_ids)
    q_ids = tuple((target, 5) for target in target_ids)
    p_logs = tuple((math.log(0.6), math.log(0.4)) for _ in target_ids)
    p_ids = q_ids
    contrast = construct_skill_contrast_target(
        hinted_topk_logprobs=q_logs,
        hinted_topk_token_ids=q_ids,
        unhinted_topk_logprobs=p_logs,
        unhinted_topk_token_ids=p_ids,
        target_token_ids=target_ids,
        config=SkillContrastConfig(minimum_support_mass=0.9),
    )
    state_hash = "state"
    action_hash = assistant_action_hash(action)
    raw_hash = canonical_hash(
        TeacherTargetRecord.raw_hash_payload(
            teacher_checkpoint="current",
            teacher_hint_hash="hint",
            state_hash=state_hash,
            teacher_action_hash=action_hash,
            target_token_ids=target_ids,
            hinted_topk_logprobs=q_logs,
            hinted_topk_token_ids=q_ids,
        )
    )
    final_hash = canonical_hash(
        TeacherTargetRecord.target_hash_payload(
            raw_teacher_target_hash=raw_hash,
            sharpened_topk_logprobs=contrast.sharpened_topk_logprobs,
            sharpened_topk_token_ids=contrast.sharpened_topk_token_ids,
            sharpening_temperatures=contrast.sharpening_temperatures,
        )
    )
    return TeacherTargetRecord(
        schema_version=2,
        state_hash=state_hash,
        teacher_action_hash=action_hash,
        raw_teacher_target_hash=raw_hash,
        teacher_target_hash=final_hash,
        teacher_checkpoint="current",
        teacher_hint_hash="hint",
        student_visible_messages=tuple(ordinary),
        hinted_teacher_messages=tuple(hinted),
        teacher_action=deepcopy(action),
        target_token_ids=target_ids,
        target_loss_mask=(1, 1),
        hinted_topk_logprobs=q_logs,
        hinted_topk_token_ids=q_ids,
        hinted_support_mass=(1.0, 1.0),
        unhinted_reference_checkpoint="current",
        unhinted_reference_topk_logprobs=p_logs,
        unhinted_reference_topk_token_ids=p_ids,
        unhinted_reference_support_mass=(1.0, 1.0),
        skill_contrast_scores=contrast.skill_contrast_scores,
        skill_gate_values=contrast.skill_gate_values,
        sharpening_temperatures=contrast.sharpening_temperatures,
        sharpened_topk_logprobs=contrast.sharpened_topk_logprobs,
        sharpened_topk_token_ids=contrast.sharpened_topk_token_ids,
        sharpened_support_mass=contrast.sharpened_support_mass,
        raw_teacher_entropy=contrast.raw_teacher_entropy,
        sharpened_teacher_entropy=contrast.sharpened_teacher_entropy,
    )


def training_row(action=None):
    target = target_record(action)
    return {
        "schema_version": 4,
        "messages": swift_on_policy_prompt_messages(
            list(target.student_visible_messages)
        ),
        "training_target": "natural_hint_on_policy_jsd",
        "teacher_target_record": target.to_dict(),
        "teacher_target_hash": target.teacher_target_hash,
    }


@pytest.mark.parametrize(
    "action",
    [
        {"role": "assistant", "content": "Please confirm."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "first", "arguments": "{}"},
                },
                {
                    "id": "two",
                    "type": "function",
                    "function": {"name": "second", "arguments": '{"x": 1}'},
                },
            ],
        },
    ],
)
def test_schema_v4_accepts_decision_states_with_text_and_tool_references(action):
    validate_student_training_row(training_row(action))


def test_reference_tool_action_is_not_copied_into_on_policy_prompt():
    row = training_row(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "ignored-by-tokenizer",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"user_id":"abc"}',
                    },
                }
            ],
        }
    )
    assert row["messages"][-1] == {
        "role": "assistant",
        "content": "",
    }
    assert "response_token_ids" not in row
    validate_student_training_row(row)


def test_openai_tool_call_conversion_preserves_function_and_arguments():
    rows = swift_training_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "ignored-by-tokenizer",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"user_id":"abc"}',
                        },
                    }
                ],
            }
        ]
    )
    assert rows == [
        {
            "role": "tool_call",
            "content": '{"name":"lookup","arguments":{"user_id":"abc"}}',
        }
    ]


def test_precomputed_response_token_ids_are_rejected():
    row = training_row()
    row["response_token_ids"] = [11, 12]
    with pytest.raises(ValueError, match="must not contain"):
        validate_student_training_row(row)


def test_arrow_loss_metadata_does_not_change_teacher_action_identity():
    row = training_row()
    for message in row["messages"]:
        message["loss"] = None
    validate_student_training_row(row)


def test_legacy_student_dataset_fails_with_actionable_version_error():
    row = training_row()
    row.pop("schema_version")
    with pytest.raises(ValueError, match="Recollect or migrate"):
        validate_student_training_row(row)


def test_teacher_view_uses_private_hint_but_excludes_reference_action():
    trainer = object.__new__(NaturalDecisionStudentTrainer)
    teacher_rows = trainer._build_opsd_teacher_data([training_row()])

    assert len(teacher_rows) == 1
    messages = teacher_rows[0]["messages"]
    assert "private_teacher_hint" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Cancel it."}
    assert all(message.get("content") != "Please confirm." for message in messages)


def test_trainer_inherits_ms_swift_on_policy_training_step():
    from swift.rlhf_trainers import GKDTrainer

    assert NaturalDecisionStudentTrainer.training_step is GKDTrainer.training_step
