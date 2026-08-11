from copy import deepcopy
from types import MethodType, SimpleNamespace
import math

import pytest
import torch

from coevo.artifacts import assistant_action_hash, canonical_hash
from coevo.scoring.skill_contrast import (
    SkillContrastConfig,
    construct_skill_contrast_target,
)
from coevo.scoring.teacher_target import TeacherTargetRecord
from coevo.training.gated_gkd import (
    NaturalDecisionStudentTrainer,
    cached_target_distillation,
    validate_student_training_row,
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
        "messages": list(target.student_visible_messages),
        "training_target": "skill_contrast_teacher_distill",
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
def test_schema_v4_accepts_complete_text_and_tool_targets(action):
    validate_student_training_row(training_row(action))


def test_legacy_student_dataset_fails_with_actionable_version_error():
    row = training_row()
    row.pop("schema_version")
    with pytest.raises(ValueError, match="Recollect or migrate"):
        validate_student_training_row(row)


def test_cached_target_is_loaded_before_optimization():
    trainer = object.__new__(NaturalDecisionStudentTrainer)

    def fake_prepare(self, rows, encode_prompt_only=False):
        return {
            "input_ids": torch.tensor([[1, 2, 11, 12]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 11, 12]]),
        }

    # Patch the base-class call site through the instance-visible method used by
    # the regression test, without contacting any Teacher endpoint.
    from swift.rlhf_trainers import GKDTrainer

    original = GKDTrainer._prepare_batch_inputs
    GKDTrainer._prepare_batch_inputs = fake_prepare
    try:
        encoded = trainer._prepare_cached_target_inputs([training_row()])
    finally:
        GKDTrainer._prepare_batch_inputs = original

    assert len(encoded["_teacher_target_records"]) == 1
    assert encoded["_teacher_target_records"][0].teacher_target_hash


def test_student_loss_is_finite_unweighted_and_has_target_gradients():
    target = target_record()
    logits = torch.zeros(2, 20, requires_grad=True)
    loss, metrics = cached_target_distillation(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0
    assert metrics["skill_gate_mean"].item() > 0


def test_closed_target_returns_graph_connected_zero():
    logits = torch.randn(2, 20, requires_grad=True)
    target = SimpleNamespace(target_token_ids=(11, 12), target_loss_mask=(0, 0))
    loss, _ = cached_target_distillation(logits, target)
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() == 0.0
