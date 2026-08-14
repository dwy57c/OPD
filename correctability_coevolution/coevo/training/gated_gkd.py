from copy import deepcopy
import math

import torch
import torch.nn.functional as F
from trl import SFTTrainer as HFSFTTrainer

from swift.rlhf_trainers import GKDTrainer
from swift.rlhf_trainers.gkd_trainer import DataSource

from coevo.artifacts import assistant_action_hash
from coevo.rollout.views import swift_cached_target_messages
from coevo.scoring.teacher_target import TeacherTargetRecord


DATASET_SCHEMA_VERSION = 4
TRAINING_TARGET = "skill_contrast_teacher_distill"


def validate_student_training_row(row: dict) -> TeacherTargetRecord:
    version = row.get("schema_version")
    if version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            "Student dataset schema_version=4 is required for cached skill-contrast "
            f"targets; got {version!r}. Recollect or migrate the dataset."
        )
    if row.get("training_target") != TRAINING_TARGET:
        raise ValueError(
            f"training_target must be {TRAINING_TARGET!r}; "
            f"got {row.get('training_target')!r}"
        )
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("schema v4 requires a messages list")
    target_value = row.get("teacher_target_record")
    if not isinstance(target_value, dict):
        raise ValueError("schema v4 row is missing teacher_target_record")
    target = TeacherTargetRecord.from_dict(target_value)
    if assistant_action_hash(target.teacher_action) != target.teacher_action_hash:
        raise ValueError("cached Teacher action hash is invalid")
    expected_messages = swift_cached_target_messages(
        list(target.student_visible_messages)
    )
    token_bearing_messages = [
        {"role": message.get("role"), "content": message.get("content") or ""}
        for message in messages
    ]
    if token_bearing_messages != expected_messages:
        raise ValueError(
            "training messages differ from the cached Teacher trajectory after "
            "Swift tool-call normalization"
        )
    response_token_ids = row.get("response_token_ids")
    if not isinstance(response_token_ids, list) or tuple(response_token_ids) != (
        target.target_token_ids
    ):
        raise ValueError(
            "response_token_ids must exactly equal the frozen Teacher target tokens"
        )
    if row.get("teacher_target_hash") != target.teacher_target_hash:
        raise ValueError("row and cached Teacher target hashes do not match")
    return target


def cached_target_distillation(
    student_logits: torch.Tensor,
    target: TeacherTargetRecord,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean forward KL to the cached target on support plus aggregate tail."""
    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [target_tokens, vocabulary]")
    if student_logits.shape[0] != len(target.target_token_ids):
        raise ValueError("Student logits and cached target token counts do not align")
    losses = []
    support_masses = []
    for row_index, active in enumerate(target.target_loss_mask):
        if not active:
            continue
        token_ids = torch.tensor(
            target.sharpened_topk_token_ids[row_index],
            device=student_logits.device,
            dtype=torch.long,
        )
        if token_ids.numel() == 0:
            raise ValueError("cached Teacher support is empty")
        if token_ids.min() < 0 or token_ids.max() >= student_logits.shape[-1]:
            raise ValueError("cached Teacher support contains an invalid token ID")
        if token_ids.unique().numel() != token_ids.numel():
            raise ValueError("cached Teacher support contains duplicate token IDs")
        target_logs = torch.tensor(
            target.sharpened_topk_logprobs[row_index],
            device=student_logits.device,
            dtype=student_logits.dtype,
        ).detach()
        q_explicit = target_logs.exp()
        q_mass = q_explicit.sum()
        if float(q_mass.detach().cpu()) > 1 + 1e-4:
            raise ValueError("cached Teacher support mass exceeds one")
        q_tail = torch.clamp(1.0 - q_mass, min=0.0)

        student_logs = F.log_softmax(student_logits[row_index], dim=-1)
        p_logs = student_logs.gather(0, token_ids)
        p_explicit = p_logs.exp()
        p_tail = torch.clamp(1.0 - p_explicit.sum(), min=epsilon)
        p_tail_log = p_tail.log()

        explicit_kl = (
            q_explicit
            * (target_logs - p_logs)
        ).sum()
        tail_kl = torch.where(
            q_tail > 0,
            q_tail * (torch.clamp(q_tail, min=epsilon).log() - p_tail_log),
            q_tail,
        )
        losses.append(explicit_kl + tail_kl)
        support_masses.append(q_mass)
    if not losses:
        zero = student_logits.sum() * 0.0
        return zero, {
            "target_support_mass": zero.detach(),
            "skill_gate_mean": zero.detach(),
            "skill_contrast_mean": zero.detach(),
            "sharpening_temperature_mean": zero.detach(),
        }
    loss = torch.stack(losses).mean()
    metrics = {
        "target_support_mass": torch.stack(support_masses).mean().detach(),
        "skill_gate_mean": torch.tensor(
            sum(target.skill_gate_values) / len(target.skill_gate_values),
            device=student_logits.device,
        ),
        "skill_contrast_mean": torch.tensor(
            sum(target.skill_contrast_scores) / len(target.skill_contrast_scores),
            device=student_logits.device,
        ),
        "sharpening_temperature_mean": torch.tensor(
            sum(target.sharpening_temperatures)
            / len(target.sharpening_temperatures),
            device=student_logits.device,
        ),
    }
    return loss, metrics


def mask_only_cached_target(
    input_ids: torch.Tensor, targets: list[TeacherTargetRecord]
) -> torch.Tensor:
    """Mask every token except the final exact frozen Teacher target span."""
    if input_ids.ndim != 2 or input_ids.shape[0] != len(targets):
        raise ValueError("input_ids and cached targets must form one aligned batch")
    labels = torch.full_like(input_ids, -100)
    for batch_index, target in enumerate(targets):
        sequence = input_ids[batch_index].tolist()
        needle = list(target.target_token_ids)
        starts = [
            index
            for index in range(len(sequence) - len(needle) + 1)
            if sequence[index : index + len(needle)] == needle
        ]
        if not starts:
            raise ValueError(
                "frozen Teacher target token IDs are absent from the encoded sequence"
            )
        start = starts[-1]
        stop = start + len(needle)
        labels[batch_index, start:stop] = input_ids[batch_index, start:stop]
    return labels


class NaturalDecisionStudentTrainer(GKDTrainer):
    """Distill cached skill-contrast Teacher targets at natural boundaries.

    Target construction is detached and happens once during collection. Neither
    stage progress nor terminal Teacher quality enters this loss.
    """

    _VIEW_FIELDS = (
        "tools",
        "images",
        "audios",
        "videos",
        "response_token_ids",
    )

    @classmethod
    def _view_row(cls, row: dict) -> dict:
        view = {
            key: deepcopy(row[key])
            for key in cls._VIEW_FIELDS
            if key in row
        }
        view["messages"] = deepcopy(row["messages"])
        view["add_eos"] = False
        return view

    def _prepare_cached_target_inputs(self, inputs: list[dict]) -> dict:
        targets = [validate_student_training_row(row) for row in inputs]
        rows = [self._view_row(row) for row in inputs]
        encoded = GKDTrainer._prepare_batch_inputs(
            self, rows, encode_prompt_only=False
        )
        encoded["labels"] = mask_only_cached_target(
            encoded["input_ids"], targets
        )
        encoded["_data_source"] = DataSource.DATASET
        encoded["_teacher_target_records"] = targets
        return encoded

    def training_step(self, model, inputs, num_items_in_batch=None):
        encoded_inputs = self._prepare_cached_target_inputs(inputs)
        with self.template.forward_context(self.model, encoded_inputs):
            return HFSFTTrainer.training_step(
                self, model, encoded_inputs, num_items_in_batch
            )

    def _prepare_batch_inputs(self, inputs, encode_prompt_only=False):
        rows = [self._view_row(row) for row in inputs]
        return GKDTrainer._prepare_batch_inputs(
            self, rows, encode_prompt_only=encode_prompt_only
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        inputs.pop("_data_source", None)
        targets = inputs.pop("_teacher_target_records", None)
        if targets is None:
            raise RuntimeError(
                "cached Teacher supervision was not loaded before compute_loss"
            )
        labels = inputs["labels"]
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"prompt", "labels"} and not key.startswith("_")
        }
        outputs = model(**model_inputs)
        if outputs.logits.shape[:2] != labels.shape:
            raise ValueError(
                "cached-target training requires full sequence logits; set "
                "--use_logits_to_keep false"
            )
        if len(targets) != labels.shape[0]:
            raise ValueError("cached target records do not align with the batch")

        losses = []
        metric_rows: dict[str, list[torch.Tensor]] = {}
        for batch_index, target in enumerate(targets):
            if isinstance(target, dict):
                target = TeacherTargetRecord.from_dict(target)
            prediction_mask = labels[batch_index, 1:].ne(-100)
            target_ids = labels[batch_index, 1:][prediction_mask].tolist()
            if tuple(target_ids) != target.target_token_ids:
                raise ValueError(
                    "locally tokenized action IDs differ from cached Teacher target; "
                    "verify tokenizer and chat-template identity: "
                    f"local_len={len(target_ids)}, cached_len={len(target.target_token_ids)}, "
                    f"local_prefix={target_ids[:16]}, "
                    f"cached_prefix={list(target.target_token_ids[:16])}"
                )
            active_logits = outputs.logits[batch_index, :-1][prediction_mask]
            row_loss, row_metrics = cached_target_distillation(
                active_logits,
                target,
                epsilon=float(getattr(self.args, "epsilon", 1e-8)),
            )
            losses.append(row_loss)
            for name, value in row_metrics.items():
                metric_rows.setdefault(name, []).append(value)
        loss = (
            torch.stack(losses).mean()
            if losses
            else outputs.logits.sum() * 0.0
        )
        train_metrics = getattr(self, "_metrics", {}).get("train")
        if train_metrics is not None:
            for name, values in metric_rows.items():
                train_metrics[name].append(
                    float(torch.stack(values).mean().detach().cpu())
                )
        if not torch.isfinite(loss):
            raise FloatingPointError("cached Teacher distillation loss is not finite")
        if return_outputs:
            return loss, outputs
        return loss
