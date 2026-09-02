from copy import deepcopy

import torch
import torch.nn.functional as F
from swift.rlhf_trainers import GKDTrainer
from swift.rlhf_trainers.gkd_trainer import DataSource, TeacherOutput

from coevo.artifacts import assistant_action_hash
from coevo.rollout.views import (
    swift_on_policy_prompt_messages,
    swift_training_messages,
)
from coevo.scoring.teacher_target import TeacherTargetRecord
from coevo.hints import HintLevel


DATASET_SCHEMA_VERSION = 4
TRAINING_TARGET = "natural_hint_on_policy_jsd"


def truncate_rows_to_active_token_budget(
    rows: list[dict], budget: int
) -> tuple[list[dict], int]:
    """Deterministically cap a dosage arm by reference active-token count.

    The current on-policy completion length is unknown before training.  The
    frozen reference action's ``target_token_count`` is therefore the auditable
    pre-training proxy shared by every arm.  The row that would cross the cap is
    excluded; no example is partially tokenized.
    """

    if budget < 1:
        raise ValueError("active-token budget must be positive")
    selected: list[dict] = []
    used = 0
    for row in rows:
        count = int(row.get("target_token_count", 0))
        if count < 1:
            raise ValueError("each training row must have target_token_count > 0")
        if used + count > budget:
            continue
        selected.append(row)
        used += count
        if used == budget:
            break
    return selected, used


def validate_student_training_row(row: dict) -> TeacherTargetRecord:
    """Validate one decision-state prompt for true on-policy distillation.

    The recorded Teacher action remains an auditable reference action. It is
    deliberately not the optimization target: the Student samples a fresh
    action at training time and the hinted frozen Teacher force-decodes those
    exact Student tokens.
    """
    version = row.get("schema_version")
    if version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            "Student dataset schema_version=4 is required for natural-hint "
            f"on-policy targets; got {version!r}. Recollect or migrate the dataset."
        )
    if row.get("training_target") != TRAINING_TARGET:
        raise ValueError(
            f"training_target must be {TRAINING_TARGET!r}; "
            f"got {row.get('training_target')!r}"
        )
    dataset_level = str(row.get("hint_level", HintLevel.L3_ORACLE.value))
    sample_level = HintLevel.parse(row.get("sample_hint_level", dataset_level))
    if dataset_level not in {"MIXED", sample_level.value}:
        raise ValueError(
            "row hint-level contract must equal sample_hint_level or MIXED"
        )
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("schema v4 requires a messages list")
    target_value = row.get("teacher_target_record")
    if not isinstance(target_value, dict):
        raise ValueError("schema v4 row is missing teacher_target_record")
    target = TeacherTargetRecord.from_dict(target_value)
    if assistant_action_hash(target.teacher_action) != target.teacher_action_hash:
        raise ValueError("recorded Teacher action hash is invalid")
    expected_messages = swift_on_policy_prompt_messages(
        list(target.student_visible_messages)
    )
    token_bearing_messages = [
        {"role": message.get("role"), "content": message.get("content") or ""}
        for message in messages
    ]
    if token_bearing_messages != expected_messages:
        raise ValueError(
            "training messages differ from the recorded Student decision state"
        )
    if row.get("response_token_ids"):
        raise ValueError(
            "on-policy rows must not contain precomputed response_token_ids; "
            "the current Student must generate the action during training"
        )
    if row.get("teacher_target_hash") != target.teacher_target_hash:
        raise ValueError("row and recorded Teacher target hashes do not match")
    return target


class NaturalDecisionStudentTrainer(GKDTrainer):
    """Step-level agentic OPD with a private natural Teacher note.

    ``GKDTrainer.training_step`` performs the Student rollout because the
    launcher sets ``lmbda=1``. This subclass supplies a second, privileged
    prompt view for force-decoding the sampled Student action. Gradients flow
    only through the unhinted Student and the loss is ms-swift's beta-JSD.
    """

    _VIEW_FIELDS = ("tools", "images", "audios", "videos", "add_eos")

    @classmethod
    def _view_row(cls, row: dict) -> dict:
        view = {
            key: deepcopy(row[key])
            for key in cls._VIEW_FIELDS
            if key in row
        }
        view["messages"] = deepcopy(row["messages"])
        return view

    def _build_opsd_teacher_data(self, inputs: list[dict]) -> list[dict]:
        """Build hinted prompts; the base trainer appends Student rollouts."""
        targets = [validate_student_training_row(row) for row in inputs]
        teacher_rows = []
        for source, target in zip(inputs, targets):
            row = self._view_row(source)
            # Exclude the recorded reference action. The base GKD trainer
            # appends newly sampled Student tokens before querying the frozen
            # Teacher endpoint.
            row["messages"] = swift_training_messages(
                list(target.hinted_teacher_messages[:-1])
            )
            if not row["messages"] or row["messages"][0].get("role") != "system":
                raise ValueError("hinted Teacher view must start with a system prompt")
            teacher_rows.append(row)
        return teacher_rows

    def _prepare_batch_inputs(self, inputs, encode_prompt_only=False):
        # Strip audit-only fields, especially any legacy cached token IDs,
        # before ms-swift encodes or generates from the prompt.
        rows = [self._view_row(row) for row in inputs]
        return GKDTrainer._prepare_batch_inputs(
            self, rows, encode_prompt_only=encode_prompt_only
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Run sparse Teacher JSD with unequal privileged/public prefix lengths.

        ms-swift 4.1.3 applies the Student's boolean ``logits_to_keep`` mask to
        Teacher API tensors before applying the separate OPSD masks. That only
        works when both prompts have equal length. A natural private note makes
        the Teacher prefix longer, so keep the full sparse Teacher grid and let
        ``_compute_jsd_loss`` select each view's response positions separately.
        """
        inputs = dict(inputs)
        data_source = inputs.pop("_data_source", DataSource.DATASET)
        teacher_logprobs = inputs.pop("_teacher_api_logprobs", None)
        teacher_indices = inputs.pop("_teacher_api_indices", None)
        teacher_inputs = inputs.pop("_opsd_teacher_inputs", None)
        inputs.pop("_opsd_teacher_messages", None)
        if not self.use_teacher_api or teacher_logprobs is None:
            raise RuntimeError(
                "natural-hint OPD requires sparse logits from a frozen Teacher API"
            )
        if teacher_inputs is None:
            raise RuntimeError("natural-hint OPD requires a privileged Teacher view")

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"prompt", "labels"}
        }
        if self.get_use_logits_to_keep(True):
            self.prepare_logits_to_keep(inputs)
            model_inputs["logits_to_keep"] = inputs["logits_to_keep"]
        if self.args.sft_alpha > 0:
            model_inputs["labels"] = inputs["labels"]
        outputs = model(**model_inputs)

        teacher_logprobs = F.pad(
            teacher_logprobs, (0, 0, 0, 1), value=float("-inf")
        )
        teacher_indices = F.pad(teacher_indices, (0, 0, 0, 1), value=0)
        teacher_output = TeacherOutput(
            topk_logprobs=teacher_logprobs,
            topk_indices=teacher_indices,
            opsd_teacher_labels=teacher_inputs["labels"],
        )
        loss = self._compute_jsd_loss(
            outputs.logits, teacher_output, inputs["labels"]
        )
        if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
            loss = loss + self.args.sft_alpha * outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError("natural-hint on-policy JSD loss is not finite")
        return (loss, outputs) if return_outputs else loss
