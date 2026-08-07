from trl import SFTTrainer as HFSFTTrainer

from swift.rlhf_trainers import GKDTrainer
from swift.rlhf_trainers.gkd_trainer import DataSource


class NaturalDecisionStudentTrainer(GKDTrainer):
    """Repair-then-distill training on verified natural decision states.

    Collection has already replaced the Student action with the one-action Teacher
    repair and filtered out non-positive intervention advantages. The first method
    therefore performs ordinary token-level distillation on that complete action;
    it does not reconstruct token cutoffs or apply a second soft weighting scheme.
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not all(
            item.get("training_target") == "repair_then_distill"
            and item.get("messages")
            and item["messages"][-1]["role"] == "assistant"
            and float(item.get("intervention_advantage", 0.0)) > 0
            for item in inputs
        ):
            raise ValueError(
                "Student training requires positive natural-decision "
                "repair_then_distill rows"
            )

        collected = []
        for item in inputs:
            row = dict(item)
            row["messages"] = [dict(message) for message in item["messages"]]
            row["add_eos"] = False
            collected.append(row)

        encoded_inputs = self._prepare_batch_inputs(collected, encode_prompt_only=False)
        encoded_inputs["_data_source"] = DataSource.STUDENT
        with self.template.forward_context(self.model, encoded_inputs):
            return HFSFTTrainer.training_step(
                self, model, encoded_inputs, num_items_in_batch
            )

    def _prepare_batch_inputs(self, inputs, encode_prompt_only=False):
        encoded = super()._prepare_batch_inputs(inputs, encode_prompt_only)
        for key in (
            "training_target",
            "original_branch_messages",
            "teacher_hint",
            "intervention_advantage",
            "student_value",
            "teacher_value",
            "state_hash",
            "sample_hash",
            "domain",
            "task_split",
            "task_id",
        ):
            encoded.pop(key, None)
        return encoded
