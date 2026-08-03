from trl import SFTTrainer as HFSFTTrainer

from swift.rlhf_trainers import GKDTrainer
from swift.rlhf_trainers.gkd_trainer import DataSource, TeacherOutput

from coevo.training.gates import gated_example_mean


class CorrectabilityGKDTrainer(GKDTrainer):
    """Collected on-policy OPSD with an absolute per-example correctability gate."""

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not all(
            item.get("messages") and item["messages"][-1]["role"] == "assistant"
            for item in inputs
        ):
            return super().training_step(model, inputs, num_items_in_batch)

        self._correctability_gates = [
            float(item["correctability"]) for item in inputs
        ]
        collected = []
        for item in inputs:
            row = dict(item)
            row["messages"] = [dict(message) for message in item["messages"]]
            row["add_eos"] = False
            collected.append(row)

        teacher_data = self._build_opsd_teacher_data(collected)
        if teacher_data is None:
            raise ValueError("Collected OPSD rows require teacher_prompt")
        for row, teacher_row in zip(collected, teacher_data):
            teacher_row["messages"].append(dict(row["messages"][-1]))
            teacher_row["add_eos"] = False

        encoded_inputs = self._prepare_batch_inputs(collected, encode_prompt_only=False)
        encoded_inputs["_data_source"] = DataSource.STUDENT
        encoded_inputs["_opsd_teacher_messages"] = [
            row["messages"] for row in teacher_data
        ]
        encoded_inputs["_opsd_teacher_inputs"] = self._prepare_batch_inputs(
            teacher_data, encode_prompt_only=False
        )
        if self.use_teacher_api:
            teacher_logprobs, teacher_indices = self._fetch_teacher_logprobs_from_api(
                encoded_inputs, raw_inputs=collected
            )
            encoded_inputs["_teacher_api_logprobs"] = teacher_logprobs
            encoded_inputs["_teacher_api_indices"] = teacher_indices

        with self.template.forward_context(self.model, encoded_inputs):
            return HFSFTTrainer.training_step(
                self, model, encoded_inputs, num_items_in_batch
            )

    def _prepare_batch_inputs(self, inputs, encode_prompt_only=False):
        encoded = super()._prepare_batch_inputs(inputs, encode_prompt_only)
        for key in (
            "teacher_prompt",
            "correctability",
            "cutoff_count",
            "domain",
            "task_id",
        ):
            encoded.pop(key, None)
        return encoded

    @staticmethod
    def _teacher_slice(teacher_output: TeacherOutput, index: int) -> TeacherOutput:
        def take(value):
            return value[index : index + 1] if value is not None else None

        return TeacherOutput(
            full_logits=take(teacher_output.full_logits),
            topk_logprobs=take(teacher_output.topk_logprobs),
            topk_indices=take(teacher_output.topk_indices),
            opsd_teacher_labels=take(teacher_output.opsd_teacher_labels),
        )

    def _compute_jsd_loss(self, student_logits, teacher_output, labels):
        batch_size = student_logits.shape[0]
        gates = getattr(self, "_correctability_gates", [1.0] * batch_size)
        if len(gates) != batch_size:
            raise ValueError(f"gate count {len(gates)} != batch size {batch_size}")
        losses = []
        for index in range(batch_size):
            sample_loss = super()._compute_jsd_loss(
                student_logits[index : index + 1],
                self._teacher_slice(teacher_output, index),
                labels[index : index + 1],
            )
            losses.append(sample_loss)
        return gated_example_mean(losses, gates)
