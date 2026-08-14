import os

import torch
from swift.rlhf_trainers import GRPOTrainer


_QWEN35_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
_QWEN35_SAFE_ATTN_IMPLS = frozenset({"flash_attn", "flash_attention_2"})


def validate_buyer_attention_backend(model_type, attn_impl) -> None:
    """Reject the Qwen3.5 SDPA path that produced non-finite GRPO updates."""
    normalized_model_type = str(model_type or "").strip().lower()
    if normalized_model_type not in _QWEN35_MODEL_TYPES:
        return
    normalized_attn_impl = str(attn_impl or "").strip().lower()
    if normalized_attn_impl not in _QWEN35_SAFE_ATTN_IMPLS:
        supported = ", ".join(sorted(_QWEN35_SAFE_ATTN_IMPLS))
        raise ValueError(
            "Qwen3.5 Buyer training requires an explicitly selected "
            f"FlashAttention2 backend ({supported}); got "
            f"attn_impl={attn_impl!r}. Default SDPA produced non-finite "
            "full-attention gradients in this runtime."
        )


class CoevolutionGRPOTrainer(GRPOTrainer):
    """GRPO trainer that can trust an explicitly preloaded identical vLLM base.

    Swift normally synchronizes the complete base model before its first LoRA
    rollout.  A 32B base is already loaded by our external rollout server, so
    that gather is redundant and can exceed one A100's memory under ZeRO-3.
    The opt-in is deliberately strict: only LoRA plus a LoRA-enabled external
    vLLM server may skip the base sync.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_buyer_attention_backend(
            getattr(self.args, "model_type", None),
            getattr(self.args, "attn_impl", None),
        )
        self.nonfinite_gradient_values_sanitized = 0
        self.nonfinite_gradient_tensors_sanitized = 0
        self._install_nonfinite_gradient_guard()
        if os.getenv("COEVO_VLLM_BASE_PRELOADED") != "1":
            return
        if not (
            self.args.tuner_type == "lora"
            and self.args.use_vllm
            and self.vllm_mode == "server"
            and self.rollout_enable_lora
        ):
            raise ValueError(
                "COEVO_VLLM_BASE_PRELOADED=1 requires LoRA training against a "
                "LoRA-enabled external vLLM server"
            )
        self.base_sync_done = True

    def _install_nonfinite_gradient_guard(self) -> None:
        action = os.getenv("COEVO_NONFINITE_GRADIENT_ACTION", "error").strip()
        if action not in {"error", "zero"}:
            raise ValueError(
                "COEVO_NONFINITE_GRADIENT_ACTION must be 'error' or 'zero'"
            )

        def guard(name, gradient):
            finite = torch.isfinite(gradient)
            if bool(finite.all()):
                return gradient
            count = int((~finite).sum().item())
            if action == "error":
                raise FloatingPointError(
                    f"non-finite Buyer gradient in {name}: {count} values"
                )
            self.nonfinite_gradient_values_sanitized += count
            self.nonfinite_gradient_tensors_sanitized += 1
            return torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                parameter.register_hook(
                    lambda gradient, name=name: guard(name, gradient)
                )

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        logs["nonfinite_gradient_values_sanitized"] = (
            self.nonfinite_gradient_values_sanitized
        )
        logs["nonfinite_gradient_tensors_sanitized"] = (
            self.nonfinite_gradient_tensors_sanitized
        )
        return super().log(logs, *args, **kwargs)
