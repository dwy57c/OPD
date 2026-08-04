import os

from swift.rlhf_trainers import GRPOTrainer


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
