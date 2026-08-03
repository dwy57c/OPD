#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/OPD/correctability_coevolution
DATA=${1:-$ROOT/artifacts/smoke/buyer_grpo.jsonl}
OUT=${2:-$ROOT/artifacts/smoke/buyer_adapter}
export HF_HOME="$ROOT/runtime/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$ROOT/runtime/modelscope_cache"
mkdir -p "$HF_HOME"

PYTHONPATH="$ROOT:/workspace/OPD/tau2-bench/src" \
CUDA_VISIBLE_DEVICES=5 \
python -m swift.cli.rlhf \
  --rlhf_type grpo \
  --model /models/Qwen3-4B \
  --external_plugins "$ROOT/coevo/training/swift_plugin.py" \
  --dataset "$DATA" \
  --remove_unused_columns false \
  --reward_funcs tau2_correctability \
  --multi_turn_scheduler tau2_buyer \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules all-linear \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.45 \
  --vllm_max_model_len 4096 \
  --num_generations 4 \
  --max_turns 2 \
  --max_completion_length 128 \
  --temperature 0.8 \
  --enable_thinking false \
  --beta 0.01 \
  --scale_rewards none \
  --torch_dtype bfloat16 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_steps 1 \
  --save_only_model true \
  --report_to none \
  --output_dir "$OUT"
