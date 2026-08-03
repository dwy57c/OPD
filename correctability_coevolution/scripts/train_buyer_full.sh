#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/OPD/correctability_coevolution
DATA=${1:-$ROOT/artifacts/round/buyer_grpo.jsonl}
OUT=${2:-$ROOT/artifacts/round/buyer_full}
STEPS=${3:-2}
export HF_HOME="$ROOT/runtime/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$ROOT/runtime/modelscope_cache"
mkdir -p "$HF_HOME"

PYTHONPATH="$ROOT:/workspace/OPD/tau2-bench/src" \
CUDA_VISIBLE_DEVICES="${COEVO_BUYER_TRAIN_GPUS:-5}" \
python -m swift.cli.rlhf \
  --rlhf_type grpo \
  --model "${COEVO_BUYER_BASE_MODEL:-/models/Qwen3-4B}" \
  --external_plugins "$ROOT/coevo/training/swift_plugin.py" \
  --dataset "$DATA" \
  --remove_unused_columns false \
  --reward_funcs tau2_correctability \
  --multi_turn_scheduler tau2_buyer \
  --tuner_type full \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8003 \
  --num_generations 4 \
  --max_turns 2 \
  --max_completion_length 128 \
  --temperature 0.8 \
  --enable_thinking false \
  --beta 0.01 \
  --scale_rewards none \
  --torch_dtype bfloat16 \
  --max_steps "$STEPS" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_steps "$STEPS" \
  --save_only_model true \
  --report_to none \
  --output_dir "$OUT"
