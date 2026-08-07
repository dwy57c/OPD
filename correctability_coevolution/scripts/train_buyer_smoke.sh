#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

DATA=${1:-$COEVO_ROOT/artifacts/smoke/buyer_grpo.jsonl}
OUT=${2:-$COEVO_ROOT/artifacts/smoke/buyer_adapter}
export HF_HOME="$COEVO_ROOT/runtime/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$COEVO_ROOT/runtime/modelscope_cache"
mkdir -p "$HF_HOME"
coevo_require_nonempty_file "$DATA"
python "$COEVO_ROOT/scripts/wait_for_servers.py" \
  "http://127.0.0.1:${COEVO_BUYER_ROLLOUT_PORT:-8003}" \
  --timeout 30 \
  --health-path /health/

CUDA_VISIBLE_DEVICES="${COEVO_BUYER_TRAIN_GPUS:-3}" \
python -m swift.cli.rlhf \
  --rlhf_type grpo \
  --model "${COEVO_BUYER_BASE_MODEL:-$COEVO_BUYER_PATH}" \
  --model_type "${COEVO_MODEL_TYPE:-qwen3}" \
  --template "${COEVO_BUYER_TEMPLATE_TYPE:-qwen3}" \
  --external_plugins "$COEVO_ROOT/coevo/training/swift_plugin.py" \
  --dataset "$DATA" \
  --remove_unused_columns false \
  --reward_funcs tau2_buyer_utility \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules all-linear \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "${COEVO_BUYER_ROLLOUT_PORT:-8003}" \
  --vllm_server_pass_dataset true \
  --num_generations "${COEVO_BUYER_NUM_GENERATIONS:-4}" \
  --max_completion_length "${COEVO_BUYER_MAX_COMPLETION_LENGTH:-512}" \
  --temperature 0.8 \
  --enable_thinking "${COEVO_BUYER_ENABLE_THINKING:-true}" \
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
