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
NUM_GENERATIONS=${COEVO_BUYER_NUM_GENERATIONS:-4}
GRAD_ACCUM_STEPS=${COEVO_BUYER_GRAD_ACCUM_STEPS:-4}
BUYER_ATTN_IMPL=${COEVO_BUYER_ATTN_IMPL:-flash_attn}
BUYER_BETA=${COEVO_BUYER_BETA:-0.01}
MIN_PROMPTS=$(((GRAD_ACCUM_STEPS + NUM_GENERATIONS - 1) / NUM_GENERATIONS))
DATA_ROWS=$(wc -l < "$DATA")
if (( DATA_ROWS < MIN_PROMPTS )); then
  echo "Buyer smoke needs at least $MIN_PROMPTS prompt rows for num_generations=$NUM_GENERATIONS and gradient_accumulation_steps=$GRAD_ACCUM_STEPS; got $DATA_ROWS" >&2
  exit 2
fi
python "$COEVO_ROOT/scripts/wait_for_servers.py" \
  "http://127.0.0.1:${COEVO_BUYER_ROLLOUT_PORT:-8003}" \
  --timeout 30 \
  --health-path /health/

CUDA_VISIBLE_DEVICES="${COEVO_BUYER_TRAIN_GPUS:-3}" \
python -m swift.cli.rlhf \
  --rlhf_type grpo \
  --model "${COEVO_BUYER_BASE_MODEL:-$COEVO_BUYER_PATH}" \
  --model_type "${COEVO_BUYER_MODEL_TYPE:-${COEVO_MODEL_TYPE:-qwen3}}" \
  --template "${COEVO_BUYER_TEMPLATE_TYPE:-qwen3}" \
  --external_plugins "$COEVO_ROOT/coevo/training/swift_plugin.py" \
  --dataset "$DATA" \
  --remove_unused_columns false \
  --reward_funcs tau2_stage_learning_progress \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules all-linear \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "${COEVO_BUYER_ROLLOUT_PORT:-8003}" \
  --vllm_server_pass_dataset true \
  --num_generations "$NUM_GENERATIONS" \
  --max_completion_length "${COEVO_BUYER_MAX_COMPLETION_LENGTH:-512}" \
  --temperature 0.8 \
  --enable_thinking "${COEVO_BUYER_ENABLE_THINKING:-true}" \
  --beta "$BUYER_BETA" \
  --scale_rewards group \
  --torch_dtype bfloat16 \
  --attn_impl "$BUYER_ATTN_IMPL" \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --learning_rate 1e-5 \
  --gradient_checkpointing "${COEVO_BUYER_GRADIENT_CHECKPOINTING:-true}" \
  --logging_nan_inf_filter false \
  --logging_steps 1 \
  --save_steps 1 \
  --save_only_model true \
  --report_to "$COEVO_REPORT_TO" \
  --run_name "${COEVO_WANDB_RUN_NAME:-buyer-infra-smoke}" \
  --output_dir "$OUT"

LATEST_ADAPTER=$(find "$OUT" -type f -name adapter_model.safetensors \
  -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -z ${LATEST_ADAPTER:-} ]]; then
  echo "Buyer LoRA smoke did not produce adapter_model.safetensors" >&2
  exit 1
fi
python "$COEVO_ROOT/scripts/check_adapter_finite.py" "$LATEST_ADAPTER"
