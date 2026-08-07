#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

DATA=${1:-$COEVO_ROOT/artifacts/round/buyer_grpo.jsonl}
OUT=${2:-$COEVO_ROOT/artifacts/round/buyer_full}
STEPS=${3:-2}
export HF_HOME="$COEVO_ROOT/runtime/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$COEVO_ROOT/runtime/modelscope_cache"
mkdir -p "$HF_HOME"
coevo_require_nonempty_file "$DATA"
python "$COEVO_ROOT/scripts/wait_for_servers.py" \
  "http://127.0.0.1:${COEVO_BUYER_ROLLOUT_PORT:-8003}" \
  --timeout 30 \
  --health-path /health/

TRAIN_GPUS=${COEVO_BUYER_TRAIN_GPUS:-3}
IFS=, read -r -a TRAIN_GPU_IDS <<< "$TRAIN_GPUS"
NPROC=${COEVO_BUYER_TRAIN_NPROC:-${#TRAIN_GPU_IDS[@]}}
if [[ $NPROC -ne ${#TRAIN_GPU_IDS[@]} ]]; then
  echo "COEVO_BUYER_TRAIN_NPROC=$NPROC does not match GPU count ${#TRAIN_GPU_IDS[@]}" >&2
  exit 2
fi

TUNER_TYPE=${COEVO_BUYER_TUNER_TYPE:-full}
if [[ $TUNER_TYPE == lora ]]; then
  LEARNING_RATE=${COEVO_BUYER_LEARNING_RATE:-1e-5}
else
  LEARNING_RATE=${COEVO_BUYER_LEARNING_RATE:-1e-6}
fi
TRAINER_ARGS=(
  --rlhf_type grpo \
  --model "${COEVO_BUYER_BASE_MODEL:-$COEVO_BUYER_PATH}" \
  --model_type "${COEVO_MODEL_TYPE:-qwen3}" \
  --template "${COEVO_BUYER_TEMPLATE_TYPE:-qwen3}" \
  --external_plugins "$COEVO_ROOT/coevo/training/swift_plugin.py" \
  --dataset "$DATA" \
  --remove_unused_columns false \
  --reward_funcs tau2_buyer_utility \
  --tuner_type "$TUNER_TYPE" \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "${COEVO_BUYER_ROLLOUT_PORT:-8003}" \
  --vllm_server_pass_dataset true \
  --num_generations "${COEVO_BUYER_NUM_GENERATIONS:-4}" \
  --max_completion_length "${COEVO_BUYER_MAX_COMPLETION_LENGTH:-512}" \
  --temperature 0.8 \
  --enable_thinking "${COEVO_BUYER_ENABLE_THINKING:-true}" \
  --beta "${COEVO_BUYER_BETA:-0.01}" \
  --scale_rewards none \
  --torch_dtype bfloat16 \
  --max_steps "$STEPS" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${COEVO_BUYER_GRADIENT_ACCUMULATION_STEPS:-4}" \
  --learning_rate "$LEARNING_RATE" \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_steps "${COEVO_BUYER_SAVE_STEPS:-$STEPS}" \
  --save_total_limit "${COEVO_BUYER_SAVE_TOTAL_LIMIT:-5}" \
  --save_only_model true \
  --report_to none \
  --output_dir "$OUT"
)

if ((NPROC > 1)); then
  TRAINER_ARGS+=(--deepspeed "${COEVO_BUYER_DEEPSPEED:-zero3}")
fi
if [[ $TUNER_TYPE == lora ]]; then
  TRAINER_ARGS+=(
    --target_modules "${COEVO_BUYER_LORA_TARGET_MODULES:-all-linear}"
    --lora_rank "${COEVO_BUYER_LORA_RANK:-16}"
    --lora_alpha "${COEVO_BUYER_LORA_ALPHA:-32}"
    --vllm_enable_lora true
    --move_model_batches "${COEVO_BUYER_MOVE_MODEL_BATCHES:-64}"
  )
fi
if [[ -n ${COEVO_BUYER_RESUME_FROM_CHECKPOINT:-} ]]; then
  TRAINER_ARGS+=(
    --resume_from_checkpoint "$COEVO_BUYER_RESUME_FROM_CHECKPOINT"
    --resume_only_model "${COEVO_BUYER_RESUME_ONLY_MODEL:-true}"
  )
fi

if ((NPROC > 1)); then
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NPROC" \
    -m swift.cli.rlhf "${TRAINER_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -m swift.cli.rlhf "${TRAINER_ARGS[@]}"
fi
