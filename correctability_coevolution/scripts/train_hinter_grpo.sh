#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

DATA=${1:?usage: train_hinter_grpo.sh DATA OUTPUT_DIR [STEPS]}
OUT=${2:?usage: train_hinter_grpo.sh DATA OUTPUT_DIR [STEPS]}
STEPS=${3:-20}
coevo_require_nonempty_file "$DATA"
: "${COEVO_HINTER_BASE_MODEL:?COEVO_HINTER_BASE_MODEL is required}"
: "${COEVO_POLICY_PATH:?COEVO_POLICY_PATH is required}"
export COEVO_TEACHER_HINT_MODE=none

# The current Student endpoint is frozen throughout this GRPO call. Its
# four teacher-forced views supply lift/copy/dose; pass@k stays out of reward.
python "$COEVO_ROOT/scripts/wait_for_servers.py" \
  "${COEVO_POLICY_URL:-http://127.0.0.1:${COEVO_POLICY_PORT:-8000}}" \
  --timeout 30 \
  --model "${COEVO_POLICY_MODEL:-Qwen3-4B}"
TRAIN_GPUS=${COEVO_HINTER_TRAIN_GPUS:-2,3,4,5,6,7}
IFS=, read -r -a TRAIN_GPU_IDS <<< "$TRAIN_GPUS"
NPROC=${COEVO_HINTER_TRAIN_NPROC:-${#TRAIN_GPU_IDS[@]}}
HINTER_TUNER_TYPE=${COEVO_HINTER_TUNER_TYPE:-full}
if [[ $NPROC -ne ${#TRAIN_GPU_IDS[@]} ]]; then
  echo "COEVO_HINTER_TRAIN_NPROC=$NPROC does not match GPU count ${#TRAIN_GPU_IDS[@]}" >&2
  exit 2
fi

ARGS=(
  --rlhf_type grpo
  --model "$COEVO_HINTER_BASE_MODEL"
  --model_type "${COEVO_HINTER_MODEL_TYPE:-qwen3}"
  --template "${COEVO_HINTER_TEMPLATE:-qwen3_nothinking}"
  --external_plugins "$COEVO_ROOT/coevo/training/swift_hinter_plugin.py"
  --dataset "$DATA"
  --remove_unused_columns false
  --reward_funcs hinter_composite
  --tuner_type "$HINTER_TUNER_TYPE"
  --num_generations "${COEVO_HINTER_NUM_GENERATIONS:-4}"
  --max_completion_length "${COEVO_HINTER_MAX_HINT_TOKENS:-192}"
  --temperature "${COEVO_HINTER_TEMPERATURE:-0.8}"
  --beta "${COEVO_HINTER_ANCHOR_BETA:-0.01}"
  --enable_thinking false
  --max_steps "$STEPS"
  --per_device_train_batch_size 1
  --gradient_accumulation_steps "${COEVO_HINTER_GRADIENT_ACCUMULATION_STEPS:-4}"
  --learning_rate "${COEVO_HINTER_LEARNING_RATE:-1e-5}"
  --gradient_checkpointing true
  --logging_steps 1
  --save_steps "${COEVO_HINTER_SAVE_STEPS:-$STEPS}"
  --report_to "$COEVO_REPORT_TO"
  --run_name "${COEVO_WANDB_RUN_NAME:-hinter-composite-grpo}"
  --output_dir "$OUT"
)
if [[ $HINTER_TUNER_TYPE == lora ]]; then
  ARGS+=(
    --target_modules "${COEVO_HINTER_LORA_TARGET_MODULES:-all-linear}"
    --lora_rank "${COEVO_HINTER_LORA_RANK:-16}"
    --lora_alpha "${COEVO_HINTER_LORA_ALPHA:-32}"
  )
fi
if ((NPROC > 1)); then
  ARGS+=(--deepspeed "${COEVO_HINTER_DEEPSPEED:-zero3}")
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun --standalone --nproc_per_node "$NPROC" \
    -m swift.cli.rlhf "${ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -m swift.cli.rlhf "${ARGS[@]}"
fi
