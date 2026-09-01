#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

DATA=${1:-$COEVO_ROOT/artifacts/round/student_gkd.jsonl}
OUT=${2:-$COEVO_ROOT/artifacts/round/student_full}
STEPS=${3:-2}
export HF_HOME="$COEVO_ROOT/runtime/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$COEVO_ROOT/runtime/modelscope_cache"
mkdir -p "$HF_HOME"
coevo_require_nonempty_file "$DATA"
if [[ -n ${COEVO_ACTIVE_TOKEN_BUDGET:-} ]]; then
  BUDGET_DATA="$OUT/equal_budget_student_gkd.jsonl"
  python "$COEVO_ROOT/scripts/prepare_active_token_budget.py" \
    "$DATA" "$BUDGET_DATA" --budget "$COEVO_ACTIVE_TOKEN_BUDGET"
  DATA=$BUDGET_DATA
  coevo_require_nonempty_file "$DATA"
fi
python "$COEVO_ROOT/scripts/wait_for_servers.py" \
  "${COEVO_POLICY_URL:-http://127.0.0.1:${COEVO_POLICY_PORT:-8000}}" \
  --timeout 30 \
  --model "${COEVO_POLICY_MODEL:-Qwen3-4B}"

TRAIN_GPUS=${COEVO_POLICY_TRAIN_GPUS:-1,2,3,4,5,6,7}
IFS=, read -r -a TRAIN_GPU_IDS <<< "$TRAIN_GPUS"
NPROC=${COEVO_POLICY_TRAIN_NPROC:-${#TRAIN_GPU_IDS[@]}}
if [[ $NPROC -ne ${#TRAIN_GPU_IDS[@]} ]]; then
  echo "COEVO_POLICY_TRAIN_NPROC=$NPROC does not match GPU count ${#TRAIN_GPU_IDS[@]}" >&2
  exit 2
fi
if [[ -n ${COEVO_POLICY_GRADIENT_ACCUMULATION_STEPS:-} ]]; then
  GRADIENT_ACCUMULATION_STEPS=$COEVO_POLICY_GRADIENT_ACCUMULATION_STEPS
else
  # Keep MAD-OPD's effective batch of 128 as closely as an arbitrary world
  # size permits. Seven train ranks plus one frozen-Teacher GPU gives 126.
  GRADIENT_ACCUMULATION_STEPS=$(((128 + NPROC / 2) / NPROC))
fi

TRAINER_ARGS=(
  --rlhf_type gkd \
  --model "${COEVO_POLICY_BASE_MODEL:-$COEVO_POLICY_PATH}" \
  --model_type "${COEVO_POLICY_MODEL_TYPE:-${COEVO_MODEL_TYPE:-qwen3}}" \
  --template "${COEVO_TEMPLATE_TYPE:-qwen3_nothinking}" \
  --teacher_model_server "${COEVO_POLICY_URL:-http://127.0.0.1:${COEVO_POLICY_PORT:-8000}}" \
  --gkd_logits_topk "${COEVO_TEACHER_GAP_TOPK:-20}" \
  --use_logits_to_keep true \
  --external_plugins "$COEVO_ROOT/coevo/training/swift_plugin.py" \
  --dataset "$DATA" \
  --remove_unused_columns false \
  --tuner_type full \
  --seq_kd false \
  --lmbda 1 \
  --beta 0.5 \
  --temperature 0.7 \
  --top_p 0.8 \
  --top_k 20 \
  --repetition_penalty 1.2 \
  --enable_thinking false \
  --sft_alpha 0 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --max_steps "$STEPS" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "${COEVO_POLICY_LEARNING_RATE:-1e-5}" \
  --weight_decay "${COEVO_POLICY_WEIGHT_DECAY:-0.01}" \
  --lr_scheduler_type "${COEVO_POLICY_LR_SCHEDULER_TYPE:-cosine}" \
  --warmup_ratio "${COEVO_POLICY_WARMUP_RATIO:-0.05}" \
  --max_length "${COEVO_POLICY_TRAIN_MAX_LENGTH:-16384}" \
  --max_completion_length "${COEVO_POLICY_TRAIN_MAX_COMPLETION_LENGTH:-4096}" \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${COEVO_POLICY_SAVE_STEPS:-5}" \
  --save_only_model true \
  --report_to "$COEVO_REPORT_TO" \
  --run_name "${COEVO_WANDB_RUN_NAME:-student-round-${COEVO_ROUND_INDEX:-0}}" \
  --output_dir "$OUT"
)

if ((NPROC > 1)); then
  TRAINER_ARGS+=(--deepspeed "${COEVO_POLICY_DEEPSPEED:-zero3}")
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NPROC" \
    -m swift.cli.rlhf "${TRAINER_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -m swift.cli.rlhf "${TRAINER_ARGS[@]}"
fi
