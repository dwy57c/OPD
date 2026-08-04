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
python "$COEVO_ROOT/scripts/wait_for_servers.py" \
  "${COEVO_TEACHER_URL:-http://127.0.0.1:${COEVO_TEACHER_PORT:-8000}}" \
  --timeout 30 \
  --model "${COEVO_TEACHER_MODEL:-Qwen3-32B}"

TRAIN_GPUS=${COEVO_STUDENT_TRAIN_GPUS:-5}
IFS=, read -r -a TRAIN_GPU_IDS <<< "$TRAIN_GPUS"
NPROC=${COEVO_STUDENT_TRAIN_NPROC:-${#TRAIN_GPU_IDS[@]}}
if [[ $NPROC -ne ${#TRAIN_GPU_IDS[@]} ]]; then
  echo "COEVO_STUDENT_TRAIN_NPROC=$NPROC does not match GPU count ${#TRAIN_GPU_IDS[@]}" >&2
  exit 2
fi

TRAINER_ARGS=(
  --rlhf_type gkd \
  --model "${COEVO_STUDENT_BASE_MODEL:-$COEVO_STUDENT_PATH}" \
  --model_type "${COEVO_MODEL_TYPE:-qwen3}" \
  --template "${COEVO_TEMPLATE_TYPE:-qwen3_nothinking}" \
  --teacher_model_server "${COEVO_TEACHER_URL:-http://127.0.0.1:${COEVO_TEACHER_PORT:-8000}}" \
  --gkd_logits_topk 20 \
  --use_logits_to_keep false \
  --external_plugins "$COEVO_ROOT/coevo/training/swift_plugin.py" \
  --dataset "$DATA" \
  --remove_unused_columns false \
  --tuner_type full \
  --lmbda 0 \
  --beta 0.5 \
  --temperature 1.0 \
  --sft_alpha 0 \
  --torch_dtype bfloat16 \
  --max_steps "$STEPS" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --max_length 8192 \
  --max_completion_length 256 \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_steps "$STEPS" \
  --save_only_model true \
  --report_to none \
  --output_dir "$OUT"
)

if ((NPROC > 1)); then
  TRAINER_ARGS+=(--deepspeed "${COEVO_STUDENT_DEEPSPEED:-zero3}")
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NPROC" \
    -m swift.cli.rlhf "${TRAINER_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python -m swift.cli.rlhf "${TRAINER_ARGS[@]}"
fi
