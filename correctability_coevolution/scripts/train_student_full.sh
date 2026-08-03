#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/OPD/correctability_coevolution
DATA=${1:-$ROOT/artifacts/round/student_opd.jsonl}
OUT=${2:-$ROOT/artifacts/round/student_full}
STEPS=${3:-2}
export HF_HOME="$ROOT/runtime/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$ROOT/runtime/modelscope_cache"
mkdir -p "$HF_HOME"

PYTHONPATH="$ROOT:/workspace/OPD/tau2-bench/src" \
CUDA_VISIBLE_DEVICES="${COEVO_STUDENT_TRAIN_GPUS:-5}" \
python -m swift.cli.rlhf \
  --rlhf_type gkd \
  --model "${COEVO_STUDENT_BASE_MODEL:-/models/Qwen3-4B}" \
  --teacher_model_server "${COEVO_TEACHER_URL:-http://127.0.0.1:8000}" \
  --gkd_logits_topk 20 \
  --use_logits_to_keep false \
  --external_plugins "$ROOT/coevo/training/swift_plugin.py" \
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
