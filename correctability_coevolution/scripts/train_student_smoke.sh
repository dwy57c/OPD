#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/OPD/correctability_coevolution
DATA=${1:-$ROOT/artifacts/smoke/student_gkd.jsonl}
OUT=${2:-$ROOT/artifacts/smoke/student_adapter}
export HF_HOME="$ROOT/runtime/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$ROOT/runtime/modelscope_cache"
mkdir -p "$HF_HOME"

PYTHONPATH="$ROOT:/workspace/OPD/tau2-bench/src" \
CUDA_VISIBLE_DEVICES=4 \
python -m swift.cli.rlhf \
  --rlhf_type gkd \
  --model /models/Qwen3-4B \
  --teacher_model_server http://127.0.0.1:8000 \
  --gkd_logits_topk 20 \
  --use_logits_to_keep false \
  --external_plugins "$ROOT/coevo/training/swift_plugin.py" \
  --dataset "$DATA" \
  --remove_unused_columns false \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules all-linear \
  --lmbda 1 \
  --beta 0.5 \
  --temperature 1.0 \
  --sft_alpha 0 \
  --torch_dtype bfloat16 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --max_length 8192 \
  --max_completion_length 128 \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_steps 1 \
  --save_only_model true \
  --report_to none \
  --output_dir "$OUT"
