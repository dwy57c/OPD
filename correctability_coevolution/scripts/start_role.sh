#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/OPD/correctability_coevolution
ROLE=$1
MODEL_PATH=$2
RUNTIME=$ROOT/runtime
mkdir -p "$RUNTIME/logs" "$RUNTIME/pids"

case "$ROLE" in
  teacher)
    GPU=${COEVO_TEACHER_GPUS:-0,1}
    PORT=8000
    CUDA_VISIBLE_DEVICES="$GPU" nohup python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name Qwen3-32B \
      --tensor-parallel-size 2 --port "$PORT" --max-model-len 16384 \
      --gpu-memory-utilization 0.88 --max-num-seqs 8 --max-logprobs 20 \
      --enable-auto-tool-choice --tool-call-parser hermes \
      > "$RUNTIME/logs/$ROLE.log" 2>&1 &
    ;;
  student)
    GPU=${COEVO_STUDENT_GPU:-2}
    PORT=8001
    CUDA_VISIBLE_DEVICES="$GPU" nohup python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name Qwen3-4B \
      --port "$PORT" --max-model-len 16384 --gpu-memory-utilization 0.80 \
      --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes \
      > "$RUNTIME/logs/$ROLE.log" 2>&1 &
    ;;
  buyer)
    GPU=${COEVO_BUYER_GPU:-3}
    PORT=8002
    CUDA_VISIBLE_DEVICES="$GPU" nohup python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name Qwen3-4B \
      --port "$PORT" --max-model-len 16384 --gpu-memory-utilization 0.80 \
      --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes \
      > "$RUNTIME/logs/$ROLE.log" 2>&1 &
    ;;
  rollout)
    GPU=${COEVO_BUYER_ROLLOUT_GPU:-4}
    PORT=8003
    CUDA_VISIBLE_DEVICES="$GPU" nohup python -m swift.cli.rollout \
      --model "$MODEL_PATH" --served_model_name Qwen3-4B --port "$PORT" \
      --vllm_max_model_len 4096 --vllm_gpu_memory_utilization 0.80 \
      --vllm_max_num_seqs 8 --vllm_enable_prefix_caching true \
      > "$RUNTIME/logs/$ROLE.log" 2>&1 &
    ;;
  *)
    echo "unknown role: $ROLE" >&2
    exit 2
    ;;
esac
echo $! > "$RUNTIME/pids/$ROLE.pid"
python "$ROOT/scripts/wait_for_servers.py" "http://127.0.0.1:$PORT"
