#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

ROLE=${1:?usage: start_role.sh ROLE MODEL_PATH}
MODEL_PATH=${2:?usage: start_role.sh ROLE MODEL_PATH}
ROLE_KEY=$ROLE
if [[ -n ${COEVO_ROLE_INSTANCE:-} ]]; then
  ROLE_KEY="$ROLE-${COEVO_ROLE_INSTANCE}"
fi
RUNTIME=$COEVO_ROOT/runtime
mkdir -p "$RUNTIME/logs" "$RUNTIME/pids"

case "$ROLE" in
  policy)
    GPU=${COEVO_POLICY_GPUS:-0}
    PORT=${COEVO_POLICY_PORT:-8000}
    IFS=, read -r -a POLICY_GPU_IDS <<< "$GPU"
    TP_SIZE=${COEVO_POLICY_TP_SIZE:-${#POLICY_GPU_IDS[@]}}
    SERVED_MODEL=${COEVO_POLICY_MODEL:-Qwen3-4B}
    CUDA_VISIBLE_DEVICES="$GPU" nohup setsid python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
      --tensor-parallel-size "$TP_SIZE" --port "$PORT" \
      --max-model-len "${COEVO_POLICY_MAX_MODEL_LEN:-16384}" \
      --gpu-memory-utilization "${COEVO_POLICY_GPU_MEMORY_UTILIZATION:-0.88}" \
      --max-num-seqs "${COEVO_POLICY_MAX_NUM_SEQS:-8}" \
      --max-logprobs "${COEVO_TEACHER_GAP_TOPK:-20}" \
      --disable-custom-all-reduce \
      --enable-auto-tool-choice --tool-call-parser hermes \
      > "$RUNTIME/logs/$ROLE_KEY.log" 2>&1 &
    ;;
  policy_previous)
    GPU=${COEVO_PREVIOUS_POLICY_GPUS:-4}
    PORT=${COEVO_PREVIOUS_POLICY_PORT:-8001}
    IFS=, read -r -a PREVIOUS_POLICY_GPU_IDS <<< "$GPU"
    TP_SIZE=${COEVO_PREVIOUS_POLICY_TP_SIZE:-${#PREVIOUS_POLICY_GPU_IDS[@]}}
    SERVED_MODEL=${COEVO_PREVIOUS_POLICY_MODEL:-${COEVO_POLICY_MODEL:-Qwen3-4B}}
    CUDA_VISIBLE_DEVICES="$GPU" nohup setsid python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
      --tensor-parallel-size "$TP_SIZE" --port "$PORT" \
      --max-model-len "${COEVO_PREVIOUS_POLICY_MAX_MODEL_LEN:-16384}" \
      --gpu-memory-utilization \
        "${COEVO_PREVIOUS_POLICY_GPU_MEMORY_UTILIZATION:-0.88}" \
      --max-num-seqs "${COEVO_PREVIOUS_POLICY_MAX_NUM_SEQS:-8}" \
      --max-logprobs "${COEVO_TEACHER_GAP_TOPK:-20}" \
      --disable-custom-all-reduce \
      --enable-auto-tool-choice --tool-call-parser hermes \
      > "$RUNTIME/logs/$ROLE_KEY.log" 2>&1 &
    ;;
  teacher|student)
    echo "role '$ROLE' was removed; Student and Teacher both use role 'policy'" >&2
    exit 2
    ;;
  buyer)
    GPU=${COEVO_BUYER_GPUS:-${COEVO_BUYER_GPU:-1}}
    IFS=, read -r -a BUYER_GPU_IDS <<< "$GPU"
    TP_SIZE=${COEVO_BUYER_TP_SIZE:-${#BUYER_GPU_IDS[@]}}
    PORT=${COEVO_BUYER_PORT:-8002}
    SERVED_MODEL=${COEVO_BUYER_MODEL:-Qwen3-4B}
    CUDA_VISIBLE_DEVICES="$GPU" nohup setsid python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
      --tensor-parallel-size "$TP_SIZE" --port "$PORT" \
      --max-model-len "${COEVO_BUYER_MAX_MODEL_LEN:-16384}" \
      --gpu-memory-utilization "${COEVO_BUYER_GPU_MEMORY_UTILIZATION:-0.88}" \
      --max-num-seqs "${COEVO_BUYER_MAX_NUM_SEQS:-8}" \
      --disable-custom-all-reduce \
      --enable-auto-tool-choice --tool-call-parser hermes \
      > "$RUNTIME/logs/$ROLE_KEY.log" 2>&1 &
    ;;
  rollout)
    GPU=${COEVO_BUYER_ROLLOUT_GPUS:-${COEVO_BUYER_ROLLOUT_GPU:-2}}
    IFS=, read -r -a ROLLOUT_GPU_IDS <<< "$GPU"
    TP_SIZE=${COEVO_BUYER_ROLLOUT_TP_SIZE:-${#ROLLOUT_GPU_IDS[@]}}
    PORT=${COEVO_BUYER_ROLLOUT_PORT:-8003}
    SERVED_MODEL=${COEVO_BUYER_MODEL:-Qwen3-4B}
    ROLLOUT_LORA_ARGS=()
    if [[ ${COEVO_BUYER_ROLLOUT_ENABLE_LORA:-false} == true ]]; then
      ROLLOUT_LORA_ARGS+=(
        --vllm_enable_lora true
        --vllm_max_lora_rank "${COEVO_BUYER_LORA_RANK:-16}"
      )
    fi
    # Swift silently advances to the next port when the requested socket is
    # still busy (including a just-stopped server).  The controller would then
    # wait on the wrong port, so require the configured port to be bindable
    # before launch.
    python "$COEVO_ROOT/scripts/wait_for_free_port.py" "$PORT" \
      --timeout "${COEVO_PORT_RELEASE_TIMEOUT:-60}"
    CUDA_VISIBLE_DEVICES="$GPU" nohup setsid python -m swift.cli.rollout \
      --model "$MODEL_PATH" \
      --model_type "${COEVO_BUYER_MODEL_TYPE:-${COEVO_MODEL_TYPE:-qwen3}}" \
      --template "${COEVO_BUYER_TEMPLATE_TYPE:-qwen3}" \
      --enable_thinking "${COEVO_BUYER_ENABLE_THINKING:-true}" \
      --served_model_name "$SERVED_MODEL" --port "$PORT" \
      --external_plugins "$COEVO_ROOT/coevo/training/swift_plugin.py" \
      --multi_turn_scheduler tau2_buyer \
      --max_turns "${COEVO_BUYER_MAX_TURNS:-2}" \
      --vllm_tensor_parallel_size "$TP_SIZE" \
      --vllm_max_model_len "${COEVO_BUYER_ROLLOUT_MAX_MODEL_LEN:-4096}" \
      --vllm_gpu_memory_utilization \
        "${COEVO_BUYER_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.88}" \
      --vllm_max_num_seqs "${COEVO_BUYER_ROLLOUT_MAX_NUM_SEQS:-8}" \
      --vllm_enable_prefix_caching true \
      "${ROLLOUT_LORA_ARGS[@]}" \
      > "$RUNTIME/logs/$ROLE_KEY.log" 2>&1 &
    ;;
  *)
    echo "unknown role: $ROLE" >&2
    exit 2
    ;;
esac
PID=$!
START_TICKS=$(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || true)
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
if [[ -z "$START_TICKS" || "$PGID" != "$PID" ]]; then
  echo "failed to isolate role $ROLE in its own process group" >&2
  kill "$PID" 2>/dev/null || true
  exit 1
fi
printf '%s %s %s\n' "$PID" "$START_TICKS" "$PGID" > "$RUNTIME/pids/$ROLE_KEY.pid"
WAIT_ARGS=(
  "http://127.0.0.1:$PORT" \
  --timeout "${COEVO_SERVER_START_TIMEOUT:-900}" \
  --pid "$PID" \
  --log-file "$RUNTIME/logs/$ROLE_KEY.log"
)
if [[ "$ROLE" == rollout ]]; then
  WAIT_ARGS+=(--health-path /health/)
else
  WAIT_ARGS+=(--model "$SERVED_MODEL")
fi
if ! python "$COEVO_ROOT/scripts/wait_for_servers.py" "${WAIT_ARGS[@]}"; then
  kill -- "-$PGID" 2>/dev/null || true
  rm -f "$RUNTIME/pids/$ROLE_KEY.pid"
  exit 1
fi
