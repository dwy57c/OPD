#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

REPLICAS=${COEVO_COLLECTION_REPLICAS:-4}
TEACHER_PORT_BASE=${COEVO_COLLECTION_TEACHER_PORT_BASE:-8100}
STUDENT_PORT_BASE=${COEVO_COLLECTION_STUDENT_PORT_BASE:-8200}

if ((REPLICAS < 1 || REPLICAS > 4)); then
  echo "COEVO_COLLECTION_REPLICAS must be between 1 and 4" >&2
  exit 2
fi

cleanup() {
  "$COEVO_ROOT/scripts/stop_collection_replicas.sh" || true
}
trap cleanup ERR INT TERM

teacher_starts=()
for ((index = 0; index < REPLICAS; index++)); do
  env \
    COEVO_ROLE_INSTANCE="collect-$index" \
    COEVO_TEACHER_GPUS="$index" \
    COEVO_TEACHER_PORT="$((TEACHER_PORT_BASE + index))" \
    COEVO_TEACHER_MAX_MODEL_LEN="${COEVO_TEACHER_MAX_MODEL_LEN:-40960}" \
    COEVO_TEACHER_MAX_NUM_SEQS="${COEVO_TEACHER_MAX_NUM_SEQS:-1}" \
    COEVO_TEACHER_GPU_MEMORY_UTILIZATION=0.92 \
    "$COEVO_ROOT/scripts/start_role.sh" teacher "$COEVO_TEACHER_PATH" &
  teacher_starts+=("$!")
done
for pid in "${teacher_starts[@]}"; do
  wait "$pid"
done

student_starts=()
for ((index = 0; index < REPLICAS; index++)); do
  env \
    COEVO_ROLE_INSTANCE="collect-$index" \
    COEVO_STUDENT_GPU="$((index + 4))" \
    COEVO_STUDENT_PORT="$((STUDENT_PORT_BASE + index))" \
    COEVO_STUDENT_MAX_MODEL_LEN="${COEVO_STUDENT_MAX_MODEL_LEN:-40960}" \
    COEVO_STUDENT_MAX_NUM_SEQS="${COEVO_STUDENT_MAX_NUM_SEQS:-1}" \
    "$COEVO_ROOT/scripts/start_role.sh" student "$COEVO_STUDENT_PATH" &
  student_starts+=("$!")
done
for pid in "${student_starts[@]}"; do
  wait "$pid"
done

trap - ERR INT TERM
