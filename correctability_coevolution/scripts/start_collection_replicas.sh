#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

REPLICAS=${COEVO_COLLECTION_REPLICAS:-4}
POLICY_PORT_BASE=${COEVO_COLLECTION_POLICY_PORT_BASE:-8100}

if ((REPLICAS < 1 || REPLICAS > 4)); then
  echo "COEVO_COLLECTION_REPLICAS must be between 1 and 4" >&2
  exit 2
fi

cleanup() {
  "$COEVO_ROOT/scripts/stop_collection_replicas.sh" || true
}
trap cleanup ERR INT TERM

policy_starts=()
for ((index = 0; index < REPLICAS; index++)); do
  env \
    COEVO_ROLE_INSTANCE="collect-$index" \
    COEVO_POLICY_GPUS="$index" \
    COEVO_POLICY_PORT="$((POLICY_PORT_BASE + index))" \
    COEVO_POLICY_MAX_MODEL_LEN="${COEVO_POLICY_MAX_MODEL_LEN:-40960}" \
    COEVO_POLICY_MAX_NUM_SEQS="${COEVO_POLICY_MAX_NUM_SEQS:-1}" \
    COEVO_POLICY_GPU_MEMORY_UTILIZATION=0.92 \
    "$COEVO_ROOT/scripts/start_role.sh" policy "$COEVO_POLICY_PATH" &
  policy_starts+=("$!")
done
for pid in "${policy_starts[@]}"; do
  wait "$pid"
done

trap - ERR INT TERM
