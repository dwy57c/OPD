#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

started=()
cleanup_partial_start() {
  local index
  for ((index=${#started[@]} - 1; index >= 0; index--)); do
    "$COEVO_ROOT/scripts/stop_role.sh" "${started[$index]}" || true
  done
}
trap cleanup_partial_start ERR INT TERM

python "$COEVO_ROOT/scripts/preflight.py" start
"$COEVO_ROOT/scripts/start_role.sh" policy "$COEVO_POLICY_PATH"
started+=(policy)
"$COEVO_ROOT/scripts/start_role.sh" buyer "$COEVO_BUYER_PATH"
started+=(buyer)
"$COEVO_ROOT/scripts/start_role.sh" rollout "$COEVO_BUYER_PATH"
started+=(rollout)

trap - ERR INT TERM
