#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

REPLICAS=${COEVO_COLLECTION_REPLICAS:-4}
status=0
for ((index = 0; index < REPLICAS; index++)); do
  "$COEVO_ROOT/scripts/stop_role.sh" "policy-collect-$index" || status=1
done
exit "$status"
