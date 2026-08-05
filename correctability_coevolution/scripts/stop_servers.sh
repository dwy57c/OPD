#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

status=0
for name in rollout buyer policy; do
  "$COEVO_ROOT/scripts/stop_role.sh" "$name" || status=1
done
exit "$status"
