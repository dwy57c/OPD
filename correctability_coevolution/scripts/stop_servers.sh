#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/OPD/correctability_coevolution
for name in teacher student buyer rollout; do
  "$ROOT/scripts/stop_role.sh" "$name"
done
