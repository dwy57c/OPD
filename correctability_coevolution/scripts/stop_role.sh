#!/usr/bin/env bash
set -euo pipefail

ROLE=$1
PID_FILE=/workspace/OPD/correctability_coevolution/runtime/pids/$ROLE.pid
[[ -f "$PID_FILE" ]] || exit 0
PID=$(<"$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 30); do
    STATE=$(ps -o stat= -p "$PID" 2>/dev/null || true)
    [[ -z "$STATE" || "$STATE" == Z* ]] && break
    sleep 1
  done
fi
rm -f "$PID_FILE"
