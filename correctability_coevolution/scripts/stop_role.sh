#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

ROLE=${1:?usage: stop_role.sh ROLE}
ROLE_KEY=$ROLE
if [[ -n ${COEVO_ROLE_INSTANCE:-} ]]; then
  ROLE_KEY="$ROLE-${COEVO_ROLE_INSTANCE}"
fi
PID_FILE=$COEVO_ROOT/runtime/pids/$ROLE_KEY.pid
[[ -f "$PID_FILE" ]] || exit 0
read -r PID EXPECTED_START_TICKS EXPECTED_PGID < "$PID_FILE"
if kill -0 "$PID" 2>/dev/null; then
  CURRENT_START_TICKS=$(awk '{print $22}' "/proc/$PID/stat" 2>/dev/null || true)
  if [[ -n ${EXPECTED_START_TICKS:-} && "$CURRENT_START_TICKS" != "$EXPECTED_START_TICKS" ]]; then
    echo "refusing to stop reused PID $PID for role $ROLE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
  TARGET=$PID
  if [[ -n ${EXPECTED_PGID:-} ]]; then
    CURRENT_PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
    if [[ "$CURRENT_PGID" != "$EXPECTED_PGID" || "$EXPECTED_PGID" != "$PID" ]]; then
      echo "refusing to stop unexpected process group for role $ROLE" >&2
      rm -f "$PID_FILE"
      exit 1
    fi
    TARGET=-$EXPECTED_PGID
  fi
  kill -- "$TARGET"
  for _ in $(seq 1 30); do
    if [[ -n ${EXPECTED_PGID:-} ]]; then
      ALIVE=$(ps -eo pgid=,stat= | awk -v pgid="$EXPECTED_PGID" \
        '$1 == pgid && $2 !~ /^Z/ { found = 1 } END { print found + 0 }')
      [[ "$ALIVE" == 0 ]] && break
    else
      STATE=$(ps -o stat= -p "$PID" 2>/dev/null || true)
      [[ -z "$STATE" || "$STATE" == Z* ]] && break
    fi
    sleep 1
  done
  if [[ -n ${EXPECTED_PGID:-} ]]; then
    ALIVE=$(ps -eo pgid=,stat= | awk -v pgid="$EXPECTED_PGID" \
      '$1 == pgid && $2 !~ /^Z/ { found = 1 } END { print found + 0 }')
  else
    STATE=$(ps -o stat= -p "$PID" 2>/dev/null || true)
    ALIVE=$([[ -n "$STATE" && "$STATE" != Z* ]] && echo 1 || echo 0)
  fi
  if [[ "$ALIVE" == 1 ]]; then
    echo "role $ROLE did not stop after 30 seconds; sending SIGKILL" >&2
    kill -KILL -- "$TARGET"
    for _ in $(seq 1 5); do
      if [[ -n ${EXPECTED_PGID:-} ]]; then
        ALIVE=$(ps -eo pgid=,stat= | awk -v pgid="$EXPECTED_PGID" \
          '$1 == pgid && $2 !~ /^Z/ { found = 1 } END { print found + 0 }')
      else
        STATE=$(ps -o stat= -p "$PID" 2>/dev/null || true)
        ALIVE=$([[ -n "$STATE" && "$STATE" != Z* ]] && echo 1 || echo 0)
      fi
      [[ "$ALIVE" == 0 ]] && break
      sleep 1
    done
    if [[ "$ALIVE" == 1 ]]; then
      echo "role $ROLE is still alive; retaining $PID_FILE" >&2
      exit 1
    fi
  fi
fi
rm -f "$PID_FILE"
