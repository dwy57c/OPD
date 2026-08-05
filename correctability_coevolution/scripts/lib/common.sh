#!/usr/bin/env bash

# Shared path/configuration discovery for every shell entrypoint.
COEVO_SCRIPT_LIB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COEVO_ROOT=${COEVO_ROOT:-$(cd -- "$COEVO_SCRIPT_LIB_DIR/../.." && pwd)}
COEVO_REPO_ROOT=$(cd -- "$COEVO_ROOT/.." && pwd)

if [[ -z ${COEVO_TAU2_SRC:-} ]]; then
  for candidate in \
    "$COEVO_REPO_ROOT/tau2-bench/src" \
    "$COEVO_REPO_ROOT/third_party/tau2-bench/src" \
    "$COEVO_ROOT/third_party/tau2-bench/src"; do
    if [[ -d "$candidate/tau2" ]]; then
      COEVO_TAU2_SRC=$candidate
      break
    fi
  done
fi
COEVO_TAU2_SRC=${COEVO_TAU2_SRC:-$COEVO_REPO_ROOT/third_party/tau2-bench/src}
TAU2_DATA_DIR=${TAU2_DATA_DIR:-$(cd -- "$COEVO_TAU2_SRC/.." && pwd)/data}

COEVO_MODEL_ROOT=${COEVO_MODEL_ROOT:-/models}
COEVO_POLICY_PATH=${COEVO_POLICY_PATH:-$COEVO_MODEL_ROOT/policy}
COEVO_BUYER_PATH=${COEVO_BUYER_PATH:-$COEVO_POLICY_PATH}

export COEVO_ROOT COEVO_REPO_ROOT COEVO_TAU2_SRC
export TAU2_DATA_DIR
export COEVO_MODEL_ROOT COEVO_POLICY_PATH COEVO_BUYER_PATH
export PYTHONPATH="$COEVO_ROOT:$COEVO_TAU2_SRC${PYTHONPATH:+:$PYTHONPATH}"

coevo_require_nonempty_file() {
  local path=$1
  if [[ ! -s "$path" ]]; then
    echo "required non-empty file is missing: $path" >&2
    return 1
  fi
}
