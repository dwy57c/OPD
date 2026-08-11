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
TAU2_DATA_DIR=${TAU2_DATA_DIR:-$(dirname -- "$COEVO_TAU2_SRC")/data}

COEVO_MODEL_ROOT=${COEVO_MODEL_ROOT:-/models}
COEVO_POLICY_PATH=${COEVO_POLICY_PATH:-$COEVO_MODEL_ROOT/policy}
COEVO_BUYER_PATH=${COEVO_BUYER_PATH:-$COEVO_POLICY_PATH}

# W&B is the only experiment reporter used by this project.  ms-swift imports
# SwanLab opportunistically when the package is installed, so disable it
# explicitly without changing the selected Trainer reporter.
COEVO_REPORT_TO=${COEVO_REPORT_TO:-wandb}
COEVO_WANDB_PROJECT=${COEVO_WANDB_PROJECT:-opd-stage-curriculum}
export SWANLAB_MODE=disabled
export WANDB_PROJECT=${WANDB_PROJECT:-$COEVO_WANDB_PROJECT}
# ms-swift's Teacher API client uses requests internally.  Ensure managed
# loopback services never inherit an unrelated host proxy route.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

# Reproducible runs are local-only by default.  A missing model or tokenizer must
# fail explicitly instead of triggering an implicit Hub/ModelScope download.
if [[ ${COEVO_ALLOW_DOWNLOADS:-0} != 1 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export MODELSCOPE_OFFLINE=1
  export LITELLM_LOCAL_MODEL_COST_MAP=True
fi

export COEVO_ROOT COEVO_REPO_ROOT COEVO_TAU2_SRC
export TAU2_DATA_DIR
export COEVO_MODEL_ROOT COEVO_POLICY_PATH COEVO_BUYER_PATH
export COEVO_REPORT_TO COEVO_WANDB_PROJECT
export PYTHONPATH="$COEVO_ROOT:$COEVO_TAU2_SRC${PYTHONPATH:+:$PYTHONPATH}"

coevo_require_nonempty_file() {
  local path=$1
  if [[ ! -s "$path" ]]; then
    echo "required non-empty file is missing: $path" >&2
    return 1
  fi
}
