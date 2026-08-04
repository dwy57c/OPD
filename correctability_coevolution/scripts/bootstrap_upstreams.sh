#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/versions.env"

THIRD_PARTY=${COEVO_THIRD_PARTY_DIR:-$COEVO_REPO_ROOT/third_party}
INSTALL=false
WITH_SWIFT_SOURCE=false
for argument in "$@"; do
  case "$argument" in
    --install)
      INSTALL=true
      WITH_SWIFT_SOURCE=true
      ;;
    --with-swift-source)
      WITH_SWIFT_SOURCE=true
      ;;
    *)
      echo "usage: bootstrap_upstreams.sh [--install] [--with-swift-source]" >&2
      exit 2
      ;;
  esac
done

checkout_revision() {
  local name=$1
  local url=$2
  local revision=$3
  local destination=$4
  shift 4
  local sparse_paths=("$@")

  if [[ -e "$destination" ]]; then
    if [[ ! -d "$destination/.git" ]]; then
      echo "$destination exists but is not a git checkout" >&2
      return 1
    fi
    local current
    current=$(git -C "$destination" rev-parse HEAD)
    if [[ "$current" != "$revision" ]]; then
      echo "$name is at $current; expected $revision" >&2
      return 1
    fi
    git -C "$destination" sparse-checkout set "${sparse_paths[@]}"
    echo "$name already pinned at $revision"
    return
  fi

  git clone --filter=blob:none --no-checkout "$url" "$destination"
  git -C "$destination" sparse-checkout init --cone
  git -C "$destination" sparse-checkout set "${sparse_paths[@]}"
  git -C "$destination" fetch --depth 1 origin "$revision"
  git -C "$destination" checkout --detach "$revision"
  echo "$name checked out at $revision"
}

mkdir -p "$THIRD_PARTY"
checkout_revision \
  tau2-bench \
  https://github.com/sierra-research/tau2-bench.git \
  "$COEVO_TAU2_COMMIT" \
  "$THIRD_PARTY/tau2-bench" \
  src/tau2 \
  data/tau2/domains/airline \
  data/tau2/domains/mock \
  data/tau2/domains/retail \
  data/tau2/domains/telecom \
  data/tau2/user_simulator
if [[ "$WITH_SWIFT_SOURCE" == true ]]; then
  checkout_revision \
    ms-swift \
    https://github.com/modelscope/ms-swift.git \
    "$COEVO_SWIFT_COMMIT" \
    "$THIRD_PARTY/ms-swift" \
    requirements swift
fi

if [[ "$INSTALL" == true ]]; then
  # The runtime image is Python 3.11 because Swift/vLLM and their compiled
  # dependencies are already installed there. τ² v1 declares Python >=3.12,
  # so this is the same explicitly tested compatibility override as Dockerfile.
  python -m pip install --ignore-requires-python -e "$THIRD_PARTY/tau2-bench[voice]"
  python -m pip install -e "$THIRD_PARTY/ms-swift"
  python -m pip install -r "$COEVO_REPO_ROOT/requirements-dev.txt"
fi
