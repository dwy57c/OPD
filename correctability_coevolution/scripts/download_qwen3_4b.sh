#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/versions.env"

DESTINATION=${1:?usage: download_qwen3_4b.sh /absolute/path/to/Qwen3-4B}
if [[ "$DESTINATION" != /* ]]; then
  echo "model destination must be an absolute path: $DESTINATION" >&2
  exit 2
fi
mkdir -p "$DESTINATION"
DESTINATION=$(cd -- "$DESTINATION" && pwd)

RUNTIME_IMAGE=${COEVO_RUNTIME_IMAGE:-coevo-swift:4.1.3}
DOCKER_ENV=(-e MODELSCOPE_CACHE=/download/.cache/modelscope)
for variable_name in HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY MODELSCOPE_API_TOKEN; do
  if [[ -n ${!variable_name:-} ]]; then
    DOCKER_ENV+=(-e "$variable_name")
  fi
done

docker run --rm --network host \
  "${DOCKER_ENV[@]}" \
  --mount "type=bind,src=$DESTINATION,dst=/download" \
  --entrypoint python \
  "$RUNTIME_IMAGE" \
  -c '
from pathlib import Path
import hashlib
import json
import sys

from modelscope import snapshot_download

modelscope_revision = sys.argv[1]
huggingface_revision = sys.argv[2]
expected_hashes = {
    "model-00001-of-00003.safetensors": sys.argv[3],
    "model-00002-of-00003.safetensors": sys.argv[4],
    "model-00003-of-00003.safetensors": sys.argv[5],
    "config.json": sys.argv[6],
    "generation_config.json": sys.argv[7],
    "model.safetensors.index.json": sys.argv[8],
    "tokenizer.json": sys.argv[9],
    "tokenizer_config.json": sys.argv[10],
    "merges.txt": sys.argv[11],
    "vocab.json": sys.argv[12],
}
destination = Path("/download")
snapshot_download(
    "Qwen/Qwen3-4B",
    revision=modelscope_revision,
    local_dir=destination,
    max_workers=8,
)
config = json.loads((destination / "config.json").read_text())
if config.get("model_type") != "qwen3":
    raise SystemExit(f"unexpected model_type: {config.get('model_type')}")
for filename, expected_hash in expected_hashes.items():
    path = destination / filename
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise SystemExit(
            f"checksum mismatch for {filename}: {actual_hash} != {expected_hash}"
        )
(destination / ".coevo-revision").write_text(huggingface_revision + "\n")
print(
    f"Qwen3-4B ready at {destination} "
    f"({len(expected_hashes)} verified files, "
    f"Hugging Face revision {huggingface_revision})"
)
' \
  "$COEVO_QWEN3_4B_MODELSCOPE_REVISION" \
  "$COEVO_QWEN3_4B_REVISION" \
  "$COEVO_QWEN3_4B_SHARD_1_SHA256" \
  "$COEVO_QWEN3_4B_SHARD_2_SHA256" \
  "$COEVO_QWEN3_4B_SHARD_3_SHA256" \
  "$COEVO_QWEN3_4B_CONFIG_SHA256" \
  "$COEVO_QWEN3_4B_GENERATION_CONFIG_SHA256" \
  "$COEVO_QWEN3_4B_INDEX_SHA256" \
  "$COEVO_QWEN3_4B_TOKENIZER_SHA256" \
  "$COEVO_QWEN3_4B_TOKENIZER_CONFIG_SHA256" \
  "$COEVO_QWEN3_4B_MERGES_SHA256" \
  "$COEVO_QWEN3_4B_VOCAB_SHA256"
