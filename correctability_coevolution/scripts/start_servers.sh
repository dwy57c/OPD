#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/OPD/correctability_coevolution
"$ROOT/scripts/start_role.sh" teacher "${COEVO_TEACHER_PATH:-/models/Qwen3-32B}"
"$ROOT/scripts/start_role.sh" student "${COEVO_STUDENT_PATH:-/models/Qwen3-4B}"
"$ROOT/scripts/start_role.sh" buyer "${COEVO_BUYER_PATH:-/models/Qwen3-4B}"
"$ROOT/scripts/start_role.sh" rollout "${COEVO_BUYER_PATH:-/models/Qwen3-4B}"
