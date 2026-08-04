#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/common.sh"

DOMAIN=${COEVO_EVAL_DOMAIN:-${COEVO_DOMAIN:-airline}}
TASK_SPLIT=${COEVO_EVAL_TASK_SPLIT:-test}
STUDENT_MODEL=${COEVO_STUDENT_MODEL:-Qwen3-4B}
STUDENT_URL=${COEVO_STUDENT_URL:-http://127.0.0.1:${COEVO_STUDENT_PORT:-8001}}
USER_MODEL=${COEVO_EVAL_USER_MODEL:-gpt-4.1}
USER_ARGS=${COEVO_EVAL_USER_ARGS:-'{"temperature": 0.0}'}
NUM_TRIALS=${COEVO_EVAL_NUM_TRIALS:-1}
MAX_CONCURRENCY=${COEVO_EVAL_MAX_CONCURRENCY:-1}
MAX_STEPS=${COEVO_EVAL_MAX_STEPS:-200}

if [[ "$USER_MODEL" == gpt-* || "$USER_MODEL" == openai/* ]]; then
  : "${OPENAI_API_KEY:?OPENAI_API_KEY is required for COEVO_EVAL_USER_MODEL=$USER_MODEL}"
fi

python "$SCRIPT_DIR/wait_for_servers.py" \
  "$STUDENT_URL" \
  --timeout "${COEVO_SERVER_READY_TIMEOUT:-30}" \
  --model "$STUDENT_MODEL"

AGENT_ARGS=$(python - "$STUDENT_URL" <<'PY'
import json
import sys

print(
    json.dumps(
        {
            "api_base": sys.argv[1].rstrip("/") + "/v1",
            "api_key": "EMPTY",
            "temperature": 0.0,
            "max_tokens": 256,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            },
        }
    )
)
PY
)

COMMAND=(
  tau2 run
  --domain "$DOMAIN"
  --task-split-name "$TASK_SPLIT"
  --agent-llm "hosted_vllm/$STUDENT_MODEL"
  --agent-llm-args "$AGENT_ARGS"
  --user-llm "$USER_MODEL"
  --user-llm-args "$USER_ARGS"
  --num-trials "$NUM_TRIALS"
  --max-concurrency "$MAX_CONCURRENCY"
  --max-steps "$MAX_STEPS"
)

if (($#)); then
  COMMAND+=(--task-ids "$@")
elif [[ -n ${COEVO_EVAL_NUM_TASKS:-} ]]; then
  COMMAND+=(--num-tasks "$COEVO_EVAL_NUM_TASKS")
fi
if [[ -n ${COEVO_EVAL_SAVE_TO:-} ]]; then
  COMMAND+=(--save-to "$COEVO_EVAL_SAVE_TO")
fi

"${COMMAND[@]}"
