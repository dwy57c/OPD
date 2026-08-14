# Local setup and execution

The runtime is local-model only by default. Do not permit Hugging Face or
ModelScope downloads unless the user explicitly changes that policy.

## 1. Configure local paths

From the repository root:

```bash
cp .env.example .env
```

Set at least:

```bash
COEVO_POLICY_MODEL_DIR=/absolute/host/path/to/Qwen3-4B
COEVO_BUYER_MODEL_DIR=/absolute/host/path/to/Buyer-or-User-model
COEVO_ALLOW_DOWNLOADS=0
COEVO_REPORT_TO=wandb
```

Inside the container the Student is mounted read-only at `/models/policy` and
the Buyer/User at `/models/buyer`. If `COEVO_BUYER_MODEL_DIR` is unset, Compose
mounts the policy model for both roles. The controller fills
`COEVO_PREVIOUS_POLICY_PATH` only after the Student update.

Do not place API keys in `.env` committed to source control. The closed-model
hint service and W&B credentials must be injected through the runtime
environment.

## 2. Start the existing runtime

```bash
docker compose up -d coevo
docker compose exec coevo bash
```

Inside the container:

```bash
cd /workspace/OPD/correctability_coevolution
python scripts/preflight.py python
python scripts/preflight.py start
pytest -q
```

`preflight.py` verifies pinned Python packages, τ² data/revision, local model
files, GPU assignment, port conflicts, and `report_to=wandb`.

## 3. Service layout

| Role | Port | Default GPU variable | Purpose |
|---|---:|---|---|
| `policy` | 8000 | `COEVO_POLICY_GPUS` | current unprivileged Student `S_(k+1)` |
| `policy_previous` | 8001 | `COEVO_PREVIOUS_POLICY_GPUS` | frozen `S_k`; also the `S_k+skill` Teacher anchor during Buyer GRPO |
| `buyer` | 8002 | `COEVO_BUYER_GPUS` | Buyer reference endpoint |
| `rollout` | 8003 | `COEVO_BUYER_ROLLOUT_GPUS` | Swift online rollout server |

At initial collection there is no previous role. After `S_k -> S_(k+1)`, the
controller starts `S_k` on port 8001 and `S_(k+1)` on port 8000, verifies both,
sets `COEVO_TEACHER_ANCHOR=previous`, then starts Buyer GRPO. Collection uses
`COEVO_TEACHER_ANCHOR=current` so Student supervision for round `k` still comes
from `S_k+skill`. The controller does not start the separate rollout replica
during collection; it starts that role only after the checkpoint pair is ready.

Manual role commands:

```bash
./scripts/start_role.sh policy /models/policy
./scripts/start_role.sh policy_previous /path/to/pre-update-checkpoint
./scripts/start_role.sh buyer /models/policy
./scripts/start_role.sh rollout /models/policy

./scripts/stop_role.sh rollout
./scripts/stop_role.sh policy_previous
./scripts/stop_role.sh policy
./scripts/stop_role.sh buyer
```

Role scripts only stop PIDs recorded in this project's runtime registry.

## 4. Required objective configuration

```bash
COEVO_TEACHER_GAP_TOPK=20
COEVO_TEACHER_GAP_MIN_SUPPORT_MASS=0.95
COEVO_TEACHER_GAP_EPS=1e-8

COEVO_SKILL_GATE_METRIC=forward_kl
COEVO_SKILL_GATE_LOW=0.0
COEVO_SKILL_GATE_HIGH=0.05
COEVO_SKILL_SHARPEN_T_MIN=0.7

# Analysis only; main collection and Buyer reward do not run this continuation.
COEVO_TEACHER_VALIDATION_CONTINUATIONS=1

COEVO_DATASET_SCHEMA_VERSION=4
COEVO_TARGET_SCHEMA_VERSION=2
COEVO_TEACHER_TARGET_VERSION=skill-contrast-sharpened-v2
```

Only one Buyer reward is registered: `tau2_stage_learning_progress`. There is no
runtime reward-mode selector or single-checkpoint fallback. Both Buyer training
scripts pass `--scale_rewards group`.

Skill thresholds and `T_min` should be calibrated once on a held-out calibration
split and frozen before matched scientific comparisons.

## 5. Collect a Student dataset

With the current policy, Buyer reference, and hint service healthy:

```bash
python scripts/collect_round.py \
  --output-dir artifacts/round_0000/data \
  --task-ids 1
```

Outputs:

```text
trajectories.jsonl   full audit records, including rejected targets
student_gkd.jsonl    accepted cached TeacherTargetRecord rows
buyer_grpo.jsonl     Buyer prompts for later online GRPO
summary.json         schema, provenance, counts, hashes, and errors
```

Student eligibility requires a complete Teacher action, exact target alignment,
valid hinted/unhinted distributions, and sufficient Teacher support mass.
The main collector does not run a Teacher takeover continuation or terminal
quality scorer.

## 6. Run one complete alternating round

The canonical entry point is:

```bash
python scripts/run_coevolution.py \
  --output-dir artifacts/stage_run \
  --rounds 1 \
  --student-steps 1 \
  --buyer-steps 1 \
  --task-ids 1 \
  --start-services
```

The controller performs:

```text
collect D_k
train Student S_k -> S_(k+1)
serve S_k and S_(k+1)
preflight both endpoints
train Buyer B_k -> B_(k+1)
atomically commit manifest
```

Use `--resume --start-services` only when reusing an existing output directory.
Resume validates schema, target version, reward formula, task IDs, and exact
checkpoint identities before any work.

## 7. Isolated Student smoke

Start the current policy role, then build a fixture from real current-policy
prompt logits:

```bash
./scripts/start_role.sh policy /models/policy

python scripts/make_student_smoke_fixture.py \
  --model-path /models/policy \
  --policy-url http://127.0.0.1:8000 \
  --output artifacts/infra_smoke/student_gkd.jsonl

WANDB_MODE=offline ./scripts/train_student_smoke.sh \
  artifacts/infra_smoke/student_gkd.jsonl \
  artifacts/infra_smoke/student_adapter
```

This verifies real logits, target caching, finite unweighted forward-KL,
non-zero target gradients, W&B reporting, checkpoint save, and reload. It is not
a τ² capability result.

## 8. Buyer smoke prerequisites

Buyer smoke requires two distinct checkpoints:

```bash
export COEVO_PREVIOUS_POLICY_PATH=/path/to/S_k
export COEVO_CURRENT_POLICY_CHECKPOINT=/path/to/S_k_plus_1
export COEVO_POLICY_PATH=/path/to/S_k_plus_1
export COEVO_TEACHER_ANCHOR=previous
```

Start previous, current, Buyer, and rollout roles, run service preflight, then:

```bash
WANDB_MODE=offline ./scripts/train_buyer_smoke.sh \
  artifacts/round_0000/data/buyer_grpo.jsonl \
  artifacts/infra_smoke/buyer_adapter
```

The rollout must report `checkpoint_teacher_anchor == checkpoint_previous`,
finite `previous_gaps`, `current_gaps`, raw LP, positive-LP reward, shared target
hashes, and only Buyer-token response masks.

Buyer training defaults to `COEVO_BUYER_BETA=0.01`; reference KL is an optimizer
regularizer, while stage learning progress remains the only scalar Buyer
reward. Qwen3.5 Buyer training must use
`COEVO_BUYER_ATTN_IMPL=flash_attn` (FlashAttention2). The Trainer rejects
Qwen3.5 SDPA/eager runs before optimization, because that path produced
non-finite full-attention gradients in this runtime. NaN/Inf loss filtering is
disabled so failures remain visible in logs.

Every LoRA entrypoint runs `check_adapter_finite.py` after save and fails if any
adapter value is NaN or infinity. The gradient guard defaults to fail-closed
(`COEVO_NONFINITE_GRADIENT_ACTION=error`). `zero` is an explicit diagnostic
mode for kernels that emit a small number of non-finite gradient elements; its
sanitized value/tensor counts are logged to W&B and must be reported.

## 9. Verification commands

```bash
pytest -q
bash -n scripts/*.sh scripts/lib/*.sh
git diff --check
pytest -q tests/test_no_obsolete_objectives.py
```

The objective-regression test must pass. No test or smoke command may
download a model. All Trainer entry points use W&B; set `WANDB_MODE=offline` when
network access is not part of the test.
