# Stage-Conditioned Environment–Student Co-Evolution for OPD

This repository implements a closed-loop environment policy that changes the
distribution of on-policy distillation data as the Student changes. The active
implementation is under [`correctability_coevolution/`](correctability_coevolution/)
and uses complete natural Agent actions, a privileged skill view over the same
policy checkpoint, τ²-Bench transitions, and round-synchronous alternating
optimization.

The central mechanism is:

> The privileged skill defines a skill-contrast-gated sharpened Teacher target.
> The pre-update checkpoint `S_k+skill` defines that target once. Consecutive
> unprivileged Students `S_k` and `S_(k+1)` define stage learning progress
> relative to the fixed target. GRPO trains the environment to generate the
> next round's data in regions with positive recent progress.

This is a moving-distribution signal. Positive stage progress does not claim
that an individual newly generated example caused the checkpoint improvement.

## Research contract

For a complete Student-visible state `s` and one complete privileged Teacher
action `a_T`, the frozen pre-update checkpoint `S_k` scores the same target
tokens twice:

```text
q_h = p(S_k | s, private skill, a_T prefix)
p_0 = p(S_k | s,                a_T prefix)
```

The only difference is the private skill. At target token `j`:

```text
c_j = KL(q_h,j || p_0,j)
g_j = clip((c_j - tau_low) / (tau_high - tau_low), 0, 1)
T_j = 1 - g_j * (1 - T_min)
q_tilde_h,j = softmax(z_h,j / T_j)
```

`g=0` reproduces the raw hinted distribution; `g=1` reaches the configured
sharpening temperature. Temperature sharpening preserves the Teacher ordering
and argmax. It constructs the target; it is not a second scalar loss weight.

The exact same detached `q_tilde_h`, target tokens, and Student-visible state are
then scored under two consecutive unprivileged Students:

```text
d_previous = KL(q_tilde_h || p_previous)
d_current  = KL(q_tilde_h || p_current)
LP         = d_previous - d_current
```

The per-decision reward and trajectory reward are exactly:

```text
r(x) = max(LP(x), 0)
R_i  = trajectory_validity * mean_x r(x)
```

There is no Teacher-quality multiplier, residual-gap gate, exploration bonus,
Student-branch subtraction, or takeover continuation in this training reward.
For each task, GRPO samples `G` Buyer trajectories and uses group reward
normalization after fail-closed all-zero group handling.

The Student objective is independently:

```text
mean_j KL(q_tilde_h,j || p_student,j)
```

No Buyer reward, learning progress, or terminal score filters or weights Student
supervision.

## Exact round ordering

```text
Round k starts with Student S_k and Buyer B_k

1. B_k and S_k collect Teacher-supervised data D_k.
2. Train Student on D_k: S_k -> S_(k+1).
3. Serve previous=S_k and current=S_(k+1) simultaneously.
4. Let current `S_(k+1)` generate Buyer-rollout states, but use frozen
   `S_k+skill` to generate each Teacher demonstration and target.
5. Train Buyer online with progress from unhinted `S_k` to `S_(k+1)` against
   that fixed `S_k+skill` target.
6. Update B_k -> B_(k+1).
7. The next round collects with S_(k+1), B_(k+1).
```

Student and Buyer never change in the same optimizer step. The pre-update
Student remains immutable and online until Buyer training and the round manifest
are committed.

## One canonical Teacher target

Every accepted Student row contains a versioned `TeacherTargetRecord` with:

- canonical state, action, raw-target, sharpened-target, hint, and checkpoint
  identities;
- Student-visible and privileged prompt views ending in the identical action;
- exact target token IDs and loss mask;
- raw hinted and same-checkpoint unhinted sparse distributions and support mass;
- per-token forward skill contrast, gate, temperature, and entropy;
- the final sharpened sparse target plus an explicit aggregate tail treatment.

The cache key binds checkpoint IDs, state, action, hint, tokenizer, and gate
configuration. Scoring is fail-closed: a timeout, token mismatch, missing actual
target token, or insufficient Teacher support produces no positive Buyer reward.
Rejected collection rows remain in trajectory audit data with an explicit
reason.

## Runtime components

| Path | Purpose |
|---|---|
| [`correctability_coevolution/coevo/scoring/skill_contrast.py`](correctability_coevolution/coevo/scoring/skill_contrast.py) | forward skill contrast and target sharpening |
| [`correctability_coevolution/coevo/scoring/teacher_target.py`](correctability_coevolution/coevo/scoring/teacher_target.py) | canonical target record and no-continuation Teacher target labeling; takeover validation is analysis-only |
| [`correctability_coevolution/coevo/scoring/stage_gap.py`](correctability_coevolution/coevo/scoring/stage_gap.py) | cached three-view teacher-forced scoring |
| [`correctability_coevolution/coevo/rewards/stage_progress.py`](correctability_coevolution/coevo/rewards/stage_progress.py) | forward-KL gaps and environment reward |
| [`correctability_coevolution/coevo/training/gated_gkd.py`](correctability_coevolution/coevo/training/gated_gkd.py) | cached-target Student distillation |
| [`correctability_coevolution/coevo/training/buyer_scheduler.py`](correctability_coevolution/coevo/training/buyer_scheduler.py) | online τ² Buyer rollout and reward aggregation |
| [`correctability_coevolution/scripts/run_coevolution.py`](correctability_coevolution/scripts/run_coevolution.py) | checkpoint rotation, resume, rollback, and alternating training |
| [`correctability_coevolution/FULL_INFRA.md`](correctability_coevolution/FULL_INFRA.md) | implementation invariants and dataflow |
| [`correctability_coevolution/SETUP.md`](correctability_coevolution/SETUP.md) | local setup and commands |

The private structured Buyer plan predicts `predicted_learning_progress` and is
rendered by a frozen renderer. The plan and private Teacher hint never enter the
Student-visible, Buyer-visible public, continuation-user, or verifier history.

## Artifact and service contracts

Dataset schema v4 and Teacher-target schema v2 are mandatory. Trajectories,
Student rows, Buyer rows, summaries, and manifests identify the tokenizer,
target construction, reward formula, checkpoint revisions, and ordered
previous/current pair. Merging or resume refuses incompatible identities.

Default service topology during Buyer training:

| Role | Port | Meaning |
|---|---:|---|
| `policy` | 8000 | current unprivileged Student `S_(k+1)` |
| `policy_previous` | 8001 | frozen `S_k`, used both unprivileged and as `S_k+skill` Teacher anchor |
| `buyer` | 8002 | Buyer reference endpoint |
| `rollout` | 8003 | Swift online Buyer rollout server |

The controller rejects a missing or accidentally identical checkpoint pair.
Teacher-anchor/current prompt-logprob failures are surfaced and never replaced
by a moving `S_(k+1)+skill` target or single-checkpoint proxy.

## Local-only execution

No model download is needed when local weights already exist:

```bash
cp .env.example .env
# Set COEVO_POLICY_MODEL_DIR to an existing local checkpoint.
# Keep COEVO_ALLOW_DOWNLOADS=0.
docker compose up -d coevo
docker compose exec coevo bash
python scripts/preflight.py python
pytest -q
```

Run one controller-managed round:

```bash
python scripts/run_coevolution.py \
  --output-dir artifacts/run \
  --rounds 1 \
  --student-steps 1 \
  --buyer-steps 1 \
  --task-ids 1 \
  --start-services
```

All Trainer scripts use Weights & Biases through `--report_to wandb`. Use
`WANDB_MODE=offline` for isolated smoke tests. SwanLab reporting is disabled.
Credentials belong only in the runtime environment.

## Evaluation boundary

The learned Buyer is never used as the benchmark user. Final Student results use
the official held-out τ² split, a fixed independent user simulator, and fixed
verifier settings. Buyer reward is an optimization diagnostic, not a held-out
capability result.

## Development rules

- Read relevant code and documentation before editing.
- Do not add unrelated refactors or features.
- Run tests, shell syntax checks, and `git diff --check` after changes.
- Keep models local unless download authorization is explicit.
- Do not commit secrets, `.codex/auth.json`, or `secrets/`.
- Do not commit, push, upload, or open a pull request without authorization.
