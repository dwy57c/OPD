# Contingent-tutoring setup

## 1. Runtime

Copy the example environment file and set local model paths. Keep credentials in
the process environment, not in the repository.

```bash
cp ../.env.example ../.env
export COEVO_POLICY_PATH=/absolute/path/to/student-checkpoint
export COEVO_BUYER_PATH=/absolute/path/to/fixed-user-model
export COEVO_TEACHER_HINT_URL=https://gateway.example/v1
export COEVO_TEACHER_HINT_API_KEY=...
export COEVO_TEACHER_HINT_MODEL=gemini-3.1-pro-preview
export COEVO_SHARPEN_ENABLED=0
```

The active E1–E3 path needs the policy endpoint on port 8000 and the fixed τ²
user endpoint on port 8002. It does not need `policy_previous` or the Buyer GRPO
rollout server.

```bash
export COEVO_START_ROLLOUT=false
bash scripts/start_servers.sh
python scripts/preflight.py services
```

## 2. Tests

```bash
pytest -q
ruff check coevo scripts tests
```

## 3. E1 static audit

First collect or select an immutable public-state pool. Then generate every hint
level against those same states:

```bash
python scripts/collect_round.py \
  --output-dir artifacts/public_states \
  --task-ids 1 2 3 \
  --hint-level L0_NONE \
  --no-sharpen-enabled

python scripts/audit_hint_ladder.py \
  --from-trajectories artifacts/public_states/trajectories.jsonl \
  --output-dir artifacts/e1_hint_audit
```

L3 standard actions are continuation-validated by default and only perfect
actions are eligible for the fixed GRPO target. Change
`--standard-quality-threshold` explicitly if a softer audited target is needed.

The grounding judge is enabled by default. Use
`--no-use-grounding-judge` only for infrastructure smoke tests; such output is
not sufficient for a behavioral claim.

## 4. E2 equal-budget dose response

The runner collects one L0 public-state pool per seed, re-labels the exact same
states at L1/L2/L3, chooses the smallest available reference-token total, and
trains every arm to that budget.
It also records per-task and aggregate L0/L1/L2/L3 Teacher success rates using
turn-refreshed hints, so Student results can be separated from Teacher quality.

```bash
python scripts/run_dosage_experiment.py \
  --output-dir artifacts/e2_dosage \
  --task-ids 1 2 3 \
  --seeds 42 43 44 \
  --student-steps 100
```

Evaluate each resulting checkpoint with the fixed held-out user, save the τ²
conversations, and run:

```bash
python scripts/evaluate_behavior.py evaluation.jsonl \
  --output artifacts/e2_behavior.json
```

## 5. E3 h* controller

```bash
python scripts/run_dosage_curriculum.py \
  --output-dir artifacts/e3_hstar \
  --task-ids 1 2 3 \
  --k 8 \
  --policy hstar
```

Baselines use `--policy fixed:L3`, `fixed:L2`, or `random`.

The controller manifest is executable. Consume its sampling weights and
per-task levels into one `MIXED` dataset:

```bash
python scripts/collect_dosage_curriculum.py \
  --manifest artifacts/e3_hstar/dosage_manifest.json \
  --source-trajectories artifacts/public_states/trajectories.jsonl \
  --output-dir artifacts/e3_hstar/student_data \
  --samples 100
```

## 6. Hinter GRPO and alternation

Build one fixed-standard-action row per audited state. The default uses the
eligible L3 action as the standard trajectory, but candidate hints cannot change
that target:

```bash
python scripts/build_hinter_grpo_dataset.py \
  artifacts/e1_hint_audit/audit_rows.jsonl \
  artifacts/hinter_grpo.jsonl
```

Before each hinter update, regenerate actual Student macro-actions under several
hints for every state. Generate the three mandatory controls, create same-state
pairwise labels, and initialize a Student-sized scalar-head discriminator from
the current Student checkpoint:

```bash
python scripts/collect_discriminator_controls.py \
  artifacts/hinter_grpo.jsonl \
  --audit-rows artifacts/e1_hint_audit/audit_rows.jsonl \
  --output-dir artifacts/discriminator_controls_t

python scripts/collect_behavior_hint_samples.py \
  artifacts/hinter_grpo.jsonl artifacts/current_behavior_samples.jsonl \
  --hints-per-state 4

python scripts/build_copying_discriminator_dataset.py \
  artifacts/current_behavior_samples.jsonl \
  artifacts/copying_discriminator_round_t.jsonl \
  --explicit-copy-controls artifacts/discriminator_controls_t/explicit_copy_controls.jsonl \
  --useless-controls artifacts/discriminator_controls_t/useless_controls.jsonl \
  --natural-copy-pairs artifacts/discriminator_controls_t/explicit_copy_natural_pairs.jsonl

python scripts/train_behavior_discriminator.py \
  --student-checkpoint /path/to/current-student \
  --pairs artifacts/copying_discriminator_round_t.jsonl \
  --output-dir artifacts/copying_discriminator_t

CUDA_VISIBLE_DEVICES=7 python scripts/serve_behavior_discriminator.py \
  --model artifacts/copying_discriminator_t --port 8010
```

Then freeze that discriminator and run the only hinter reward:

```bash
export COEVO_HINTER_BASE_MODEL=/path/to/open-hinter
export COEVO_HINTER_TUNER_TYPE=full
export COEVO_HINTER_DISCRIMINATOR_URL=http://127.0.0.1:8010
export COEVO_HINTER_COPY_WEIGHT=1.0
export COEVO_HINTER_LENGTH_WEIGHT=0.002
export COEVO_HINTER_RULE_LEAK_FLOOR=1.0
export COEVO_HINTER_ANCHOR_BETA=0.01
export COEVO_HINTER_REWARD_TRACE_PATH=artifacts/hinter_reward_round_t.jsonl
bash scripts/train_hinter_grpo.sh \
  artifacts/hinter_grpo.jsonl artifacts/hinter_round_t 20
```

`coevo/hinter_training/alternating_loop.py` fixes the outer order: Student N
steps, one pass@k measurement for both delayed acceptance and curriculum,
fresh discriminator retraining, then hinter GRPO. A regression rolls back the
Student segment and the prior hinter candidate; no pass@k or observed
distillation gain is passed to GRPO.

Every few rounds, rerun `train_behavior_discriminator.py` from the same current
Student with a different seed and empty output directory, then compare it with
the active scorer using `compare_behavior_discriminators.py`. This independent
auditor is not reused for training rewards.

For production alternation, `scripts/run_alternating_rounds.py` turns every
callback into a checked subprocess stage, persists per-round manifests and
state, and executes rollback commands. Its `--commands` JSON contains argv
templates for the nine named stages; every stage receives its required output
path as both `{output}` and `COEVO_STAGE_OUTPUT`.

The repository includes a schema-compatible command set. Before every Student
update it consumes the previous h* manifest, asks the currently served
`hinter_under_test` through `teacher_hint_mode=open_hinter`, rebuilds the mixed
Student dataset, and trains from `{student}`. Both Student and hinter therefore
accumulate across rounds.

```bash
export COEVO_ALTERNATING_TASK_IDS="1 2 3"
export COEVO_ALTERNATING_SAMPLES=100
export COEVO_HINTER_GRPO_DATASET=/path/to/hinter_grpo.jsonl
export COEVO_HINT_AUDIT_ROWS=/path/to/e1/audit_rows.jsonl
export COEVO_HINTER_URL=http://127.0.0.1:8004
export COEVO_HINTER_MODEL=hinter

python scripts/run_alternating_rounds.py \
  --commands configs/alternating_commands.example.json \
  --scenario-pool /path/to/scenario_pool.json \
  --source-trajectories /path/to/public_state_pool/trajectories.jsonl \
  --bootstrap-curriculum-manifest /path/to/bootstrap/dosage_manifest.json \
  --student-checkpoint /path/to/student \
  --hinter-checkpoint /path/to/hinter \
  --output-dir artifacts/alternating \
  --rounds 3
```

The adapters under `scripts/stages/` translate existing checkpoint, dosage,
discriminator, and independent-auditor artifacts into the driver's strict JSON
contracts. A stage that fails its gate or omits its output aborts the round with
a failed manifest.

## 7. Detector validation

Human-label at least 200 rows per detector and report agreement and Cohen's
kappa:

```bash
python scripts/validate_behavior_detectors.py annotations.jsonl \
  --output artifacts/detector_validation.json \
  --minimum-rows 200
```

The retired Buyer/LP setup is preserved in
`../docs/archive/SETUP_BUYER_LP.md` for historical reproduction only.
