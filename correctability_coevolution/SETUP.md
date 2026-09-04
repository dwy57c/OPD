# Contingent-tutoring setup

## Runtime

Keep credentials outside the repository. The E4 eight-GPU layout is fixed:

| role | GPU |
|---|---|
| frozen policy service | 0 |
| open-hinter service | 1 |
| Student or hinter training | 2–7 |

```bash
cp ../.env.example ../.env
export COEVO_POLICY_PATH=/path/to/current-student
export COEVO_POLICY_GPUS=0
export COEVO_HINTER_BASE_MODEL=/path/to/open-hinter
export COEVO_HINTER_URL=http://127.0.0.1:8004
export COEVO_HINTER_MODEL=hinter
export COEVO_HINTER_GPUS=1
export COEVO_POLICY_TRAIN_GPUS=2,3,4,5,6,7
export COEVO_HINTER_TRAIN_GPUS=2,3,4,5,6,7
```

Run checks with:

```bash
ruff check coevo scripts tests
pytest -q
```

## E1: fixed-ladder audit

Collect one immutable L0 public-state pool, then audit L1/L2/L3 on the same
states. L2 is blind-written from the public state, goal, and public domain
policy, and is not fact-checked afterward. L3 must state hidden facts explicitly.

```bash
python scripts/collect_round.py \
  --output-dir artifacts/public_states \
  --task-ids 1 2 3 \
  --hint-level L0_NONE \
  --no-sharpen-enabled

python scripts/audit_hint_ladder.py \
  --from-trajectories artifacts/public_states/trajectories.jsonl \
  --student-profile-manifest artifacts/e1_profile/dosage_manifest.json \
  --output-dir artifacts/e1_hint_audit
```

The audit runs the same four teacher-forced views used by GRPO and reports
mean lift, mean analytical copy, and copy/transferable fractions. It records the
coarse dose value only as an internal penalty diagnostic, not a paper metric.
`recommended_copying_weight_from_l3` sets lambda from the observed L3 anchor.
`analytical_scoring_panel` reports how many states were skipped because no L3
standard action survived the strict contract and continuation validation.
Provide one environment-valid alternative answer for the same `state_id` and
`task_id`, marked `plausible_alternative=true`. Regenerate the open hinter under
the original and changed facts, then render the two E1 figures:

```bash
python scripts/generate_hint_counterfactuals.py \
  artifacts/e1_hint_audit/audit_rows.jsonl \
  --counterfactuals artifacts/e1_same_task_alternatives.jsonl \
  --output-dir artifacts/e1_open_counterfactual
python scripts/evaluate_hint_counterfactuals.py \
  artifacts/e1_open_counterfactual/original.jsonl \
  artifacts/e1_open_counterfactual/counterfactual.jsonl \
  --output artifacts/e1_counterfactual.json
python scripts/plot_e1_hint_metrics.py \
  artifacts/e1_hint_audit/summary.json \
  artifacts/e1_counterfactual.json \
  --output-dir artifacts/e1_figures
```

## E2: equal-budget source and sink controls

The runner factorially compares raw versus Purified-OPSD targets at every hint
level while retaining identical states, seeds, and active-token budgets. It
also reports teacher-side pass rates for every fixed level.

```bash
python scripts/run_dosage_experiment.py \
  --output-dir artifacts/e2_dosage \
  --task-ids 1 2 3 \
  --seeds 42 43 44 \
  --target-operators raw purified \
  --student-steps 100
```

The purified sink centers `log q(s,h) - log p(h)`, applies
`c*tanh(delta/c)` with `COEVO_PURIFIED_CLIP=10`, and then forms the target.

## E3: curriculum sensor

The fixed-ladder experiment still measures L0–L3 and supports fixed/random
baselines. During coevolution, set `COEVO_HINT_LEVEL=HINTER`; the controller
then measures only L0 versus the current open hinter. It uses those results to
classify and weight scenarios without selecting a hint level.

```bash
COEVO_HINT_LEVEL=HINTER COEVO_TEACHER_HINT_MODE=open_hinter \
COEVO_TEACHER_HINT_URL="$COEVO_HINTER_URL" \
COEVO_TEACHER_HINT_MODEL="$COEVO_HINTER_MODEL" \
COEVO_TEACHER_HINT_API_KEY="${COEVO_HINTER_API_KEY:-EMPTY}" \
python scripts/run_dosage_curriculum.py \
  --output-dir artifacts/e3_hstar --task-ids 1 2 3 --k 8 \
  --student-profile-manifest artifacts/e3_previous/dosage_manifest.json
```

`collect_dosage_curriculum.py` consumes only the sampling weights. Mastered/L0
rows are excluded, and every selected training row has
`sample_hint_level=HINTER`.

## Cold-start hinter SFT

Cold start is fail-closed: it requires low-copy rows from at least two Student
checkpoints and at least two non-zero minimal sufficient doses. The prompt gets
an explicit `student_profile={unhinted_success, curriculum_band}` from h*;
success is rounded to a 0.1 bucket and checkpoint/revision/round strings are
never exposed. Thus conflicting h* labels do not share an identical input. Each
`--source` contains the checkpoint identity, its E1 rows, and its h* manifest.

```bash
python scripts/build_hinter_cold_start_dataset.py \
  --source student_0 artifacts/e1_s0/audit_rows.jsonl artifacts/hstar_s0.json \
  --source student_1 artifacts/e1_s1/audit_rows.jsonl artifacts/hstar_s1.json \
  --max-mean-copy 0.1 \
  --output artifacts/hinter_cold_start.jsonl
```

## E4: four-view GRPO and alternation

Build one GRPO row per audited task session. Each row contains every eligible
L3 standard action in that session, while the hinter emits only one task hint:

```bash
python scripts/build_hinter_grpo_dataset.py \
  artifacts/e1_hint_audit/audit_rows.jsonl artifacts/hinter_grpo.jsonl
```

The reward uses four parallel teacher-forced calls to the same frozen Student:

```text
lift_t = clip(log q(a*_t | s,h) - log p(a*_t | s), -c, c)
copy_t = clip(max(log p(a*_t | h) - log p(a*_t | empty), 0), 0, c)
dose   = max(mean_t KL(q_h || p) - bandwidth, 0)
R      = mean(lift) - lambda mean(copy) - nu dose - mu tokens(h)
```

The empty view contains the ordinary system prompt but no hint or dialogue.
Therefore copy isolates the hint's effect instead of charging a no-state model
prior. `mean_copy` is clipped nats per target token, not an 0–1 probability.
E1 then averages turn scores within each session and gives sessions equal
weight. GRPO applies one candidate task hint to every standard decision turn
and returns the same session-average reward for that hint.

The sparse dose KL is a stable coarse-grained lower bound on shared explicit
support plus one tail bucket. Treat it only as an internal penalty proxy, not as
a reported KL metric. Rule-detected fact/tool leakage still caps reward
at a negative floor. No Student rollout, learned discriminator, pass@k, or
post-distillation gain enters GRPO.

```bash
export COEVO_HINTER_COPY_WEIGHT=1.0
export COEVO_HINTER_DOSE_WEIGHT=1.0
export COEVO_HINTER_LENGTH_WEIGHT=0.002
export COEVO_HINTER_TOKEN_CLIP=5.0
export COEVO_HINTER_DOSE_BANDWIDTH=0.05
export COEVO_HINTER_RULE_LEAK_FLOOR=1.0
export COEVO_HINTER_REWARD_TRACE_PATH=artifacts/hinter_reward_round_t.jsonl
bash scripts/train_hinter_grpo.sh \
  artifacts/hinter_grpo.jsonl artifacts/hinter_round_t 20
```

The open hinter receives exactly `{domain_policy,
authoritative_oracle_steps}` as privileged context. Its public state uses the
same canonical serializer in serving and GRPO. It self-reports `level: L1`,
`L2`, or `L3` for fading plots, but the label is neither required nor rewarded.

The outer loop is only:

```text
distill Student N steps with the current hinter candidate
measure deployment pass@k once
accept or roll back the Student and prior hinter; reuse the panel for scheduling
train the next hinter candidate for a few analytical-reward GRPO steps
```

```bash
python scripts/run_alternating_rounds.py \
  --commands configs/alternating_commands.example.json \
  --scenario-pool /path/to/scenario_pool.json \
  --source-trajectories /path/to/public_state_pool/trajectories.jsonl \
  --bootstrap-curriculum-manifest /path/to/bootstrap/dosage_manifest.json \
  --student-checkpoint /path/to/student \
  --hinter-checkpoint /path/to/hinter \
  --output-dir artifacts/alternating --rounds 3
```

## ALFWorld

Use AgentGym/ETO train, `valid_seen`, and `valid_unseen` splits. Normalize each
record with `privilege_from_agentgym_eto_record`; the required privileged source
is the expert action trajectory plus simulator hidden state:

```json
{
  "goal_object_locations": {"mug": "coffeemachine 1"},
  "destination_receptacle": "cabinet 4",
  "unobserved_states": {"mug": "cold"}
}
```

The ALFWorld behavior audit reports whether the first navigation target directly
hits the true hidden location and whether any look/open/inspect action occurred
before pickup.

The retired Buyer/LP and discriminator designs remain available only through
Git history and archived method notes; active code does not import them.
