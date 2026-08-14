# TODO: Stage-Conditioned Teacher-Gap Curriculum with Skill-Contrast Sharpening

> Implementation backlog for the current natural-action Swift baseline under `correctability_coevolution/`.
>
> This revision intentionally removes the previous shadow-OPD, utility-critic, intervention-advantage curriculum, per-sample virtual-update, and hand-engineered learnability proposals. The environment policy is trained with one direct signal: **the decrease in Teacher–Student discrepancy between two consecutive Student checkpoints on newly generated data**. Separately, the Student target uses a **skill-contrast sharpening gate** derived from the logit difference between the skill-conditioned self-Teacher and the same checkpoint without the skill.

---

## 0. Research contract

The project should implement the following stage-conditioned curriculum loop.

At curriculum stage `k`, keep two frozen unprivileged Student checkpoints available:

- previous Student: \(S_{k-1}\);
- current Student: \(S_k\).

For a newly generated natural-decision example \(x\), first score the same target tokens with the same checkpoint under two prompt views:

\[
q_{h,j}(\cdot\mid x)=p_{S_k}(\cdot\mid x,h)_j,
\qquad
p_{0,j}(\cdot\mid x)=p_{S_k}(\cdot\mid x)_j.
\]

The two views must share model parameters, tokenizer, target tokens, and prefix. Their only difference is the private skill/hint. Define a detached token-level skill contrast:

\[
c_j(x)
=
D_{\mathrm{KL}}\!\left(q_{h,j}\,\|\,p_{0,j}\right),
\]

and a bounded gate:

\[
g_j(x)
=
\operatorname{clip}\!\left(
\frac{c_j(x)-\tau_{\mathrm{low}}}
{\tau_{\mathrm{high}}-\tau_{\mathrm{low}}},
0,1
\right).
\]

Use the gate to lower the Teacher temperature only where the skill materially changes the distribution:

\[
T_j(x)=1-g_j(x)\left(1-T_{\min}\right),
\qquad 0<T_{\min}<1,
\]

\[
\widetilde q_{h,j}(\cdot\mid x)
=
\operatorname{softmax}\!\left(
\frac{z^h_j(x)}{T_j(x)}
\right).
\]

Thus \(g_j=0\) leaves the raw hinted Teacher distribution unchanged, while \(g_j=1\) maps it to the configured sharper endpoint. Equivalently, with \(\alpha_j=1/T_j\), \(\widetilde q_{h,j}\propto q_{h,j}^{\alpha_j}\); this preserves vocabulary ordering and the Teacher argmax while reducing entropy whenever \(T_j<1\). The gate is part of **Teacher-target construction**, not a separate environment reward term.

Score the **same detached sharpened target, same target tokens, and same Student-visible state** under the previous and current unprivileged Students:

\[
d_{k-1}(x)
=
D_{\mathrm{KL}}\!\left(\widetilde q_h(\cdot\mid x)\,\|\,p_{S_{k-1}}(\cdot\mid x)\right),
\]

\[
d_k(x)
=
D_{\mathrm{KL}}\!\left(\widetilde q_h(\cdot\mid x)\,\|\,p_{S_k}(\cdot\mid x)\right).
\]

Define stage learning progress as:

\[
LP_k(x)=d_{k-1}(x)-d_k(x).
\]

The environment/Buyer reward is:

\[
R_{\mathrm{env}}(x,k)
=
q_{\mathrm{valid}}(x)
\left[
q_{\mathrm{teacher}}(x)
\,[LP_k(x)]_+
\,g_{\mathrm{residual}}\!\left(d_k(x)\right)
+
\beta b_{\mathrm{explore}}(x)
\right].
\]

This reward means:

> The newly generated example lies in a region where the current Student is closer to the privileged Teacher than the previous Student, while a non-trivial Teacher gap remains.

It does **not** mean:

> The newly generated example caused the improvement from \(S_{k-1}\) to \(S_k\).

The environment is learning a moving data distribution, not estimating the causal value of each individual sample.

### Non-negotiable signal separation

- [ ] Teacher supervision is the Student’s direct training signal.
- [ ] The skill-contrast gate is computed from hinted versus unhinted logits of the same checkpoint and only constructs the sharpened Teacher target.
- [ ] The same detached sharpened target `q_h_tilde` is used by Student distillation and by both cross-checkpoint gap calculations.
- [ ] Stage learning progress is the Buyer/environment GRPO reward.
- [ ] Buyer reward must never weight the Student token loss.
- [ ] Student loss must never be multiplied by `intervention_advantage`, `learning_progress`, Buyer reward, `q_teacher`, or verifier reward.
- [ ] `q_teacher` is used only in the environment reward; it must not filter or weight Student supervision.
- [ ] A newly generated example is evaluated as a probe of the Student’s current learning region; it is not credited with causing the checkpoint change.
- [ ] Final Student evaluation must use a fixed evaluator and a fixed user simulator independent of the learned Buyer.

---

## 1. Exact stage and round semantics

Use the following round ordering. It is compatible with the repository’s current high-level `collect -> train Student -> train Buyer` controller, but requires preserving the pre-update Student checkpoint.

```text
Round k starts with Student S_k and Buyer B_k

1. B_k and S_k collect Teacher-supervised Student data D_k.
2. Train Student on D_k:
       S_k -> S_{k+1}
3. Keep both checkpoints online:
       previous = S_k
       current  = S_{k+1}
4. Train Buyer online with GRPO.
   Every newly generated example x is scored by:
       raw q_h from the current skill-conditioned self-Teacher
       p_0 from the same current checkpoint without the skill
       q_h_tilde from the skill-contrast sharpening gate
       d_previous against S_k
       d_current  against S_{k+1}
       LP = d_previous - d_current
5. Update Buyer:
       B_k -> B_{k+1}
6. Round k+1 collects Student data with S_{k+1} and B_{k+1}.
```

This creates a one-stage curriculum feedback loop:

- the latest Student update determines which regions show learning progress;
- Buyer GRPO learns how to generate more examples from those regions;
- the updated Buyer changes the next round’s Student training distribution.

- [ ] Document this ordering in `README.md`, `FULL_INFRA.md`, and `run_coevolution.py`.
- [ ] Store the exact `(previous_checkpoint, current_checkpoint, buyer_checkpoint)` tuple in every round manifest.
- [ ] Prevent Buyer training when the checkpoint pair is missing, identical by mistake, or reversed.
- [ ] Do not silently substitute a single-checkpoint reward.

---

# P0 — Delete the obsolete objective stack

## 2. Remove shadow OPD and the utility critic completely

These components are no longer part of the method, not even as an online fallback or calibration path.

- [ ] Delete `coevo/training/shadow_opd.py`.
- [ ] Delete `coevo/rewards/utility_critic.py`.
- [ ] Remove `ShadowOPDConfig`, `ShadowOPDEvaluator`, `ShadowOPDResult`, and `ShadowTrainStats` imports and tests.
- [ ] Remove `LinearUtilityCritic`, `UtilityFeatures`, and `UtilityLabel` imports and tests.
- [ ] Remove their exports from `coevo/rewards/__init__.py`.
- [ ] Remove every occurrence of:

  ```text
  opd_utility_gain
  opd_utility_source
  shadow_opd
  shadow_gain_per_token
  predicted_shadow_probe_gain_per_token
  utility_critic
  ```

- [ ] Delete documentation sections that describe disposable LoRA updates, probe-set gain, or learned utility prediction.
- [ ] Add a repository test or CI grep that fails if these obsolete identifiers are reintroduced unintentionally.

Acceptance criterion:

```bash
rg -n "shadow_opd|opd_utility|UtilityCritic|UtilityFeatures|shadow_gain" \
  correctability_coevolution
```

returns no implementation references.

---

## 3. Remove intervention advantage as the Buyer objective

The current `Tau2BuyerScheduler._rollout_infos()` uses mean intervention advantage and optionally switches to `opd_utility_gain`. Replace this logic rather than adding another reward mode beside it.

- [ ] Delete `fast_intervention_reward`.
- [ ] Delete `mean_intervention_advantage` from the Buyer reward path.
- [ ] Delete `turn_intervention_advantages` from Buyer rollout state.
- [ ] Delete implicit reward-source selection based on dictionary keys.
- [ ] Do not retain `intervention|learning_progress|hybrid` runtime modes in the final baseline.
- [ ] Register one explicit Buyer reward function, for example:

  ```text
  tau2_stage_learning_progress
  ```

- [ ] Rename `BuyerUtilityReward` to `BuyerStageProgressReward`.
- [ ] Keep group-level all-zero skipping because \([LP]_+\) can make an entire GRPO group zero.
- [ ] Remove `reward_lcb` handling unless it is used by a separately specified statistical estimator.

Files:

```text
coevo/training/buyer_scheduler.py
coevo/training/swift_plugin.py
coevo/rewards/buyer.py
scripts/train_buyer_full.sh
scripts/train_buyer_smoke.sh
```

---

## 4. Remove paired Student-vs-Teacher takeover scoring

The new environment reward does not use:

\[
V_T-V_S.
\]

Therefore the current paired branch machinery should not remain in the main training path.

- [ ] Delete or replace `coevo/intervention/action_branch.py`.
- [ ] Remove `BranchEvaluation.advantage`.
- [ ] Remove `ActionBranchResult.intervention_advantage`.
- [ ] Remove paired Student continuation execution used only to compute `V_T-V_S`.
- [ ] Remove `student_value` from curriculum records.
- [ ] Remove `intervention_advantage` from Student and Buyer datasets.
- [ ] Remove `decision_interventions` and `intervention_advantages` from `summary.json`.
- [ ] Remove tests whose only contract is paired-advantage computation.

Keep only the parts still needed by the new method:

- `DecisionState` for complete natural-action boundaries;
- `TeacherActionGenerator` for one privileged Teacher target;
- optional Teacher-target terminal validation for \(q_{\mathrm{teacher}}\).

If absolute Teacher validation is retained, replace the paired runner with a one-sided component such as:

```text
coevo/scoring/teacher_target.py
```

It should execute only:

```text
state -> Teacher action -> current unprivileged Student continuation -> terminal score
```

and return `teacher_quality`, never an intervention difference.

---

## 5. Remove intervention-dependent Student filtering

The current collector discards rows with `intervention_advantage <= 0`, and `NaturalDecisionStudentTrainer` requires a positive advantage. Both constraints become invalid once intervention advantage is removed.

- [ ] Remove the positive-advantage filter from `NaturalDecisionCollector.student_rows()`.
- [ ] Remove the `intervention_advantage > 0` assertion from `NaturalDecisionStudentTrainer.training_step()`.
- [ ] Remove `original_branch_messages` unless another active experiment consumes it.
- [ ] Define Student-row eligibility only through:
  - a non-empty complete Teacher action;
  - exact token alignment between hinted and same-checkpoint unhinted views;
  - valid raw hinted, unhinted-reference, and sharpened Teacher distributions.
- [ ] Never use `q_teacher` to filter or weight Student rows in this baseline.
- [ ] Preserve rejected rows in audit data with an explicit rejection reason.

Files:

```text
coevo/rollout/collector.py
coevo/training/gated_gkd.py
coevo/orchestration/collection.py
```

---

# P0 — Make one reusable Teacher target

## 6. Define the curriculum example `x`

For each complete Student natural decision, define:

\[
x=(s, a^T),
\]

where:

- `s` is the complete Student-visible pre-action state;
- `a^T` is one complete Teacher action generated under the private hint;
- the action is text, one tool call, or a protocol-valid parallel tool-call group;
- the Teacher action tokens are teacher-forced under every scoring view.

- [ ] Add a versioned `TeacherTargetRecord` dataclass.
- [ ] Store:

  ```text
  schema_version
  state_hash
  teacher_action_hash
  raw_teacher_target_hash
  teacher_target_hash
  teacher_checkpoint
  teacher_hint_hash
  student_visible_messages
  hinted_teacher_messages
  teacher_action
  target_token_ids
  target_loss_mask
  hinted_topk_logprobs
  hinted_topk_token_ids
  hinted_support_mass
  unhinted_reference_checkpoint
  unhinted_reference_topk_logprobs
  unhinted_reference_topk_token_ids
  unhinted_reference_support_mass
  skill_contrast_scores
  skill_gate_values
  sharpening_temperatures
  sharpened_topk_logprobs
  sharpened_topk_token_ids
  sharpened_support_mass
  raw_teacher_entropy
  sharpened_teacher_entropy
  teacher_quality
  ```

- [ ] Never expose `hinted_teacher_messages` or raw hints to Student-visible or Buyer-visible dialogue history.
- [ ] Ensure text and tool-call serialization is identical across previous, current, and hinted views.
- [ ] Reject a record if the target token IDs differ across hinted, same-checkpoint unhinted, previous-Student, or current-Student views.
- [ ] Define `raw_teacher_target_hash` over `q_h` and the canonical `teacher_target_hash` over `q_h_tilde`, including the per-token temperatures.
- [ ] Normalize all per-example distances by active target-token count.

Primary files:

```text
coevo/intervention/decision_state.py
coevo/intervention/teacher_action.py
coevo/models/hinted_teacher.py
coevo/rollout/collector.py
```

---

## 7. Cache the raw and skill-gated Teacher target once

The same detached sharpened Teacher target must be used for Student distillation and both checkpoint comparisons.

- [ ] Generate the Teacher action and raw hinted distribution `q_h` once per `x`.
- [ ] Score the same target tokens with the same checkpoint without the skill to obtain `p_0`.
- [ ] Compute token-level skill contrast, gate values, temperatures, and `q_h_tilde` once per `x`.
- [ ] Freeze/detach `q_h`, `p_0`, the gate, and `q_h_tilde`; no gradient may flow into target construction during Student or Buyer optimization.
- [ ] Do not independently regenerate or re-sharpen a Teacher target for the previous and current Students.
- [ ] Do not compare `S_{k-1}` to a previous Teacher and `S_k` to a current Teacher.
- [ ] Cache by:

  ```text
  (teacher_checkpoint_id, unhinted_reference_checkpoint_id, state_hash, teacher_action_hash, hint_hash, tokenizer_hash, gate_config_hash)
  ```

- [ ] Record cache hits, misses, and scoring failures.
- [ ] Fail closed: a missing or misaligned target yields zero Buyer reward, not an intervention fallback.

---

## 8. Use a simple, auditable Teacher-gap estimator

For token position `j`, let `q_h_tilde_j` be the detached, skill-gated sharpened Teacher distribution and let `p^-_j`, `p^+_j` be the previous and current unhinted Student distributions.

The exact learning-progress difference can be written without separately estimating Teacher entropy:

\[
LP_k(x)
=
\frac{1}{L}
\sum_{j=1}^{L}
\sum_v \widetilde q_{h,j}(v)
\left[
\log p^+_j(v)-\log p^-_j(v)
\right].
\]

This is equivalent to:

\[
D_{\mathrm{KL}}(\widetilde q_h\|p^-)-D_{\mathrm{KL}}(\widetilde q_h\|p^+).
\]

- [ ] Add `coevo/rewards/stage_progress.py`.
- [ ] Implement a typed result object containing:

  ```text
  previous_gap
  current_gap
  learning_progress
  positive_learning_progress
  residual_gate
  teacher_quality
  validity
  exploration_bonus
  decision_reward
  ```

- [ ] Use forward KL, `KL(sharpened Teacher || Student)`, consistently.
- [ ] Use a fixed Teacher top-k support plus an explicit tail treatment.
- [ ] Include the actual target token in the support even when it is outside Teacher top-k.
- [ ] Log Teacher support mass and reject examples below a configured minimum coverage.
- [ ] Use the same tokenizer and chat template for all three policy views.
- [ ] Add a hard-target NLL implementation only as a diagnostic fallback, not as the default metric.

Suggested initial configuration:

```text
COEVO_TEACHER_GAP_TOPK=20
COEVO_TEACHER_GAP_MIN_SUPPORT_MASS=0.95
COEVO_TEACHER_GAP_EPS=1e-8
```

---

# P0 — Add previous/current checkpoint scoring

## 9. Add a frozen previous-Student endpoint

The current repository exposes only one policy endpoint and stops it during refresh. Buyer training requires both checkpoints simultaneously.

- [ ] Extend `InfraConfig` with:

  ```text
  previous_policy: ModelEndpoint
  ```

- [ ] Add environment variables:

  ```text
  COEVO_PREVIOUS_POLICY_URL
  COEVO_PREVIOUS_POLICY_PORT
  COEVO_PREVIOUS_POLICY_MODEL
  COEVO_PREVIOUS_POLICY_PATH
  COEVO_PREVIOUS_POLICY_GPUS
  ```

- [ ] Add `policy_previous` to `scripts/start_role.sh` and `scripts/stop_role.sh`.
- [ ] Use a non-conflicting default port, for example `8001`.
- [ ] Start the previous checkpoint before Buyer GRPO begins.
- [ ] Keep the current policy endpoint on the existing policy port.
- [ ] Add both endpoints to service preflight checks.
- [ ] Include model path, served-model name, tokenizer revision, and checkpoint hash in rollout metadata.
- [ ] Add batched teacher-forced scoring requests for both endpoints.

Files:

```text
coevo/config.py
scripts/start_role.sh
scripts/start_servers.sh
scripts/stop_role.sh
scripts/stop_servers.sh
scripts/preflight.py
```

---

## 10. Implement a three-view scorer

Add a scorer that evaluates one target under exactly three views:

1. current skill-conditioned self-Teacher: produces raw `q_h`;
2. previous unhinted Student: produces `p_previous`;
3. current unhinted Student: produces `p_current`, which also serves as `p_0` for the skill-contrast gate because it shares the Teacher checkpoint and differs only by removal of the skill.

- [ ] Add `coevo/scoring/stage_gap.py`.
- [ ] Build all views from the same canonical serialized state and target token sequence.
- [ ] Construct `q_h_tilde` once from `(q_h, p_current)` before computing either checkpoint gap.
- [ ] Assert exact token-position alignment before computing any distance.
- [ ] Batch previous/current requests whenever multiple decisions are scored in one trajectory.
- [ ] Cache repeated state/target/checkpoint combinations.
- [ ] Add bounded retries and explicit failure reasons.
- [ ] Never fall back to `V_T-V_S`, raw Student failure, or current gap alone.

Acceptance criteria:

- identical previous/current checkpoints produce `LP == 0` up to numerical tolerance;
- replacing the current logits with logits closer to `q_h_tilde` produces positive LP;
- swapping previous/current checkpoints flips the sign of raw LP;
- Teacher target hashes are identical in both distance calculations.

---

# P0 — Implement the Buyer reward directly

## 11. Define the minimal reward terms

### Validity

Reuse the existing hard Buyer validity semantics:

\[
q_{\mathrm{valid}}(x)\in\{0,1\}.
\]

- malformed Buyer plan;
- illegal user tool call;
- failed Buyer-originated transition;
- truncated rollout;
- missing Student natural action;

must yield zero reward.

Student mistakes must not invalidate the Buyer rollout.

### Teacher quality

Use a simple absolute Teacher reliability term:

\[
q_{\mathrm{teacher}}(x)\in[0,1].
\]

For the first implementation:

- [ ] define `q_teacher` as the category-balanced terminal score obtained after inserting the Teacher action and returning continuation control to the current unprivileged Student;
- [ ] average over `K` Teacher-validation continuations if `K > 1`;
- [ ] do not subtract a Student-branch score;
- [ ] if Teacher validation is disabled in an ablation, set `q_teacher = 1` explicitly and log the ablation.

### Residual gap

The residual term only prevents already-mastered examples from remaining rewarding.

Use a simple hard gate first:

\[
g_{\mathrm{residual}}(d_k)
=
\mathbf{1}[d_k>\tau_{\mathrm{residual}}].
\]

- [ ] Do not introduce a multi-term handcrafted difficulty function.
- [ ] Log the ungated current gap even when the gate is closed.

### Exploration bonus

Exploration is optional and secondary.

- [ ] Set `beta = 0` in the first end-to-end baseline.
- [ ] After the core reward works, implement a bounded count bonus over a stable state/skill hash:

  \[
  b_{\mathrm{explore}}(x)=\frac{1}{\sqrt{1+N(\mathrm{bucket}(x))}}.
  \]

- [ ] Apply the validity gate to the exploration term as well; invalid rollouts must always receive zero total reward.
- [ ] Keep the bonus small enough that an example with zero learning progress cannot dominate a genuinely progressing example.

Suggested configuration:

```text
COEVO_RESIDUAL_GAP_THRESHOLD
COEVO_EXPLORATION_BETA=0
COEVO_TEACHER_VALIDATION_CONTINUATIONS=1
```

---

## 12. Aggregate decision rewards into one Buyer trajectory reward

For a Buyer rollout with valid natural-decision examples `x_1, ..., x_n`, use the arithmetic mean:

\[
R_{\mathrm{Buyer}}
=
\frac{1}{n}\sum_{i=1}^{n}R_{\mathrm{env}}(x_i,k).
\]

- [ ] Use one documented aggregation rule; do not tune among mean, max, or positive-only mean after seeing benchmark results.
- [ ] Return zero if no valid decision can be scored.
- [ ] Keep raw negative LP for diagnostics, but use `[LP]_+` in the reward.
- [ ] Preserve the existing Buyer-only response loss mask.
- [ ] Preserve all-zero GRPO group skipping.
- [ ] Rename scheduler fields to:

  ```text
  buyer_reward
  reward_source = "stage_learning_progress"
  trajectory_validity
  decision_count
  previous_gaps
  current_gaps
  learning_progresses
  positive_learning_progresses
  residual_gates
  teacher_qualities
  decision_rewards
  checkpoint_previous
  checkpoint_current
  teacher_target_hashes
  skill_contrast_scores
  skill_gate_values
  sharpening_temperatures
  raw_teacher_entropies
  sharpened_teacher_entropies
  scoring_errors
  ```

Files:

```text
coevo/training/buyer_scheduler.py
coevo/training/swift_plugin.py
coevo/rewards/buyer.py
coevo/rewards/stage_progress.py
```

---

## 13. Update the structured Buyer plan vocabulary

The current private Buyer schema predicts takeover gain, which no longer matches the objective.

- [ ] Rename:

  ```text
  predicted_takeover_gain
  ```

  to:

  ```text
  predicted_learning_progress
  ```

- [ ] Update `BuyerPlan`, JSON validation, planner prompt, serialization, renderer tests, and training fixtures.
- [ ] Delete `observed_intervention_advantage` from `buyer_plan_aux.py`.
- [ ] Either:
  - [ ] delete `buyer_plan_aux.py` if it is not used by the active trainer; or
  - [ ] rewrite it to log `observed_learning_progress` only.
- [ ] Do not add an auxiliary critic that predicts post-update utility.

Files:

```text
coevo/models/buyer_plan.py
coevo/training/buyer_plan_aux.py
coevo/models/frozen_renderer.py
tests/test_buyer_plan.py
tests/test_buyer_scheduler.py
```

---

# P0 — Keep Student supervision independent

## 14. Add the skill-contrast sharpening gate

The Student should not distill the raw hinted distribution uniformly. The same checkpoint must score each Teacher target both with and without the private skill so the implementation can isolate the skill-induced distribution shift.

For every active target token `j`:

\[
q_{h,j}=p_{S_k}(\cdot\mid x,h)_j,
\qquad
p_{0,j}=p_{S_k}(\cdot\mid x)_j,
\]

\[
c_j=D_{\mathrm{KL}}(q_{h,j}\|p_{0,j}),
\]

\[
g_j=\operatorname{clip}\!\left(
\frac{c_j-\tau_{\mathrm{low}}}
{\tau_{\mathrm{high}}-\tau_{\mathrm{low}}},
0,1
\right),
\]

\[
T_j=1-g_j(1-T_{\min}),
\qquad
\widetilde q_{h,j}=\operatorname{softmax}(z^h_j/T_j).
\]

This is a target-construction gate:

- `g_j = 0` keeps the original hinted Teacher distribution;
- `g_j = 1` applies the maximum configured sharpening;
- intermediate values interpolate through the temperature;
- the gate never determines the sign of an RL update and is not multiplied into the Buyer reward;
- all values are detached before Student optimization.

Implementation tasks:

- [ ] Add `coevo/scoring/skill_contrast.py` with typed outputs for raw contrast, normalized gate, temperature, raw entropy, sharpened entropy, and support coverage.
- [ ] Compute contrast from the same-checkpoint hinted and unhinted distributions; never compare different checkpoints when constructing the gate.
- [ ] Use forward KL, `KL(hinted || unhinted)`, as the first baseline metric.
- [ ] Normalize and clamp the gate with fixed predeclared thresholds.
- [ ] Apply sharpening only on active Teacher-target tokens, not prompt or environment tokens.
- [ ] Guarantee `T_j in [T_min, 1]` and numerically stable normalization.
- [ ] Guarantee sharpening preserves the raw Teacher ordering and argmax; it may increase confidence but must not invent a different mode.
- [ ] Store both raw `q_h` and final `q_h_tilde` for auditability.
- [ ] Use `q_h_tilde` as the sole Student distillation target.
- [ ] Use the exact same `q_h_tilde` in `d_previous`, `d_current`, and `LP`; do not measure curriculum progress against a different raw target.
- [ ] Do not add a second multiplicative token-loss gate in the baseline; the temperature transformation is the gate. Evaluate explicit loss weighting only as a separate ablation.
- [ ] Fail closed when hinted/unhinted token alignment or support coverage is invalid.

Suggested initial configuration:

```text
COEVO_SKILL_GATE_METRIC=forward_kl
COEVO_SKILL_GATE_LOW
COEVO_SKILL_GATE_HIGH
COEVO_SKILL_SHARPEN_T_MIN
COEVO_SKILL_GATE_EPS=1e-8
```

The thresholds and `T_min` must be calibrated on a held-out calibration split and frozen before the main comparison.

## 15. Fix the Student Teacher-logit path

The current `NaturalDecisionStudentTrainer.training_step()` calls the Hugging Face SFT training step directly after encoding, which can bypass the expected ms-swift GKD Teacher-logit retrieval path.

- [ ] Add a one-batch regression test that proves hinted Teacher logits are actually fetched or loaded.
- [ ] Refactor the trainer so the Student is optimized against the cached/current skill-gated Teacher target `q_h_tilde`.
- [ ] Use the same canonical `TeacherTargetRecord` used by the stage-gap scorer.
- [ ] Remove all intervention-dependent row checks.
- [ ] Do not weight Student loss by LP, residual gap, Buyer reward, `q_teacher`, or teacher terminal score. The skill gate acts only by constructing `q_h_tilde`, never as a second scalar loss weight.
- [ ] Verify finite loss and non-zero gradients on target tokens.
- [ ] Verify a closed/invalid target produces a graph-connected zero loss.
- [ ] Verify text, single-tool-call, and parallel-tool-call targets.

Student objective:

\[
\mathcal{L}_{\mathrm{Student}}
=
\frac{1}{L}
\sum_{j=1}^{L}
D_{\mathrm{KL}}\!\left(\widetilde q_{h,j}\,\|\,p_{S,j}\right).
\]

This TODO includes skill-contrast sharpening as part of the baseline Teacher-target construction. It does not add Student-side GRPO, AgentOPSD credit reshaping, or outcome-weighted distillation; those remain separate experiments after the curriculum baseline is correct.

Files:

```text
coevo/training/gated_gkd.py
scripts/train_student_full.sh
scripts/train_student_smoke.sh
tests/test_student_training.py
```

---

# P0 — Orchestration and checkpoint rotation

## 16. Preserve the pre-update Student during policy refresh

The current controller stops the old policy and replaces it with the new checkpoint before Buyer training. Change refresh semantics as follows.

- [ ] Before Student training, record `student_checkpoint_before`.
- [ ] After Student training, resolve `student_checkpoint_after`.
- [ ] Start `student_checkpoint_before` as `policy_previous`.
- [ ] Start `student_checkpoint_after` as the current `policy`.
- [ ] Run preflight against both endpoints.
- [ ] Train Buyer only after both endpoints are healthy.
- [ ] Stop `policy_previous` only after Buyer training and manifest commit complete.
- [ ] On failure, restore the previous current-policy service and preserve the manifest’s checkpoint pair.
- [ ] On resume, refuse to recompute Buyer reward with a different checkpoint pair.

Required manifest fields:

```text
round
student_checkpoint_before
student_checkpoint_after
previous_policy_endpoint
current_policy_endpoint
teacher_checkpoint
buyer_checkpoint_before
buyer_checkpoint_after
reward_name
reward_formula_version
target_schema_version
tokenizer_hash
status
phase
```

Files:

```text
scripts/run_coevolution.py
scripts/run_full_trainset.py
scripts/run_online_policy_trainset.py
scripts/start_role.sh
scripts/stop_role.sh
```

---

# P1 — Data schemas and observability

## 17. Replace intervention artifacts with stage-progress artifacts

- [ ] Version `trajectories.jsonl`, `student_gkd.jsonl`, `buyer_grpo.jsonl`, `summary.json`, and manifests.
- [ ] Remove obsolete fields:

  ```text
  intervention_advantage
  intervention_advantages
  mean_intervention_advantage
  turn_intervention_advantages
  student_value
  original_branch_messages
  opd_utility_gain
  opd_utility_source
  predicted_takeover_gain
  ```

- [ ] Add:

  ```text
  previous_gap
  current_gap
  learning_progress
  positive_learning_progress
  residual_gate
  teacher_quality
  stage_progress_reward
  checkpoint_previous
  checkpoint_current
  raw_teacher_target_hash
  teacher_target_hash
  target_token_count
  teacher_support_mass
  unhinted_reference_checkpoint
  skill_contrast_scores
  skill_gate_values
  sharpening_temperatures
  raw_teacher_entropy
  sharpened_teacher_entropy
  ```

- [ ] Refuse to merge artifacts with incompatible schema versions, tokenizers, target constructions, or checkpoint ordering.
- [ ] Add a CLI that prints one decision’s state, Teacher action, raw and sharpened target hashes, skill contrast, gate/temperature, entropy change, previous/current gaps, LP, residual gate, and final reward.

---

## 18. Log the curriculum state, not only scalar reward

Per round, report:

- [ ] distribution of `d_previous`;
- [ ] distribution of `d_current`;
- [ ] raw LP distribution, including negative values;
- [ ] fraction with positive LP;
- [ ] fraction suppressed as already mastered;
- [ ] Teacher-quality distribution;
- [ ] skill-contrast and gate-value distributions;
- [ ] raw-versus-sharpened Teacher entropy;
- [ ] fraction of target tokens near `g=0` and near `g=1`;
- [ ] invalid and unscorable rollout rates;
- [ ] all-zero GRPO group rate;
- [ ] generated domain/task/skill distribution;
- [ ] state-hash or skill-bucket diversity;
- [ ] Student held-out task score under the fixed evaluator;
- [ ] Student held-out Teacher gap;
- [ ] token, rollout, and wall-clock cost.

Classify generated examples into four diagnostic regions:

```text
mastered:    current gap <= residual threshold
progressing: current gap > threshold and LP > 0
stagnant:    current gap > threshold and LP ~= 0
regressing:  LP < 0
```

Do not convert these labels into additional online reward terms in the first baseline.

---

# P1 — Tests

## 19. Unit tests for the reward mathematics

- [ ] `S_previous == S_current` gives `LP == 0`.
- [ ] Current Student closer to Teacher gives positive LP.
- [ ] Current Student farther from Teacher gives negative raw LP and zero positive-LP reward.
- [ ] `current_gap <= threshold` closes the residual gate.
- [ ] Invalid trajectory gives zero reward.
- [ ] Zero Teacher quality gives zero reward.
- [ ] Identical Teacher target hash is used for previous/current scoring.
- [ ] Token-length normalization removes trivial sequence-length scaling.
- [ ] Top-k support alignment is deterministic.
- [ ] Missing support mass or token mismatch fails closed.
- [ ] A fully zero GRPO group remains zero after normalization.
- [ ] Identical hinted/unhinted logits give `contrast == 0`, `g == 0`, `T == 1`, and `q_h_tilde == q_h`.
- [ ] Increasing skill contrast monotonically lowers temperature within bounds.
- [ ] Sharpening never increases entropy for a fixed non-uniform hinted distribution.
- [ ] Sharpening preserves the raw Teacher token ordering and argmax.
- [ ] `T_min <= T <= 1` for every active token.
- [ ] Prompt tokens receive no skill gate or Student loss.
- [ ] The cached sharpened target hash is identical in Student loss and both gap calculations.

## 20. Scheduler tests

- [ ] Every complete Student action created by the Buyer rollout can produce one `TeacherTargetRecord`.
- [ ] Student errors do not invalidate Buyer rollout validity.
- [ ] Buyer-originated invalid transitions do invalidate the rollout.
- [ ] Final Buyer action is executed and scored.
- [ ] Previous/current endpoint failures are surfaced explicitly.
- [ ] Scheduler never calls the deleted intervention-advantage path.
- [ ] Scheduler emits only Buyer tokens in the response loss mask.

## 21. Real two-checkpoint smoke

- [ ] Collect one real τ² Student dataset with `S_0` and `B_0`.
- [ ] Perform one real Student update to obtain `S_1`.
- [ ] Serve `S_0` as previous and `S_1` as current.
- [ ] Run one Buyer GRPO group that generates at least one natural Student decision.
- [ ] Generate and cache one raw hinted Teacher target and its same-checkpoint unhinted reference.
- [ ] Compute the skill contrast, gate, temperature, and sharpened Teacher target.
- [ ] Score the same sharpened target under `S_0` and `S_1`.
- [ ] Compute finite `d_0`, `d_1`, LP, and Buyer reward.
- [ ] Execute one Buyer optimizer step.
- [ ] Save and reload both Student and Buyer checkpoints.
- [ ] Run fixed-user held-out Student evaluation.

---

# P1 — Experiments

## 22. Minimum scientific comparison

Keep budgets matched across:

- Student initialization;
- Buyer initialization;
- task split;
- Student optimizer steps;
- Buyer optimizer steps;
- Teacher-token budget;
- environment interactions;
- random seeds;
- evaluator and checkpoint-selection rule.

Required conditions:

- [ ] **Fixed Buyer:** no environment learning.
- [ ] **Current-gap only:** reward proportional to remaining `d_current`.
- [ ] **Stage learning progress:** `[d_previous - d_current]_+` with residual gate.
- [ ] **No residual gate:** positive LP without mastered-state suppression.
- [ ] **No Teacher-quality term:** set `q_teacher = 1`.
- [ ] **Exploration ablation:** `beta = 0` versus the bounded count bonus.
- [ ] **Teacher-target ablation:** raw hinted target versus fixed-temperature sharpening versus skill-contrast-gated sharpening.

The deleted intervention and shadow-OPD systems may be evaluated from an archived commit for historical comparison, but they should not remain as maintained runtime branches in the new implementation.

Primary result:

- [ ] held-out Student task score versus generated tokens, Teacher tokens, environment interactions, and wall-clock cost.

Secondary results:

- [ ] movement of the generated task/skill distribution across stages;
- [ ] held-out Teacher-gap reduction;
- [ ] percentage of generated examples in mastered/progressing/stagnant/regressing regions;
- [ ] Buyer collapse or repetition rate;
- [ ] generalization under a fixed user simulator not used during Buyer training.

---

## 23. Test the central hypothesis directly

Main hypothesis:

> A GRPO-trained environment can learn a stage-conditioned data curriculum from the decrease in privileged Teacher discrepancy across consecutive Student checkpoints. It should shift probability mass toward regions with positive recent learning progress and a remaining Teacher gap, without requiring a shadow Student update or a learned utility critic. Skill contrast is used only to construct the sharpened Teacher target and is not an additional environment-reward term.

- [ ] Show that the Buyer’s generated distribution changes when the `(previous, current)` Student pair changes.
- [ ] Show that swapping the checkpoint pair reverses raw LP signs as expected.
- [ ] Show that a later Student suppresses examples already mastered through the residual gate.
- [ ] Show that persistently unlearned regions receive near-zero reward because LP remains near zero.
- [ ] Show that an early-stage Buyer replayed against a later checkpoint pair receives a different reward landscape.
- [ ] Report cases where LP reward fails to predict later Student gains; do not add new online components without a separate preregistered experiment.

---

# P0 — Documentation cleanup

## 24. Rewrite the repository documentation around one objective

- [ ] Rewrite the root `README.md` in academic English.
- [ ] Rewrite `correctability_coevolution/FULL_INFRA.md`.
- [ ] Update `SETUP.md`, `.env.example`, and the research-design document.
- [ ] Remove all claims that the environment searches for “correctable difficulty” through `V_T - V_S`.
- [ ] Remove all claims that successful takeover proves a sample is worth learning.
- [ ] Remove shadow OPD and utility-critic diagrams and configuration.
- [ ] Remove intervention-weighted Student-loss descriptions.
- [ ] State the central mechanism exactly:

  > The privileged skill defines a skill-contrast-gated sharpened Teacher target. Consecutive Student checkpoints define stage learning progress relative to that same target. The GRPO environment learns where to generate the next round of data.

- [ ] Clearly distinguish:
  - Student supervision: skill-contrast-gated sharpened Teacher distribution;
  - environment reward: cross-checkpoint Teacher-gap decrease;
  - task evaluation: fixed τ² verifier under an independent user simulator.
- [ ] Record the exact Git commit and upstream dependency revisions for every experiment.

---

# Definition of done

The first stage-conditioned curriculum baseline is complete only when:

- [ ] shadow OPD and utility critic code are deleted;
- [ ] Buyer reward contains no intervention advantage or hidden fallback;
- [ ] paired Student-vs-Teacher continuation scoring is removed from the main path;
- [ ] Student data collection no longer requires positive intervention advantage;
- [ ] one detached raw hinted distribution and same-checkpoint unhinted reference construct a reproducible skill-contrast gate;
- [ ] one detached sharpened Teacher target is shared by Student distillation and both checkpoint comparisons;
- [ ] zero skill contrast exactly recovers ordinary hinted-Teacher distillation;
- [ ] active sharpening reduces target entropy without changing the Teacher argmax;
- [ ] previous and current Students score identical target tokens under the unhinted view;
- [ ] `LP = d_previous - d_current` is the implemented Buyer signal;
- [ ] mastered examples are suppressed by one simple residual gate;
- [ ] Buyer GRPO runs with both Student checkpoints online;
- [ ] Student training remains independent skill-gated Teacher distillation;
- [ ] a real two-checkpoint smoke completes Student update, checkpoint rotation, Buyer GRPO, and fixed-user evaluation;
- [ ] documentation, configuration, code, datasets, and tests use the same terminology and equations;
- [ ] held-out Student improvement, not Buyer reward, is the primary reported outcome.

---

## Explicit non-goals for this baseline

- [ ] No shadow Student update.
- [ ] No disposable LoRA probe.
- [ ] No learned teaching-value critic.
- [ ] No per-sample causal attribution of checkpoint improvement.
- [ ] No intervention-advantage Buyer reward.
- [ ] No product of many handcrafted learnability heuristics.
- [ ] No skill-contrast term added directly to Buyer reward.
- [ ] No additional Student sample gate or scalar token-loss weight derived from skill contrast; the contrast only changes target temperature.
- [ ] No Student-side GRPO or AgentOPSD integration in the first implementation.
- [ ] No claim that positive LP proves an individual example caused learning.
- [ ] No private Teacher hint in public interaction history.
