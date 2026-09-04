# Contingent Tutoring: scientific and implementation contract

## 1. Question and claims

For public state `s`, hidden task information `h`, hint constructor `phi`, and
self-Teacher `q(a | s, phi(h))`, forward distillation teaches the unprivileged
Student the hidden-state marginal of the hinted Teacher. A good hint improves
the Student's game; a bad hint changes the game to one whose privileged optimum
cannot be imitated.

The falsifiable claims are:

1. Instance-fact dosage suppresses information acquisition and increases
   unsupported action commitments in multi-turn agents.
2. Blind procedural hints prevent source-side leakage; Purified OPSD provides
   an orthogonal sink-side control.
3. An open hinter can learn its own contingent dose as the Student changes when
   rewarded for transferable lift and charged for copying, excessive
   distribution shift, and length.

## 2. Fixed ladder for phenomenon experiments

- L1 receives no privileged facts and emits only a general policy reminder.
- L2 receives only the public state and goal. It must teach unbiased evidence
  acquisition and must not use an incomplete, privileged candidate ordering.
- L3 receives structured hidden facts and must state them explicitly.

L1/L2 are not fact-checked after generation because they never received the
answer; only their format contract is checked. L3 and the open hinter are
fact-audited because they did receive privilege. For tau2 the audit detects
dates, amounts, database values, ordinary identifiers, and
mixed capital-and-digit identifiers such as `ZX99AB`. For ALFWorld it checks
the goal object's true receptacle, the destination instance, and unobserved
states against both class and instance aliases.

## 3. Three-view analytical reward

For a fixed current Student, public state `s`, candidate hint `h`, and fixed
standard action trajectory `tau*`, score the same target tokens in parallel:

```text
p_t = p_theta(a*_t | s)
q_t = p_theta(a*_t | s,h)
r_t = p_theta(a*_t | h)       # system prompt + hint, no dialogue history

lift_t = clip(log q_t - log p_t, -c, c)
copy_t = clip(max(log r_t - log p_t, 0), 0, c)
dose   = max(mean_t KL(q_t || p_t) - bandwidth, 0)

R(h) = mean(lift_t) - lambda mean(copy_t) - nu dose - mu tokens(h)
```

The dose estimator is the coarse-grained forward KL on shared explicit sparse
support plus a tail bucket, a stable lower bound that avoids top-k membership
flips dominating the signal. It is an optimizer penalty proxy, not a reported
KL metric or paper claim. Lambda is calibrated from E1's natural L3 hints:
the L3 mean-copy anchor must cancel its positive mean lift before dose and
length costs are added.

Machine-checkable fact or tool-name leakage remains a hard rule gate and caps
the total reward at a negative floor. The optimizer uses no agent rollout,
learned discriminator, pass@k result, or post-distillation gain.

The decomposition also defines E1's plot quantities:

```text
copy mass         = sum_t copy_t
transferable mass = sum_t max(lift_t - copy_t, 0)
```

Report their normalized fractions for L1/L2/L3. A second E1 intervention tests
the open hinter, which really sees privilege: keep the public state and task
fixed, replace the answer with an environment-valid alternative for that same
task, and regenerate. Do not create counterfactuals by borrowing another task's
answer.

## 4. Purified sink control

The E2 sink-side control follows Purified OPSD. With clean base distribution
`P0`, full hinted Teacher `q(s,h)`, and hint-only reference `p(h)`. The
vocabulary residual is centered and soft-clipped before exponentiation:

```text
delta(v) = log q(v | s,h) - log p(v | h)
bounded(v) = c * tanh((delta(v) - mean_v delta(v)) / c)
P_target(v) proportional to P0(v) * exp(bounded(v) / beta).
```

E2 crosses `raw/purified` target operators with the fixed hint levels. This
separates source contract effects, sink purification effects, and their
combination under identical states, seeds, and active-token budgets.

## 5. Emergent open hinter

The trained hinter has no externally assigned dose. Every sample is marked
`sample_hint_level=HINTER`; h* chooses scenarios only. Mastered/L0 rows are
excluded so zero-gradient examples cannot consume the budget.

The serving and GRPO prompt paths share one canonical serializer. The hinter
receives public state separately and exactly two privileged keys:
`domain_policy` and `authoritative_oracle_steps`. Extra keys fail closed.

The hinter may begin its output with `level: L1`, `level: L2`, or `level: L3`.
This self-report is stored for fading curves but is not required, validated, or
used by the reward. Open-hinter acceptance uses one unlevelled 140-word gate
for format, exact tool names, and hidden facts.

Cold-start SFT is a necessary precondition, not an optional convenience. Its
builder requires low-copy examples from at least two Student checkpoints and
at least two non-zero minimal sufficient levels. Each prompt includes a public
`student_profile` with the checkpoint and measured h* scores, so different
Students do not produce contradictory targets for an identical input. Rows are
selected by h* rather than by raw closed-model volume.

## 6. Alternation and acceptance

One round is:

```text
distill Student N steps using the current hinter candidate
freeze and measure deployment pass@k once
accept or roll back the Student and prior hinter update
reuse that same panel to schedule scenarios
run a few hinter GRPO steps with the current frozen Student
```

Pass@k performs only scheduling and delayed empirical acceptance. It never
enters GRPO. The measured post-distillation gain is logged but never optimized.
The Student and hinter both continue from their accepted previous checkpoints.

## 7. ALFWorld protocol

Use AgentGym/ETO expert trajectories and the canonical train, `valid_seen`, and
`valid_unseen` partitions. Privilege is the rule-expert action sequence plus
the simulator's hidden object-location/state table.

The domain behavior endpoints are:

- unsupported commitment: the first navigation target directly hits the true
  hidden goal-object location;
- information acquisition: a look/open/examine action occurs before pickup.

These replace customer-service clarification and database-query counters while
preserving the same behavioral-toxicity question.

## 8. Stop rules

- E1 premise absent: revise contracts before dosage training.
- Detector/human agreement inadequate: repair measurement before downstream
  claims.
- E2 flat: report signal calibration rather than toxicity.
- Open hinter unstable or unable to beat fixed controls: retain the phenomenon,
  h* scheduler, and source/sink controls as the fallback paper.

The retired Buyer learning-progress and learned behavior-discriminator methods
are not part of the active runtime.
