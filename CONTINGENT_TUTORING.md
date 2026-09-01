# Contingent Tutoring: scientific and implementation contract

## 1. Question and claims

For public state `s`, hidden task information `h`, hint constructor `phi`, and
self-Teacher `q(a | s, phi(h))`, forward distillation can only teach an
unprivileged Student the marginal behavior

```text
p*(a | s) = E_{h | s}[q(a | s, phi(h))].
```

A good hint makes that marginal a deployable policy. A bad hint makes the
Teacher solve a different game whose privileged optimum cannot be imitated.

The falsifiable claims are:

1. Increasing instance-fact dosage suppresses clarification and lookup while
   increasing unsupported factual assertions.
2. A fact-to-procedure contract and the minimum sufficient per-state dose `h*`
   preserve usefulness with less behavior copying.
3. A hinter should evolve with the Student using one cheap local reward and an
   independently refreshed behavior discriminator.

## 2. Three measurable conditions

| Condition | Failure mode | Estimator |
|---|---|---|
| Content: Teacher behavior should reveal little additional information about `h` once `s` is known | marginal policy becomes confident guessing | conditional probe AUC minus s-only AUC; behavior audit |
| Dose: hinted target must remain within the Student's absorbable region | hindsight support gap | frozen-checkpoint L0–L3 pass@k and `h*` |
| State distribution: training states must match deployment and retain hidden-state diversity | memorization or irrelevant curriculum | fixed longitudinal panel plus held-out twin scenarios |

The operational content criterion is public support: a claim is safe only when
it follows from public history, domain policy, or observed tool output.

## 3. Experiments

### E1 — static audit

Generate one Teacher macro-action for each fixed public state and each hint
level. Report hint length, clarification rate, lookup rate, unsupported-claim
rate, conditional leakage AUC, s-only AUC, and their difference. If L3 targets
are not behaviorally dirtier than L2 targets, stop and revise the contract
before training.

### E2 — equal-budget dose response

Train L1, L2, and L3 arms with the same states, seeds, base checkpoint, and
reference active-token budget. L0 is the untouched base checkpoint. Disable
skill-contrast temperature sharpening. Evaluate held-out task success and all
three behavior endpoints with three seeds.

### E3 — minimum sufficient dose

At a frozen checkpoint, estimate L0–L3 success for each scenario with `k`
independent trials. Select the smallest level meeting the sufficiency threshold.
Classify scenarios as mastered, frontier, scaffolded, or out-of-reach. Schedule
frontier/scaffolded states primarily, with explicit review and exploration mass.
Use held-out twin scenarios for promotion and retirement decisions.

The controller records non-monotone empirical dose curves. It must not assume
that L3 always dominates L2.

### E4 — one alternating hinter objective

For a fixed public state `s`, fixed current Student `theta`, fixed standard
action trajectory `tau*`, and candidate hint `h`, the only hinter GRPO reward is

```text
useful(h) = log p_theta(tau* | s, h) - log p_theta(tau* | s)
reward(h) = useful(h) - lambda_copy * D_t(s, operation_theta(s,h), h)
                       - beta_len * tokens(h)
```

The usefulness term requires exactly two teacher-forced Student scoring views
and no environment rollout. Separately, each GRPO candidate hint produces one
actual frozen-Student macro-action as its operation record. The copying
discriminator has the same base-model scale as the Student plus one scalar head.
It sees `(public state, operation record, candidate hint)` and is trained with
the pairwise loss

```text
-log sigmoid(score(true used hint) - score(same-state unused hint)).
```

The state and operation record are identical within a pair; only the hint
changes. The discriminator is initialized from the current Student with a fresh
score head and retrained on new behaviors every round, then frozen during hinter
GRPO. The copying penalty is the mean pairwise probability that the true hint
beats the other hints sampled for the same state.

Neither pass@k nor observed post-distillation improvement appears in this
reward. Hint length is charged directly, so as the Student improves a long hint
must earn enough additional log-probability gain to pay for every extra token.
This is the mechanism expected to produce fading.

The only alternating loop uses one pass@k panel per round. A hinter candidate
created in round `t` is accepted or rejected after the Student has actually
distilled from it at the start of round `t+1`:

```text
distill Student for N steps with the previous round's hinter candidate
freeze the new Student and measure pass@k once
use that measurement to accept/rollback the prior Student+hinter update
use the same measurement to schedule the four curriculum bands
collect fresh behavior/hint pairs and retrain the discriminator
run a few hinter GRPO steps with usefulness - copying - length
carry the new hinter candidate into the next Student segment for real acceptance
```

pass@k is used only by the scheduler and acceptance fuse. The measured
cross-round distillation gain is logged and used for rollback but is never an
optimizer reward.

Three gates protect the copying signal:

- explicit-answer-copy hints must be identified with high accuracy;
- useless generic hints must stay near chance, otherwise the pair construction
  is leaking task identity;
- every few rounds an independently initialized same-size discriminator is
  trained and must agree with the active discriminator.

Hinter drift is constrained only by GRPO reference KL (`beta`); no second anchor
model is trained.

## 4. Measurement validation

Before behavioral claims are used downstream:

- manually label at least 200 examples per detector and report agreement and
  Cohen's kappa;
- report fresh-probe versus training-probe AUC and correlation with public
  grounding judgments;
- repeat verifier decisions under the same seed and manually audit scenarios
  that remain indefinitely in the frontier band.

## 5. Stop rules

- E1 premise absent: revise the contracts before any dosage training.
- Detector agreement inadequate: pause downstream training and repair measurement.
- E2 flat: publish signal-calibration findings rather than claiming toxicity.
- Hinter GRPO unstable or unable to beat the fixed controller: fall back
  to the phenomenon plus h* controller paper.

## 6. Legacy boundary

The consecutive-checkpoint Buyer reward

```text
max(KL(q_tilde || S_previous) - KL(q_tilde || S_current), 0)
```

is archived. Its sparse-support epsilon and independent vLLM endpoints can
produce differences larger than the observed reward, and its positive-only
feedback does not implement a variance-preserving curriculum. New measurements
do not import this reward or start the previous-policy service.
