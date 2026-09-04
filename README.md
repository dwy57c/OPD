# Contingent Tutoring for On-Policy Agent Distillation

This repository studies **what privileged context a self-teacher should receive**
during multi-turn on-policy distillation: what it may contain, how much help a
state needs, and how the hinter should change as the Student learns.

The central hypothesis is specific to interactive agents. Oracle facts in a
private hint can improve the Teacher while simultaneously teaching the Student
to replace clarification and tool use with confident unsupported claims. The
project therefore measures both task success and information-acquisition
behavior.

## Active research path

```text
E1 static hint audit
  -> E2 equal-budget L1/L2/L3 dose response
  -> E3 per-state minimum sufficient dose h*
  -> alternating Student distillation and hinter GRPO
```

The hint ladder is shared by every experiment:

| Level | Private context contract |
|---|---|
| `L0_NONE` | No hint and no hinter API call. Evaluation-only base control. |
| `L1_POLICY` | 15–40 word general policy or safety reminder; oracle steps are removed before the request. |
| `L2_PROCEDURAL` | Blind-written from public state, goal, and public policy; no hidden-fact audit is applied. |
| `L3_ORACLE` | Full natural-language oracle note; supplied instance facts must be stated. |

`COEVO_SHARPEN_ENABLED=0` is the experimental default. Student training retains
the repository's true on-policy GKD path: the Student samples a fresh macro-action
and the frozen hinted Teacher force-decodes those tokens. L0 is not trained in
the dose-response experiment; it is the untouched base checkpoint.

## Main modules

| Path | Purpose |
|---|---|
| `correctability_coevolution/coevo/hints/ladder.py` | L0–L3 contracts and fail-closed validators |
| `correctability_coevolution/coevo/hinter_prompt.py` | one prompt contract shared by GRPO, sampling, and open-hinter collection |
| `correctability_coevolution/coevo/audit/behavior.py` | clarification, lookup, and public-grounding metrics |
| `correctability_coevolution/coevo/audit/leakage_probe.py` | conditional `(s,a,h)` leakage probe and s-only baseline |
| `correctability_coevolution/coevo/curriculum/hstar.py` | frozen-checkpoint pass@k probes, h*, and four-band scheduling |
| `correctability_coevolution/coevo/hinter_training/grpo_reward.py` | three-view analytical lift/copy/dose/length reward |
| `correctability_coevolution/coevo/hinter_training/cold_start.py` | multi-checkpoint, minimal-dose-diverse hinter SFT selection |
| `correctability_coevolution/coevo/hinter_training/alternating_loop.py` | Student N steps, pass@k scheduling, hinter GRPO, acceptance rollback |
| `correctability_coevolution/scripts/audit_hint_ladder.py` | E1 runner |
| `correctability_coevolution/scripts/run_dosage_experiment.py` | E2 equal-active-token runner |
| `correctability_coevolution/scripts/run_dosage_curriculum.py` | E3 controller and baselines |
| `correctability_coevolution/scripts/collect_dosage_curriculum.py` | weighted scenario sampler for unlevelled `HINTER` rows |
| `correctability_coevolution/scripts/run_alternating_rounds.py` | manifest-backed Student/hinter subprocess driver |
| `correctability_coevolution/scripts/evaluate_behavior.py` | behavior audit for evaluation conversations |

The detailed scientific and implementation contract is in
[`CONTINGENT_TUTORING.md`](CONTINGENT_TUTORING.md).

## Quick checks

```bash
cd correctability_coevolution
pytest -q

# E1 from a previously collected public-state pool
python scripts/audit_hint_ladder.py \
  --from-trajectories artifacts/base_states/trajectories.jsonl \
  --output-dir artifacts/e1_hint_audit

# E2; policy and closed-hinter endpoints must already be available
python scripts/run_dosage_experiment.py \
  --output-dir artifacts/e2_dosage \
  --task-ids 1 2 3 \
  --seeds 42 43 44

# E3 frozen-checkpoint probing
python scripts/run_dosage_curriculum.py \
  --output-dir artifacts/e3_hstar \
  --task-ids 1 2 3 \
  --k 8
```

## Archived baseline

The previous Buyer-GRPO / consecutive-checkpoint learning-progress method is
retained for reproducibility but is no longer an active training entry point.
Its code remains under `coevo/rewards/stage_progress.py`,
`coevo/scoring/stage_gap.py`, `coevo/training/buyer_*`, and
`scripts/run_coevolution.py`. Historical design documents live in
`docs/archive/`.

The separate GLM-to-Qwen environment-policy distillation work is preserved in
`correctability_coevolution/ENVIRONMENT_DISTILLATION.md`; it is not the hinter
co-evolution loop.

## Research invariants

- Define every measurement on one frozen Student checkpoint.
- Compare hint levels on the same public states and active-token budget.
- Condition leakage probes on the public state and report advantage over an
  s-only baseline.
- Calibrate analytical copy weight from natural E1 L3 hints and retain the hard rule gate.
- Never mix hint levels or incompatible target contracts in one artifact merge.
- Evaluate the Student without private hints and with a fixed independent user
  simulator.
