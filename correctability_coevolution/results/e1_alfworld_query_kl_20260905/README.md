# Fixed-hint query dependence experiment

Reanalysis of the three saved ALFWorld mug-heating sessions from
`e1_alfworld_glm53flash_oracle3_20260905` on repository base `2999c60`.
The nine GLM-5.3-Flash hints, task states, and expert action tokens were held
fixed. No external model calls or new hint generation were needed. This
experiment does not change the training reward.

At each expert action-token prefix, compare the same Qwen3-4B weights under:

    Q = p_theta(next token | system, hint, query, action prefix)
    R = p_theta(next token | system, hint, action prefix)
    query_KL = sum_v Q(v) * (log Q(v) - log R(v))

Query means the exact input used in the preceding E1: goal, last six history
messages, current observation, and admissible command list. The hint-only view
removes that whole user message. Both views use the same non-thinking assistant
generation prefix and exactly the same teacher-forced action tokens, checked
against the previous E1 traces. Only action content is scored, without EOS.
The model input includes target tokens only up to the preceding token; no future
action or observation is supplied.

KL is summed over all 151,936 vocabulary tokens, without top-k approximation or
clipping. Model inference uses BF16 with FP32 log-softmax and reductions, via
PyTorch 2.10.0 / Transformers 5.5.3 / SDPA on one A100. Means are formed within
actions, then across turns, then equally across sessions. These are mean
conditional token KLs on an expert-state panel, not KL of autonomous rollouts.
The first action token is also reported separately to expose prefix effects.

| Hint | Mean query KL (nats/token) | Mean first-token KL | Initial-state KL |
|---|---:|---:|---:|
| None, control | 6.758936 | 3.403179 | 0.671753 |
| L1 | 6.197160 | 8.395065 | 5.169088 |
| L2 | 6.273030 | 7.669009 | 10.462669 |
| L3 | 3.342131 | 5.489667 | 3.507177 |

| Scene | L2 mean KL | L3 mean KL | L3 minus L2 |
|---|---:|---:|---:|
| Cabinet | 4.393317 | 3.051256 | -1.342061 |
| CoffeeMachine | 8.004090 | 3.031915 | -4.972174 |
| SideTable | 6.421683 | 3.943222 | -2.478461 |

L3 is lower than L2 in all three whole-session comparisons (46.72% lower on
average), and in all three initial-state comparisons. The reverse KL and JSD
also have lower overall means for L3. This supports the exploratory hypothesis
that these L3 hints replace more of the query information than these L2 hints.

It does not establish a copying detector. There are only three distinct games
of one task family and no independent repeated seeds: the earlier extraction
stored seed labels without applying them to the environment. In the SideTable
scene, the whole-session first-token KL is higher for L3 than L2. After pickup,
the mean first-token KL is also higher for L3 (13.678760 vs 11.119252), consistent
with action selection still depending on state. The low initial-state KL of
the no-hint control (0.671753) is itself a counterexample to treating every low
query KL as evidence of copying. Content, hint length, and instruction style
remain confounded across the three dose levels.

Validation: 428 action rows (107 turns times four levels), all original target
IDs aligned, no missing/invalid hints. Known-distribution and identity checks
verify KL direction and JSD bounds. Input file SHA256 values are in summary.json;
token_rows.jsonl contains individual divergences and target log probabilities.
The original E1 used vLLM; compared with its saved target probabilities, this
Transformers run has mean absolute differences of 0.0933 nats (with query) and
0.1082 nats (hint only). All comparisons in this experiment use the same new
backend, not differences across the two backends.

Reproduce in the configured container from correctability_coevolution:

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python scripts/measure_alfworld_query_kl.py \
  --input-dir artifacts/e1_alfworld_glm53flash_oracle3_20260905 \
  --output-dir artifacts/e1_alfworld_query_kl_reproduction \
  --model /models/policy
```

The model process exited and released the GPU after writing the results.
