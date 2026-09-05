# ALFWorld location-fact intervention

Exploratory experiment on Qwen3-4B, based on repository commit 751c973. Three
ALFWorld mug-heating scenes, two actually applied policy/environment seeds
(42 and 43), seven conditions per scene/seed: 42 executed short rollouts and
42 separate elicited grounding questions. Sampling temperature is 0.7 and each
rollout has at most 12 policy actions. No student or hinter was trained.

## Intervention

The true and false conditions differ only in the location string:

    A mug is located at {location}. Find a mug, heat it using a microwave,
    and place it at the destination specified by the task. Act one step
    at a time.

These are controlled templates, not the previous long GLM-generated hints.
True locations were checked against simulator relations and actual visual-text
observations. False locations are cabinets in the same room that can contain
mugs, have not been inspected in the initial public state, and actually contain
no mug. The environment is held unchanged under the false hint. This is a
misinformation intervention, not a claim to have relocated an object.

Cases and both inspection records are saved in cases.json. Hidden simulator
facts and the auditor's false-location inspection never enter policy prompts.
Public prompts contain the complete real branch history and the current
admissible commands. Paired branches verify the same initial state hash. Seeds
are passed to TextWorld and to each policy request; entity-label shuffling is
disabled. The two seeds are correlated repeats within three tasks, not six
independent task samples.

Controls are no hint, procedural hint, a same-meaning reordering of the true
fact sentence, and two public-evidence conditions. The latter replay an actual
visit/open to the true location before testing the true or false hint. Thus
the disclosed-fact pair has identical public observations too.

## Results

All six true/false pairs switched their first navigation from the true source
to the substituted false source. All six paraphrase controls retained the
true-hint first navigation. This is evidence of location dependence, which is
not sufficient by itself to call navigation a hallucination.

The separate grounding question asks which location has been confirmed by
the public observations, with 'unknown' permitted. It is scored against the
actual public observations, not simply against the world's hidden truth.

| Condition | First navigation to true source | First navigation to false source | Unsupported answer to grounding question | Wins within 12 policy steps |
|---|---:|---:|---:|---:|
| No hint | 2/6 | 0/6 | 2/6 | 0/6 |
| Procedural | 0/6 | 0/6 | 0/6 | 0/6 |
| True fact | 6/6 | 0/6 | 6/6 | 6/6 |
| False fact | 0/6 | 6/6 | 6/6 | 0/6 |
| Paraphrase true | 6/6 | 0/6 | 6/6 | 3/6 |
| Public evidence + true fact | 0/6 | 0/6 | 0/6 | 6/6 |
| Public evidence + false fact | 0/6 | 1/6 | 6/6 | 4/6 |

True/false hints both made the elicited answer follow the hinted location in
all six pairs. Even after a real observation exposed the true source, the
false-hint grounding answer still named the false source in all six tests.
This is stronger evidence of unsupported factual adoption than a distribution
distance alone. It is an elicited-answer result, not a spontaneous assertion
made while executing commands.

In the actual rollouts, however, none of the 33 mug pickup attempts lacked
support in the observations or the public command menu. Navigation to an open
surface returns a contents observation and is counted as observation. Every
step is genuinely executed; independent actions are not stitched into a
fictional history. There were 102 other invalid actions in 410 executed policy
steps, so zero unsupported pickups does not mean the policies were reliable.
The false-hint condition returned to an already refuted cabinet three times.

One example (Cabinet scene, seed 42): the false hint sent the agent to cabinet
2, which it opened and found to contain a saltshaker and soap bottle. It later
found and picked up a mug only after visiting the coffee machine. In the
public-evidence/false-hint control, the factual probe answered 'cabinet 2' while
the actual agent began by correctly taking a visibly observed mug from
countertop 1. Answer-grounding and action-grounding are different endpoints.

## Interpretation and limits

The paired test identifies which hidden fact influences decisions and detects
unsupported adoption in a grounding question. It does not establish that the
model skips observation before pickup, nor that distilling it would create
such behavior. The reward has not been changed.

The paraphrase control preserves initial navigation but not full-horizon
success (3/6 vs 6/6), demonstrating that the later rollout is sensitive to
wording. The input command menu can itself provide evidence and reduce
unsupported action opportunities. Success is bounded at 12 policy steps;
public-evidence controls also receive a setup visit/open before that horizon.
Their win rates therefore must not be treated as a fair general capability
comparison. The sample is too small and narrow for a population-level claim.

## Verification and reproduction

verification.json records replay of all 42 branches and all 410 policy steps
without any model calls. Every observation, admissible menu, environment
reward, terminal flag and pickup-grounding label is checked. The scripts also
test that only the location differs in the factual pair and that observing
the contents of a mug is not mistaken for inspecting the surrounding surface.

The first attempt used threads and failed in TextWorld's global parser; its
partial directory is excluded. The reported run uses isolated worker processes.
The environment is loaded from the installed ALFWorld source at /alfworld, via
the swift-seed-alfworld:4.0.1 runtime. The policy is a local vLLM Qwen3-4B server.

```bash
PYTHONHASHSEED=42 ALFWORLD_DATA=/alfworld/data PYTHONPATH=/alfworld \
python scripts/experiment_alfworld_fact_swap.py \
  --sessions results/e1_fact_swap_20260905_v2/input_sessions.jsonl \
  --output-dir artifacts/fact_swap_reproduction \
  --policy-url http://127.0.0.1:8000 --model Qwen3-4B \
  --seeds 42 43 --temperature 0.7 --max-steps 12

PYTHONHASHSEED=42 ALFWORLD_DATA=/alfworld/data PYTHONPATH=/alfworld \
python scripts/verify_alfworld_fact_swap.py artifacts/fact_swap_reproduction
```

GPU4 was released after collecting the branches. All raw prompts can be
reconstructed from the saved initial observations, hints, actual histories,
and before-action menus in cases.json and branches/.
