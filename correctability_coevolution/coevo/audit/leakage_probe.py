from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

from json_repair import repair_json
from openai import OpenAI

from coevo.config import ModelEndpoint


@dataclass(frozen=True)
class LeakageProbeExample:
    state_id: str
    state: Any
    action: Any
    hidden: Any
    matched: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeakageProbeReport:
    conditional_auc: float
    state_only_auc: float
    conditional_advantage: float
    examples: int
    positive_examples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_matched_shuffled_examples(
    rows: Sequence[Mapping[str, Any]], seed: int = 42
) -> list[LeakageProbeExample]:
    """Create matched positives and deranged-hidden negatives."""

    if len(rows) < 2:
        raise ValueError("leakage probing requires at least two states")
    rng = random.Random(seed)
    order = list(range(len(rows)))
    groups = [str(row.get("group_id", row.get("state_id", index))) for index, row in enumerate(rows)]
    for _ in range(100):
        rng.shuffle(order)
        if all(groups[source] != groups[target] for source, target in enumerate(order)):
            break
    else:
        raise ValueError("cannot construct negatives from different hidden-state groups")
    examples: list[LeakageProbeExample] = []
    for index, row in enumerate(rows):
        state_id = str(row.get("state_id", index))
        examples.append(
            LeakageProbeExample(
                state_id=state_id,
                state=row["state"],
                action=row["action"],
                hidden=row["hidden"],
                matched=True,
            )
        )
        examples.append(
            LeakageProbeExample(
                state_id=state_id,
                state=row["state"],
                action=row["action"],
                hidden=rows[order[index]]["hidden"],
                matched=False,
            )
        )
    return examples


def roc_auc(labels: Sequence[bool | int], scores: Sequence[float]) -> float:
    """Mann-Whitney AUC with exact tie handling and no sklearn dependency."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("AUC labels and scores must be non-empty and aligned")
    positives = [float(score) for label, score in zip(labels, scores) if bool(label)]
    negatives = [float(score) for label, score in zip(labels, scores) if not bool(label)]
    if not positives or not negatives:
        raise ValueError("AUC requires both positive and negative examples")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


class NLLeakageJudge:
    """V0 conditional probe: estimate whether a hidden record matches (s, a)."""

    def __init__(self, endpoint: ModelEndpoint, retries: int = 3):
        self.endpoint = endpoint
        self.retries = retries
        base_url = endpoint.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self.client = OpenAI(base_url=base_url, api_key=endpoint.api_key)

    @staticmethod
    def response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "conditional_leakage_score",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "probability_matched": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["probability_matched", "reason"],
                    "additionalProperties": False,
                },
            },
        }

    def score(self, example: LeakageProbeExample, *, include_action: bool = True) -> float:
        payload = {"public_state": example.state, "candidate_hidden": example.hidden}
        if include_action:
            payload["teacher_action"] = example.action
        last_error: Exception | None = None
        for _ in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.endpoint.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Estimate whether the candidate hidden record is the "
                                "one paired with the public state"
                                + (" and teacher action." if include_action else ".")
                                + " Use only statistical compatibility; return a probability."
                            ),
                        },
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0,
                    max_tokens=self.endpoint.max_tokens,
                    response_format=self.response_format(),
                )
                content = response.choices[0].message.content or ""
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    result = repair_json(content, return_objects=True)
                score = float(result["probability_matched"])
                if not 0 <= score <= 1:
                    raise ValueError("probe probability is outside [0, 1]")
                return score
            except Exception as error:
                last_error = error
        raise RuntimeError(
            f"leakage judge failed after {self.retries} attempts: {last_error}"
        ) from last_error


class ConditionalLeakageProbe:
    def __init__(
        self,
        conditional_score: Callable[[LeakageProbeExample], float],
        state_only_score: Callable[[LeakageProbeExample], float],
    ):
        self.conditional_score = conditional_score
        self.state_only_score = state_only_score

    def evaluate(self, examples: Iterable[LeakageProbeExample]) -> LeakageProbeReport:
        rows = list(examples)
        labels = [row.matched for row in rows]
        conditional = [float(self.conditional_score(row)) for row in rows]
        state_only = [float(self.state_only_score(row)) for row in rows]
        conditional_auc = roc_auc(labels, conditional)
        state_only_auc = roc_auc(labels, state_only)
        return LeakageProbeReport(
            conditional_auc=conditional_auc,
            state_only_auc=state_only_auc,
            conditional_advantage=conditional_auc - state_only_auc,
            examples=len(rows),
            positive_examples=sum(labels),
        )
