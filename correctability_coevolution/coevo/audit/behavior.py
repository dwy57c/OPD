from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Iterable, Mapping, Protocol, Sequence

from json_repair import repair_json
from openai import OpenAI

from coevo.config import ModelEndpoint


_QUESTION = re.compile(r"[?？]|\b(?:could|would|can|may|do|did|is|are|what|which|when|where|who|how)\b", re.I)
_CLARIFY = re.compile(
    r"\b(?:please (?:provide|confirm|clarify)|could you|would you|can you|"
    r"what (?:is|was|date|number|amount)|which (?:order|booking|flight|account)|"
    r"do you mean|to confirm|may i have|need (?:your|the))\b",
    re.I,
)
_LOOKUP_TOOL = re.compile(
    r"(?:^|_)(?:get|search|lookup|list|find|retrieve|read|check|fetch|query)(?:_|$)",
    re.I,
)
_MUTATION_TOOL = re.compile(
    r"(?:^|_)(?:book|cancel|update|create|delete|modify|change|send|transfer)(?:_|$)",
    re.I,
)
_FACT_SPAN = re.compile(
    r"(?:[$€£¥]\s*\d+(?:\.\d+)?|\b\d{1,4}(?:[-/:]\d{1,4})+\b|"
    r"\b[A-Z]{1,3}\d{2,8}\b|\b\d{5,}\b)",
    re.I,
)


def _row(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json", exclude_none=True)
    raise TypeError(f"unsupported message type: {type(message).__name__}")


def _tool_names(message: Mapping[str, Any]) -> tuple[str, ...]:
    result = []
    for call in message.get("tool_calls") or []:
        name = call.get("name")
        if name is None:
            function = call.get("function") or {}
            name = function.get("name")
        if name:
            result.append(str(name))
    return tuple(result)


def _public_text(messages: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for message in messages:
        if message.get("role") in {"system", "user", "tool"}:
            parts.append(str(message.get("content") or ""))
    return "\n".join(parts)


class GroundingJudge(Protocol):
    def __call__(
        self, public_context: Sequence[Mapping[str, Any]], action: Mapping[str, Any]
    ) -> "GroundingJudgment | list[str]": ...


@dataclass(frozen=True)
class GroundingJudgment:
    factual_assertions: int
    ungrounded_assertions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.factual_assertions < len(self.ungrounded_assertions):
            raise ValueError("ungrounded claims cannot exceed all factual claims")


@dataclass(frozen=True)
class BehaviorAction:
    index: int
    clarifying: bool
    lookup: bool
    tool_names: tuple[str, ...]
    factual_assertions: int
    ungrounded_assertions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tool_names"] = list(self.tool_names)
        value["ungrounded_assertions"] = list(self.ungrounded_assertions)
        return value


@dataclass(frozen=True)
class BehaviorReport:
    decisions: int
    clarifying_decisions: int
    lookup_decisions: int
    factual_assertions: int
    ungrounded_assertion_count: int
    actions: tuple[BehaviorAction, ...]

    @property
    def clarification_rate(self) -> float:
        return self.clarifying_decisions / self.decisions if self.decisions else 0.0

    @property
    def lookup_rate(self) -> float:
        return self.lookup_decisions / self.decisions if self.decisions else 0.0

    @property
    def ungrounded_assertion_rate(self) -> float:
        return (
            self.ungrounded_assertion_count / self.factual_assertions
            if self.factual_assertions
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "clarifying_decisions": self.clarifying_decisions,
            "lookup_decisions": self.lookup_decisions,
            "factual_assertions": self.factual_assertions,
            "ungrounded_assertion_count": self.ungrounded_assertion_count,
            "clarification_rate": self.clarification_rate,
            "lookup_rate": self.lookup_rate,
            "ungrounded_assertion_rate": self.ungrounded_assertion_rate,
            "actions": [action.to_dict() for action in self.actions],
        }


def heuristic_ungrounded_assertions(
    public_context: Sequence[Mapping[str, Any]], action: Mapping[str, Any]
) -> list[str]:
    """Conservative first pass: novel instance-like literals are unsupported."""

    content = str(action.get("content") or "")
    evidence = _public_text(public_context).casefold()
    unsupported = []
    for match in _FACT_SPAN.finditer(content):
        claim = match.group(0).strip()
        if claim.casefold() not in evidence:
            unsupported.append(claim)
    return list(dict.fromkeys(unsupported))


def ungrounded_assertions(
    public_context: Sequence[Mapping[str, Any]],
    action: Mapping[str, Any],
    judge: GroundingJudge | None = None,
) -> list[str]:
    heuristic = heuristic_ungrounded_assertions(public_context, action)
    if judge is None:
        return heuristic
    judged = judge(public_context, action)
    values = (
        judged.ungrounded_assertions
        if isinstance(judged, GroundingJudgment)
        else judged
    )
    return list(dict.fromkeys([*heuristic, *values]))


class OpenAIGroundingJudge:
    """NL-judge fallback with bounded retries and a strict response schema."""

    def __init__(self, endpoint: ModelEndpoint, retries: int = 3):
        if retries < 1:
            raise ValueError("retries must be positive")
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
                "name": "public_grounding_audit",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "grounded": {"type": "boolean"},
                                    "evidence": {"type": "string"},
                                },
                                "required": ["text", "grounded", "evidence"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["claims"],
                    "additionalProperties": False,
                },
            },
        }

    def __call__(
        self, public_context: Sequence[Mapping[str, Any]], action: Mapping[str, Any]
    ) -> GroundingJudgment:
        payload = {
            "public_history_and_tool_results": list(public_context),
            "assistant_action": dict(action),
        }
        last_error: Exception | None = None
        for _ in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.endpoint.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Extract externally checkable factual claims in the "
                                "assistant action. Mark a claim grounded only when it "
                                "follows from the supplied public history, policy, or "
                                "tool results. Do not use hidden task knowledge."
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
                claims = result.get("claims") if isinstance(result, dict) else None
                if not isinstance(claims, list):
                    raise ValueError("grounding judge omitted claims")
                ungrounded = [
                    str(claim.get("text") or "").strip()
                    for claim in claims
                    if isinstance(claim, dict)
                    and claim.get("grounded") is False
                    and str(claim.get("text") or "").strip()
                ]
                return GroundingJudgment(len(claims), tuple(ungrounded))
            except Exception as error:
                last_error = error
        raise RuntimeError(
            f"grounding judge failed after {self.retries} attempts: {last_error}"
        ) from last_error


class BehaviorAuditor:
    def __init__(self, grounding_judge: GroundingJudge | None = None):
        self.grounding_judge = grounding_judge

    def analyze(self, messages: Iterable[Any]) -> BehaviorReport:
        rows = [_row(message) for message in messages]
        actions: list[BehaviorAction] = []
        for index, action in enumerate(rows):
            if action.get("role") != "assistant":
                continue
            content = str(action.get("content") or "").strip()
            names = _tool_names(action)
            if not content and not names:
                continue
            clarifying = bool(content and _QUESTION.search(content) and _CLARIFY.search(content))
            lookup = any(_LOOKUP_TOOL.search(name) for name in names)
            if names and not lookup and not any(_MUTATION_TOOL.search(name) for name in names):
                lookup = True
            factual = len(_FACT_SPAN.findall(content))
            heuristic = heuristic_ungrounded_assertions(rows[:index], action)
            unsupported = list(heuristic)
            if self.grounding_judge is not None:
                judged = self.grounding_judge(rows[:index], action)
                if isinstance(judged, GroundingJudgment):
                    factual = max(factual, judged.factual_assertions)
                    judged_unsupported = judged.ungrounded_assertions
                else:
                    judged_unsupported = judged
                unsupported = list(
                    dict.fromkeys([*unsupported, *judged_unsupported])
                )
            factual = max(factual, len(unsupported))
            actions.append(
                BehaviorAction(
                    index=index,
                    clarifying=clarifying,
                    lookup=lookup,
                    tool_names=names,
                    factual_assertions=factual,
                    ungrounded_assertions=tuple(unsupported),
                )
            )
        return BehaviorReport(
            decisions=len(actions),
            clarifying_decisions=sum(action.clarifying for action in actions),
            lookup_decisions=sum(action.lookup for action in actions),
            factual_assertions=sum(action.factual_assertions for action in actions),
            ungrounded_assertion_count=sum(
                len(action.ungrounded_assertions) for action in actions
            ),
            actions=tuple(actions),
        )
