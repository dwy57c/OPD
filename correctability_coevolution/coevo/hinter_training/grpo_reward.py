from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from openai import OpenAI
import requests
from swift.rewards import ORM

from coevo.artifacts import assistant_action_hash, canonical_hash
from coevo.config import InfraConfig
from coevo.models.hinted_teacher import format_teacher_system_prompt_with_hint
from coevo.scoring.stage_gap import SparseTargetView, TeacherTargetBuilder

from .behavior_discriminator import pairwise_copy_probability
from .discriminator_data import format_discriminator_input


@dataclass(frozen=True)
class HinterRewardConfig:
    copying_weight: float = 1.0
    length_weight: float = 0.01
    max_hint_tokens: int = 192

    def __post_init__(self) -> None:
        if self.copying_weight < 0 or self.length_weight < 0:
            raise ValueError("hinter reward weights must be non-negative")
        if self.max_hint_tokens < 1:
            raise ValueError("max_hint_tokens must be positive")

    @classmethod
    def from_env(cls) -> "HinterRewardConfig":
        return cls(
            copying_weight=float(os.getenv("COEVO_HINTER_COPY_WEIGHT", "1.0")),
            length_weight=float(os.getenv("COEVO_HINTER_LENGTH_WEIGHT", "0.01")),
            max_hint_tokens=int(os.getenv("COEVO_HINTER_MAX_HINT_TOKENS", "192")),
        )


@dataclass(frozen=True)
class TeacherForcedProbabilityTrace:
    state_hash: str
    standard_action_hash: str
    target_token_ids: tuple[int, ...]
    actual_token_logprobs: tuple[float, ...]
    top1_token_ids: tuple[int, ...]
    top1_logprobs: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {key: list(item) if isinstance(item, tuple) else item for key, item in value.items()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherForcedProbabilityTrace":
        return cls(
            state_hash=str(value["state_hash"]),
            standard_action_hash=str(value["standard_action_hash"]),
            target_token_ids=tuple(int(item) for item in value["target_token_ids"]),
            actual_token_logprobs=tuple(
                float(item) for item in value["actual_token_logprobs"]
            ),
            top1_token_ids=tuple(int(item) for item in value["top1_token_ids"]),
            top1_logprobs=tuple(float(item) for item in value["top1_logprobs"]),
        )


@dataclass(frozen=True)
class TeacherForcedUsefulness:
    unhinted_log_probability: float
    hinted_log_probability: float
    log_probability_gain: float
    target_token_count: int
    probability_trace: TeacherForcedProbabilityTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "unhinted_log_probability": self.unhinted_log_probability,
            "hinted_log_probability": self.hinted_log_probability,
            "log_probability_gain": self.log_probability_gain,
            "target_token_count": self.target_token_count,
            "probability_trace": self.probability_trace.to_dict(),
        }


@dataclass(frozen=True)
class HinterRewardBreakdown:
    usefulness: float
    copying_probability: float
    hint_tokens: int
    copying_weight: float
    length_weight: float

    @property
    def copying_penalty(self) -> float:
        return self.copying_weight * self.copying_probability

    @property
    def length_penalty(self) -> float:
        return self.length_weight * self.hint_tokens

    @property
    def reward(self) -> float:
        return self.usefulness - self.copying_penalty - self.length_penalty

    def to_dict(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "copying_penalty": self.copying_penalty,
            "length_penalty": self.length_penalty,
            "reward": self.reward,
        }


def score_hinter_hint(
    *,
    usefulness: float,
    copying_probability: float,
    hint_tokens: int,
    config: HinterRewardConfig,
) -> HinterRewardBreakdown:
    if not 0 <= copying_probability <= 1:
        raise ValueError("copying_probability must be in [0, 1]")
    if hint_tokens < 1:
        raise ValueError("a hint must contain at least one token")
    if hint_tokens > config.max_hint_tokens:
        raise ValueError("candidate hint exceeds the configured hard token cap")
    return HinterRewardBreakdown(
        usefulness=float(usefulness),
        copying_probability=float(copying_probability),
        hint_tokens=int(hint_tokens),
        copying_weight=config.copying_weight,
        length_weight=config.length_weight,
    )


def _actual_logprobs(view: SparseTargetView) -> tuple[float, ...]:
    values = []
    for actual, token_ids, logprobs in zip(
        view.target_input_ids, view.topk_token_ids, view.topk_logprobs
    ):
        lookup = {int(token_id): float(logprob) for token_id, logprob in zip(token_ids, logprobs)}
        if int(actual) not in lookup:
            raise ValueError("actual standard-trajectory token is absent from Student support")
        values.append(lookup[int(actual)])
    return tuple(values)


def _probability_trace(
    view: SparseTargetView, *, state_hash: str, standard_action_hash: str
) -> TeacherForcedProbabilityTrace:
    actual = _actual_logprobs(view)
    top1_ids = []
    top1_logs = []
    for token_ids, logprobs in zip(view.topk_token_ids, view.topk_logprobs):
        index = max(range(len(logprobs)), key=lambda value: float(logprobs[value]))
        top1_ids.append(int(token_ids[index]))
        top1_logs.append(float(logprobs[index]))
    return TeacherForcedProbabilityTrace(
        state_hash=state_hash,
        standard_action_hash=standard_action_hash,
        target_token_ids=tuple(int(value) for value in view.target_input_ids),
        actual_token_logprobs=actual,
        top1_token_ids=tuple(top1_ids),
        top1_logprobs=tuple(top1_logs),
    )


class TeacherForcedUsefulnessScorer:
    """Score one standard action twice under the same frozen current Student."""

    def __init__(
        self,
        config: InfraConfig,
        *,
        target_builder: TeacherTargetBuilder | None = None,
    ):
        self.config = config
        self.target_builder = target_builder or TeacherTargetBuilder(config)

    def score(
        self,
        *,
        student_visible_messages: Sequence[Mapping[str, Any]],
        hint: str,
        state_hash: str,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    ) -> TeacherForcedUsefulness:
        messages = deepcopy(list(student_visible_messages))
        if not messages or messages[0].get("role") != "system":
            raise ValueError("standard trajectory must start with a system message")
        if messages[-1].get("role") != "assistant":
            raise ValueError("standard trajectory must end with an assistant action")
        if not hint.strip():
            raise ValueError("candidate hint must be non-empty")
        action_hash = assistant_action_hash(messages[-1])
        checkpoint = self.config.current_policy_checkpoint or self.config.policy.model
        unhinted = self.target_builder.score_view(
            endpoint=self.config.policy,
            checkpoint_id=checkpoint,
            state_hash=state_hash,
            action_hash=action_hash,
            information_view="hinter_reward_unhinted",
            messages=messages,
            tool_schemas=list(tool_schemas or []),
        )
        hinted_messages = deepcopy(messages)
        hinted_messages[0]["content"] = format_teacher_system_prompt_with_hint(
            str(messages[0].get("content") or ""), {"plan": hint.strip()}
        )
        hint_hash = canonical_hash({"hint": hint.strip()})
        hinted = self.target_builder.score_view(
            endpoint=self.config.policy,
            checkpoint_id=checkpoint,
            state_hash=state_hash,
            action_hash=action_hash,
            information_view=f"hinter_reward_hinted:{hint_hash}",
            messages=hinted_messages,
            tool_schemas=list(tool_schemas or []),
        )
        if hinted.target_input_ids != unhinted.target_input_ids:
            raise ValueError("hinted and unhinted standard-trajectory tokens differ")
        unhinted_logs = _actual_logprobs(unhinted)
        hinted_logs = _actual_logprobs(hinted)
        unhinted_total = sum(unhinted_logs)
        hinted_total = sum(hinted_logs)
        return TeacherForcedUsefulness(
            unhinted_log_probability=unhinted_total,
            hinted_log_probability=hinted_total,
            log_probability_gain=hinted_total - unhinted_total,
            target_token_count=len(hinted_logs),
            probability_trace=_probability_trace(
                hinted, state_hash=state_hash, standard_action_hash=action_hash
            ),
        )


class StudentMacroActionGenerator:
    """Generate one actual frozen-Student operation record for a candidate hint."""

    def __init__(self, config: InfraConfig):
        self.config = config
        base_url = config.policy.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self.client = OpenAI(base_url=base_url, api_key=config.policy.api_key)

    def generate(
        self,
        *,
        student_visible_messages: Sequence[Mapping[str, Any]],
        hint: str,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        messages = deepcopy(list(student_visible_messages[:-1]))
        if not messages or messages[0].get("role") != "system":
            raise ValueError("Student behavior prompt must start with a system message")
        messages[0]["content"] = format_teacher_system_prompt_with_hint(
            str(messages[0].get("content") or ""), {"plan": hint.strip()}
        )
        arguments = {
            "model": self.config.policy.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.config.branch_max_tokens,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            },
        }
        if tool_schemas:
            arguments["tools"] = list(tool_schemas)
        response = self.client.chat.completions.create(**arguments)
        message = response.choices[0].message
        result: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        if not result["content"] and not result.get("tool_calls"):
            raise ValueError("frozen Student produced an empty operation record")
        return result


class BehaviorCopyingDiscriminator:
    """Client for the same-size scalar-head discriminator service."""

    def __init__(self, *, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("behavior discriminator URL is required")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def score_texts(self, texts: Sequence[str]) -> list[float]:
        response = self.session.post(
            self.base_url + "/score",
            json={"inputs": list(texts)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        scores = [float(value) for value in response.json()["scores"]]
        if len(scores) != len(texts):
            raise ValueError("discriminator server returned the wrong score count")
        return scores

    def copy_probability(
        self,
        *,
        public_state: Any,
        student_behavior: Any,
        true_hint: str,
        alternative_hints: Sequence[str],
    ) -> float:
        alternatives = [hint for hint in alternative_hints if hint != true_hint]
        if not alternatives:
            raise ValueError("copying penalty requires a same-state alternative hint")
        texts = [
            format_discriminator_input(
                public_state=public_state,
                student_behavior=student_behavior,
                candidate_hint=hint,
            )
            for hint in [true_hint, *alternatives]
        ]
        scores = self.score_texts(texts)
        return sum(
            pairwise_copy_probability(scores[0], negative)
            for negative in scores[1:]
        ) / (len(scores) - 1)


def validate_hinter_reward_row(row: Mapping[str, Any]) -> None:
    required = (
        "state_hash",
        "public_state",
        "student_visible_messages",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"hinter GRPO row is missing fields: {', '.join(missing)}")
    messages = row["student_visible_messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("student_visible_messages must be a non-empty list")
    if messages[-1].get("role") != "assistant":
        raise ValueError("student_visible_messages must end in the standard action")


class HinterCompositeReward(ORM):
    """GRPO reward = teacher-forced usefulness - copying - token length."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_config = HinterRewardConfig.from_env()
        self.infra = InfraConfig.from_env()
        self.usefulness = TeacherForcedUsefulnessScorer(self.infra)
        self.student_behavior = StudentMacroActionGenerator(self.infra)
        self.discriminator = BehaviorCopyingDiscriminator(
            base_url=os.getenv("COEVO_HINTER_DISCRIMINATOR_URL", ""),
            timeout=float(os.getenv("COEVO_HINTER_DISCRIMINATOR_TIMEOUT", "120")),
        )
        from transformers import AutoTokenizer

        model_path = Path(os.environ["COEVO_HINTER_BASE_MODEL"]).expanduser()
        self.hinter_tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        self.trace_path = os.getenv("COEVO_HINTER_REWARD_TRACE_PATH", "").strip()
        self._trace_lock = Lock()

    @staticmethod
    def _hint_text(completion: Any) -> str:
        if isinstance(completion, str):
            return completion.strip()
        if isinstance(completion, list) and completion:
            item = completion[-1]
            if isinstance(item, Mapping):
                return str(item.get("content") or "").strip()
        if isinstance(completion, Mapping):
            return str(completion.get("content") or "").strip()
        return str(completion).strip()

    @staticmethod
    def _item(kwargs: Mapping[str, Any], key: str, index: int, count: int) -> Any:
        value = kwargs.get(key)
        if (
            key in {"public_state", "student_visible_messages", "tools"}
            and isinstance(value, list)
            and value
            and isinstance(value[0], Mapping)
        ):
            return value
        if isinstance(value, list) and len(value) == count:
            return value[index]
        return value

    def _record(self, row: dict[str, Any]) -> None:
        if not self.trace_path:
            return
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def __call__(self, completions, **kwargs):
        count = len(completions)
        candidates = []
        for index, completion in enumerate(completions):
            hint = self._hint_text(completion)
            row = {
                key: self._item(kwargs, key, index, count)
                for key in (
                    "state_hash",
                    "public_state",
                    "student_visible_messages",
                    "tools",
                )
            }
            validate_hinter_reward_row(row)
            usefulness = self.usefulness.score(
                student_visible_messages=row["student_visible_messages"],
                hint=hint,
                state_hash=str(row["state_hash"]),
                tool_schemas=row.get("tools") or [],
            )
            student_behavior = self.student_behavior.generate(
                student_visible_messages=row["student_visible_messages"],
                hint=hint,
                tool_schemas=row.get("tools") or [],
            )
            hint_tokens = len(
                self.hinter_tokenizer.encode(hint, add_special_tokens=False)
            )
            candidates.append(
                {
                    "row": row,
                    "hint": hint,
                    "usefulness": usefulness,
                    "student_behavior": student_behavior,
                    "hint_tokens": hint_tokens,
                }
            )

        grouped: dict[str, list[int]] = {}
        for index, candidate in enumerate(candidates):
            grouped.setdefault(str(candidate["row"]["state_hash"]), []).append(index)

        rewards = [0.0] * count
        for state_hash, indexes in grouped.items():
            group_hints = [str(candidates[index]["hint"]) for index in indexes]
            if len(set(group_hints)) < 2:
                raise ValueError(
                    f"GRPO state {state_hash!r} needs at least two distinct hints"
                )
            for index in indexes:
                candidate = candidates[index]
                copying = self.discriminator.copy_probability(
                    public_state=candidate["row"]["public_state"],
                    student_behavior=candidate["student_behavior"],
                    true_hint=str(candidate["hint"]),
                    alternative_hints=group_hints,
                )
                usefulness = candidate["usefulness"]
                breakdown = score_hinter_hint(
                    usefulness=usefulness.log_probability_gain,
                    copying_probability=copying,
                    hint_tokens=int(candidate["hint_tokens"]),
                    config=self.reward_config,
                )
                rewards[index] = breakdown.reward
                self._record(
                    {
                        "state_hash": candidate["row"]["state_hash"],
                        "public_state": candidate["row"]["public_state"],
                        "hint": candidate["hint"],
                        "student_behavior": candidate["student_behavior"],
                        "usefulness": usefulness.to_dict(),
                        "reward": breakdown.to_dict(),
                    }
                )
        return rewards
