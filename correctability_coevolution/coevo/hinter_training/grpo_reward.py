from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from swift.rewards import ORM

from coevo.artifacts import assistant_action_hash, canonical_hash
from coevo.config import InfraConfig
from coevo.hints import hint_fact_leaks
from coevo.models.hinted_teacher import format_teacher_system_prompt_with_hint
from coevo.scoring.stage_gap import SparseTargetView, TeacherTargetBuilder


@dataclass(frozen=True)
class HinterRewardConfig:
    """Weights for the three-view analytical hinter objective."""

    copying_weight: float = 1.0
    dose_weight: float = 1.0
    length_weight: float = 0.002
    token_clip: float = 5.0
    dose_bandwidth: float = 0.05
    max_hint_tokens: int = 192
    rule_leak_floor: float = 1.0

    def __post_init__(self) -> None:
        if min(self.copying_weight, self.dose_weight, self.length_weight) < 0:
            raise ValueError("hinter reward weights must be non-negative")
        if not math.isfinite(self.token_clip) or self.token_clip <= 0:
            raise ValueError("token_clip must be finite and positive")
        if not math.isfinite(self.dose_bandwidth) or self.dose_bandwidth < 0:
            raise ValueError("dose_bandwidth must be finite and non-negative")
        if self.max_hint_tokens < 1:
            raise ValueError("max_hint_tokens must be positive")
        if self.rule_leak_floor <= 0:
            raise ValueError("rule_leak_floor must be positive")

    @classmethod
    def from_env(cls) -> "HinterRewardConfig":
        return cls(
            copying_weight=float(os.getenv("COEVO_HINTER_COPY_WEIGHT", "1.0")),
            dose_weight=float(os.getenv("COEVO_HINTER_DOSE_WEIGHT", "1.0")),
            length_weight=float(os.getenv("COEVO_HINTER_LENGTH_WEIGHT", "0.002")),
            token_clip=float(os.getenv("COEVO_HINTER_TOKEN_CLIP", "5.0")),
            dose_bandwidth=float(
                os.getenv("COEVO_HINTER_DOSE_BANDWIDTH", "0.05")
            ),
            max_hint_tokens=int(os.getenv("COEVO_HINTER_MAX_HINT_TOKENS", "192")),
            rule_leak_floor=float(
                os.getenv("COEVO_HINTER_RULE_LEAK_FLOOR", "1.0")
            ),
        )


@dataclass(frozen=True)
class TeacherForcedProbabilityTrace:
    state_hash: str
    standard_action_hash: str
    target_token_ids: tuple[int, ...]
    unhinted_actual_logprobs: tuple[float, ...]
    hinted_actual_logprobs: tuple[float, ...]
    hint_only_actual_logprobs: tuple[float, ...]
    token_lifts: tuple[float, ...]
    token_copies: tuple[float, ...]
    token_dose_kls: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {
            key: list(item) if isinstance(item, tuple) else item
            for key, item in value.items()
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherForcedProbabilityTrace":
        return cls(
            state_hash=str(value["state_hash"]),
            standard_action_hash=str(value["standard_action_hash"]),
            target_token_ids=tuple(int(item) for item in value["target_token_ids"]),
            unhinted_actual_logprobs=tuple(
                float(item) for item in value["unhinted_actual_logprobs"]
            ),
            hinted_actual_logprobs=tuple(
                float(item) for item in value["hinted_actual_logprobs"]
            ),
            hint_only_actual_logprobs=tuple(
                float(item) for item in value["hint_only_actual_logprobs"]
            ),
            token_lifts=tuple(float(item) for item in value["token_lifts"]),
            token_copies=tuple(float(item) for item in value["token_copies"]),
            token_dose_kls=tuple(float(item) for item in value["token_dose_kls"]),
        )


@dataclass(frozen=True)
class TeacherForcedUsefulness:
    """Three-view token signals for one fixed standard action trajectory."""

    unhinted_log_probability: float
    hinted_log_probability: float
    hint_only_log_probability: float
    mean_lift: float
    mean_copy: float
    mean_dose_kl: float
    target_token_count: int
    probability_trace: TeacherForcedProbabilityTrace

    @property
    def log_probability_gain(self) -> float:
        return self.hinted_log_probability - self.unhinted_log_probability

    @property
    def per_token_gain(self) -> float:
        return self.mean_lift

    @property
    def transferable_mass(self) -> float:
        return sum(
            max(lift - copy, 0.0)
            for lift, copy in zip(
                self.probability_trace.token_lifts,
                self.probability_trace.token_copies,
            )
        )

    @property
    def copy_mass(self) -> float:
        return sum(self.probability_trace.token_copies)

    @property
    def copy_fraction(self) -> float:
        denominator = self.copy_mass + self.transferable_mass
        return self.copy_mass / denominator if denominator > 0 else 0.0

    @property
    def transferable_fraction(self) -> float:
        denominator = self.copy_mass + self.transferable_mass
        return self.transferable_mass / denominator if denominator > 0 else 0.0

    def dose_excess(self, bandwidth: float) -> float:
        return max(0.0, self.mean_dose_kl - float(bandwidth))

    def to_dict(self) -> dict[str, Any]:
        return {
            "unhinted_log_probability": self.unhinted_log_probability,
            "hinted_log_probability": self.hinted_log_probability,
            "hint_only_log_probability": self.hint_only_log_probability,
            "log_probability_gain": self.log_probability_gain,
            "mean_lift": self.mean_lift,
            "mean_copy": self.mean_copy,
            "mean_dose_kl": self.mean_dose_kl,
            "copy_fraction": self.copy_fraction,
            "transferable_fraction": self.transferable_fraction,
            "target_token_count": self.target_token_count,
            "probability_trace": self.probability_trace.to_dict(),
        }


@dataclass(frozen=True)
class HinterRewardBreakdown:
    mean_lift: float
    mean_copy: float
    dose: float
    hint_tokens: int
    copying_weight: float
    dose_weight: float
    length_weight: float
    rule_leaks: tuple[str, ...] = ()
    rule_leak_floor: float = 1.0

    @property
    def copying_penalty(self) -> float:
        return self.copying_weight * self.mean_copy

    @property
    def dose_penalty(self) -> float:
        return self.dose_weight * self.dose

    @property
    def length_penalty(self) -> float:
        return self.length_weight * self.hint_tokens

    @property
    def reward(self) -> float:
        base = (
            self.mean_lift
            - self.copying_penalty
            - self.dose_penalty
            - self.length_penalty
        )
        if self.rule_leaks:
            return min(base, -self.rule_leak_floor)
        return base

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rule_leaks"] = list(self.rule_leaks)
        value["copying_penalty"] = self.copying_penalty
        value["dose_penalty"] = self.dose_penalty
        value["length_penalty"] = self.length_penalty
        value["reward"] = self.reward
        return value


def score_hinter_hint(
    *,
    mean_lift: float,
    mean_copy: float,
    mean_dose_kl: float,
    hint_tokens: int,
    config: HinterRewardConfig,
    rule_leaks: Sequence[str] = (),
) -> HinterRewardBreakdown:
    if mean_copy < 0 or mean_dose_kl < 0:
        raise ValueError("copy and dose KL signals must be non-negative")
    if hint_tokens < 1:
        raise ValueError("a hint must contain at least one token")
    if hint_tokens > config.max_hint_tokens:
        raise ValueError("candidate hint exceeds the configured hard token cap")
    return HinterRewardBreakdown(
        mean_lift=float(mean_lift),
        mean_copy=float(mean_copy),
        dose=max(0.0, float(mean_dose_kl) - config.dose_bandwidth),
        hint_tokens=int(hint_tokens),
        copying_weight=config.copying_weight,
        dose_weight=config.dose_weight,
        length_weight=config.length_weight,
        rule_leaks=tuple(str(value) for value in rule_leaks),
        rule_leak_floor=config.rule_leak_floor,
    )


def calibrate_copying_weight(
    *, l3_mean_lift: float, l3_mean_copy: float, target_margin: float = 0.0
) -> float:
    """Choose lambda so the E1 L3 anchor is neutral or worse before other costs."""

    if l3_mean_copy <= 0:
        raise ValueError("the E1 L3 anchor must have positive analytical copy")
    if target_margin < 0:
        raise ValueError("target_margin must be non-negative")
    return (max(0.0, float(l3_mean_lift)) + target_margin) / float(l3_mean_copy)


def _actual_logprobs(view: SparseTargetView) -> tuple[float, ...]:
    values = []
    for actual, token_ids, logprobs in zip(
        view.target_input_ids, view.topk_token_ids, view.topk_logprobs
    ):
        lookup = {
            int(token_id): float(logprob)
            for token_id, logprob in zip(token_ids, logprobs)
        }
        if int(actual) not in lookup:
            raise ValueError(
                "actual standard-trajectory token is absent from Student support"
            )
        values.append(lookup[int(actual)])
    return tuple(values)


def _distribution(view: SparseTargetView, index: int) -> dict[int, float]:
    return {
        int(token_id): math.exp(float(logprob))
        for token_id, logprob in zip(
            view.topk_token_ids[index], view.topk_logprobs[index]
        )
        if math.isfinite(float(logprob))
    }


def _coarse_token_kl(
    hinted: SparseTargetView, unhinted: SparseTargetView
) -> tuple[float, ...]:
    """Stable lower-bound KL on shared explicit support plus one tail bucket."""

    if hinted.target_input_ids != unhinted.target_input_ids:
        raise ValueError("dose views must share target token IDs")
    values = []
    for index in range(len(hinted.target_input_ids)):
        q = _distribution(hinted, index)
        p = _distribution(unhinted, index)
        shared = sorted(set(q) & set(p))
        q_values = [q[token_id] for token_id in shared]
        p_values = [p[token_id] for token_id in shared]
        q_values.append(max(0.0, 1.0 - sum(q_values)))
        p_values.append(max(0.0, 1.0 - sum(p_values)))
        kl = sum(
            q_value * (math.log(q_value) - math.log(max(p_value, 1e-12)))
            for q_value, p_value in zip(q_values, p_values)
            if q_value > 0
        )
        values.append(max(0.0, kl))
    return tuple(values)


def _clip(value: float, limit: float) -> float:
    return min(limit, max(-limit, float(value)))


class TeacherForcedUsefulnessScorer:
    """Score tau* under state, state+hint, and hint-only views."""

    def __init__(
        self,
        config: InfraConfig,
        *,
        target_builder: TeacherTargetBuilder | None = None,
        token_clip: float | None = None,
    ):
        self.config = config
        self.target_builder = target_builder or TeacherTargetBuilder(config)
        self.token_clip = float(
            token_clip
            if token_clip is not None
            else os.getenv("COEVO_HINTER_TOKEN_CLIP", "5.0")
        )
        if self.token_clip <= 0:
            raise ValueError("token_clip must be positive")

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
        hint_hash = canonical_hash({"hint": hint.strip()})
        hinted_system = format_teacher_system_prompt_with_hint(
            str(messages[0].get("content") or ""), {"plan": hint.strip()}
        )
        hinted_messages = deepcopy(messages)
        hinted_messages[0]["content"] = hinted_system
        hint_only_messages = [
            {**deepcopy(messages[0]), "content": hinted_system},
            deepcopy(messages[-1]),
        ]
        common = {
            "endpoint": self.config.policy,
            "checkpoint_id": checkpoint,
            "state_hash": state_hash,
            "action_hash": action_hash,
            "tool_schemas": list(tool_schemas or []),
        }
        requests = {
            "unhinted": {
                **common,
                "information_view": "hinter_reward_unhinted",
                "messages": messages,
            },
            "hinted": {
                **common,
                "information_view": f"hinter_reward_hinted:{hint_hash}",
                "messages": hinted_messages,
            },
            "hint_only": {
                **common,
                "information_view": f"hinter_reward_hint_only:{hint_hash}",
                "messages": hint_only_messages,
            },
        }
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                name: executor.submit(self.target_builder.score_view, **arguments)
                for name, arguments in requests.items()
            }
            views = {name: future.result() for name, future in futures.items()}

        unhinted = views["unhinted"]
        hinted = views["hinted"]
        hint_only = views["hint_only"]
        if not (
            hinted.target_input_ids
            == unhinted.target_input_ids
            == hint_only.target_input_ids
        ):
            raise ValueError("all three standard-trajectory tokenizations must align")

        p_logs = _actual_logprobs(unhinted)
        q_logs = _actual_logprobs(hinted)
        h_logs = _actual_logprobs(hint_only)
        lifts = tuple(
            _clip(q_value - p_value, self.token_clip)
            for q_value, p_value in zip(q_logs, p_logs)
        )
        copies = tuple(
            max(0.0, _clip(h_value - p_value, self.token_clip))
            for h_value, p_value in zip(h_logs, p_logs)
        )
        dose_kls = _coarse_token_kl(hinted, unhinted)
        count = len(lifts)
        trace = TeacherForcedProbabilityTrace(
            state_hash=state_hash,
            standard_action_hash=action_hash,
            target_token_ids=hinted.target_input_ids,
            unhinted_actual_logprobs=p_logs,
            hinted_actual_logprobs=q_logs,
            hint_only_actual_logprobs=h_logs,
            token_lifts=lifts,
            token_copies=copies,
            token_dose_kls=dose_kls,
        )
        return TeacherForcedUsefulness(
            unhinted_log_probability=sum(p_logs),
            hinted_log_probability=sum(q_logs),
            hint_only_log_probability=sum(h_logs),
            mean_lift=sum(lifts) / count,
            mean_copy=sum(copies) / count,
            mean_dose_kl=sum(dose_kls) / count,
            target_token_count=count,
            probability_trace=trace,
        )


def validate_hinter_reward_row(row: Mapping[str, Any]) -> None:
    required = ("state_hash", "public_state", "student_visible_messages")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"hinter GRPO row is missing fields: {', '.join(missing)}")
    messages = row["student_visible_messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("student_visible_messages must be a non-empty list")
    if messages[-1].get("role") != "assistant":
        raise ValueError("student_visible_messages must end in the standard action")


class HinterCompositeReward(ORM):
    """mean lift - lambda*mean copy - nu*dose - mu*length."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_config = HinterRewardConfig.from_env()
        self.infra = InfraConfig.from_env()
        self.usefulness = TeacherForcedUsefulnessScorer(
            self.infra, token_clip=self.reward_config.token_clip
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
        if key in {"privileged_context", "fact_audit_context"} and isinstance(
            value, list
        ) and len(value) == count:
            return value[index]
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
        rewards = []
        for index, completion in enumerate(completions):
            hint = self._hint_text(completion)
            row = {
                key: self._item(kwargs, key, index, count)
                for key in (
                    "state_hash",
                    "public_state",
                    "student_visible_messages",
                    "tools",
                    "privileged_context",
                    "fact_audit_context",
                )
            }
            validate_hinter_reward_row(row)
            signals = self.usefulness.score(
                student_visible_messages=row["student_visible_messages"],
                hint=hint,
                state_hash=str(row["state_hash"]),
                tool_schemas=row.get("tools") or [],
            )
            hint_tokens = len(
                self.hinter_tokenizer.encode(hint, add_special_tokens=False)
            )
            privileged = row.get("privileged_context")
            audit_context = row.get("fact_audit_context")
            leak_payload = (
                dict(audit_context) if isinstance(audit_context, Mapping) else {}
            )
            leak_payload.setdefault("available_tools", row.get("tools") or [])
            if isinstance(privileged, Mapping):
                leak_payload.setdefault(
                    "authoritative_oracle_steps",
                    privileged.get("authoritative_oracle_steps", ""),
                )
            rule_leaks = hint_fact_leaks(hint, leak_payload)
            breakdown = score_hinter_hint(
                mean_lift=signals.mean_lift,
                mean_copy=signals.mean_copy,
                mean_dose_kl=signals.mean_dose_kl,
                hint_tokens=hint_tokens,
                config=self.reward_config,
                rule_leaks=rule_leaks,
            )
            rewards.append(breakdown.reward)
            self._record(
                {
                    "state_hash": row["state_hash"],
                    "public_state": row["public_state"],
                    "hint": hint,
                    "signals": signals.to_dict(),
                    "reward": breakdown.to_dict(),
                }
            )
        return rewards
