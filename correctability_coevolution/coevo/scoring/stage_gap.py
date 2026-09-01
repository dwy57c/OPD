"""Deprecated three-view LP scorer retained for historical reproduction only."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any

import requests
import torch

from coevo.artifacts import assistant_action_hash, canonical_hash
from coevo.config import InfraConfig, ModelEndpoint
from coevo.rewards.stage_progress import (
    StageProgressResult,
    mean_forward_kl,
    score_stage_progress,
)
from coevo.scoring.skill_contrast import (
    SkillContrastConfig,
    construct_skill_contrast_target,
)
from coevo.scoring.teacher_target import (
    TEACHER_TARGET_SCHEMA_VERSION,
    TeacherTargetRecord,
)


@dataclass(frozen=True)
class TargetTokenization:
    full_input_ids: tuple[int, ...]
    target_input_ids: tuple[int, ...]
    target_start: int


@dataclass(frozen=True)
class SparseTargetView:
    target_input_ids: tuple[int, ...]
    topk_logprobs: tuple[tuple[float, ...], ...]
    topk_token_ids: tuple[tuple[int, ...], ...]
    support_mass: tuple[float, ...]


@dataclass(frozen=True)
class StageGapScore:
    teacher_target: TeacherTargetRecord
    current_view: SparseTargetView
    progress: StageProgressResult
    checkpoint_teacher_anchor: str
    checkpoint_previous: str
    checkpoint_current: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.progress.to_dict(),
            "checkpoint_teacher_anchor": self.checkpoint_teacher_anchor,
            "checkpoint_previous": self.checkpoint_previous,
            "checkpoint_current": self.checkpoint_current,
            "raw_teacher_target_hash": self.teacher_target.raw_teacher_target_hash,
            "teacher_target_hash": self.teacher_target.teacher_target_hash,
            "target_token_count": sum(self.teacher_target.target_loss_mask),
            "teacher_support_mass": list(
                self.teacher_target.hinted_support_mass
            ),
            "skill_contrast_scores": list(
                self.teacher_target.skill_contrast_scores
            ),
            "skill_gate_values": list(self.teacher_target.skill_gate_values),
            "sharpening_temperatures": list(
                self.teacher_target.sharpening_temperatures
            ),
            "raw_teacher_entropy": list(
                self.teacher_target.raw_teacher_entropy
            ),
            "sharpened_teacher_entropy": list(
                self.teacher_target.sharpened_teacher_entropy
            ),
        }


class PromptLogprobClient:
    """Bounded-retry vLLM prompt-logprob client with no scoring fallback."""

    def __init__(self, *, topk: int = 20, timeout: float = 120.0, retries: int = 3):
        if topk < 1:
            raise ValueError("topk must be positive")
        if retries < 1:
            raise ValueError("retries must be positive")
        self.topk = topk
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.trust_env = False

    def fetch(
        self, endpoint: ModelEndpoint, input_ids: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        headers = {}
        if endpoint.api_key and endpoint.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        base_url = endpoint.base_url.rstrip("/")
        completion_url = (
            base_url + "/completions"
            if base_url.endswith("/v1")
            else base_url + "/v1/completions"
        )
        last_error: Exception | None = None
        for _ in range(self.retries):
            try:
                response = self.session.post(
                    completion_url,
                    headers=headers,
                    json={
                        "model": endpoint.model,
                        "prompt": list(input_ids),
                        "max_tokens": 1,
                        "temperature": 0,
                        "prompt_logprobs": self.topk,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                raw = response.json()["choices"][0]["prompt_logprobs"]
                return self._parse(raw, input_ids)
            except Exception as error:
                last_error = error
        raise RuntimeError(
            "policy prompt-logprob request failed after bounded retries: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _parse(self, raw, input_ids: tuple[int, ...]):
        if len(raw) != len(input_ids):
            raise RuntimeError(
                "policy prompt-logprob response length mismatch: "
                f"expected={len(input_ids)}, actual={len(raw)}"
            )
        width = self.topk + 1
        rows_logprobs = []
        rows_token_ids = []
        for position_index, position in enumerate(raw[1:], start=1):
            if not isinstance(position, dict):
                raise RuntimeError(
                    f"missing prompt log-probabilities at position {position_index}"
                )
            items = [
                (int(token_id), float(payload["logprob"]))
                for token_id, payload in position.items()
                if math.isfinite(float(payload["logprob"]))
            ]
            items.sort(key=lambda item: (-item[1], item[0]))
            selected = items[: self.topk]
            actual = int(input_ids[position_index])
            actual_item = next((item for item in items if item[0] == actual), None)
            if actual_item is None:
                raise RuntimeError(
                    f"actual prompt token {actual} is absent at position {position_index}"
                )
            if actual not in {item[0] for item in selected}:
                selected.append(actual_item)
            padding = width - len(selected)
            rows_token_ids.append(
                [item[0] for item in selected] + [0] * max(0, padding)
            )
            rows_logprobs.append(
                [item[1] for item in selected]
                + [float("-inf")] * max(0, padding)
            )
        return (
            torch.tensor(rows_logprobs, dtype=torch.float32),
            torch.tensor(rows_token_ids, dtype=torch.long),
        )


class TeacherTargetBuilder:
    """Build and cache one raw/sharpened target from the configured anchor."""

    def __init__(
        self,
        config: InfraConfig,
        *,
        tokenizer=None,
        client: PromptLogprobClient | None = None,
    ):
        self.config = config
        if tokenizer is None:
            from transformers import AutoTokenizer

            model_path = Path(os.environ["COEVO_POLICY_PATH"]).expanduser()
            if not model_path.is_dir():
                raise FileNotFoundError(
                    "COEVO_POLICY_PATH must be a local tokenizer directory"
                )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True
            )
        self.tokenizer = tokenizer
        self.client = client or PromptLogprobClient(
            topk=config.teacher_gap_topk,
            timeout=float(os.getenv("COEVO_POLICY_SCORE_TIMEOUT", "120")),
            retries=int(os.getenv("COEVO_POLICY_SCORE_RETRIES", "3")),
        )
        self.skill_config = SkillContrastConfig(
            low=config.skill_gate_low,
            high=config.skill_gate_high,
            minimum_temperature=config.skill_sharpen_t_min,
            minimum_support_mass=config.teacher_gap_min_support_mass,
            epsilon=config.skill_gate_eps,
            sharpen_enabled=config.sharpen_enabled,
        )
        self._target_cache: dict[tuple[str, ...], TeacherTargetRecord] = {}
        self._view_cache: dict[tuple[str, ...], SparseTargetView] = {}
        self._lock = Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        self.scoring_failures = 0
        self._gate_config_hash = canonical_hash(asdict(self.skill_config))
        self._tokenizer_hash = canonical_hash(
            {
                "tokenizer_id": config.tokenizer_id,
                "chat_template": getattr(tokenizer, "chat_template", None),
            }
        )

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "scoring_failures": self.scoring_failures,
        }

    def tokenize_target(
        self,
        messages: list[dict],
        tool_schemas: list[dict] | None = None,
    ) -> TargetTokenization:
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("policy scoring messages must end in an assistant target")
        kwargs = {"enable_thinking": False}
        if tool_schemas:
            # Qwen renders the available functions into the system turn.  The
            # exact same schemas must therefore be present for both Teacher
            # scoring and Swift's later cached-target tokenization.
            kwargs["tools"] = deepcopy(tool_schemas)
        tool_target = bool(messages[-1].get("tool_calls"))
        try:
            full = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                # Transformers truncates at the final message's textual content
                # when continue_final_message=True. A pure tool-call action has
                # empty text, so that mode silently drops the entire tool call.
                continue_final_message=not tool_target,
                **kwargs,
            )
            prefix = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True, **kwargs
            )
        except TypeError as error:
            # Older tokenizer implementations may not expose Qwen's
            # enable_thinking extension.  Retrying without that flag is safe;
            # silently dropping tools is not, because it changes the sequence
            # whose logits are distilled.
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("enable_thinking", None)
            try:
                full = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    continue_final_message=not tool_target,
                    **fallback_kwargs,
                )
                prefix = self.tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                    **fallback_kwargs,
                )
            except TypeError:
                if tool_schemas:
                    raise TypeError(
                        "policy tokenizer must support tool schemas for "
                        "tools-aware Teacher scoring"
                    ) from error
                raise
        full = self._input_ids(full)
        prefix = self._input_ids(prefix)
        full_ids = tuple(int(item) for item in full)
        prefix_ids = tuple(int(item) for item in prefix)
        if tool_target and hasattr(self.tokenizer, "encode"):
            # The ordinary Qwen rendering keeps the turn terminator. Swift's
            # add_eos=False training labels exclude it, as does the continued
            # text-action rendering above. Strip exactly that template suffix
            # so text and tool macro-actions share one target convention.
            terminator = tuple(
                int(item)
                for item in self.tokenizer.encode(
                    "<|im_end|>\n", add_special_tokens=False
                )
            )
            if not terminator or full_ids[-len(terminator) :] != terminator:
                raise ValueError(
                    "tool-call target does not end in the expected Qwen turn terminator"
                )
            full_ids = full_ids[: -len(terminator)]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "chat template does not preserve the assistant generation prefix"
            )
        target = full_ids[len(prefix_ids) :]
        if not target:
            raise ValueError("assistant target tokenization is empty")
        return TargetTokenization(full_ids, target, len(prefix_ids))

    @staticmethod
    def _input_ids(value):
        if isinstance(value, dict) or hasattr(value, "keys"):
            value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], (list, tuple)):
            if len(value) != 1:
                raise ValueError("chat template returned an unexpected batch")
            value = value[0]
        return value

    @staticmethod
    def _slice_view(
        tokenized: TargetTokenization,
        logprobs: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> SparseTargetView:
        if logprobs.shape != token_ids.shape or logprobs.ndim != 2:
            raise ValueError("policy sparse tensors must have shape [sequence, topk]")
        expected = len(tokenized.full_input_ids) - 1
        if logprobs.shape[0] != expected:
            raise ValueError(
                "policy scores do not align with tokenized prompt: "
                f"expected={expected}, actual={logprobs.shape[0]}"
            )
        start = tokenized.target_start - 1
        stop = start + len(tokenized.target_input_ids)
        if start < 0 or stop > logprobs.shape[0]:
            raise ValueError("target span is outside returned prompt scores")
        row_logs = []
        row_ids = []
        support_mass = []
        for row_index, actual_target in enumerate(tokenized.target_input_ids):
            logs = logprobs[start + row_index].tolist()
            ids = token_ids[start + row_index].tolist()
            unique: dict[int, float] = {}
            order = []
            for token_id, logprob in zip(ids, logs):
                if not math.isfinite(float(logprob)):
                    continue
                token_id = int(token_id)
                if token_id not in unique:
                    order.append(token_id)
                    unique[token_id] = float(logprob)
                else:
                    unique[token_id] = max(unique[token_id], float(logprob))
            if int(actual_target) not in unique:
                raise ValueError(
                    f"actual target token missing from sparse view at row {row_index}"
                )
            row_ids.append(tuple(order))
            row_logs.append(tuple(unique[token_id] for token_id in order))
            mass = sum(math.exp(unique[token_id]) for token_id in order)
            if mass > 1 + 1e-5:
                raise ValueError("sparse support probability mass exceeds one")
            support_mass.append(min(1.0, mass))
        return SparseTargetView(
            tokenized.target_input_ids,
            tuple(row_logs),
            tuple(row_ids),
            tuple(support_mass),
        )

    def score_view(
        self,
        *,
        endpoint: ModelEndpoint,
        checkpoint_id: str,
        state_hash: str,
        action_hash: str,
        information_view: str,
        messages: list[dict],
        tool_schemas: list[dict] | None = None,
    ) -> SparseTargetView:
        tool_schema_hash = canonical_hash(tool_schemas or [])
        key = (
            checkpoint_id,
            state_hash,
            action_hash,
            information_view,
            self._tokenizer_hash,
            tool_schema_hash,
        )
        with self._lock:
            cached = self._view_cache.get(key)
        if cached is not None:
            return cached
        tokenized = self.tokenize_target(messages, tool_schemas)
        logprobs, token_ids = self.client.fetch(endpoint, tokenized.full_input_ids)
        result = self._slice_view(tokenized, logprobs, token_ids)
        with self._lock:
            self._view_cache[key] = result
        return result

    def build(
        self,
        *,
        student_visible_messages: list[dict],
        hinted_teacher_messages: list[dict],
        teacher_action: dict,
        state_hash: str,
        teacher_hint_hash: str,
        tool_schemas: list[dict] | None = None,
    ) -> TeacherTargetRecord:
        action_hash = assistant_action_hash(teacher_action)
        teacher_checkpoint = self.config.teacher_anchor_checkpoint
        key = (
            teacher_checkpoint,
            teacher_checkpoint,
            state_hash,
            action_hash,
            teacher_hint_hash,
            self._tokenizer_hash,
            canonical_hash(tool_schemas or []),
            self._gate_config_hash,
        )
        with self._lock:
            cached = self._target_cache.get(key)
            if cached is not None:
                self.cache_hits += 1
        if cached is not None:
            return cached
        with self._lock:
            self.cache_misses += 1
        try:
            requests_to_score = {
                "hinted": dict(
                    endpoint=self.config.teacher,
                    checkpoint_id=teacher_checkpoint,
                    state_hash=state_hash,
                    action_hash=action_hash,
                    information_view="teacher_anchor_hinted",
                    messages=hinted_teacher_messages,
                    tool_schemas=tool_schemas,
                ),
                "unhinted": dict(
                    endpoint=self.config.teacher,
                    checkpoint_id=teacher_checkpoint,
                    state_hash=state_hash,
                    action_hash=action_hash,
                    information_view="teacher_anchor_unhinted",
                    messages=student_visible_messages,
                    tool_schemas=tool_schemas,
                ),
            }
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    name: executor.submit(self.score_view, **arguments)
                    for name, arguments in requests_to_score.items()
                }
                views = {name: future.result() for name, future in futures.items()}
            hinted = views["hinted"]
            unhinted = views["unhinted"]
            if hinted.target_input_ids != unhinted.target_input_ids:
                raise ValueError("hinted and unhinted target token IDs do not align")
            contrast = construct_skill_contrast_target(
                hinted_topk_logprobs=hinted.topk_logprobs,
                hinted_topk_token_ids=hinted.topk_token_ids,
                unhinted_topk_logprobs=unhinted.topk_logprobs,
                unhinted_topk_token_ids=unhinted.topk_token_ids,
                target_token_ids=hinted.target_input_ids,
                config=self.skill_config,
            )
            raw_hash = canonical_hash(
                TeacherTargetRecord.raw_hash_payload(
                    teacher_checkpoint=teacher_checkpoint,
                    teacher_hint_hash=teacher_hint_hash,
                    state_hash=state_hash,
                    teacher_action_hash=action_hash,
                    target_token_ids=hinted.target_input_ids,
                    hinted_topk_logprobs=hinted.topk_logprobs,
                    hinted_topk_token_ids=hinted.topk_token_ids,
                )
            )
            target_hash = canonical_hash(
                TeacherTargetRecord.target_hash_payload(
                    raw_teacher_target_hash=raw_hash,
                    sharpened_topk_logprobs=contrast.sharpened_topk_logprobs,
                    sharpened_topk_token_ids=contrast.sharpened_topk_token_ids,
                    sharpening_temperatures=contrast.sharpening_temperatures,
                )
            )
            record = TeacherTargetRecord(
                schema_version=TEACHER_TARGET_SCHEMA_VERSION,
                state_hash=state_hash,
                teacher_action_hash=action_hash,
                raw_teacher_target_hash=raw_hash,
                teacher_target_hash=target_hash,
                teacher_checkpoint=teacher_checkpoint,
                teacher_hint_hash=teacher_hint_hash,
                student_visible_messages=tuple(deepcopy(student_visible_messages)),
                hinted_teacher_messages=tuple(deepcopy(hinted_teacher_messages)),
                teacher_action=deepcopy(teacher_action),
                target_token_ids=hinted.target_input_ids,
                target_loss_mask=tuple(1 for _ in hinted.target_input_ids),
                hinted_topk_logprobs=hinted.topk_logprobs,
                hinted_topk_token_ids=hinted.topk_token_ids,
                hinted_support_mass=hinted.support_mass,
                unhinted_reference_checkpoint=teacher_checkpoint,
                unhinted_reference_topk_logprobs=unhinted.topk_logprobs,
                unhinted_reference_topk_token_ids=unhinted.topk_token_ids,
                unhinted_reference_support_mass=unhinted.support_mass,
                skill_contrast_scores=contrast.skill_contrast_scores,
                skill_gate_values=contrast.skill_gate_values,
                sharpening_temperatures=contrast.sharpening_temperatures,
                sharpened_topk_logprobs=contrast.sharpened_topk_logprobs,
                sharpened_topk_token_ids=contrast.sharpened_topk_token_ids,
                sharpened_support_mass=contrast.sharpened_support_mass,
                raw_teacher_entropy=contrast.raw_teacher_entropy,
                sharpened_teacher_entropy=contrast.sharpened_teacher_entropy,
            )
            record.assert_hashes()
        except Exception:
            with self._lock:
                self.scoring_failures += 1
            raise
        with self._lock:
            self._target_cache[key] = record
        return record


class StageGapScorer:
    """Score an S_k+skill target under unhinted S_k and S_(k+1)."""

    def __init__(
        self,
        config: InfraConfig,
        *,
        target_builder: TeacherTargetBuilder | None = None,
    ):
        self.config = config
        self.target_builder = target_builder or TeacherTargetBuilder(config)

    def validate_checkpoint_pair(self) -> None:
        if self.config.previous_policy is None:
            raise ValueError("Buyer stage scoring requires a previous-policy endpoint")
        previous = self.config.previous_policy_checkpoint
        current = self.config.current_policy_checkpoint
        if not previous or not current:
            raise ValueError("Buyer stage scoring requires both checkpoint identities")
        if previous == current:
            raise ValueError("previous and current checkpoints must be distinct")
        if self.config.teacher_anchor != "previous":
            raise ValueError(
                "Buyer stage scoring requires teacher_anchor='previous' so the "
                "Teacher target remains fixed at S_k+skill"
            )
        if self.config.teacher_anchor_checkpoint != previous:
            raise ValueError(
                "Teacher anchor checkpoint must equal the previous Student checkpoint"
            )

    def score(
        self,
        *,
        student_visible_messages: list[dict],
        hinted_teacher_messages: list[dict],
        teacher_action: dict,
        state_hash: str,
        teacher_hint_hash: str,
        tool_schemas: list[dict] | None = None,
    ) -> StageGapScore:
        self.validate_checkpoint_pair()
        if self.config.previous_policy is None:  # guarded above
            raise ValueError("previous-policy endpoint is not configured")
        record = self.target_builder.build(
            student_visible_messages=student_visible_messages,
            hinted_teacher_messages=hinted_teacher_messages,
            teacher_action=teacher_action,
            state_hash=state_hash,
            teacher_hint_hash=teacher_hint_hash,
            tool_schemas=tool_schemas,
        )
        previous_checkpoint = (
            self.config.previous_policy_checkpoint
            or self.config.previous_policy.model
        )
        current_checkpoint = (
            self.config.current_policy_checkpoint or self.config.policy.model
        )
        if record.teacher_checkpoint != previous_checkpoint:
            raise ValueError(
                "Teacher target must be anchored to the previous Student checkpoint"
            )
        current = self.target_builder.score_view(
            endpoint=self.config.policy,
            checkpoint_id=current_checkpoint,
            state_hash=state_hash,
            action_hash=record.teacher_action_hash,
            information_view="current_unhinted_against_previous_teacher",
            messages=student_visible_messages,
            tool_schemas=tool_schemas,
        )
        if current.target_input_ids != record.target_token_ids:
            raise ValueError("current Student target token IDs do not align")
        previous_gap = mean_forward_kl(
            teacher_logprobs=record.sharpened_topk_logprobs,
            teacher_token_ids=record.sharpened_topk_token_ids,
            student_logprobs=record.unhinted_reference_topk_logprobs,
            student_token_ids=record.unhinted_reference_topk_token_ids,
            target_token_ids=record.target_token_ids,
            target_loss_mask=record.target_loss_mask,
            epsilon=self.config.teacher_gap_eps,
        )
        current_gap = mean_forward_kl(
            teacher_logprobs=record.sharpened_topk_logprobs,
            teacher_token_ids=record.sharpened_topk_token_ids,
            student_logprobs=current.topk_logprobs,
            student_token_ids=current.topk_token_ids,
            target_token_ids=record.target_token_ids,
            target_loss_mask=record.target_loss_mask,
            epsilon=self.config.teacher_gap_eps,
        )
        progress = score_stage_progress(
            previous_gap=previous_gap,
            current_gap=current_gap,
        )
        return StageGapScore(
            teacher_target=record,
            current_view=current,
            progress=progress,
            checkpoint_teacher_anchor=record.teacher_checkpoint,
            checkpoint_previous=previous_checkpoint,
            checkpoint_current=current_checkpoint,
        )
