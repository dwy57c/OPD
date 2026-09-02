from dataclasses import asdict, dataclass
import hashlib
import json
import time
from typing import Any

from openai import OpenAI
from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    LLMAgent,
    LLMAgentStateType,
    LLMGTAgent,
    SYSTEM_PROMPT,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
)
from tau2.data_model.tasks import Task

from coevo.config import HintEndpoint
from coevo.hints import (
    HintLevel,
    hint_instruction,
    prepare_hint_payload,
    validate_hint_note,
)
from coevo.hinter_prompt import build_hinter_messages


# Compatibility export for callers that benchmark the historical full-oracle dose.
HINTER_INSTRUCTION = hint_instruction(HintLevel.L3_ORACLE)


@dataclass(frozen=True)
class TeacherHintResult:
    hint: dict[str, Any]
    model: str
    latency_ms: int
    sha256: str
    level: str = HintLevel.L3_ORACLE.value
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def oracle_steps_from_task(task: Task) -> str:
    criteria = task.evaluation_criteria
    actions = criteria.actions if criteria is not None else None
    return "\n".join(
        f"[Step {index}] "
        + LLMGTAgent.make_agent_instructions_from_action(
            action, include_function_args=True
        )
        for index, action in enumerate(actions or [], start=1)
    )


def _private_hint_note(
    value: "TeacherHintResult | dict[str, Any] | None",
) -> str | None:
    payload = private_hint_payload(value)
    if not payload:
        return None
    note = payload.get("plan")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("teacher hint payload must contain a non-empty plan string")
    return note.strip()


def _validate_natural_note(plan: str, payload: dict[str, Any]) -> None:
    """Compatibility validator for the historical L3 contract."""
    validate_hint_note(plan, payload, HintLevel.L3_ORACLE)


def format_teacher_query_with_hint(
    query: str,
    hint: "TeacherHintResult | dict[str, Any] | None",
) -> str:
    """Build the Teacher-only query used by OPSD's frozen policy API."""
    note = _private_hint_note(hint)
    if note is None:
        return query
    return (
        f"{query}\n\n"
        "<private_teacher_note>\n"
        f"{note}\n"
        "</private_teacher_note>\n"
        "Treat this as advisory reasoning. Follow the policy and visible evidence, "
        "choose the concrete next action yourself, and never expose the note."
    )


def private_hint_payload(
    value: "TeacherHintResult | dict[str, Any] | None",
) -> dict[str, Any] | None:
    """Return only the private plan, excluding audit metadata.

    Collected rows store ``TeacherHintResult.to_dict()`` so model name, latency,
    and hashes remain auditable.  Only the nested ``hint`` object is allowed into
    the privileged Teacher prompt.
    """
    if value is None:
        return None
    if isinstance(value, TeacherHintResult):
        return value.hint
    if not isinstance(value, dict):
        raise TypeError(f"teacher hint must be a mapping, got {type(value).__name__}")
    if "hint" in value:
        nested = value["hint"]
        if nested is None:
            return None
        if not isinstance(nested, dict):
            raise TypeError("teacher hint payload must be a mapping")
        return nested
    return value


def format_teacher_system_prompt_with_hint(
    system_prompt: str,
    hint: "TeacherHintResult | dict[str, Any] | None",
) -> str:
    """Materialize the privileged information view without changing history.

    A missing hint intentionally returns the ordinary prompt.  This makes a
    no-hint row produce identical information views and therefore a closed token
    gate instead of inventing supervision.
    """
    payload = private_hint_payload(hint)
    if not payload:
        return system_prompt
    note = _private_hint_note(payload)
    return (
        f"{system_prompt}\n"
        "<private_teacher_note>\n"
        f"{note}\n"
        "</private_teacher_note>\n"
        "Treat this as advisory reasoning. Follow the policy and visible evidence, "
        "choose the concrete next action yourself, and never expose the note."
    )


class ClosedModelTeacherHinter:
    def __init__(self, endpoint: HintEndpoint):
        self.endpoint = endpoint
        self.client = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key or "EMPTY",
            timeout=endpoint.timeout,
            max_retries=0,
        )

    def hint(
        self,
        payload: dict[str, Any],
        level: HintLevel | str = HintLevel.L3_ORACLE,
    ) -> TeacherHintResult:
        parsed_level = HintLevel.parse(level)
        if parsed_level is HintLevel.L0_NONE:
            raise ValueError("L0 must bypass the hinter API")
        request_payload = prepare_hint_payload(payload, parsed_level)
        started = time.monotonic()
        last_error = None
        correction = ""
        for attempt in range(self.endpoint.retries):
            try:
                user_content = json.dumps(request_payload, ensure_ascii=False)
                if correction:
                    user_content += (
                        "\n\nThe previous draft was rejected because it violated: "
                        f"{correction}. Rewrite it from scratch and remove that "
                        "violation while preserving useful procedural guidance."
                    )
                response = self.client.chat.completions.create(
                    model=self.endpoint.model,
                    messages=[
                        {
                            "role": "system",
                            "content": hint_instruction(
                                parsed_level, request_payload.get("domain")
                            ),
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ],
                    temperature=0 if attempt == 0 else 0.7,
                    max_tokens=self.endpoint.max_tokens,
                )
                plan = (response.choices[0].message.content or "").strip()
                validate_hint_note(plan, request_payload, parsed_level)
                hint = {"plan": plan}
                canonical = json.dumps(
                    {"level": parsed_level.value, "hint": hint},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return TeacherHintResult(
                    hint=hint,
                    model=self.endpoint.model,
                    latency_ms=round((time.monotonic() - started) * 1000),
                    sha256=hashlib.sha256(canonical.encode()).hexdigest()[:16],
                    level=parsed_level.value,
                )
            except Exception as error:
                last_error = error
                if isinstance(error, ValueError):
                    correction = str(error)
                if attempt + 1 < self.endpoint.retries:
                    time.sleep(min(2**attempt, 4))
        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown error"
        )
        error_payload = {
            "type": type(last_error).__name__ if last_error is not None else "Unknown",
            "message": detail,
            "attempts": self.endpoint.retries,
        }
        canonical = json.dumps(
            {"level": parsed_level.value, "hint": {}, "error": error_payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return TeacherHintResult(
            hint={},
            model=self.endpoint.model,
            latency_ms=round((time.monotonic() - started) * 1000),
            sha256=hashlib.sha256(canonical.encode()).hexdigest()[:16],
            level=parsed_level.value,
            error=error_payload,
        )


class OpenModelTeacherHinter(ClosedModelTeacherHinter):
    """Open hinter queried with the exact prompt used by GRPO and sampling."""

    def hint(
        self,
        payload: dict[str, Any],
        level: HintLevel | str = HintLevel.L3_ORACLE,
    ) -> TeacherHintResult:
        parsed_level = HintLevel.parse(level)
        if parsed_level is HintLevel.L0_NONE:
            raise ValueError("L0 must bypass the hinter API")
        prepared = prepare_hint_payload(payload, parsed_level)
        public_state = prepared.get("current_history") or []
        privileged_context = {
            key: value
            for key, value in prepared.items()
            if key != "current_history"
        }
        messages = build_hinter_messages(public_state, privileged_context)
        started = time.monotonic()
        last_error = None
        correction = ""
        for attempt in range(self.endpoint.retries):
            try:
                request_messages = list(messages)
                if correction:
                    request_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous draft violated this contract: "
                                f"{correction}. Rewrite it without the violation."
                            ),
                        }
                    )
                response = self.client.chat.completions.create(
                    model=self.endpoint.model,
                    messages=request_messages,
                    temperature=0 if attempt == 0 else 0.7,
                    max_tokens=self.endpoint.max_tokens,
                )
                plan = (response.choices[0].message.content or "").strip()
                validate_hint_note(plan, prepared, parsed_level)
                hint = {"plan": plan}
                canonical = json.dumps(
                    {"level": parsed_level.value, "hint": hint},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return TeacherHintResult(
                    hint=hint,
                    model=self.endpoint.model,
                    latency_ms=round((time.monotonic() - started) * 1000),
                    sha256=hashlib.sha256(canonical.encode()).hexdigest()[:16],
                    level=parsed_level.value,
                )
            except Exception as error:
                last_error = error
                if isinstance(error, ValueError):
                    correction = str(error)
                if attempt + 1 < self.endpoint.retries:
                    time.sleep(min(2**attempt, 4))
        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown error"
        )
        error_payload = {
            "type": type(last_error).__name__ if last_error is not None else "Unknown",
            "message": detail,
            "attempts": self.endpoint.retries,
        }
        canonical = json.dumps(
            {"level": parsed_level.value, "hint": {}, "error": error_payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return TeacherHintResult(
            hint={},
            model=self.endpoint.model,
            latency_ms=round((time.monotonic() - started) * 1000),
            sha256=hashlib.sha256(canonical.encode()).hexdigest()[:16],
            level=parsed_level.value,
            error=error_payload,
        )


def _message_rows(messages: list[APICompatibleMessage]) -> list[dict]:
    rows = []
    for message in messages:
        if isinstance(message, MultiToolMessage):
            rows.extend(
                item.model_dump(mode="json", exclude_none=True)
                for item in message.tool_messages
            )
        else:
            rows.append(message.model_dump(mode="json", exclude_none=True))
    return rows


class HintedTeacherAgent(LLMAgent):
    """The shared policy model with one private plan per dialogue branch."""

    def __init__(
        self,
        *args,
        task: Task,
        hinter_endpoint: HintEndpoint,
        hinter=None,
        initial_hint: TeacherHintResult | None = None,
        refresh_hint_each_turn: bool = False,
        hint_level: HintLevel | str = HintLevel.L3_ORACLE,
        hinter_mode: str = "closed_model",
        hint_domain: str = "tau2",
        structured_instance_facts: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if hinter_mode not in {"closed_model", "open_hinter"}:
            raise ValueError("hinter_mode must be closed_model or open_hinter")
        self.task = task
        if hinter is not None:
            self.hinter = hinter
        elif hinter_mode == "open_hinter":
            self.hinter = OpenModelTeacherHinter(hinter_endpoint)
        else:
            self.hinter = ClosedModelTeacherHinter(hinter_endpoint)
        self.hint_records: list[dict] = []
        self._session_hint = initial_hint
        self.refresh_hint_each_turn = refresh_hint_each_turn
        self.hint_level = HintLevel.parse(hint_level)
        self.hint_domain = hint_domain
        self.structured_instance_facts = dict(structured_instance_facts or {})

    def _hint_payload(self, history: list[APICompatibleMessage]) -> dict[str, Any]:
        return {
            "task_id": self.task.id,
            "domain": self.hint_domain,
            "domain_policy": self.domain_policy,
            "available_tools": [tool.openai_schema for tool in self.tools],
            "authoritative_oracle_steps": oracle_steps_from_task(self.task),
            "current_history": _message_rows(history),
            **self.structured_instance_facts,
        }

    def hint_for_history(
        self, history: list[APICompatibleMessage]
    ) -> TeacherHintResult | None:
        if self.hint_level is HintLevel.L0_NONE:
            return None
        payload = self._hint_payload(history)
        try:
            result = self.hinter.hint(payload, self.hint_level)
        except TypeError:
            # Test doubles and legacy hinter adapters may only accept payload.
            result = self.hinter.hint(prepare_hint_payload(payload, self.hint_level))
        self.hint_records.append(
            {"turn": len(self.hint_records) + 1, **result.to_dict()}
        )
        return result

    def system_prompt_with_hint(self, result: TeacherHintResult | None) -> str:
        base_prompt = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )
        return format_teacher_system_prompt_with_hint(base_prompt, result)

    def plan_for_session(
        self, history: list[APICompatibleMessage]
    ) -> TeacherHintResult | None:
        if self.hint_level is HintLevel.L0_NONE:
            return None
        if self._session_hint is None:
            self._session_hint = self.hint_for_history(history)
        return self._session_hint

    def plan_for_history(
        self, history: list[APICompatibleMessage]
    ) -> TeacherHintResult | None:
        """Return the plan view configured for this rollout.

        Collection creates a fresh one-action Teacher branch at every Student
        decision, so its default session cache is already decision-local.  A
        full-dialogue Teacher benchmark must opt into refreshing the plan at
        every Agent turn to reproduce that treatment over one orchestrator.
        """
        if self.refresh_hint_each_turn:
            self._session_hint = self.hint_for_history(history)
            return self._session_hint
        return self.plan_for_session(history)

    def hinted_system_prompt_for_history(
        self, history: list[APICompatibleMessage]
    ) -> tuple[str, TeacherHintResult | None]:
        result = self.plan_for_history(history)
        return self.system_prompt_with_hint(result), result

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentStateType
    ) -> tuple[AssistantMessage, LLMAgentStateType]:
        incoming = (
            message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        )
        result = self.plan_for_history([*state.messages, *incoming])
        state.system_messages = [
            SystemMessage(role="system", content=self.system_prompt_with_hint(result))
        ]
        return super().generate_next_message(message, state)
