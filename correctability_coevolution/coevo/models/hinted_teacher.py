from dataclasses import asdict, dataclass
import hashlib
import json
import re
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


HINTER_INSTRUCTION = """
Write a short private decision note that helps a customer-service policy model
choose its next turn. Write like a careful expert reasoning in ordinary prose,
not like a controller issuing an answer. The note is private and will never be
shown to the customer.

Use the dialogue, tool results, domain policy, and privileged resolution facts
as evidence. Establish what is already known, what remains unresolved, and what
policy constraint matters now. When more than one move is genuinely plausible,
briefly consider two or three alternatives and the tradeoff or evidence that
favors one. End with a tentative semantic direction and any important
confirmation or safety guardrail, while leaving the policy model to choose and
express the concrete action itself.

Do not emit an exact function or tool name, argument key, JSON, code, schema,
quoted call syntax, or any other copyable API invocation. Do not quote or
mechanically restate the oracle steps. Do not use headings, numbered fields,
bullet lists, or a rigid template. You may mention identifiers, dates, routes,
or amounts only as facts in natural sentences when they are necessary to reason
correctly. Do not write the public reply. Use complete sentences and finish the
note cleanly. Keep it concise, usually 60-140 words.
""".strip()


_RIGID_HEADING = re.compile(
    r"^(?:#{1,6}\s*)?(?:state|user intent|next action|remaining plan|policy checks)\s*:?$",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r"^(?:[-*]\s+|\d+[.)]\s+)")


@dataclass(frozen=True)
class TeacherHintResult:
    hint: dict[str, Any]
    model: str
    latency_ms: int
    sha256: str

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
    """Reject truncated or action-serialization-like closed-model notes."""
    if not plan:
        raise ValueError("closed-model hinter returned an empty note")
    if plan[-1] not in ".!?。！？":
        raise ValueError("closed-model hinter returned an incomplete note")
    if len(plan.split()) > 220:
        raise ValueError("closed-model hinter returned an overlong note")
    if "```" in plan or plan.lstrip().startswith(("{", "[")):
        raise ValueError("closed-model hinter returned structured output")
    lines = [line.strip() for line in plan.splitlines() if line.strip()]
    if any(_RIGID_HEADING.match(line) or _LIST_ITEM.match(line) for line in lines):
        raise ValueError("closed-model hinter returned a rigid template")

    tool_names = []
    for schema in payload.get("available_tools") or []:
        function = schema.get("function") if isinstance(schema, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            tool_names.append(name)
    for name in tool_names:
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", plan):
            raise ValueError(
                "closed-model hinter copied an exact tool name into the note"
            )


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

    def hint(self, payload: dict[str, Any]) -> TeacherHintResult:
        started = time.monotonic()
        last_error = None
        for attempt in range(self.endpoint.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.endpoint.model,
                    messages=[
                        {"role": "system", "content": HINTER_INSTRUCTION},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                    temperature=0,
                    max_tokens=self.endpoint.max_tokens,
                )
                plan = (response.choices[0].message.content or "").strip()
                _validate_natural_note(plan, payload)
                hint = {"plan": plan}
                canonical = json.dumps(
                    hint, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                return TeacherHintResult(
                    hint=hint,
                    model=self.endpoint.model,
                    latency_ms=round((time.monotonic() - started) * 1000),
                    sha256=hashlib.sha256(canonical.encode()).hexdigest()[:16],
                )
            except Exception as error:
                last_error = error
                if attempt + 1 < self.endpoint.retries:
                    time.sleep(min(2**attempt, 4))
        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown error"
        )
        raise RuntimeError(
            f"Teacher hinter failed after {self.endpoint.retries} attempts: {detail}"
        ) from last_error


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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task = task
        self.hinter = hinter or ClosedModelTeacherHinter(hinter_endpoint)
        self.hint_records: list[dict] = []
        self._session_hint = initial_hint
        self.refresh_hint_each_turn = refresh_hint_each_turn

    def _hint_payload(self, history: list[APICompatibleMessage]) -> dict[str, Any]:
        return {
            "task_id": self.task.id,
            "domain_policy": self.domain_policy,
            "available_tools": [tool.openai_schema for tool in self.tools],
            "authoritative_oracle_steps": oracle_steps_from_task(self.task),
            "current_history": _message_rows(history),
        }

    def hint_for_history(
        self, history: list[APICompatibleMessage]
    ) -> TeacherHintResult:
        result = self.hinter.hint(self._hint_payload(history))
        self.hint_records.append(
            {"turn": len(self.hint_records) + 1, **result.to_dict()}
        )
        return result

    def system_prompt_with_hint(self, result: TeacherHintResult) -> str:
        base_prompt = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )
        return format_teacher_system_prompt_with_hint(base_prompt, result)

    def plan_for_session(
        self, history: list[APICompatibleMessage]
    ) -> TeacherHintResult:
        if self._session_hint is None:
            self._session_hint = self.hint_for_history(history)
        return self._session_hint

    def plan_for_history(
        self, history: list[APICompatibleMessage]
    ) -> TeacherHintResult:
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
    ) -> tuple[str, TeacherHintResult]:
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
