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


HINT_SCHEMA = {
    "type": "object",
    "properties": {
        "current_state": {"type": "string"},
        "latest_user_intent": {"type": "string"},
        "completed_or_obsolete_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "next_step": {"type": "string"},
        "remaining_steps": {"type": "array", "items": {"type": "string"}},
        "policy_checks": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "current_state",
        "latest_user_intent",
        "completed_or_obsolete_steps",
        "next_step",
        "remaining_steps",
        "policy_checks",
    ],
    "additionalProperties": False,
}

HINTER_INSTRUCTION = """
You are a private hint generator for a customer-service policy model. The same
policy model is used for both Student and Teacher. The Teacher differs only in
receiving your hint.

Given the visible dialogue, tool results, domain policy, available tool schemas,
and privileged oracle resolution steps, produce a concise hint for the policy
model's next turn. Do not produce the public response and do not reveal
chain-of-thought.

Rules:
1. Treat oracle tool names and arguments as authoritative; never invent or alter
   their IDs, dates, amounts, passenger data, or payment methods.
2. Reconcile the oracle steps with actions and tool results already present in
   history. Mark completed or no-longer-applicable steps explicitly.
3. The next step must obey policy prerequisites such as identification,
   clarification, disclosure of action details, and explicit confirmation.
4. Choose exactly one immediate next action: ask/say one thing, or issue one tool
   call. Put later work in remaining_steps.
5. If the latest input is a tool result, use it to decide the next action. If it
   is a user query, answer or clarify it before advancing the workflow.
6. Do not recommend side-effecting actions outside the oracle resolution steps.
7. Return only JSON matching the supplied schema.
""".strip()


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


def format_teacher_query_with_hint(query: str, hint: dict[str, Any]) -> str:
    """Build the Teacher-only query used by OPSD's frozen policy API."""
    hint_text = json.dumps(hint, ensure_ascii=False, indent=2)
    return (
        f"{query}\n\n"
        "<private_teacher_hint>\n"
        f"{hint_text}\n"
        "</private_teacher_hint>\n"
        "Use the private hint to answer the user while following the domain policy. "
        "Do not mention or expose the hint."
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

    @staticmethod
    def _parse_hint(content: str) -> dict[str, Any]:
        start = (content or "").find("{")
        if start < 0:
            raise ValueError("closed-model hinter returned no JSON object")
        value, _ = json.JSONDecoder().raw_decode(content[start:])
        if not isinstance(value, dict) or set(value) != set(HINT_SCHEMA["required"]):
            raise ValueError("closed-model hinter returned an invalid hint shape")
        scalar_keys = {"current_state", "latest_user_intent", "next_step"}
        if any(
            not isinstance(value[key], str) or not value[key].strip()
            for key in scalar_keys
        ):
            raise ValueError("closed-model hinter returned an empty scalar hint field")
        list_keys = {
            "completed_or_obsolete_steps",
            "remaining_steps",
            "policy_checks",
        }
        if any(
            not isinstance(value[key], list)
            or any(not isinstance(item, str) or not item.strip() for item in value[key])
            for key in list_keys
        ):
            raise ValueError("closed-model hinter returned an invalid list hint field")
        return value

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
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "tau2_teacher_hint",
                            "strict": True,
                            "schema": HINT_SCHEMA,
                        },
                    },
                )
                hint = self._parse_hint(response.choices[0].message.content or "")
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
        raise RuntimeError(
            f"Teacher hinter failed after {self.endpoint.retries} attempts"
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
    """The shared policy model with a private closed-model hint per turn."""

    def __init__(self, *args, task: Task, hinter_endpoint: HintEndpoint, hinter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self.hinter = hinter or ClosedModelTeacherHinter(hinter_endpoint)
        self.hint_records: list[dict] = []

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

    def _hinted_system_prompt(self, result: TeacherHintResult) -> str:
        base_prompt = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )
        hint_text = json.dumps(result.hint, ensure_ascii=False, indent=2)
        return (
            f"{base_prompt}\n"
            "<private_teacher_hint>\n"
            f"{hint_text}\n"
            "</private_teacher_hint>\n"
            "Use this private hint as guidance for the immediate next action. "
            "Follow the policy and visible dialogue, and never mention the hint."
        )

    def hinted_system_prompt_for_history(
        self, history: list[APICompatibleMessage]
    ) -> tuple[str, TeacherHintResult]:
        result = self.hint_for_history(history)
        return self._hinted_system_prompt(result), result

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentStateType
    ) -> tuple[AssistantMessage, LLMAgentStateType]:
        incoming = (
            message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        )
        result = self.hint_for_history([*state.messages, *incoming])
        state.system_messages = [
            SystemMessage(role="system", content=self._hinted_system_prompt(result))
        ]
        return super().generate_next_message(message, state)
