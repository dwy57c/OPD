from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Iterable

from tau2.data_model.message import ToolCall, UserMessage

from coevo.models.buyer_plan import BUYER_ACTION_PAYLOAD_FIELDS, BuyerPlan


_VALID_STOP_REASONS = {
    "task_complete",
    "scenario_requires_stop",
    "user_abandoned",
    "environment_blocked",
}
_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "developer message",
    "<|system|>",
)


@dataclass(frozen=True)
class BuyerRenderContext:
    available_user_tools: tuple[str, ...] = ()
    scenario_text: str = ""
    turn_index: int = 0


@dataclass(frozen=True)
class RenderedBuyerAction:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_message(self) -> UserMessage:
        return UserMessage(
            role="user",
            content=self.content,
            tool_calls=list(deepcopy(self.tool_calls)) or None,
        )

    def to_dict(self) -> dict:
        return self.to_message().model_dump(mode="json")


class FrozenRenderer:
    """Deterministic, non-trainable renderer from a private plan to public action."""

    def __init__(
        self,
        *,
        public_text_guard: Callable[[str], bool] | None = None,
        scenario_fact_guard: Callable[[BuyerPlan, BuyerRenderContext], bool] | None = None,
    ):
        self.public_text_guard = public_text_guard or self._default_text_guard
        self.scenario_fact_guard = scenario_fact_guard

    def render(
        self, plan: BuyerPlan, context: BuyerRenderContext
    ) -> RenderedBuyerAction:
        self._validate_payload(plan, context)
        if (
            self.scenario_fact_guard is not None
            and not self.scenario_fact_guard(plan, context)
        ):
            raise ValueError("Buyer plan failed scenario-fidelity validation")
        if plan.next_move == "reveal_hidden_constraint" and context.scenario_text:
            constraint = " ".join(plan.payload["constraint"].lower().split())
            scenario = " ".join(context.scenario_text.lower().split())
            if constraint not in scenario:
                raise ValueError("Revealed constraint is absent from the hidden scenario")

        payload = plan.payload
        move = plan.next_move
        if move == "execute_user_tool":
            arguments = deepcopy(payload["arguments"])
            call_payload = json.dumps(
                {
                    "name": payload["tool_name"],
                    "arguments": arguments,
                    "turn_index": context.turn_index,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            call_id = "buyer-" + sha256(call_payload.encode("utf-8")).hexdigest()[:16]
            return RenderedBuyerAction(
                content=None,
                tool_calls=(
                    ToolCall(
                        id=call_id,
                        name=payload["tool_name"],
                        arguments=arguments,
                        requestor="user",
                    ),
                ),
            )

        text = self._render_text(move, payload)
        if not self.public_text_guard(text):
            raise ValueError("Rendered public text failed prompt-injection audit")
        return RenderedBuyerAction(content=text)

    @staticmethod
    def _validate_payload(plan: BuyerPlan, context: BuyerRenderContext) -> None:
        required = set(BUYER_ACTION_PAYLOAD_FIELDS[plan.next_move])
        actual = set(plan.payload)
        if actual != required:
            raise ValueError(
                f"payload fields for {plan.next_move!r} must be {sorted(required)}, "
                f"got {sorted(actual)}"
            )
        for key, value in plan.payload.items():
            if key != "arguments" and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"payload.{key} must be a non-empty string")
        if plan.next_move == "execute_user_tool":
            if not isinstance(plan.payload["arguments"], dict):
                raise ValueError("payload.arguments must be an object")
            if plan.payload["tool_name"] not in set(context.available_user_tools):
                raise ValueError("Buyer plan requested an unavailable user tool")
        if plan.next_move == "stop" and plan.payload["stop_reason"] not in _VALID_STOP_REASONS:
            raise ValueError("Buyer plan contains an invalid stop reason")

    @staticmethod
    def _render_text(move: str, payload: dict) -> str:
        if move == "answer_normally":
            return payload["answer"]
        if move == "reveal_hidden_constraint":
            return f"I should also mention that {payload['constraint']}"
        if move == "withhold_information":
            return "Please continue using the information I have already provided."
        if move == "clarify_previous_statement":
            return f"To clarify: {payload['clarification']}"
        if move == "challenge_student_assumption":
            return f"Why are you assuming that {payload['assumption']}?"
        if move == "request_alternative":
            return f"Could you offer another option that meets {payload['criteria']}?"
        if move == "accept_proposal":
            return "That works for me. Please proceed."
        if move == "reject_proposal":
            return f"That does not work for me because {payload['reason']}"
        if move == "confirm_action":
            return f"Yes, please proceed with {payload['action']}."
        if move == "ask_about_cost":
            return "What will the total cost be, including all fees?"
        if move == "ask_about_policy":
            return "What policy or restrictions apply before I proceed?"
        if move == "stop":
            return "###STOP###"
        raise ValueError(f"Unsupported Buyer move: {move!r}")

    @staticmethod
    def _default_text_guard(text: str) -> bool:
        lowered = text.lower()
        return bool(text.strip()) and not any(marker in lowered for marker in _INJECTION_MARKERS)


def available_tool_names(tools: Iterable) -> tuple[str, ...]:
    names = []
    for tool in tools:
        name = getattr(tool, "name", None)
        schema = getattr(tool, "openai_schema", None)
        if name is None and isinstance(schema, dict):
            name = schema.get("function", {}).get("name")
        if name:
            names.append(str(name))
    return tuple(sorted(set(names)))
