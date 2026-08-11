from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json

from tau2.data_model.message import AssistantMessage, Message

from coevo.environment.tau2 import dump_messages


def _digest(messages: list[Message]) -> str:
    payload = json.dumps(
        dump_messages(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionState:
    """One complete, natural Student macro-action and its pre-action state.

    A macro-action is a complete ``AssistantMessage``: text, one tool call, or a
    protocol-supported group of parallel tool calls. It is never a token/character
    prefix. All supervision and counterfactual branching therefore use the same
    protocol-level decision boundary as the deployed Student.
    """

    history_before: tuple[Message, ...]
    student_action: AssistantMessage
    message_index: int
    state_hash: str
    sample_hash: str

    @classmethod
    def from_history(cls, history: list[Message], message_index: int) -> "DecisionState":
        if message_index < 0 or message_index >= len(history):
            raise IndexError(f"message_index {message_index} is outside the history")
        action = history[message_index]
        if not isinstance(action, AssistantMessage):
            raise TypeError("A natural Student decision must be an AssistantMessage")
        if not action.content and not action.tool_calls:
            raise ValueError("A natural Student decision cannot be empty")
        if action.content and action.tool_calls:
            raise ValueError("A Student decision cannot mix text and tool calls")

        before = deepcopy(history[:message_index])
        copied_action = deepcopy(action)
        state_hash = _digest(before)
        sample_hash = _digest([*before, copied_action])
        return cls(
            history_before=tuple(before),
            student_action=copied_action,
            message_index=message_index,
            state_hash=state_hash,
            sample_hash=sample_hash,
        )

    def branch_history(self, action: AssistantMessage | None = None) -> list[Message]:
        return [*deepcopy(self.history_before), deepcopy(action or self.student_action)]

    def to_dict(self) -> dict:
        return {
            "message_index": self.message_index,
            "state_hash": self.state_hash,
            "sample_hash": self.sample_hash,
            "history_before": dump_messages(list(self.history_before)),
            "student_action": self.student_action.model_dump(mode="json"),
        }


def extract_decision_states(
    history: list[Message],
    *,
    start_index: int = 0,
    limit: int = 0,
) -> list[DecisionState]:
    """Extract complete Student actions; ``limit=0`` means unlimited."""
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    decisions = []
    for index in range(start_index, len(history)):
        message = history[index]
        if not isinstance(message, AssistantMessage):
            continue
        # An AssistantMessage is a protocol decision.  Do not silently erase a
        # malformed empty or mixed decision from audit/collection; the caller
        # must see the same fail-closed validation as direct construction.
        decisions.append(DecisionState.from_history(history, index))
        if limit and len(decisions) >= limit:
            break
    return decisions
