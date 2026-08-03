from hashlib import sha256
import json

from tau2.data_model.message import AssistantMessage, Message

from coevo.cutoff import TeacherCutoffSelector, semantic_boundaries
from coevo.environment.tau2 import dump_messages
from coevo.rewards import CorrectabilityEstimator


class TurnCutoffScorer:
    """Select and score semantic cutoffs inside one completed Student turn."""

    def __init__(
        self,
        selector: TeacherCutoffSelector,
        estimator: CorrectabilityEstimator,
    ):
        self.selector = selector
        self.estimator = estimator

    def score_turn(self, history: list[Message], message_index: int) -> dict | None:
        message = history[message_index]
        if (
            not isinstance(message, AssistantMessage)
            or message.tool_calls
            or not message.content
        ):
            return None
        candidates = semantic_boundaries(message.content)
        if not candidates:
            return None
        history_before = history[:message_index]
        selected = self.selector.select(history_before, message.content, candidates)
        cutoffs = []
        for item in selected:
            offset = item.candidate.char_offset
            partial = message.model_copy(update={"content": message.content[:offset]})
            cutoff_history = [*history_before, partial]
            result = self.estimator.estimate(cutoff_history)
            state_payload = json.dumps(
                dump_messages(cutoff_history), ensure_ascii=False, sort_keys=True
            )
            cutoffs.append(
                {
                    "selection": item.to_dict(),
                    "history": dump_messages(cutoff_history),
                    "state_hash": sha256(state_payload.encode()).hexdigest(),
                    "correctability": result.to_dict(),
                }
            )
        turn_score = sum(
            cutoff["correctability"]["correctability"] for cutoff in cutoffs
        ) / len(cutoffs)
        return {
            "message_index": message_index,
            "student_output": message.content,
            "history_before": dump_messages(history_before),
            "cutoffs": cutoffs,
            "correctability": turn_score,
        }
