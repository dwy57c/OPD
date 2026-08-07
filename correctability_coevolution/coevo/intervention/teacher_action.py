from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from tau2.data_model.message import AssistantMessage

from coevo.intervention.decision_state import DecisionState


@dataclass(frozen=True)
class TeacherActionResult:
    action: AssistantMessage
    hint: dict | None = None

    def to_dict(self) -> dict:
        return {
            "action": self.action.model_dump(mode="json"),
            "hint": deepcopy(self.hint),
        }


class TeacherActionGenerator:
    """Generate exactly one privileged Teacher macro-action at a Student state."""

    def __init__(
        self,
        environment,
        action_provider: Callable[[DecisionState, int], TeacherActionResult] | None = None,
    ):
        self.environment = environment
        self.action_provider = action_provider

    def generate(self, decision: DecisionState, seed: int) -> TeacherActionResult:
        if self.action_provider is not None:
            result = self.action_provider(decision, seed)
            self._validate(result.action)
            return result

        history = list(decision.history_before)
        orchestrator = self.environment.orchestrator(history, "teacher", seed=seed)
        orchestrator.initialize()
        initial_size = len(orchestrator.get_trajectory())
        while not orchestrator.done:
            orchestrator.step()
            generated = orchestrator.get_trajectory()[initial_size:]
            action = next(
                (message for message in generated if isinstance(message, AssistantMessage)),
                None,
            )
            if action is not None:
                self._validate(action)
                hint = getattr(orchestrator.agent, "_session_hint", None)
                if hint is not None and hasattr(hint, "to_dict"):
                    hint = hint.to_dict()
                elif hint is not None and not isinstance(hint, dict):
                    hint = None
                return TeacherActionResult(deepcopy(action), deepcopy(hint))
        raise RuntimeError("Teacher branch terminated before producing one action")

    @staticmethod
    def _validate(action: AssistantMessage) -> None:
        if not action.content and not action.tool_calls:
            raise ValueError("Teacher produced an empty action")
        if action.content and action.tool_calls:
            raise ValueError("Teacher action cannot mix text and tool calls")
