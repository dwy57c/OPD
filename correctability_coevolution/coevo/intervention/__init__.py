"""Natural decision-boundary and privileged-action primitives."""

from .decision_state import DecisionState, extract_decision_states
from .teacher_action import TeacherActionGenerator, TeacherActionResult

__all__ = [
    "DecisionState",
    "TeacherActionGenerator",
    "TeacherActionResult",
    "extract_decision_states",
]
