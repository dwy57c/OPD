from dataclasses import dataclass

from coevo.intervention.decision_state import DecisionState
from coevo.intervention.teacher_action import TeacherActionGenerator, TeacherActionResult
from coevo.rewards.tau2_soft_score import SoftScoreResult, soft_completion_score


@dataclass(frozen=True)
class BranchEvaluation:
    seed: int
    student: SoftScoreResult
    teacher: SoftScoreResult

    @property
    def advantage(self) -> float:
        return self.teacher.score - self.student.score

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "student": self.student.to_dict(),
            "teacher": self.teacher.to_dict(),
            "advantage": self.advantage,
        }


@dataclass(frozen=True)
class ActionBranchResult:
    decision: DecisionState
    teacher_action: TeacherActionResult
    pairs: tuple[BranchEvaluation, ...]

    @property
    def intervention_advantage(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(pair.advantage for pair in self.pairs) / len(self.pairs)

    @property
    def student_value(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(pair.student.score for pair in self.pairs) / len(self.pairs)

    @property
    def teacher_value(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(pair.teacher.score for pair in self.pairs) / len(self.pairs)

    def to_dict(self) -> dict:
        action = self.decision.student_action
        return {
            "message_index": self.decision.message_index,
            "state_hash": self.decision.state_hash,
            "sample_hash": self.decision.sample_hash,
            "history_before": self.decision.to_dict()["history_before"],
            "student_action": action.model_dump(mode="json"),
            "student_output": action.content or "",
            "teacher_action": self.teacher_action.action.model_dump(mode="json"),
            "teacher_hint": self.teacher_action.hint,
            "paired_continuations": [pair.to_dict() for pair in self.pairs],
            "continuations": len(self.pairs),
            "student_value": self.student_value,
            "teacher_value": self.teacher_value,
            "intervention_advantage": self.intervention_advantage,
        }


class ActionBranchRunner:
    """Compare Student vs one-action Teacher takeover with paired continuations."""

    def __init__(
        self,
        environment,
        *,
        continuations: int = 1,
        teacher_generator: TeacherActionGenerator | None = None,
        score_fn=soft_completion_score,
    ):
        if continuations < 1:
            raise ValueError("continuations must be positive")
        self.environment = environment
        self.continuations = continuations
        self.teacher_generator = teacher_generator or TeacherActionGenerator(environment)
        self.score_fn = score_fn

    def run(self, decision: DecisionState) -> ActionBranchResult:
        base_seed = self.environment.config.seed
        teacher_action = self.teacher_generator.generate(decision, base_seed)
        pairs = []
        for sample_index in range(self.continuations):
            seed = base_seed + sample_index
            student_run = self.environment.continue_to_terminal(
                decision.branch_history(), "student", seed=seed
            )
            teacher_run = self.environment.continue_to_terminal(
                decision.branch_history(teacher_action.action), "student", seed=seed
            )
            pairs.append(
                BranchEvaluation(
                    seed=seed,
                    student=self.score_fn(student_run.reward_info),
                    teacher=self.score_fn(teacher_run.reward_info),
                )
            )
        return ActionBranchResult(decision, teacher_action, tuple(pairs))

    def score_decision(self, history, message_index: int) -> dict:
        """Score one complete natural Student action in a collected trajectory."""
        return self.run(DecisionState.from_history(history, message_index)).to_dict()
