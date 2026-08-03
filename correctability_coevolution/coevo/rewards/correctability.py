from dataclasses import asdict, dataclass

from tau2.data_model.message import Message

from coevo.environment import Tau2Environment


@dataclass(frozen=True)
class CorrectabilityResult:
    teacher_successes: int
    student_successes: int
    continuations: int
    q_teacher: float
    q_student: float
    correctability: float

    def to_dict(self) -> dict:
        return asdict(self)


class CorrectabilityEstimator:
    """Terminal verifier estimate from one shared semantic action boundary."""

    def __init__(
        self,
        environment: Tau2Environment,
        continuations: int = 1,
        beta: float = 1.0,
        continuation_runner=None,
    ):
        self.environment = environment
        self.continuations = continuations
        self.beta = beta
        self.continuation_runner = continuation_runner

    def _continue(self, history: list[Message], policy: str, seed: int):
        if self.continuation_runner is not None:
            return self.continuation_runner.run(history, policy, seed)
        return self.environment.continue_to_terminal(history, policy, seed=seed)

    def estimate(self, history: list[Message]) -> CorrectabilityResult:
        teacher_successes = 0
        student_successes = 0
        for sample_index in range(self.continuations):
            seed = self.environment.config.seed + sample_index
            teacher_successes += int(
                self._continue(history, "teacher", seed).reward_info.reward > 0
            )
            student_successes += int(
                self._continue(history, "student", seed).reward_info.reward > 0
            )
        denominator = self.continuations + 2 * self.beta
        q_teacher = (teacher_successes + self.beta) / denominator
        q_student = (student_successes + self.beta) / denominator
        return CorrectabilityResult(
            teacher_successes=teacher_successes,
            student_successes=student_successes,
            continuations=self.continuations,
            q_teacher=q_teacher,
            q_student=q_student,
            correctability=q_teacher * (1.0 - q_student),
        )
