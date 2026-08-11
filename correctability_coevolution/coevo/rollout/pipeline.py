from coevo.environment import Tau2Environment
from coevo.scoring import TeacherTargetLabeler, TeacherTargetValidator


def build_teacher_target_labeler(
    environment: Tau2Environment,
) -> TeacherTargetLabeler:
    """Main-path Teacher supervision; no takeover continuation is executed."""
    return TeacherTargetLabeler(environment)


def build_teacher_target_validator(
    environment: Tau2Environment,
) -> TeacherTargetValidator:
    """Analysis-only absolute Teacher-action continuation validator."""
    return TeacherTargetValidator(
        environment,
        continuations=environment.config.teacher_validation_continuations,
    )
