"""Teacher-target construction and consecutive-checkpoint scoring."""

from .skill_contrast import (
    SkillContrastConfig,
    SkillContrastResult,
    construct_skill_contrast_target,
)
from .stage_gap import (
    PromptLogprobClient,
    SparseTargetView,
    StageGapScore,
    StageGapScorer,
    TeacherTargetBuilder,
)
from .teacher_target import (
    TeacherTargetLabel,
    TeacherTargetLabeler,
    TeacherTargetRecord,
    TeacherTargetValidator,
    TeacherValidationResult,
)

__all__ = [
    "SkillContrastConfig",
    "SkillContrastResult",
    "construct_skill_contrast_target",
    "PromptLogprobClient",
    "SparseTargetView",
    "StageGapScore",
    "StageGapScorer",
    "TeacherTargetBuilder",
    "TeacherTargetLabel",
    "TeacherTargetLabeler",
    "TeacherTargetRecord",
    "TeacherTargetValidator",
    "TeacherValidationResult",
]
