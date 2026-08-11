from dataclasses import asdict, dataclass

from coevo.models.buyer_plan import FAILURE_TYPES, BuyerPlan


@dataclass(frozen=True)
class BuyerPlanAuxRecord:
    predicted_learning_progress: float
    observed_learning_progress: float
    predicted_failure_type: str
    observed_failure_type: str
    target_skill: str
    plan_action_consistent: bool

    def to_dict(self) -> dict:
        return asdict(self)


def huber_loss(prediction: float, target: float, delta: float = 1.0) -> float:
    if delta <= 0:
        raise ValueError("delta must be positive")
    error = abs(prediction - target)
    if error <= delta:
        return 0.5 * error * error
    return delta * (error - 0.5 * delta)


def build_aux_record(
    plan: BuyerPlan,
    *,
    observed_learning_progress: float,
    observed_failure_type: str,
    plan_action_consistent: bool,
) -> BuyerPlanAuxRecord:
    if observed_failure_type not in FAILURE_TYPES:
        raise ValueError(f"Unknown observed failure type: {observed_failure_type!r}")
    return BuyerPlanAuxRecord(
        predicted_learning_progress=plan.predicted_learning_progress,
        observed_learning_progress=observed_learning_progress,
        predicted_failure_type=plan.diagnosis.failure_type,
        observed_failure_type=observed_failure_type,
        target_skill=plan.target_skill,
        plan_action_consistent=bool(plan_action_consistent),
    )
