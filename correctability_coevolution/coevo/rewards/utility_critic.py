from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class UtilityFeatures:
    mean_intervention_advantage: float
    positive_fraction: float
    harmful_fraction: float
    decision_count: float
    effective_opd_tokens: float

    @classmethod
    def from_advantages(
        cls, advantages: list[float], effective_opd_tokens: int = 0
    ) -> "UtilityFeatures":
        count = len(advantages)
        return cls(
            mean_intervention_advantage=(sum(advantages) / count if count else 0.0),
            positive_fraction=(
                sum(value > 0 for value in advantages) / count if count else 0.0
            ),
            harmful_fraction=(
                sum(value < 0 for value in advantages) / count if count else 0.0
            ),
            decision_count=float(count),
            effective_opd_tokens=float(effective_opd_tokens),
        )

    def vector(self) -> tuple[float, ...]:
        return (
            self.mean_intervention_advantage,
            self.positive_fraction,
            self.harmful_fraction,
            math.log1p(self.decision_count),
            math.log1p(self.effective_opd_tokens),
        )


@dataclass(frozen=True)
class UtilityLabel:
    features: UtilityFeatures
    shadow_gain_per_token: float


class LinearUtilityCritic:
    """Small calibrated critic for fast online reward prediction.

    This intentionally keeps training separate from Buyer GRPO. True shadow-OPD
    labels can periodically update the critic without adding hand-tuned terms to
    the scalar Buyer reward.
    """

    def __init__(self, learning_rate: float = 0.05, l2: float = 1e-4):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights = [0.0] * 5
        self.bias = 0.0
        self.label_count = 0

    def predict(self, features: UtilityFeatures) -> float:
        return self.bias + sum(
            weight * value for weight, value in zip(self.weights, features.vector())
        )

    def fit(self, labels: list[UtilityLabel], epochs: int = 200) -> None:
        if not labels:
            raise ValueError("At least one shadow-OPD label is required")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        scale = 1.0 / len(labels)
        for _ in range(epochs):
            weight_gradients = [0.0] * len(self.weights)
            bias_gradient = 0.0
            for label in labels:
                vector = label.features.vector()
                error = self.predict(label.features) - label.shadow_gain_per_token
                bias_gradient += 2.0 * error * scale
                for index, value in enumerate(vector):
                    weight_gradients[index] += 2.0 * error * value * scale
            for index in range(len(self.weights)):
                gradient = weight_gradients[index] + 2.0 * self.l2 * self.weights[index]
                self.weights[index] -= self.learning_rate * gradient
            self.bias -= self.learning_rate * bias_gradient
        self.label_count += len(labels)

    def state_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "weights": list(self.weights),
            "bias": self.bias,
            "label_count": self.label_count,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "LinearUtilityCritic":
        critic = cls(float(state["learning_rate"]), float(state["l2"]))
        weights = [float(value) for value in state["weights"]]
        if len(weights) != len(critic.weights):
            raise ValueError("Utility critic state has the wrong feature dimension")
        critic.weights = weights
        critic.bias = float(state["bias"])
        critic.label_count = int(state.get("label_count", 0))
        return critic


def utility_label_to_dict(label: UtilityLabel) -> dict:
    return {
        "features": asdict(label.features),
        "shadow_gain_per_token": label.shadow_gain_per_token,
    }
