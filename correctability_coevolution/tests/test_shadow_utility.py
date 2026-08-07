from contextlib import contextmanager

import pytest

from coevo.rewards import (
    LinearUtilityCritic,
    UtilityFeatures,
    UtilityLabel,
    absolute_group_is_trainable,
    allocate_turn_credit,
    apply_absolute_group_skip,
)
from coevo.training.buyer_plan_aux import huber_loss
from coevo.training.shadow_opd import (
    ShadowOPDConfig,
    ShadowOPDEvaluator,
    ShadowTrainStats,
)


def test_shadow_opd_uses_fixed_budget_and_always_deletes_adapter():
    lifecycle = []
    config = ShadowOPDConfig(2, 1e-4, 100, 8, 2)

    @contextmanager
    def adapter_factory(base_model, shadow_config):
        adapter = {"loss": base_model["loss"]}
        lifecycle.append("created")
        try:
            yield adapter
        finally:
            lifecycle.append("deleted")

    def trainer(adapter, dataset, shadow_config):
        adapter["loss"] = 2.0
        return ShadowTrainStats(optimizer_steps=2, effective_tokens=50)

    evaluator = ShadowOPDEvaluator(
        config=config,
        adapter_factory=adapter_factory,
        trainer=trainer,
        probe_loss=lambda model, probe: model["loss"],
    )

    result = evaluator.evaluate({"loss": 4.0}, ["sample"], ["probe"])

    assert result.gain_per_token == pytest.approx(0.04)
    assert lifecycle == ["created", "deleted"]


def test_utility_critic_and_positive_turn_credit():
    low = UtilityFeatures.from_advantages([0.0, -0.2], 100)
    high = UtilityFeatures.from_advantages([0.8, 0.6], 100)
    critic = LinearUtilityCritic(learning_rate=0.01)
    critic.fit([UtilityLabel(low, -0.1), UtilityLabel(high, 0.4)], epochs=500)

    assert critic.predict(high) > critic.predict(low)
    assert allocate_turn_credit(0.9, [-1.0, 0.5, 1.0], [10, 2, 1]) == [
        0.0,
        0.45,
        0.45,
    ]
    assert absolute_group_is_trainable([-0.1, 0.2])
    assert not absolute_group_is_trainable([-0.1, 0.0])
    assert apply_absolute_group_skip(
        [-0.2, 0.0, -0.1, 0.4], group_ids=["a", "a", "b", "b"]
    ) == [0.0, 0.0, -0.1, 0.4]
    assert huber_loss(2.0, 0.0, delta=1.0) == 1.5
