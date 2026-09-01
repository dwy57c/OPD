"""Swift registration for the single active hinter GRPO reward."""

from swift.rewards import orms

from coevo.hinter_training.grpo_reward import HinterCompositeReward


orms["hinter_composite"] = HinterCompositeReward
