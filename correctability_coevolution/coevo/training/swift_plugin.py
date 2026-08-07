from swift.rewards import ORM, orms
from swift.rollout.multi_turn import multi_turns
from swift.trainers.trainer_factory import TrainerFactory

from coevo.training.buyer_scheduler import Tau2BuyerScheduler
from coevo.rewards import apply_absolute_group_skip


class BuyerUtilityReward(ORM):
    def __call__(self, completions, rollout_infos, **kwargs):
        rewards = [info["buyer_reward"] for info in rollout_infos]
        lower_bounds = [
            info.get("reward_lcb", reward)
            for info, reward in zip(rollout_infos, rewards)
        ]
        group_ids = kwargs.get("prompt_id")
        group_size = getattr(self.args, "num_generations", None)
        return apply_absolute_group_skip(
            rewards,
            group_ids=group_ids,
            group_size=group_size,
            lower_bounds=lower_bounds,
        )


multi_turns["tau2_buyer"] = Tau2BuyerScheduler
orms["tau2_buyer_utility"] = BuyerUtilityReward
TrainerFactory.TRAINER_MAPPING["gkd"] = (
    "coevo.training.gated_gkd.NaturalDecisionStudentTrainer"
)
TrainerFactory.TRAINER_MAPPING["grpo"] = (
    "coevo.training.buyer_trainer.CoevolutionGRPOTrainer"
)
