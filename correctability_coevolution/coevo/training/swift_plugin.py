from swift.rewards import ORM, orms
from swift.rollout.multi_turn import multi_turns
from swift.trainers.trainer_factory import TrainerFactory

from coevo.training.buyer_scheduler import Tau2BuyerScheduler
from coevo.rewards import apply_absolute_group_skip, buyer_group_telemetry


class BuyerStageProgressReward(ORM):
    def __call__(self, completions, rollout_infos, **kwargs):
        rewards = [info["buyer_reward"] for info in rollout_infos]
        group_ids = kwargs.get("prompt_id")
        group_size = getattr(self.args, "num_generations", None)
        skipped_rewards = apply_absolute_group_skip(
            rewards,
            group_ids=group_ids,
            group_size=group_size,
        )
        telemetry = buyer_group_telemetry(
            rewards,
            skipped_rewards,
            rollout_infos,
            group_ids=group_ids,
            group_size=group_size,
        )
        for info in rollout_infos:
            info.update(telemetry)
        return skipped_rewards


multi_turns["tau2_buyer"] = Tau2BuyerScheduler
orms["tau2_stage_learning_progress"] = BuyerStageProgressReward
TrainerFactory.TRAINER_MAPPING["gkd"] = (
    "coevo.training.gated_gkd.NaturalDecisionStudentTrainer"
)
TrainerFactory.TRAINER_MAPPING["grpo"] = (
    "coevo.training.buyer_trainer.CoevolutionGRPOTrainer"
)
