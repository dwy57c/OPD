from swift.rewards import ORM, orms
from swift.rollout.multi_turn import multi_turns
from swift.trainers.trainer_factory import TrainerFactory

from coevo.training.buyer_scheduler import Tau2BuyerScheduler


class CorrectabilityReward(ORM):
    def __call__(self, completions, rollout_infos, **kwargs):
        return [info["correctability_reward"] for info in rollout_infos]


multi_turns["tau2_buyer"] = Tau2BuyerScheduler
orms["tau2_correctability"] = CorrectabilityReward
TrainerFactory.TRAINER_MAPPING["gkd"] = (
    "coevo.training.gated_gkd.CorrectabilityGKDTrainer"
)

