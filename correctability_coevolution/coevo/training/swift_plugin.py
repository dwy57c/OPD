import json
import math
import os
from pathlib import Path

from swift.rewards import ORM, orms
from swift.rollout.multi_turn import multi_turns
from swift.trainers.trainer_factory import TrainerFactory

from coevo.training.buyer_scheduler import Tau2BuyerScheduler
from coevo.rewards import apply_absolute_group_skip, buyer_group_telemetry


def _capture_buyer_reward_groups(
    rewards, skipped_rewards, rollout_infos, *, group_ids, group_size
):
    """Persist exact GRPO candidates and normalized advantages when requested."""
    trace_path = os.getenv("COEVO_BUYER_TRACE_PATH", "").strip()
    if not trace_path:
        return
    if group_ids is None:
        size = int(group_size or len(rewards) or 1)
        group_ids = [index // size for index in range(len(rewards))]
    groups = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(str(group_id), []).append(index)

    path = Path(trace_path)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if world_size > 1:
        path = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        for group_id, indexes in groups.items():
            values = [float(skipped_rewards[index]) for index in indexes]
            mean = sum(values) / len(values) if values else 0.0
            sample_std = (
                math.sqrt(
                    sum((value - mean) ** 2 for value in values)
                    / (len(values) - 1)
                )
                if len(values) > 1
                else 0.0
            )
            advantages = [
                (value - mean) / (sample_std + 1e-4) for value in values
            ]
            record = {
                "scope": (
                    "single-process-global-group"
                    if world_size == 1
                    else "rank-local-shard"
                ),
                "formula": "(post_skip_reward-group_mean)/(sample_std+1e-4)",
                "rank": rank,
                "world_size": world_size,
                "expected_global_group_size": int(group_size or len(values)),
                "group_id": group_id,
                "candidate_indexes": indexes,
                "raw_rewards": [float(rewards[index]) for index in indexes],
                "post_skip_rewards": values,
                "group_mean": mean,
                "group_sample_std": sample_std,
                "normalized_advantages": advantages,
                "rollouts": [rollout_infos[index] for index in indexes],
            }
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


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
        _capture_buyer_reward_groups(
            rewards,
            skipped_rewards,
            rollout_infos,
            group_ids=group_ids,
            group_size=group_size,
        )
        return skipped_rewards


multi_turns["tau2_buyer"] = Tau2BuyerScheduler
orms["tau2_stage_learning_progress"] = BuyerStageProgressReward
TrainerFactory.TRAINER_MAPPING["gkd"] = (
    "coevo.training.gated_gkd.NaturalDecisionStudentTrainer"
)
TrainerFactory.TRAINER_MAPPING["grpo"] = (
    "coevo.training.buyer_trainer.CoevolutionGRPOTrainer"
)
