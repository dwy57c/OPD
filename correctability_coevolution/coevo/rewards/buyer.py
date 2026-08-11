from tau2.data_model.message import Message, ToolMessage


def _groups(length: int, group_ids=None, group_size: int | None = None) -> dict:
    if group_ids is not None and len(group_ids) != length:
        raise ValueError("group_ids and rewards must align")
    if group_ids is None:
        if group_size is None or group_size < 1:
            group_size = length or 1
        group_ids = [index // group_size for index in range(length)]
    groups = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(str(group_id), []).append(index)
    return groups


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance**0.5


def buyer_group_telemetry(
    rewards: list[float],
    skipped_rewards: list[float],
    rollout_infos: list[dict],
    *,
    group_ids=None,
    group_size: int | None = None,
) -> dict[str, float]:
    if len(rewards) != len(skipped_rewards) or len(rewards) != len(rollout_infos):
        raise ValueError("rewards, skipped_rewards, and rollout_infos must align")
    groups = _groups(len(rewards), group_ids, group_size)
    raw_mean, raw_std = _mean_std([float(value) for value in rewards])
    post_mean, post_std = _mean_std([float(value) for value in skipped_rewards])
    all_zero_groups = sum(
        all(float(skipped_rewards[index]) == 0.0 for index in indexes)
        for indexes in groups.values()
    )
    progress = [
        float(value)
        for info in rollout_infos
        for value in info.get("learning_progresses", [])
    ]
    gaps = [
        float(value)
        for info in rollout_infos
        for value in info.get("current_gaps", [])
    ]
    gap_mean, gap_std = _mean_std(gaps)
    progress_count = len(progress) or 1
    rollout_count = len(rollout_infos) or 1
    return {
        "group_raw_reward_mean": raw_mean,
        "group_raw_reward_std": raw_std,
        "group_post_skip_reward_mean": post_mean,
        "group_post_skip_reward_std": post_std,
        "group_all_zero_fraction": all_zero_groups / (len(groups) or 1),
        "group_invalid_rollout_fraction": sum(
            float(info.get("trajectory_validity", 1.0)) <= 0
            for info in rollout_infos
        )
        / rollout_count,
        "group_learning_progress_positive_fraction": sum(
            value > 0 for value in progress
        )
        / progress_count,
        "group_learning_progress_zero_fraction": sum(value == 0 for value in progress)
        / progress_count,
        "group_learning_progress_negative_fraction": sum(
            value < 0 for value in progress
        )
        / progress_count,
        "group_current_gap_mean": gap_mean,
        "group_current_gap_std": gap_std,
    }


def transition_validity(messages: list[Message]) -> float:
    return float(not any(isinstance(message, ToolMessage) and message.error for message in messages))


def absolute_group_is_trainable(rewards: list[float]) -> bool:
    """A group is trainable only when at least one absolute reward is positive."""
    return bool(rewards) and max(rewards) > 0


def apply_absolute_group_skip(
    rewards: list[float],
    *,
    group_ids: list | None = None,
    group_size: int | None = None,
) -> list[float]:
    """Zero groups whose best absolute stage-progress reward is non-positive.

    Zeroing a complete group makes its normalized GRPO advantages zero while
    preserving row alignment and trainer bookkeeping.
    """
    groups = _groups(len(rewards), group_ids, group_size)
    result = list(rewards)
    for indexes in groups.values():
        if max((rewards[index] for index in indexes), default=0.0) <= 0:
            for index in indexes:
                result[index] = 0.0
    return result
