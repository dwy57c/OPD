from tau2.data_model.message import Message, ToolMessage


def transition_validity(messages: list[Message]) -> float:
    return float(not any(isinstance(message, ToolMessage) and message.error for message in messages))


def allocate_turn_credit(
    trajectory_reward: float,
    intervention_advantages: list[float],
    effective_tokens: list[int] | None = None,
) -> list[float]:
    """Allocate trajectory utility across positive local intervention advantages."""
    if effective_tokens is None:
        effective_tokens = [1] * len(intervention_advantages)
    if len(effective_tokens) != len(intervention_advantages):
        raise ValueError("effective_tokens and intervention_advantages must align")
    masses = [
        max(0.0, advantage) * max(0, tokens)
        for advantage, tokens in zip(intervention_advantages, effective_tokens)
    ]
    denominator = sum(masses)
    if denominator <= 0:
        return [0.0] * len(masses)
    return [trajectory_reward * mass / denominator for mass in masses]


def absolute_group_is_trainable(rewards: list[float], lower_bounds=None) -> bool:
    """Skip GRPO groups whose best absolute lower-confidence reward is non-positive."""
    values = rewards if lower_bounds is None else list(lower_bounds)
    if len(values) != len(rewards):
        raise ValueError("lower_bounds and rewards must align")
    return bool(values) and max(values) > 0


def apply_absolute_group_skip(
    rewards: list[float],
    *,
    group_ids: list | None = None,
    group_size: int | None = None,
    lower_bounds: list[float] | None = None,
) -> list[float]:
    """Zero groups whose best absolute reward LCB is non-positive.

    Zeroing a complete group makes its normalized GRPO advantages zero while
    preserving row alignment and trainer bookkeeping.
    """
    if lower_bounds is not None and len(lower_bounds) != len(rewards):
        raise ValueError("lower_bounds and rewards must align")
    if group_ids is not None and len(group_ids) != len(rewards):
        raise ValueError("group_ids and rewards must align")
    if group_ids is None:
        if group_size is None or group_size < 1:
            group_size = len(rewards) or 1
        group_ids = [index // group_size for index in range(len(rewards))]

    groups = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(str(group_id), []).append(index)
    result = list(rewards)
    bounds = rewards if lower_bounds is None else lower_bounds
    for indexes in groups.values():
        if max((bounds[index] for index in indexes), default=0.0) <= 0:
            for index in indexes:
                result[index] = 0.0
    return result
