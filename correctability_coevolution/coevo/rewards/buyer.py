from tau2.data_model.message import Message, ToolMessage


def transition_validity(messages: list[Message]) -> float:
    return float(not any(isinstance(message, ToolMessage) and message.error for message in messages))


def buyer_reward(correctability: float, validity: float) -> float:
    return validity * correctability


def trajectory_buyer_reward(turn_scores: list[float], validity: float) -> float:
    mean_score = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0
    return buyer_reward(mean_score, validity)
