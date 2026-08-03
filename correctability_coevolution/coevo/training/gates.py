import torch


def gated_example_mean(
    losses: list[torch.Tensor], gates: list[float]
) -> torch.Tensor:
    if len(losses) != len(gates):
        raise ValueError(f"loss count {len(losses)} != gate count {len(gates)}")
    return torch.stack(
        [loss * gate for loss, gate in zip(losses, gates)]
    ).mean()
