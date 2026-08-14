from pathlib import Path

import torch
from safetensors import safe_open


def safetensors_finite_summary(path: Path) -> dict:
    tensor_count = 0
    value_count = 0
    nonfinite_count = 0
    bad_tensors: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensor_count += 1
            value_count += tensor.numel()
            bad = int((~torch.isfinite(tensor)).sum().item())
            nonfinite_count += bad
            if bad:
                bad_tensors.append(key)
    return {
        "path": str(path),
        "tensor_count": tensor_count,
        "value_count": value_count,
        "nonfinite_count": nonfinite_count,
        "bad_tensor_count": len(bad_tensors),
        "bad_tensors": bad_tensors,
        "all_finite": nonfinite_count == 0,
    }
