from pathlib import Path

from safetensors.torch import save_file
import torch

from coevo.training.finite_check import safetensors_finite_summary


def test_safetensors_finite_summary_accepts_finite_adapter(tmp_path: Path):
    path = tmp_path / "adapter.safetensors"
    save_file({"lora_A": torch.ones(2, 3), "lora_B": torch.zeros(3, 2)}, path)

    summary = safetensors_finite_summary(path)

    assert summary["all_finite"] is True
    assert summary["tensor_count"] == 2
    assert summary["nonfinite_count"] == 0


def test_safetensors_finite_summary_rejects_nan_adapter(tmp_path: Path):
    path = tmp_path / "adapter.safetensors"
    save_file({"lora_A": torch.tensor([1.0, float("nan")])}, path)

    summary = safetensors_finite_summary(path)

    assert summary["all_finite"] is False
    assert summary["bad_tensor_count"] == 1
    assert summary["nonfinite_count"] == 1
