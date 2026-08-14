import pytest
import torch

from coevo.training.buyer_trainer import (
    CoevolutionGRPOTrainer,
    validate_buyer_attention_backend,
)


@pytest.mark.parametrize("attn_impl", ["flash_attn", "flash_attention_2"])
def test_qwen35_buyer_accepts_flash_attention_2(attn_impl):
    validate_buyer_attention_backend("qwen3_5", attn_impl)


@pytest.mark.parametrize("attn_impl", [None, "", "sdpa", "eager"])
def test_qwen35_buyer_rejects_unsafe_attention(attn_impl):
    with pytest.raises(ValueError, match="requires an explicitly selected"):
        validate_buyer_attention_backend("qwen3_5", attn_impl)


def test_other_buyer_models_keep_their_selected_attention_backend():
    validate_buyer_attention_backend("qwen3", "sdpa")


def _guard_only_trainer(monkeypatch, action: str):
    monkeypatch.setenv("COEVO_NONFINITE_GRADIENT_ACTION", action)
    trainer = object.__new__(CoevolutionGRPOTrainer)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.nonfinite_gradient_values_sanitized = 0
    trainer.nonfinite_gradient_tensors_sanitized = 0
    trainer._install_nonfinite_gradient_guard()
    return trainer


def test_nonfinite_buyer_gradient_guard_zeros_bad_values(monkeypatch):
    trainer = _guard_only_trainer(monkeypatch, "zero")

    (trainer.model.weight * torch.tensor(float("nan"))).sum().backward()

    assert torch.isfinite(trainer.model.weight.grad).all()
    assert trainer.nonfinite_gradient_values_sanitized == 1
    assert trainer.nonfinite_gradient_tensors_sanitized == 1


def test_nonfinite_buyer_gradient_guard_fails_closed_by_default(monkeypatch):
    monkeypatch.delenv("COEVO_NONFINITE_GRADIENT_ACTION", raising=False)
    trainer = object.__new__(CoevolutionGRPOTrainer)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.nonfinite_gradient_values_sanitized = 0
    trainer.nonfinite_gradient_tensors_sanitized = 0
    trainer._install_nonfinite_gradient_guard()

    with pytest.raises(FloatingPointError, match="non-finite Buyer gradient"):
        (trainer.model.weight * torch.tensor(float("nan"))).sum().backward()
