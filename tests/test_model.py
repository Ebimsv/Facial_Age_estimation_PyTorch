"""Tests for AgeEstimationModel."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import AgeEstimationModel
from src.training.losses import L1AgeLoss

BACKBONES = ["mobilenet_v3", "efficientnet_b0", "efficientnet_b3", "resnet50"]
# Note: "vit_small" is excluded from fast unit tests because it downloads ~80MB weights.


@pytest.mark.parametrize("model_name", BACKBONES)
@pytest.mark.parametrize("output_dim", [1, 101])
def test_forward_shape(model_name: str, output_dim: int) -> None:
    """Model forward pass must return the correct output shape."""
    model = AgeEstimationModel(model_name=model_name, pretrained=False, output_dim=output_dim)
    model.eval()

    batch = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (2, output_dim), (
        f"{model_name} output shape {out.shape} != (2, {output_dim})"
    )


@pytest.mark.parametrize("model_name", BACKBONES)
@pytest.mark.parametrize("bias", [-5., 105.])
def test_raw_regression_gradient_and_bounded_prediction(model_name: str, bias: float) -> None:
    """Out-of-range heads must learn; only the inference decoder bounds ages."""
    model = AgeEstimationModel(model_name=model_name, pretrained=False, output_dim=1)
    model.eval()
    final = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(bias)
    batch = torch.zeros(2, 3, 64, 64)
    targets = torch.full((2, 1), 50.)
    logits = model(batch)
    torch.testing.assert_close(logits, torch.full_like(logits, bias))
    loss_fn = L1AgeLoss()
    total, _, _ = loss_fn(logits, targets)
    total.backward()
    assert final.bias.grad.item() == (-1. if bias < 0 else 1.)
    assert torch.isfinite(final.bias.grad).all()
    torch.testing.assert_close(loss_fn.predict_age(logits), torch.full((2,), max(0., min(100., bias))))
    with torch.no_grad():
        final.bias -= .1 * final.bias.grad
        assert loss_fn(model(batch), targets)[0] < total


def test_num_parameters() -> None:
    """num_parameters property should return a positive float."""
    model = AgeEstimationModel(model_name="efficientnet_b3", pretrained=False, output_dim=101)
    params = model.num_parameters
    assert isinstance(params, float)
    assert params > 0


def test_unknown_backbone_raises() -> None:
    """Passing an unknown backbone name must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown backbone"):
        AgeEstimationModel(model_name="nonexistent_model", pretrained=False)
