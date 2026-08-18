"""Tests for AgeEstimationModel."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import AgeEstimationModel


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
def test_regression_clamped(model_name: str) -> None:
    """Regression mode (output_dim=1) output must be in [0, 100]."""
    model = AgeEstimationModel(model_name=model_name, pretrained=False, output_dim=1)
    model.eval()

    batch = torch.randn(4, 3, 224, 224) * 10  # extreme inputs
    with torch.no_grad():
        out = model(batch)

    assert out.min().item() >= 0.0, "Output below 0"
    assert out.max().item() <= 100.0, "Output above 100"


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
