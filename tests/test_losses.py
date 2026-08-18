"""Tests for loss functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.losses import L1AgeLoss, MeanVarianceLoss, get_loss_fn


def _dummy_logits(batch_size: int = 4, num_classes: int = 101) -> torch.Tensor:
    return torch.randn(batch_size, num_classes)


def _dummy_targets(batch_size: int = 4, max_age: float = 80.0) -> torch.Tensor:
    return torch.rand(batch_size, 1) * max_age


class TestMeanVarianceLoss:
    def test_output_tuple_length(self) -> None:
        loss_fn = MeanVarianceLoss()
        logits = _dummy_logits()
        targets = _dummy_targets()
        result = loss_fn(logits, targets)
        assert len(result) == 3, "MeanVarianceLoss should return (total, mean, variance)"

    def test_total_loss_positive(self) -> None:
        loss_fn = MeanVarianceLoss()
        total, _, _ = loss_fn(_dummy_logits(), _dummy_targets())
        assert total.item() > 0

    def test_backward_runs(self) -> None:
        """Backward pass must not raise any errors."""
        loss_fn = MeanVarianceLoss()
        logits = _dummy_logits().requires_grad_(True)
        total, _, _ = loss_fn(logits, _dummy_targets())
        total.backward()
        assert logits.grad is not None

    def test_predict_age_range(self) -> None:
        """Predicted ages should be within [0, 100]."""
        loss_fn = MeanVarianceLoss(start_age=0, end_age=100)
        ages = loss_fn.predict_age(_dummy_logits(batch_size=16))
        assert ages.min().item() >= 0.0
        assert ages.max().item() <= 100.0

    def test_perfect_prediction_low_loss(self) -> None:
        """When all logits concentrate mass on the true age, loss should be near 0."""
        loss_fn = MeanVarianceLoss(lambda_mean=1.0, lambda_variance=0.0)
        true_age = 30
        # One-hot logits at age 30
        logits = torch.full((4, 101), -1e9)
        logits[:, true_age] = 1e9
        targets = torch.full((4, 1), float(true_age))
        total, mean_loss, _ = loss_fn(logits, targets)
        assert mean_loss.item() < 0.01, f"Mean loss should be ~0 but got {mean_loss.item()}"


class TestL1AgeLoss:
    def test_output_tuple_length(self) -> None:
        loss_fn = L1AgeLoss()
        logits = torch.randn(4, 1)
        targets = torch.rand(4, 1) * 80
        result = loss_fn(logits, targets)
        assert len(result) == 3

    def test_backward_runs(self) -> None:
        loss_fn = L1AgeLoss()
        logits = torch.randn(4, 1).requires_grad_(True)
        total, _, _ = loss_fn(logits, torch.rand(4, 1) * 50)
        total.backward()
        assert logits.grad is not None


class TestGetLossFn:
    def test_mean_variance_factory(self) -> None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "loss": {"type": "mean_variance", "lambda_mean": 0.2,
                     "lambda_variance": 0.05, "start_age": 0, "end_age": 100}
        })
        loss_fn = get_loss_fn(cfg)
        assert isinstance(loss_fn, MeanVarianceLoss)

    def test_l1_factory(self) -> None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "loss": {"type": "l1", "lambda_mean": 0.2,
                     "lambda_variance": 0.05, "start_age": 0, "end_age": 100}
        })
        loss_fn = get_loss_fn(cfg)
        assert isinstance(loss_fn, L1AgeLoss)

    def test_unknown_loss_raises(self) -> None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "loss": {"type": "mse", "lambda_mean": 0.2,
                     "lambda_variance": 0.05, "start_age": 0, "end_age": 100}
        })
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss_fn(cfg)
