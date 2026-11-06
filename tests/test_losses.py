"""Tests for loss functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.losses import L1AgeLoss, L1MeanVarianceLoss, MeanVarianceLoss, get_loss_fn


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


@pytest.mark.parametrize("mode", ["l1_mean_variance", "mean_variance"])
def test_distribution_names_are_compatible(mode):
    from src.configs import load_config
    loss = get_loss_fn(load_config(overrides=[f"loss.type={mode}"]))
    assert isinstance(loss, L1MeanVarianceLoss)
    assert MeanVarianceLoss is L1MeanVarianceLoss


@pytest.mark.parametrize(("logits", "targets", "expected_mean", "expected_variance"), [
    ([[0., 0., 0., 0., 0.]], [1.], 1., 2.),
    ([[float("-inf"), float("-inf"), float("-inf"), float("-inf"), 0.]], [4.], 0., 0.),
])
def test_analytical_distributions(logits, targets, expected_mean, expected_variance):
    loss = L1MeanVarianceLoss(start_age=0, end_age=4)
    total, mean, variance = loss(torch.tensor(logits), torch.tensor(targets))
    assert mean.item() == pytest.approx(expected_mean)
    assert variance.item() == pytest.approx(expected_variance)
    assert total.item() == pytest.approx(.2 * expected_mean + .05 * expected_variance)


def test_two_point_shifted_distribution():
    loss = L1MeanVarianceLoss(lambda_mean=1, lambda_variance=1, start_age=10, end_age=11)
    logits = torch.tensor([[.25, .75]]).log()
    total, mean, variance = loss(logits, torch.tensor([10.5]))
    assert loss.predict_age(logits).item() == pytest.approx(10.75)
    assert mean.item() == pytest.approx(.25, abs=2e-6)
    assert variance.item() == pytest.approx(.1875, abs=2e-6)
    assert total.item() == pytest.approx(.4375, abs=2e-6)


def test_central_variance_is_stable_under_large_age_shift():
    logits = torch.zeros(1, 3)
    reference = L1MeanVarianceLoss(start_age=0, end_age=2)(logits, torch.tensor([1.]))
    shifted = L1MeanVarianceLoss(start_age=10_000, end_age=10_002)(logits, torch.tensor([10_001.]))
    assert shifted[2].item() >= 0
    assert shifted[2].item() == pytest.approx(2 / 3, abs=2e-5)
    torch.testing.assert_close(shifted[2], reference[2], atol=2e-5, rtol=2e-5)


def test_float32_loss_and_gradients_match_float64_oracle():
    values = torch.randn(3, 5, generator=torch.Generator().manual_seed(19), dtype=torch.float64)
    targets = torch.tensor([5.5, 7.2, 8.9], dtype=torch.float64)
    logits = values.clone().requires_grad_()
    loss = L1MeanVarianceLoss(start_age=5, end_age=9)
    actual = loss(logits, targets)
    actual_grad, = torch.autograd.grad(actual[0], logits)
    reference_logits = values.clone().requires_grad_()
    p = reference_logits.softmax(dim=1)
    ages = torch.arange(5, 10, dtype=torch.float64)
    mean = (p * ages).sum(1)
    mean_error = (mean - targets).abs().mean()
    variance = (p * (ages - mean[:, None]).square()).sum(1).mean()
    reference = .2 * mean_error + .05 * variance
    reference_grad, = torch.autograd.grad(reference, reference_logits)
    assert actual[0].dtype == torch.float32
    torch.testing.assert_close(actual[0].double(), reference, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_grad, reference_grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_low_precision_extreme_logits_have_finite_float32_moments(dtype):
    logits = torch.tensor([[-1000., 0., 1000.], [1000., 0., -1000.]], dtype=dtype, requires_grad=True)
    loss = L1MeanVarianceLoss(start_age=10, end_age=12)
    total, mean, variance = loss(logits, torch.tensor([11., 11.]))
    assert all(value.dtype == torch.float32 and torch.isfinite(value) for value in [total, mean, variance])
    assert variance >= 0
    assert loss.predict_age(logits).dtype == torch.float32
    total.backward()
    assert torch.isfinite(logits.grad).all()


def test_regression_loss_is_raw_and_decoder_is_float32():
    logits = torch.tensor([[-5.], [105.]], dtype=torch.float16, requires_grad=True)
    loss_fn = L1AgeLoss()
    loss, _, _ = loss_fn(logits, torch.tensor([50., 50.]))
    assert loss.dtype == torch.float32 and loss.item() == 55.
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-.5], [.5]], dtype=torch.float16))
    predictions = loss_fn.predict_age(logits)
    assert predictions.dtype == torch.float32
    torch.testing.assert_close(predictions, torch.tensor([0., 100.]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP verification requires a GPU")
@pytest.mark.parametrize("mode", ["l1", "l1_mean_variance"])
def test_cuda_autocast_loss_and_gradients(mode):
    loss_fn = L1AgeLoss() if mode == "l1" else L1MeanVarianceLoss(start_age=0, end_age=4)
    loss_fn = loss_fn.cuda()
    model = torch.nn.Linear(3, 1 if mode == "l1" else 5).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    # The default startup scale can overflow an FP16 linear weight gradient even
    # for L1 (e.g. 65536 * 1). Test objective stability at a representable scale;
    # dynamic startup backoff is GradScaler's job, not a failure of FP32 moments.
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(torch.ones(2, 3, device="cuda"))
        loss, _, _ = loss_fn(logits, torch.full((2, 1), 2., device="cuda"))
    assert loss.dtype == torch.float32
    assert loss_fn.predict_age(logits).dtype == torch.float32
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    scaler.step(optimizer)
    scaler.update()
    assert all(torch.isfinite(p).all() for p in model.parameters())
