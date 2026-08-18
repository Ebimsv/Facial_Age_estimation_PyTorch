"""Metrics for age estimation evaluation."""

from __future__ import annotations

import torch


class AverageMeter:
    """Computes and stores a running average of a scalar value."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update meter with a new value.

        Args:
            val: New value (e.g., batch-level loss or metric).
            n:   Number of samples in this batch.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: avg={self.avg:.4f} (n={self.count})"


@torch.no_grad()
def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Absolute Error between predictions and targets.

    Args:
        predictions: Predicted ages [B] or [B, 1].
        targets:     Ground-truth ages [B] or [B, 1].

    Returns:
        Scalar MAE value.
    """
    return torch.abs(predictions.view(-1) - targets.view(-1)).mean().item()


@torch.no_grad()
def compute_cumulative_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: int = 5,
) -> float:
    """Cumulative Score (CS) — fraction of samples with |pred - true| <= threshold.

    CS@5 is a standard secondary metric for age estimation benchmarks.

    Args:
        predictions: Predicted ages [B] or [B, 1].
        targets:     Ground-truth ages [B] or [B, 1].
        threshold:   Error tolerance in years (default 5).

    Returns:
        CS value in [0, 1].
    """
    errors = torch.abs(predictions.view(-1) - targets.view(-1))
    return (errors <= threshold).float().mean().item()
