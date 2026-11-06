"""Age objectives with explicit inference policies.

The repository's distribution objective is L1 on expected age plus a central
variance penalty, computed in float32. It is inspired by mean-variance methods,
but is not a reproduction of the squared-mean + softmax joint objective in
Pan et al., CVPR 2018. No UTKFace performance advantage has been established.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.config import (
    DISTRIBUTION_LOSSES,
    validate_age_support,
    validate_loss_weights,
    validate_mode,
)


def _validate_targets(targets: torch.Tensor, batch: int, start_age: float, end_age: float) -> None:
    if targets.ndim not in (1, 2) or (targets.ndim == 2 and targets.shape[1] != 1):
        raise ValueError("Age targets must have shape [B] or [B, 1]")
    if targets.shape[0] != batch or batch == 0:
        raise ValueError("Age targets and logits must have the same nonempty batch size")
    if not (torch.isfinite(targets) & (targets >= start_age) & (targets <= end_age)).all():
        raise ValueError(f"Age targets must be finite and inside [{start_age}, {end_age}]")


class L1MeanVarianceLoss(nn.Module):
    """Repository variant: L1 on expected age plus central age variance.

    This is not the full joint loss of Pan et al. (CVPR 2018). Moments and
    expected-age decoding always use float32, including under autocast.

    Args:
        lambda_mean:     Weight for the mean (expected age) loss term.
        lambda_variance: Weight for the variance (spread) loss term.
        start_age:       Minimum age in the distribution (inclusive).
        end_age:         Maximum age in the distribution (inclusive).
    """

    def __init__(
        self,
        lambda_mean: float = 0.2,
        lambda_variance: float = 0.05,
        start_age: int = 0,
        end_age: int = 100,
    ) -> None:
        super().__init__()
        validate_age_support(start_age, end_age)
        validate_loss_weights(lambda_mean, lambda_variance)
        if start_age == end_age:
            raise ValueError("MeanVarianceLoss requires at least two age classes")
        self.start_age, self.end_age = start_age, end_age
        self.lambda_mean = lambda_mean
        self.lambda_variance = lambda_variance

        # Age index tensor — registered as a buffer so it moves with the model
        ages = torch.arange(start_age, end_age + 1, dtype=torch.float32)
        self.register_buffer("ages", ages)          # shape: [num_classes]
        # Retained for compatibility with existing loss.state_dict() consumers.
        ages_sq = ages ** 2
        self.register_buffer("ages_sq", ages_sq)   # shape: [num_classes]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Mean-Variance Loss.

        Args:
            logits:  Raw model output, shape [B, num_classes].
            targets: Ground-truth ages, shape [B, 1] or [B].

        Returns:
            (total_loss, mean_loss, variance_loss) — all scalar tensors.
        """
        self._validate_logits(logits)
        _validate_targets(targets, logits.shape[0], self.start_age, self.end_age)
        targets = targets.view(-1, 1)  # [B, 1]

        # Probability distribution over ages
        probs = torch.softmax(logits, dim=1, dtype=torch.float32)  # [B, num_classes]
        ages = self.ages.float()

        # Expected age  E[age] = Σ p_i × i
        expected_age = (probs * ages).sum(dim=1, keepdim=True)  # [B, 1]

        # Mean loss: L1 between expected age and ground-truth
        mean_loss = torch.abs(expected_age - targets.float()).mean()

        # Central variance avoids cancellation between two large nearly equal moments.
        variance = (probs * (ages - expected_age).square()).sum(dim=1).mean()

        total_loss = self.lambda_mean * mean_loss + self.lambda_variance * variance
        return total_loss, mean_loss, variance

    @torch.no_grad()
    def predict_age(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to predicted age scalar(s).

        Args:
            logits: Shape [B, num_classes].

        Returns:
            Predicted ages, shape [B].
        """
        self._validate_logits(logits)
        probs = torch.softmax(logits, dim=1, dtype=torch.float32)
        return (probs * self.ages.float()).sum(dim=1)

    def _validate_logits(self, logits: torch.Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.ages.numel() or logits.shape[0] == 0:
            raise ValueError(f"Distribution logits must have shape [B, {self.ages.numel()}] with B > 0")


class L1AgeLoss(nn.Module):
    """Thin wrapper around nn.L1Loss for regression-mode models (output_dim=1).

    Returns a 3-tuple (total, total, zero) to be API-compatible with
    ``MeanVarianceLoss``, so the trainer does not need branching logic.
    """

    def __init__(self) -> None:
        super().__init__()
        self._l1 = nn.L1Loss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_logits(logits)
        _validate_targets(targets, logits.shape[0], 0, 100)
        loss = self._l1(logits.float().view(-1), targets.float().view(-1))
        zero = torch.zeros_like(loss)
        return loss, loss, zero

    @torch.no_grad()
    def predict_age(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the modern checkpoint's bounded inference policy, outside training."""
        self._validate_logits(logits)
        return logits.float().view(-1).clamp(0.0, 100.0)

    @staticmethod
    def _validate_logits(logits: torch.Tensor) -> None:
        if logits.ndim not in (1, 2) or (logits.ndim == 2 and logits.shape[1] != 1) or logits.shape[0] == 0:
            raise ValueError("Regression logits must have shape [B] or [B, 1] with B > 0")


# Historical import/API compatibility; both names denote the same L1 variant.
MeanVarianceLoss = L1MeanVarianceLoss


def training_objective_spec(loss_type: str) -> dict:
    """Identify this training implementation; do not infer old weights' provenance."""
    if loss_type in DISTRIBUTION_LOSSES:
        return {"name": "l1_mean_variance", "revision": 2, "mean_term": "absolute_expected_age_error",
                "variance": "central", "compute_dtype": "float32"}
    if loss_type == "l1":
        return {"name": "l1_raw_regression", "revision": 2, "compute_dtype": "float32"}
    raise ValueError(f"Unknown loss type: {loss_type}")


def get_loss_fn(cfg) -> L1MeanVarianceLoss | L1AgeLoss:
    """Factory that creates the appropriate loss from an OmegaConf config.

    Args:
        cfg: Config object with a ``loss`` sub-tree.

    Returns:
        Instantiated loss module.
    """
    loss_type = cfg.loss.type.lower()
    if "model" in cfg:
        validate_mode(cfg.model.output_dim, loss_type, cfg.loss.start_age, cfg.loss.end_age)
    if loss_type in DISTRIBUTION_LOSSES:
        return L1MeanVarianceLoss(
            lambda_mean=cfg.loss.lambda_mean,
            lambda_variance=cfg.loss.lambda_variance,
            start_age=cfg.loss.start_age,
            end_age=cfg.loss.end_age,
        )
    elif loss_type == "l1":
        return L1AgeLoss()
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. Choose from: l1_mean_variance | mean_variance (alias) | l1"
        )
