"""Training utilities — losses, trainer, and loss factory.

``Trainer`` is imported lazily (PEP 562) so that importing only
``MeanVarianceLoss`` or ``get_loss_fn`` does **not** trigger the full
``trainer.py`` module (which pulls in torch.amp, TensorBoard, etc.).
This keeps test startup fast and avoids unnecessary import-time side effects.
"""

from __future__ import annotations

from .losses import MeanVarianceLoss, get_loss_fn

__all__ = ["MeanVarianceLoss", "get_loss_fn", "Trainer"]


def __getattr__(name: str):
    """Lazy-load ``Trainer`` only when it is actually needed."""
    if name == "Trainer":
        from .trainer import Trainer  # noqa: PLC0415
        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
