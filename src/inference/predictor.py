"""One preprocessing and decoding path for CLI and interactive inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from src.utils.checkpoint import LoadedCheckpoint, load_checkpoint
from src.utils.config import DISTRIBUTION_LOSSES


@torch.no_grad()
def predict_image(model, loss_fn, transform, device, image: Image.Image):
    """Return age predictions and logits using the checkpoint's deterministic transform."""
    if not isinstance(image, Image.Image):
        raise ValueError("Provide a PIL image containing a cropped face")
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    return loss_fn.predict_age(logits), logits


@dataclass
class Prediction:
    age: float
    probabilities: torch.Tensor | None
    ages: torch.Tensor | None


class AgePredictor:
    """Image inference using a restored checkpoint's immutable contract."""

    def __init__(self, checkpoint: LoadedCheckpoint) -> None:
        self.checkpoint = checkpoint

    @classmethod
    def from_checkpoint(cls, path, config_path=None, **kwargs):
        """Load an artifact; kwargs follow the shared load_checkpoint policy."""
        return cls(load_checkpoint(path, config_path, **kwargs))

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Prediction:
        """Return predicted age and optional distribution on its saved age support."""
        bundle = self.checkpoint
        ages, logits = predict_image(bundle.model, bundle.loss_fn, bundle.transform, bundle.device, image)
        distribution = bundle.cfg.loss.type in DISTRIBUTION_LOSSES
        return Prediction(ages.item(), torch.softmax(logits.float(), dim=1)[0].cpu() if distribution else None,
                          bundle.loss_fn.ages.cpu() if distribution else None)
