"""Historical ResNet50 inference architecture from commit 432569f."""

from __future__ import annotations

import torch
from torch import nn

from .age_estimator import AgeEstimationModel


class LegacyResNet50(AgeEstimationModel):
    """Reproduce the historical head and raw regression output without downloads."""

    def __init__(self) -> None:
        super().__init__("resnet50", pretrained=False, output_dim=1, dropout=0.2, img_size=128)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(2048, 256), nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
