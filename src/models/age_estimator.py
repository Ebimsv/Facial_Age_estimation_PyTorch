"""Age estimation model definitions.

Supported backbones
-------------------
- ``mobilenet_v3``    MobileNetV3-Large  ~5.5 M params  (edge / mobile)
- ``efficientnet_b0`` EfficientNet-B0    ~5.3 M params  (lightweight baseline)
- ``efficientnet_b3`` EfficientNet-B3    ~12 M params   (balanced, default)
- ``resnet50``        ResNet-50          ~25.6 M params (classic baseline)
- ``vit_small``       ViT-S/14 DINOv2   ~22 M params   (highest accuracy)

Output modes
------------
- ``output_dim=101``  Probability distribution over ages [0–100] → use with
                      ``MeanVarianceLoss`` (recommended).
- ``output_dim=1``    Direct regression → use with ``nn.L1Loss`` (legacy mode).

This module is **side-effect free**: importing it does NOT instantiate any
model or touch the GPU.  Use ``build_model(cfg)`` or instantiate
``AgeEstimationModel`` directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    efficientnet_b3,
    mobilenet_v3_large,
    resnet50,
)

from src.utils.config import validate_mode, validate_model_options

try:
    import timm

    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

# ── Backbone feature-dimension lookup ────────────────────────────────────────
_BACKBONE_DIMS: dict[str, int] = {
    "mobilenet_v3": 1280,
    "efficientnet_b0": 1280,
    "efficientnet_b3": 1536,
    "resnet50": 2048,
    "vit_small": 384,
}


def _make_head(in_features: int, output_dim: int, dropout: float) -> nn.Sequential:
    """Shared classification / regression head.

    Architecture: Dropout → Linear(in, 512) → GELU → Linear(512, output_dim)
    """
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Linear(512, output_dim),
    )


class AgeEstimationModel(nn.Module):
    """Unified age estimation model supporting multiple backbones.

    Args:
        model_name:  One of the keys in ``_BACKBONE_DIMS``.
        pretrained:  Load ImageNet pretrained weights.
        output_dim:  101 for Mean-Variance Loss; 1 for regression.
        dropout:     Dropout rate applied before the first head linear layer.
        img_size:    Input resolution; ViT requires a multiple of its 14-pixel patch size.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        output_dim: int = 101,
        dropout: float = 0.3,
        img_size: int = 224,
    ) -> None:
        super().__init__()

        validate_model_options(model_name, pretrained, output_dim, dropout, img_size)

        self.model_name = model_name
        self.output_dim = output_dim
        self.img_size = img_size
        self.dropout = dropout
        in_features = _BACKBONE_DIMS[model_name]

        # ── Build backbone ────────────────────────────────────────────────
        if model_name == "mobilenet_v3":
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = mobilenet_v3_large(weights=weights)
            # Replace only the final linear in the classifier block
            backbone.classifier[-1] = _make_head(in_features, output_dim, dropout)
            self.model = backbone

        elif model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            backbone.classifier = _make_head(in_features, output_dim, dropout)
            self.model = backbone

        elif model_name == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b3(weights=weights)
            backbone.classifier = _make_head(in_features, output_dim, dropout)
            self.model = backbone

        elif model_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
            backbone.fc = _make_head(in_features, output_dim, dropout)
            self.model = backbone

        elif model_name == "vit_small":
            if not _TIMM_AVAILABLE:
                raise ImportError(
                    "timm is required for the 'vit_small' backbone. "
                    "Install it with: pip install timm"
                )
            backbone = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=pretrained,
                img_size=img_size,
            )
            backbone.head = _make_head(in_features, output_dim, dropout)
            self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, 3, H, W].

        Returns:
            Logits of shape [B, output_dim].
            - If output_dim=101: pass through ``MeanVarianceLoss`` to get ages.
            - If output_dim=1: raw regression values; inference bounding belongs
              to the checkpoint's age decoder, not the training graph.
        """
        return self.model(x)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count (in millions, rounded to 2 decimals)."""
        return round(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6, 2)


def build_model(cfg) -> AgeEstimationModel:
    """Instantiate an ``AgeEstimationModel`` from an OmegaConf config node.

    Args:
        cfg: OmegaConf config with a ``model`` sub-tree containing
             ``name``, ``pretrained``, ``output_dim``, and ``dropout``.

    Returns:
        Instantiated model (not yet moved to a device).
    """
    if "loss" in cfg:
        validate_mode(cfg.model.output_dim, cfg.loss.type, cfg.loss.start_age, cfg.loss.end_age)
    return AgeEstimationModel(
        model_name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        img_size=cfg.data.img_size if "data" in cfg else 224,
    )
