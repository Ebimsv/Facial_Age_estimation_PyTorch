"""Transform factories for train / validation / test splits.

All transforms accept an ``img_size`` parameter so the image resolution can be
changed from the config without touching any source code.
"""

from __future__ import annotations

import torchvision.transforms as T


def get_train_transforms(
    img_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> T.Compose:
    """Augmented pipeline used during training.

    Strategy:
    - Resize to (img_size + 32) then random-crop to img_size adds scale variance.
    - ColorJitter + RandomGrayscale improve lighting/domain robustness.
    - RandomErasing mimics partial occlusion (sunglasses, scarves, etc.).
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    pad_size = img_size + 32  # crop from a slightly larger image

    return T.Compose(
        [
            T.Resize((pad_size, pad_size)),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
        ]
    )


def get_val_transforms(
    img_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> T.Compose:
    """Deterministic pipeline for validation (centre-crop, no augmentation)."""
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def get_test_transforms(
    img_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> T.Compose:
    """Identical to val transforms — used for final test-set evaluation."""
    return get_val_transforms(img_size=img_size, mean=mean, std=std)
