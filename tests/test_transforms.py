"""Tests for transform pipelines."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.transforms import get_test_transforms, get_train_transforms, get_val_transforms


def _dummy_pil(size: tuple[int, int] = (300, 300)) -> Image.Image:
    return Image.fromarray((torch.randint(0, 255, (*size, 3)).numpy()).astype("uint8"))


@pytest.mark.parametrize("img_size", [112, 224, 256])
def test_train_transform_output_shape(img_size: int) -> None:
    tf = get_train_transforms(img_size=img_size)
    img = _dummy_pil()
    tensor = tf(img)
    assert tensor.shape == (3, img_size, img_size), (
        f"Expected (3, {img_size}, {img_size}), got {tensor.shape}"
    )


@pytest.mark.parametrize("img_size", [112, 224, 256])
def test_val_transform_output_shape(img_size: int) -> None:
    tf = get_val_transforms(img_size=img_size)
    img = _dummy_pil()
    tensor = tf(img)
    assert tensor.shape == (3, img_size, img_size)


def test_transforms_produce_float_tensors() -> None:
    tf = get_train_transforms()
    img = _dummy_pil()
    tensor = tf(img)
    assert tensor.dtype == torch.float32


def test_normalization_changes_range() -> None:
    """Normalized tensor should NOT be in [0, 1] anymore."""
    tf = get_val_transforms(img_size=224)
    img = _dummy_pil()
    tensor = tf(img)
    # After ImageNet normalisation, values can be negative
    assert tensor.min().item() < 0.0 or tensor.max().item() > 1.0


def test_val_and_test_transforms_are_identical() -> None:
    """Val and test transforms must produce identical outputs for the same input."""
    val_tf = get_val_transforms(img_size=224)
    test_tf = get_test_transforms(img_size=224)
    img = _dummy_pil()
    # Transforms are deterministic, so same input → same output
    assert torch.allclose(val_tf(img), test_tf(img))
