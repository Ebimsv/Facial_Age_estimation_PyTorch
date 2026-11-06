"""Tests for UTKDataset using synthetic images and CSVs only."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import UTKDataset
from src.data.transforms import get_val_transforms


def _make_dummy_dataset(tmp_path: Path, n: int = 10) -> tuple[Path, Path]:
    """Create a tiny dummy dataset in a temp folder."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    csv_path = tmp_path / "data.csv"

    rows = []
    for i in range(n):
        fname = f"{20 + i}_0_0_{i:08d}.jpg"
        # Create a tiny RGB image
        img = Image.fromarray(
            (torch.randint(0, 255, (64, 64, 3)).numpy()).astype("uint8")
        )
        img.save(img_dir / fname)
        rows.append({"image_name": fname, "age": 20 + i, "gender": "Male", "ethnicity": "White"})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "age", "gender", "ethnicity"])
        writer.writeheader()
        writer.writerows(rows)

    return img_dir, csv_path


def test_dataset_length(tmp_path: Path) -> None:
    img_dir, csv_path = _make_dummy_dataset(tmp_path, n=8)
    ds = UTKDataset(root_dir=img_dir, csv_file=csv_path)
    assert len(ds) == 8


def test_dataset_getitem_shapes(tmp_path: Path) -> None:
    img_dir, csv_path = _make_dummy_dataset(tmp_path, n=4)
    transform = get_val_transforms(img_size=64)
    ds = UTKDataset(root_dir=img_dir, csv_file=csv_path, transform=transform)
    image, age, gender, ethnicity = ds[0]
    assert image.shape == (3, 64, 64)
    assert age.shape == (1,)
    assert age.dtype == torch.float32
    assert isinstance(gender, str)
    assert isinstance(ethnicity, str)


def test_dataset_missing_column_raises(tmp_path: Path) -> None:
    """A CSV without the 'age' column should raise ValueError."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("image_name,gender\ntest.jpg,Male\n")

    with pytest.raises(ValueError, match="missing columns"):
        UTKDataset(root_dir=img_dir, csv_file=csv_path)


def test_dataset_age_tensor_range(tmp_path: Path) -> None:
    """All ages in the test dummy should be in [20, 29]."""
    img_dir, csv_path = _make_dummy_dataset(tmp_path, n=10)
    transform = get_val_transforms(img_size=32)
    ds = UTKDataset(root_dir=img_dir, csv_file=csv_path, transform=transform)
    ages = [ds[i][1].item() for i in range(len(ds))]
    assert all(20 <= a <= 29 for a in ages), f"Unexpected ages: {ages}"


def test_corrupt_image_raises_contextual_decode_error(tmp_path):
    img_dir, csv_path = _make_dummy_dataset(tmp_path, n=1)
    path = next(img_dir.iterdir())
    path.write_bytes(b"not an image")
    ds = UTKDataset(img_dir, csv_path)
    with pytest.raises(ValueError, match="Could not decode dataset image") as error:
        ds[0]
    assert path.name in str(error.value)
