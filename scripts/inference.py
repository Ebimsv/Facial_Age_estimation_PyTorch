#!/usr/bin/env python3
"""Run age estimation on a single image or a folder of images.

Usage
-----
  # Single image — saves result to output.jpg
  python scripts/inference.py --checkpoint checkpoints/best_*.pt --image img_test/30_1_2.jpg

  # Save annotated image
  python scripts/inference.py --checkpoint checkpoints/best_*.pt \
      --image img_test/30_1_2.jpg --output img_test/output.jpg

  # Batch inference on a folder
  python scripts/inference.py --checkpoint checkpoints/best_*.pt --image-dir img_test/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image, ImageDraw

from src.inference import predict_image
from src.utils.checkpoint import add_checkpoint_arguments, load_checkpoint, load_checkpoint_args

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model_and_pipeline(checkpoint_path: str, config_path: str | None = None, **kwargs):
    """Return the historical tuple API using the shared artifact loader."""
    bundle = load_checkpoint(checkpoint_path, config_path, **kwargs)
    return bundle.model, bundle.loss_fn, bundle.transform, bundle.device


@torch.no_grad()
def predict_age(model, loss_fn, transform, device, image_path: Path) -> float:
    """Return the predicted age for a single image."""
    with Image.open(image_path) as image:
        ages, _ = predict_image(model, loss_fn, transform, device, image)
    return ages.item()


def annotate_image(image_path: Path, age: float, output_path: Path) -> None:
    """Draw the predicted age on the image and save it."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    text = f"Age: {age:.1f} yrs"
    # Draw text with a simple shadow for readability
    draw.text((11, 11), text, fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 80, 80))
    image.save(output_path)
    print(f"  Annotated image saved → {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run age estimation inference")
    add_checkpoint_arguments(p)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--image-dir", type=str, help="Path to a folder of images")

    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for annotated image (single-image mode only)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_checkpoint_args(args)
    model, loss_fn, transform, device = bundle.model, bundle.loss_fn, bundle.transform, bundle.device
    print(f"Model loaded on {device}")

    if args.image:
        img_path = Path(args.image)
        age = predict_age(model, loss_fn, transform, device, img_path)
        print(f"\nImage : {img_path.name}")
        print(f"Predicted Age : {age:.1f} years")

        if args.output:
            annotate_image(img_path, age, Path(args.output))

    elif args.image_dir:
        img_dir = Path(args.image_dir)
        images = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS]
        if not images:
            print(f"No supported images found in '{img_dir}'")
            return

        print(f"\nRunning inference on {len(images)} images in '{img_dir}':")
        for img_path in images:
            age = predict_age(model, loss_fn, transform, device, img_path)
            print(f"  {img_path.name:<40} → {age:.1f} years")


if __name__ == "__main__":
    main()
