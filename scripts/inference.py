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
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from src.models import AgeEstimationModel
from src.training import get_loss_fn

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model_and_pipeline(checkpoint_path: str, config_path: str):
    """Load model, loss function, and preprocessing transform from a checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(config_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_name = ckpt.get("model_name", cfg.model.name)
    output_dim = ckpt.get("output_dim", cfg.model.output_dim)

    model = AgeEstimationModel(
        model_name=model_name,
        pretrained=False,
        output_dim=output_dim,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loss_fn = get_loss_fn(cfg).to(device)

    img_size = cfg.data.img_size
    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=list(cfg.data.mean), std=list(cfg.data.std)),
        ]
    )

    return model, loss_fn, transform, device


@torch.no_grad()
def predict_age(model, loss_fn, transform, device, image_path: Path) -> float:
    """Return the predicted age for a single image."""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    return loss_fn.predict_age(logits).item()


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
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )

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
    model, loss_fn, transform, device = load_model_and_pipeline(
        args.checkpoint, args.config
    )
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
