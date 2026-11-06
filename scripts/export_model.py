#!/usr/bin/env python3
"""Export a trained model to TorchScript (.jit) or ONNX format.

Usage
-----
  # TorchScript (default)
  python scripts/export_model.py --checkpoint checkpoints/best_*.pt

  # ONNX
  python scripts/export_model.py --checkpoint checkpoints/best_*.pt --format onnx

  # Custom output path
  python scripts/export_model.py --checkpoint checkpoints/best_*.pt --output model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.configs import load_config
from src.models import AgeEstimationModel
from src.utils.config import validate_mode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export age estimation model to TorchScript or ONNX")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument(
        "--config",
        type=str,
        default=None,
    )
    p.add_argument(
        "--format",
        choices=["torchscript", "onnx"],
        default="torchscript",
        help="Export format (default: torchscript)",
    )
    p.add_argument("--output", type=str, default=None, help="Output file path")
    p.add_argument("--img-size", type=int, default=None, help="Input image size (default: config data.img_size)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = load_config(args.config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_name = ckpt.get("model_name", cfg.model.name)
    output_dim = ckpt.get("output_dim", cfg.model.output_dim)
    validate_mode(output_dim, cfg.loss.type, cfg.loss.start_age, cfg.loss.end_age)
    img_size = args.img_size if args.img_size is not None else cfg.data.img_size

    model = AgeEstimationModel(
        model_name=model_name,
        pretrained=False,
        output_dim=output_dim,
        dropout=0.0,
        img_size=img_size,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, img_size, img_size)

    if args.format == "torchscript":
        out_path = args.output or str(ckpt_path.with_suffix(".jit"))
        # Use torch.jit.trace (faster than script for CNNs)
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)
        torch.jit.save(traced, out_path)
        print(f"✓ TorchScript model saved → {out_path}")

    elif args.format == "onnx":
        out_path = args.output or str(ckpt_path.with_suffix(".onnx"))
        torch.onnx.export(
            model,
            dummy,
            out_path,
            export_params=True,
            opset_version=17,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        print(f"✓ ONNX model saved → {out_path}")
        print("  Verify with: python -c \"import onnx; onnx.checker.check_model('" + out_path + "')\"")


if __name__ == "__main__":
    main()
