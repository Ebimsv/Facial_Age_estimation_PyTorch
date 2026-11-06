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

import json

import torch

from src.utils.checkpoint import add_checkpoint_arguments, load_checkpoint_args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export age estimation model to TorchScript or ONNX")
    add_checkpoint_arguments(p)
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
    if args.device not in (None, "cpu"):
        raise ValueError("Export currently runs on CPU; use --device cpu")
    bundle = load_checkpoint_args(args, runtime_overrides={"device": "cpu"})
    model = bundle.model
    img_size = bundle.cfg.data.img_size
    if args.img_size is not None and args.img_size != img_size:
        raise ValueError("--img-size cannot change the checkpoint preprocessing contract")

    dummy = torch.zeros(1, 3, img_size, img_size)

    if args.format == "torchscript":
        out_path = args.output or str(ckpt_path.with_suffix(".jit"))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # Use torch.jit.trace (faster than script for CNNs)
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)
        torch.jit.save(traced, out_path)
        print(f"✓ TorchScript model saved → {out_path}")

    elif args.format == "onnx":
        out_path = args.output or str(ckpt_path.with_suffix(".onnx"))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
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

    metadata = {key: bundle.metadata[key] for key in
                ["schema_version", "model_spec", "preprocessing", "predictor"]}
    metadata["output"] = "logits"
    Path(str(out_path) + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
