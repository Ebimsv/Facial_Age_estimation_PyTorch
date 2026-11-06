#!/usr/bin/env python3
"""Training entry point for facial age estimation.

Usage
-----
# Train with defaults (efficientnet_b3, l1_mean_variance loss, 100 epochs):
  python scripts/train.py

# Override any config key with dotlist syntax:
  python scripts/train.py model.name=resnet50 training.epochs=50

# Use W&B logging:
  python scripts/train.py logging.use_wandb=true

# Regression mode (L1 loss, output_dim=1):
  python scripts/train.py model.output_dim=1 loss.type=l1

# Different image resolution:
  python scripts/train.py data.img_size=256 training.batch_size=64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the project root is on the path regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf

from src.configs import get_runtime_root, load_config
from src.data import (
    build_dataloaders,
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)
from src.models import build_model
from src.training import Trainer, get_loss_fn
from src.utils import set_seed
from src.utils.checkpoint import load_training_checkpoint
from src.utils.config import merge_config, validate_training_steps
from src.utils.training_state import training_config_signature


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a facial age estimation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--print-config", action="store_true", help="Print resolved validated config and exit without data/model I/O",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (default: packaged defaults)",
    )
    p.add_argument("--resume", default=None, help="P2.1 best/last checkpoint; resume from its completed epoch")
    p.add_argument(
        "overrides",
        nargs="*",
        help="Key=value overrides, e.g. model.name=resnet50 training.epochs=50",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load & merge config ───────────────────────────────────────────────
    if args.resume:
        artifact = load_training_checkpoint(args.resume)
        cfg = load_config(updates=artifact["cfg"])
        cfg = merge_config(cfg, OmegaConf.load(args.config) if args.config else {}, overrides=args.overrides)
        if training_config_signature(cfg) != artifact["training_state"]["contract"]["config"]:
            raise ValueError("Resume configuration changed; epochs/optimizer/data settings must retain the saved schedule")
        del artifact
    else:
        cfg = load_config(args.config, overrides=args.overrides)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return

    # ── Reproducibility ───────────────────────────────────────────────────
    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)

    # ── Device ────────────────────────────────────────────────────────────
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device requests CUDA, but CUDA is unavailable; use device=cpu")
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    # ── Data ──────────────────────────────────────────────────────────────
    mean = list(cfg.data.mean)
    std = list(cfg.data.std)
    img_size = cfg.data.img_size

    train_loader, val_loader, _ = build_dataloaders(
        root_dir=get_runtime_root() / cfg.data.root_dir,
        csv_dir=get_runtime_root() / cfg.data.csv_dir,
        train_transform=get_train_transforms(img_size, mean, std),
        val_transform=get_val_transforms(img_size, mean, std),
        test_transform=get_test_transforms(img_size, mean, std),
        batch_size=cfg.training.batch_size,
        eval_batch_size=cfg.training.eval_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
        age_range=(cfg.loss.start_age, cfg.loss.end_age),
        seed=cfg.training.seed,
    )
    validate_training_steps(cfg, len(train_loader))

    print(
        f"Dataset split sizes → "
        f"train: {len(train_loader.dataset)}, "
        f"val: {len(val_loader.dataset)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cfg, pretrained=False).to(device) if args.resume else build_model(cfg).to(device)
    print(f"Model: {cfg.model.name}  ({model.num_parameters}M trainable params)")

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_fn = get_loss_fn(cfg)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(model=model, loss_fn=loss_fn, cfg=cfg, device=device,
                      data_provenance=train_loader.data_provenance)
    trainer.run(train_loader, val_loader, resume=args.resume)


if __name__ == "__main__":
    main()
