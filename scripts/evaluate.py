#!/usr/bin/env python3
"""Evaluate a trained model on the held-out test set.

Usage
-----
  python scripts/evaluate.py --checkpoint checkpoints/best_efficientnet_b3_epoch042_mae4.123.pt

Optional flags
--------------
  --config     Override the config file (default: configs/default.yaml)
  --split      Which split to evaluate: test | val | train  (default: test)
  --bias       Print per-age-decade, per-gender, and per-ethnicity MAE breakdown
  --save-csv   Save per-image predictions to a CSV file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data import build_dataloaders, get_test_transforms, get_train_transforms, get_val_transforms
from src.models import AgeEstimationModel
from src.training import get_loss_fn
from src.utils import compute_cumulative_score, compute_mae, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained age estimation model")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--bias", action="store_true", help="Print demographic bias breakdown")
    p.add_argument("--save-csv", type=str, default=None, help="Path to save prediction CSV")
    return p.parse_args()


@torch.no_grad()
def run_evaluation(
    model: AgeEstimationModel,
    loader,
    loss_fn,
    device: torch.device,
    use_amp: bool,
) -> tuple[list, list, list, list]:
    """Collect predictions, targets, genders, and ethnicities."""
    model.eval()
    all_preds, all_targets, all_genders, all_ethnicities = [], [], [], []

    for images, ages, genders, ethnicities in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)

        pred_ages = loss_fn.predict_age(logits).cpu()

        all_preds.extend(pred_ages.tolist())
        all_targets.extend(ages.cpu().view(-1).tolist())
        all_genders.extend(list(genders))
        all_ethnicities.extend(list(ethnicities))

    return all_preds, all_targets, all_genders, all_ethnicities


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: {ckpt_path.name}")
    print(f"  Trained for: {ckpt.get('epoch', '?')} epochs")
    print(f"  Best val MAE: {ckpt.get('val_mae', '?'):.3f}y")

    # ── Config ────────────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    # Restore model settings from checkpoint if available
    if "cfg" in ckpt:
        saved_cfg = OmegaConf.create(ckpt["cfg"])
        cfg = OmegaConf.merge(cfg, saved_cfg)

    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.training.use_amp and device.type == "cuda"

    # ── Model ─────────────────────────────────────────────────────────────
    model = AgeEstimationModel(
        model_name=ckpt.get("model_name", cfg.model.name),
        pretrained=False,
        output_dim=ckpt.get("output_dim", cfg.model.output_dim),
        dropout=0.0,  # disable dropout at test time
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model: {model.model_name}  (output_dim={model.output_dim})")

    # ── Loss / predictor ──────────────────────────────────────────────────
    loss_fn = get_loss_fn(cfg).to(device)

    # ── Data ──────────────────────────────────────────────────────────────
    mean, std = list(cfg.data.mean), list(cfg.data.std)
    img_size = cfg.data.img_size

    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir=PROJECT_ROOT / cfg.data.root_dir,
        csv_dir=PROJECT_ROOT / cfg.data.csv_dir,
        train_transform=get_train_transforms(img_size, mean, std),
        val_transform=get_val_transforms(img_size, mean, std),
        test_transform=get_test_transforms(img_size, mean, std),
        batch_size=cfg.training.batch_size,
        eval_batch_size=cfg.training.eval_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    # ── Evaluate ──────────────────────────────────────────────────────────
    preds, targets, genders, ethnicities = run_evaluation(
        model, loader, loss_fn, device, use_amp
    )

    preds_t = torch.tensor(preds)
    targets_t = torch.tensor(targets)

    mae = compute_mae(preds_t, targets_t)
    cs5 = compute_cumulative_score(preds_t, targets_t, threshold=5)
    cs10 = compute_cumulative_score(preds_t, targets_t, threshold=10)

    print(f"\n{'='*50}")
    print(f"  Split  : {args.split}  (n={len(preds)})")
    print(f"  MAE    : {mae:.3f} years")
    print(f"  CS@5   : {cs5:.3f}  ({cs5*100:.1f}% within 5 years)")
    print(f"  CS@10  : {cs10:.3f}  ({cs10*100:.1f}% within 10 years)")
    print(f"{'='*50}")

    # ── Bias breakdown ────────────────────────────────────────────────────
    if args.bias:
        df = pd.DataFrame(
            {
                "pred": preds,
                "target": targets,
                "error": [abs(p - t) for p, t in zip(preds, targets)],
                "gender": genders,
                "ethnicity": ethnicities,
                "decade": [int(t // 10) * 10 for t in targets],
            }
        )
        print("\n── MAE by Gender ───────────────────────────────────────")
        print(df.groupby("gender")["error"].mean().round(3).to_string())

        print("\n── MAE by Ethnicity ────────────────────────────────────")
        print(df.groupby("ethnicity")["error"].mean().round(3).to_string())

        print("\n── MAE by Age Decade ───────────────────────────────────")
        print(df.groupby("decade")["error"].mean().round(3).to_string())

    # ── Save CSV ──────────────────────────────────────────────────────────
    if args.save_csv:
        out = pd.DataFrame(
            {"predicted_age": preds, "true_age": targets, "gender": genders, "ethnicity": ethnicities}
        )
        out.to_csv(args.save_csv, index=False)
        print(f"\nPredictions saved → {args.save_csv}")


if __name__ == "__main__":
    main()
