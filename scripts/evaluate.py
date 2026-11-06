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
from tqdm import tqdm

from src.configs import get_runtime_root
from src.data import build_dataloaders
from src.models import AgeEstimationModel
from src.utils import compute_cumulative_score, compute_mae, set_seed
from src.utils.checkpoint import add_checkpoint_arguments, load_checkpoint_args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained age estimation model")
    add_checkpoint_arguments(p)
    p.add_argument("--root-dir", default=None, help="Runtime image root override")
    p.add_argument("--csv-dir", default=None, help="Runtime split CSV root override")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
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
    overrides = {}
    for flag, key in [("root_dir", "data.root_dir"), ("csv_dir", "data.csv_dir"),
                      ("num_workers", "data.num_workers"), ("eval_batch_size", "training.eval_batch_size")]:
        if getattr(args, flag) is not None:
            overrides[key] = getattr(args, flag)
    bundle = load_checkpoint_args(args, runtime_overrides=overrides)
    cfg, model, loss_fn, device = bundle.cfg, bundle.model, bundle.loss_fn, bundle.device
    use_amp = cfg.training.use_amp and device.type == "cuda"
    print(f"Loaded checkpoint: {Path(args.checkpoint).name}")
    print(f"  Epoch index: {bundle.metadata.get('epoch', 'unknown')}")
    print(f"  Validation MAE: {bundle.metadata.get('val_mae', 'unknown')}")
    set_seed(cfg.training.seed, deterministic=cfg.training.deterministic)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir=get_runtime_root() / cfg.data.root_dir,
        csv_dir=get_runtime_root() / cfg.data.csv_dir,
        train_transform=bundle.transform,
        val_transform=bundle.transform,
        test_transform=bundle.transform,
        batch_size=cfg.training.batch_size,
        eval_batch_size=cfg.training.eval_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
        age_range=(cfg.loss.start_age, cfg.loss.end_age),
        seed=cfg.training.seed,
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
