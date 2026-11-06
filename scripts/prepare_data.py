#!/usr/bin/env python3
"""Prepare UTKFace CSV files for training.

Usage
-----
  python scripts/prepare_data.py --dataset-dir dataset/utkcropped
  python scripts/prepare_data.py --dataset-dir /path/to/utkcropped --csv-dir csv_dataset

What this script does
---------------------
1. Scans the UTKFace image folder.
2. Parses age / gender / ethnicity from each filename (<age>_<gender>_<eth>_<ts>.jpg).
3. Filters out images with age >= max_age (default 85, removes very-sparse tail).
4. Saves a master CSV: csv_dataset/utkface_dataset.csv
5. Performs **stratified** train / val / test split to preserve age distribution.
6. Saves: csv_dataset/train_set.csv, valid_set.csv, test_set.csv
7. Prints per-split age statistics for a quick sanity check.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from src.configs import load_config
from src.utils.config import to_plain_config, validate_split_options

# ── Label maps ───────────────────────────────────────────────────────────────
GENDER_MAP = {0: "Male", 1: "Female"}
ETHNICITY_MAP = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare UTKFace CSV splits")
    p.add_argument("--config", default=None, help="YAML defaults; explicit preparation flags take precedence")
    p.add_argument("--print-config", action="store_true", help="Print resolved config without writing CSVs")
    p.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to the folder containing UTKFace images",
    )
    p.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="Output folder for generated CSV files",
    )
    p.add_argument(
        "--max-age",
        type=int,
        default=None,
        help="Exclude images with age >= max_age (default: 85)",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Fraction of data used for training (default: 0.80)",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Fraction of data used for validation (default: 0.10)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save age-distribution histograms to csv_dir/split_distributions.png",
    )
    return p.parse_args()


def create_master_csv(
    dataset_dir: Path, csv_path: Path, max_age: int, min_age: int = 0,
) -> pd.DataFrame:
    """Scan image files and write master CSV with labels.

    Args:
        dataset_dir: Folder containing UTKFace images.
        csv_path:    Output CSV path.
        max_age:     Exclude images with age >= max_age.
        min_age:     Inclusive lower bound from the configured age support.

    Returns:
        DataFrame with columns [image_name, age, gender, ethnicity].
    """
    rows: list[dict] = []
    skipped = 0

    for img_file in sorted(dataset_dir.iterdir()):
        parts = img_file.stem.split("_")
        if len(parts) < 3:
            skipped += 1
            continue

        try:
            age = int(parts[0])
            gender = int(parts[1])
            ethnicity = int(parts[2])
        except ValueError:
            skipped += 1
            continue

        if age < min_age or age >= max_age:
            continue

        rows.append(
            {
                "image_name": img_file.name,
                "age": age,
                "gender": GENDER_MAP.get(gender, "Unknown"),
                "ethnicity": ETHNICITY_MAP.get(ethnicity, "Others"),
            }
        )

    if not rows:
        raise RuntimeError(
            f"No valid images found in '{dataset_dir}'. "
            "Check that the folder contains UTKFace images with the naming "
            "convention <age>_<gender>_<ethnicity>_<timestamp>.jpg"
        )

    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"✓ Master CSV saved  → {csv_path}  ({len(df)} images, {skipped} skipped)")
    return df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split on the age column.

    Stratification ensures each split has the same age distribution,
    which is critical given the long-tailed age distribution in UTKFace.
    """
    validate_split_options(train_ratio, val_ratio, seed)
    test_ratio = round(1.0 - train_ratio - val_ratio, 4)

    # Step 1: hold out test set
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["age"],
        random_state=seed,
    )

    # Step 2: split remaining into train / val
    relative_val = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=relative_val,
        stratify=df_train_val["age"],
        random_state=seed,
    )

    return df_train, df_val, df_test


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    csv_dir: Path,
) -> None:
    df_train.to_csv(csv_dir / "train_set.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    df_val.to_csv(csv_dir / "valid_set.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    df_test.to_csv(csv_dir / "test_set.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"✓ train  → {len(df_train):>5} images  (mean age: {df_train['age'].mean():.1f})")
    print(f"✓ valid  → {len(df_val):>5} images  (mean age: {df_val['age'].mean():.1f})")
    print(f"✓ test   → {len(df_test):>5} images  (mean age: {df_test['age'].mean():.1f})")


def plot_distributions(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, df, title in zip(
        axes,
        [df_train, df_val, df_test],
        ["Train", "Validation", "Test"],
    ):
        ax.hist(df["age"], bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.set_title(f"{title}  (n={len(df)})")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"✓ Distribution plot saved → {out_path}")
    plt.close()


def main() -> None:
    args = parse_args()
    updates = {"data": {}, "training": {}}
    for argument, key in {"dataset_dir": "root_dir", "csv_dir": "csv_dir", "max_age": "max_age",
                          "train_ratio": "train_ratio", "val_ratio": "val_ratio"}.items():
        if getattr(args, argument) is not None:
            updates["data"][key] = getattr(args, argument)
    if args.seed is not None:
        updates["training"]["seed"] = args.seed
    cfg = load_config(args.config, updates=updates)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    dataset_dir = Path(cfg.data.root_dir)
    csv_dir = Path(cfg.data.csv_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: '{dataset_dir}'\n"
            "Download the UTKFace cropped dataset from Kaggle and point "
            "--dataset-dir at the 'utkcropped' folder."
        )

    # 1. Build master CSV
    master_csv = csv_dir / "utkface_dataset.csv"
    df = create_master_csv(dataset_dir, master_csv, max_age=cfg.data.max_age, min_age=cfg.loss.start_age)

    # 2. Split
    df_train, df_val, df_test = stratified_split(
        df,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.training.seed,
    )

    # 3. Save
    save_splits(df_train, df_val, df_test, csv_dir)
    manifest = {
        "kind": "preparation_parameters", "schema_version": 1,
        "parameters": {"min_age": cfg.loss.start_age, "max_age_exclusive": cfg.data.max_age,
                       "train_ratio": cfg.data.train_ratio, "val_ratio": cfg.data.val_ratio,
                       "test_ratio": round(1 - cfg.data.train_ratio - cfg.data.val_ratio, 4),
                       "seed": cfg.training.seed},
        "counts": {"master": len(df), "train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "config": to_plain_config(cfg),
    }
    (csv_dir / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 4. Optional plot
    if args.plot:
        plot_distributions(df_train, df_val, df_test, csv_dir / "split_distributions.png")

    print("\nAll done!  Next step: python scripts/train.py")


if __name__ == "__main__":
    main()
