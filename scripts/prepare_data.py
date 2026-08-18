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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Label maps ───────────────────────────────────────────────────────────────
GENDER_MAP = {0: "Male", 1: "Female"}
ETHNICITY_MAP = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare UTKFace CSV splits")
    p.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/utkcropped",
        help="Path to the folder containing UTKFace images",
    )
    p.add_argument(
        "--csv-dir",
        type=str,
        default="csv_dataset",
        help="Output folder for generated CSV files",
    )
    p.add_argument(
        "--max-age",
        type=int,
        default=85,
        help="Exclude images with age >= max_age (default: 85)",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Fraction of data used for training (default: 0.80)",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Fraction of data used for validation (default: 0.10)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save age-distribution histograms to csv_dir/split_distributions.png",
    )
    return p.parse_args()


def create_master_csv(dataset_dir: Path, csv_path: Path, max_age: int) -> pd.DataFrame:
    """Scan image files and write master CSV with labels.

    Args:
        dataset_dir: Folder containing UTKFace images.
        csv_path:    Output CSV path.
        max_age:     Exclude images with age >= max_age.

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

        if age >= max_age:
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
    test_ratio = round(1.0 - train_ratio - val_ratio, 4)
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

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
    dataset_dir = Path(args.dataset_dir)
    csv_dir = Path(args.csv_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: '{dataset_dir}'\n"
            "Download the UTKFace cropped dataset from Kaggle and point "
            "--dataset-dir at the 'utkcropped' folder."
        )

    # 1. Build master CSV
    master_csv = csv_dir / "utkface_dataset.csv"
    df = create_master_csv(dataset_dir, master_csv, max_age=args.max_age)

    # 2. Split
    df_train, df_val, df_test = stratified_split(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 3. Save
    save_splits(df_train, df_val, df_test, csv_dir)

    # 4. Optional plot
    if args.plot:
        plot_distributions(df_train, df_val, df_test, csv_dir / "split_distributions.png")

    print("\nAll done!  Next step: python scripts/train.py")


if __name__ == "__main__":
    main()
