#!/usr/bin/env python3
"""Prepare a new, validated UTKFace split protocol without overwriting existing outputs.

Example: python scripts/prepare_data.py --dataset-dir /path/to/utkcropped \
    --csv-dir csv_dataset/protocol_v2 --check-decode
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from src.configs import load_config
from src.data.validation import (
    COLUMNS,
    PROTOCOL,
    SPLIT_FILES,
    image_path,
    parse_utkface_name,
    sha256_file,
    validate_partition,
    validate_table,
)
from src.utils.config import to_plain_config, validate_split_options
from src.utils.environment import collect_environment


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
    p.add_argument("--check-decode", action="store_true", help="Fully decode each candidate image before splitting")
    p.add_argument("--skip-content-hash", action="store_true", help="Skip byte hashes/duplicate filtering; content leakage remains unknown")
    p.add_argument("--source", default="unknown", help="Dataset source identifier/URL, recorded as supplied")
    p.add_argument("--dataset-version", default="unknown", help="Dataset release/version, recorded as supplied")
    return p.parse_args()


def scan_images(dataset_dir: Path, max_age: int, min_age: int = 0, *, check_decode: bool = False,
                hash_images: bool = True) -> tuple[pd.DataFrame, list[dict], dict | None, list[dict]]:
    """Scan a flat UTKFace folder. Reject invalid entries and all exact-byte duplicate groups."""
    from PIL import Image

    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset folder is not a directory: {dataset_dir}")
    if min_age < 0 or max_age <= min_age:
        raise ValueError("Require 0 <= min_age < max_age (exclusive)")
    rows, rejected, hashes = [], [], {}
    for entry in sorted(dataset_dir.iterdir(), key=lambda item: item.name):
        reason = None
        try:
            path = image_path(dataset_dir, entry.name)
        except ValueError as exc:
            reason = str(exc)
        if reason is None:
            try:
                row = parse_utkface_name(entry.name)
                age = row["age"]
            except ValueError:
                reason = "invalid_filename_or_labels"
        if reason is None and not min_age <= age < max_age:
            reason = "age_filter"
        if reason is None and check_decode:
            try:
                with Image.open(path) as image:
                    image.convert("RGB").load()
            except (OSError, ValueError, Image.DecompressionBombError):
                reason = "decode_failed"
        if reason is None and hash_images:
            try:
                hashes[entry.name] = sha256_file(path)
            except OSError:
                reason = "content_read_failed"
        if reason is not None:
            rejected.append({"image_name": entry.name, "reason": reason})
            continue
        rows.append(row)
    groups = {}
    for name, digest in hashes.items():
        groups.setdefault(digest, []).append(name)
    duplicates = [{"sha256": digest, "image_names": names} for digest, names in sorted(groups.items()) if len(names) > 1]
    duplicate_names = {name for group in duplicates for name in group["image_names"]}
    # Removing every member avoids choosing an arbitrary target when labels conflict.
    rows = [row for row in rows if row["image_name"] not in duplicate_names]
    rejected.extend({"image_name": name, "reason": "duplicate_content"} for name in sorted(duplicate_names))
    hashes = {name: digest for name, digest in hashes.items() if name not in duplicate_names}
    rejected.sort(key=lambda row: row["image_name"])
    if not rows:
        raise ValueError(f"No valid images remain in {dataset_dir}; {len(rejected)} entries rejected. "
                         f"Reasons: {dict(Counter(row['reason'] for row in rejected))}")
    return pd.DataFrame(rows, columns=COLUMNS), rejected, hashes if hash_images else None, duplicates


def create_master_csv(dataset_dir: Path, csv_path: Path, max_age: int, min_age: int = 0,
                      *, check_decode: bool = False, hash_images: bool = True) -> pd.DataFrame:
    """Compatibility helper for a master-only scan; never overwrites an existing file."""
    if csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite {csv_path}; choose a new output path")
    df, _, _, _ = scan_images(dataset_dir, max_age, min_age, check_decode=check_decode, hash_images=hash_images)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, mode="x")
    return df


def stratified_split(df: pd.DataFrame, train_ratio: float, val_ratio: float,
                     seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage age stratification, with explicit feasibility errors and stable input order."""
    import math

    validate_split_options(train_ratio, val_ratio, seed)
    df = validate_table(df).sort_values("image_name").reset_index(drop=True)
    if df.empty:
        raise ValueError("Cannot stratify an empty dataset")
    train_fraction, val_fraction = Fraction(str(train_ratio)), Fraction(str(val_ratio))
    test_ratio = 1 - train_fraction - val_fraction

    def split(frame, ratio, stage):
        counts = frame["age"].value_counts().sort_index()
        held = math.ceil(len(frame) * ratio)
        if counts.min() < 2 or min(held, len(frame) - held) < len(counts):
            raise ValueError(f"Age-stratified {stage} split is infeasible: n={len(frame)}, "
                             f"held_out={held}, classes={len(counts)}, counts={counts.to_dict()}. "
                             "Use more samples or explicit different ratios/age filters; no unstratified fallback.")
        try:
            left, right = train_test_split(frame, test_size=held, stratify=frame["age"], random_state=seed)
            if set(left["age"]) != set(counts.index) or set(right["age"]) != set(counts.index):
                raise ValueError(f"some ages would disappear from a split; age counts={counts.to_dict()}")
            return left, right
        except ValueError as exc:
            raise ValueError(f"Age-stratified {stage} split is infeasible: {exc}") from exc

    train_val, test = split(df, test_ratio, "test")
    # Requested val share of the original population, relative to the remaining fraction.
    train, val = split(train_val, val_fraction / (train_fraction + val_fraction), "validation")
    result = tuple(frame.sort_values("image_name").reset_index(drop=True) for frame in (train, val, test))
    validate_partition(dict(zip(SPLIT_FILES, result)), df)
    return result


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    csv_dir: Path,
) -> None:
    for filename in SPLIT_FILES.values():
        if (csv_dir / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {csv_dir / filename}; choose a new --csv-dir")
    csv_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(csv_dir / "train_set.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, mode="x")
    df_val.to_csv(csv_dir / "valid_set.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, mode="x")
    df_test.to_csv(csv_dir / "test_set.csv", index=False, quoting=csv.QUOTE_NONNUMERIC, mode="x")

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

    if csv_dir.exists() and (not csv_dir.is_dir() or any(csv_dir.iterdir())):
        raise FileExistsError(f"Refusing to write into nonempty output {csv_dir}; choose a new --csv-dir, "
                              "for example csv_dataset/protocol_v2. Existing protocols are preserved.")
    if not args.source.strip() or not args.dataset_version.strip():
        raise ValueError("--source and --dataset-version must be nonempty (use unknown when unverified)")
    # Scan and prove split feasibility before creating output.
    df, rejected, content_hashes, duplicates = scan_images(
        dataset_dir, max_age=cfg.data.max_age, min_age=cfg.loss.start_age,
        check_decode=args.check_decode, hash_images=not args.skip_content_hash)

    # 2. Split
    df_train, df_val, df_test = stratified_split(
        df,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.training.seed,
    )
    validate_partition({"train": df_train, "val": df_val, "test": df_test}, df, root=dataset_dir)

    # 3. Save only into a new/empty directory. No overwrite escape hatch.
    csv_dir.mkdir(parents=True, exist_ok=True)
    master_csv = csv_dir / "utkface_dataset.csv"
    df.to_csv(master_csv, index=False, quoting=csv.QUOTE_NONNUMERIC, mode="x")
    save_splits(df_train, df_val, df_test, csv_dir)
    manifest = {
        "kind": "preparation_parameters", "schema_version": 1,
        "parameters": {"min_age": cfg.loss.start_age, "max_age_exclusive": cfg.data.max_age,
                       "train_ratio": cfg.data.train_ratio, "val_ratio": cfg.data.val_ratio,
                       "test_ratio": float(1 - Fraction(str(cfg.data.train_ratio)) - Fraction(str(cfg.data.val_ratio))),
                       "seed": cfg.training.seed},
        "counts": {"master": len(df), "train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "config": to_plain_config(cfg),
    }
    (csv_dir / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    reject_path = csv_dir / "rejected_samples.csv"
    pd.DataFrame(rejected, columns=["image_name", "reason"]).to_csv(reject_path, index=False, mode="x")
    tables = {"utkface_dataset.csv": df, **dict(zip(SPLIT_FILES.values(), (df_train, df_val, df_test)))}
    split_manifest = {
        "schema_version": 1, "protocol": PROTOCOL,
        "source": {"identifier": args.source, "version": args.dataset_version,
                   "root_dir": str(dataset_dir.resolve()), "annotation_quality": "not_verified"},
        "parameters": manifest["parameters"],
        "algorithm": "sorted_names_then_sklearn_two_stage_stratified_by_age",
        "allocation": "ceil_of_exact_decimal_fraction_at_each_stage",
        "validation": {"decode_checked": args.check_decode, "content_hashed": not args.skip_content_hash,
                       "duplicate_policy": "exclude_all_exact_byte_duplicates" if not args.skip_content_hash else "not_checked",
                       "identity_leakage": "unknown", "near_duplicates": "not_checked"},
        "counts": {**manifest["counts"], "rejected": len(rejected), "scanned": len(df) + len(rejected)},
        "csv_files": {name: {"sha256": sha256_file(csv_dir / name), "count": len(frame)}
                      for name, frame in tables.items()},
        "rejects": {"file": reject_path.name, "sha256": sha256_file(reject_path),
                    "by_reason": pd.Series([row["reason"] for row in rejected], dtype=str).value_counts().to_dict()},
        "content_sha256": content_hashes, "duplicate_groups": duplicates,
        "environment": collect_environment(),
    }
    (csv_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8")

    # 4. Optional plot
    if args.plot:
        plot_distributions(df_train, df_val, df_test, csv_dir / "split_distributions.png")

    print(f"\nAll done!  Next step: python scripts/train.py data.csv_dir={csv_dir} data.root_dir={dataset_dir}")


if __name__ == "__main__":
    main()
