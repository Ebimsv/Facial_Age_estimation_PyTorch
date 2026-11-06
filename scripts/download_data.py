#!/usr/bin/env python3
"""Download the UTKFace dataset from Kaggle.

Usage
-----
  python scripts/download_data.py                      # download to dataset/utkcropped
  python scripts/download_data.py --dest my/path       # custom destination
  python scripts/download_data.py --check-only         # just verify existing data

Setup (one-time)
----------------
1. Create a free Kaggle account at https://www.kaggle.com
2. Go to https://www.kaggle.com/settings  →  scroll to "API"  →  "Create New Token"
3. This downloads  kaggle.json  to your Downloads folder
4. Move it to the correct location:

   Linux / macOS:
       mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json

   Windows:
       Move kaggle.json to  C:\\Users\\<username>\\.kaggle\\kaggle.json

5. Run this script:
       python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "jangedoo/utkface-new"
EXPECTED_SUBFOLDER = "utkcropped"           # the folder we actually use
MIN_IMAGES = 18_000                          # sanity-check threshold
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "dataset"


# ── Helpers ──────────────────────────────────────────────────────────────────

def count_images(folder: Path) -> int:
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def verify_dataset(dest: Path) -> bool:
    """Return True if a valid UTKFace dataset already exists at dest."""
    target = dest / EXPECTED_SUBFOLDER
    if not target.exists():
        return False
    n = count_images(target)
    if n < MIN_IMAGES:
        print(f"  ⚠  Found only {n} images in '{target}' (expected ≥ {MIN_IMAGES}).")
        return False
    print(f"  ✓  Dataset already present: {n:,} images in '{target}'")
    return True


def check_kaggle_credentials() -> bool:
    """Return True if kaggle.json is present and readable."""
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if cred.exists():
        return True
    # Also accept KAGGLE_USERNAME + KAGGLE_KEY env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def print_setup_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║              Kaggle API credentials not found                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  One-time setup (takes ~2 minutes):                                      ║
║                                                                          ║
║  1. Create a free Kaggle account at  https://www.kaggle.com              ║
║                                                                          ║
║  2. Go to  https://www.kaggle.com/settings                               ║
║     Scroll to the "API" section → click  "Create New API Token"          ║
║     This downloads  kaggle.json                                           ║
║                                                                          ║
║  3. Move it to the right place:                                           ║
║                                                                          ║
║     Linux / macOS:                                                        ║
║       mkdir -p ~/.kaggle                                                  ║
║       mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json                   ║
║       chmod 600 ~/.kaggle/kaggle.json                                    ║
║                                                                          ║
║     Windows (PowerShell):                                                 ║
║       New-Item -ItemType Directory -Force "$HOME\\.kaggle"               ║
║       Move-Item "$HOME\\Downloads\\kaggle.json" "$HOME\\.kaggle\\"       ║
║                                                                          ║
║  4. Run this script again:                                                ║
║       python scripts/download_data.py                                    ║
║                                                                          ║
║  Alternative — set environment variables instead of kaggle.json:         ║
║     export KAGGLE_USERNAME=your_username                                  ║
║     export KAGGLE_KEY=your_api_key                                        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def download_from_kaggle(dest: Path) -> None:
    """Download and extract the UTKFace dataset using the Kaggle API."""
    try:
        import kaggle  # noqa: PLC0415  (imported here to give better error msg)
    except ImportError:
        print("  Installing kaggle library...")
        import subprocess  # noqa: PLC0415
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "-q"], check=True)
        import kaggle  # noqa: PLC0415

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "utkface-new.zip"

    print(f"\n  Downloading '{KAGGLE_DATASET}' from Kaggle...")
    print(f"  Destination: {dest}")
    print("  (This may take a few minutes — the dataset is ~400 MB)\n")

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(dest),
        unzip=False,
        quiet=False,
    )

    # Find the downloaded zip (Kaggle names it after the dataset slug)
    zips = list(dest.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"No zip file found in '{dest}' after download.")
    zip_path = zips[0]

    print(f"\n  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, dest)
            if i % 2000 == 0 or i == total:
                print(f"    {i:>6}/{total} files extracted", end="\r")
    print()

    zip_path.unlink()
    print("  ✓  Extraction complete. Zip file removed.")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download the UTKFace dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dest",
        type=str,
        default=str(DEFAULT_DEST),
        help=f"Download destination (default: {DEFAULT_DEST})",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify that the dataset exists; do not download",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest)

    print("\n🔍 Checking for existing dataset...")
    if verify_dataset(dest):
        utk_path = dest / EXPECTED_SUBFOLDER
        n = count_images(utk_path)
        print(f"\n✅ Dataset ready at: {utk_path}")
        print(f"   {n:,} images found — no download needed.\n")
        print("Next step:")
        print(f"  python scripts/prepare_data.py --dataset-dir {utk_path} --plot\n")
        return

    if args.check_only:
        print("\n❌ Dataset not found. Run without --check-only to download.\n")
        sys.exit(1)

    # ── Download ──────────────────────────────────────────────────────────────
    if not check_kaggle_credentials():
        print_setup_instructions()
        sys.exit(1)

    download_from_kaggle(dest)

    # ── Verify ────────────────────────────────────────────────────────────────
    print("\n🔍 Verifying download...")
    if not verify_dataset(dest):
        print("\n❌ Verification failed. The download may be incomplete.")
        print(f"   Please check the contents of: {dest}")
        sys.exit(1)

    utk_path = dest / EXPECTED_SUBFOLDER
    n = count_images(utk_path)
    print(f"\n✅ UTKFace dataset ready: {n:,} images at '{utk_path}'")
    print("\nNext step:")
    print(f"  python scripts/prepare_data.py --dataset-dir {utk_path} --plot\n")


if __name__ == "__main__":
    main()
