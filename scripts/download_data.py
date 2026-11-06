#!/usr/bin/env python3
"""Download UTKFace into a protected destination, or inspect it without network access.

  python scripts/download_data.py --check-only
  python scripts/download_data.py --dest dataset --dataset-version 1
  python scripts/download_data.py --image-subdir UTKFace

Install the optional dependency yourself with: python -m pip install ".[data]"
Authentication is delegated to Kaggle; see https://github.com/Kaggle/kaggle-cli
The script never installs packages or inspects credential contents.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import stat
import sys
import tempfile
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.data.validation import IMAGE_SUFFIXES, image_path, parse_utkface_name

KAGGLE_DATASET = "jangedoo/utkface-new"
ARCHIVE_NAME = "utkface-new.zip"
EXPECTED_SUBFOLDER = "utkcropped"
DEFAULT_DEST = PROJECT_ROOT / "dataset"
MIN_IMAGES = 18_000
DECODE_SAMPLE = 5
MAX_IMAGE_BYTES = 32 * 1024**2
MAX_ARCHIVE_BYTES = 8 * 1024**3
MAX_MEMBERS = 200_000


class DownloadError(RuntimeError):
    """An actionable error safe to display without SDK exception details."""


class _DiscardOutput:
    def write(self, value):
        return len(value)

    def flush(self):
        pass


def load_kaggle_api():
    """Import the optional SDK only for a requested download; suppress its auth output."""
    try:
        with redirect_stdout(_DiscardOutput()), redirect_stderr(_DiscardOutput()):
            return importlib.import_module("kaggle").api
    except ImportError:
        raise DownloadError('Kaggle is unavailable. Install it explicitly: python -m pip install ".[data]"') from None
    except Exception:
        raise DownloadError("Kaggle initialization/authentication failed. Configure credentials using the official "
                            "Kaggle guide; credential values and SDK details are not displayed.") from None


def inspect_images(folder: Path, *, minimum: int | None = None) -> int:
    """Check valid UTK filenames/paths and decode up to five images; not a full data audit."""
    minimum = MIN_IMAGES if minimum is None else minimum
    if not folder.is_dir() or folder.is_symlink():
        raise DownloadError(f"Dataset image directory is missing or is a symlink: {folder}")
    valid = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            image_path(folder, path.name)
            parse_utkface_name(path.name)
        except ValueError:
            continue
        valid.append(path)
    if len(valid) < minimum:
        raise DownloadError(f"Dataset is incomplete: {len(valid)} valid image names in {folder}; expected >= {minimum}.")
    for path in valid[:DECODE_SAMPLE]:
        try:
            with Image.open(path) as image:
                image.convert("RGB").load()
        except (OSError, ValueError, Image.DecompressionBombError):
            raise DownloadError(f"Dataset sample could not be decoded: {path.name}") from None
    return len(valid)


def verify_dataset(dest: Path) -> bool:
    """Read-only compatibility check of dest/utkcropped; never imports Kaggle."""
    try:
        count = inspect_images(dest / EXPECTED_SUBFOLDER)
    except (DownloadError, OSError) as exc:
        print(f"Dataset check failed: {exc}")
        return False
    print(f"Dataset ready: {count:,} valid image names at {dest / EXPECTED_SUBFOLDER}")
    return True


def _relative_subdir(value: str) -> str:
    if value == ".":
        return value
    path = PurePosixPath(value)
    if not value or "\x00" in value or path.is_absolute() or path.as_posix() != value or any(
        item in {".", ".."} or ":" in item or "\\" in item for item in path.parts
    ):
        raise ValueError("--image-subdir must be a canonical relative archive directory (or .)")
    return value


def select_archive_images(archive: zipfile.ZipFile, image_subdir: str | None = None) -> list[zipfile.ZipInfo]:
    """Validate archive paths/types first; select one flat cropped-image directory."""
    members = archive.infolist()
    if len(members) > MAX_MEMBERS or sum(info.file_size for info in members) > MAX_ARCHIVE_BYTES:
        raise DownloadError("Archive exceeds the supported member-count/uncompressed-size limits.")
    groups, seen = {}, set()
    for info in members:
        name = info.orig_filename
        relative = PurePosixPath(name.rstrip("/"))
        if (not name or "\x00" in name or "\\" in name or relative.is_absolute()
                or relative.as_posix() != name.rstrip("/")
                or any(part in {".", ".."} or ":" in part for part in relative.parts)):
            raise DownloadError("Archive contains an unsafe or noncanonical path.")
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise DownloadError("Archive contains a symlink or unsupported special file.")
        if relative in seen:
            raise DownloadError("Archive contains duplicate member paths.")
        seen.add(relative)
        if info.is_dir():
            continue
        if info.flag_bits & 1:
            raise DownloadError("Encrypted archives are unsupported.")
        try:
            parse_utkface_name(relative.name)
        except ValueError:
            continue
        if info.file_size > MAX_IMAGE_BYTES:
            raise DownloadError("An image exceeds the supported uncompressed-size limit.")
        groups.setdefault(relative.parent.as_posix(), []).append(info)
    if image_subdir is not None:
        selected = _relative_subdir(image_subdir)
        if len(groups.get(selected, [])) < MIN_IMAGES:
            raise DownloadError("Selected --image-subdir does not contain enough valid UTKFace image names.")
    else:
        candidates = sorted(parent for parent, items in groups.items() if len(items) >= MIN_IMAGES and
                            (parent == "." or PurePosixPath(parent).name.lower() in {"utkface", "utkcropped"}))
        if len(candidates) != 1:
            choices = [parent for parent, items in sorted(groups.items()) if len(items) >= MIN_IMAGES]
            raise DownloadError("Cropped-image layout is missing or ambiguous. Inspect the archive layout and pass "
                                f"--image-subdir explicitly; no directory is guessed. Qualifying directories: {choices[:8]!r}")
        selected = candidates[0]
    return sorted(groups[selected], key=lambda info: info.filename)


def extract_images(archive_path: Path, output: Path, image_subdir: str | None = None) -> None:
    """Extract only validated selected image members, never arbitrary archive contents."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = select_archive_images(archive, image_subdir)
            output.mkdir()
            for info in members:
                with archive.open(info) as source, (output / PurePosixPath(info.filename).name).open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
    except DownloadError:
        raise
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, NotImplementedError):
        raise DownloadError("Archive reading/extraction failed; no dataset was published.") from None


def publish_images(source: Path, target: Path) -> None:
    """Reserve the target exclusively; roll back only files created by this call on failure."""
    try:
        target.mkdir()  # A preexisting file, directory or symlink is never replaced.
    except FileExistsError:
        raise DownloadError("Destination already exists; it will not be overwritten. Choose a fresh --dest.") from None
    created = []
    try:
        for path in sorted(source.iterdir()):
            destination = target / path.name
            with path.open("rb") as stream, destination.open("xb") as output:
                created.append(destination)
                shutil.copyfileobj(stream, output, length=1024 * 1024)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        # rmdir never deletes another process's unexpected files.
        try:
            target.rmdir()
        except OSError:
            pass
        raise


def download_from_kaggle(dest: Path, *, dataset_version: int | None = None,
                         image_subdir: str | None = None, api=None) -> Path:
    """Download into this run's staging area, validate, then copy into a reserved new target."""
    if dataset_version is not None and (type(dataset_version) is not int or dataset_version < 1):
        raise DownloadError("--dataset-version must be a positive integer")
    if image_subdir is not None:
        _relative_subdir(image_subdir)
    target = dest / EXPECTED_SUBFOLDER
    if target.exists() or target.is_symlink():
        raise DownloadError("Destination already exists; it will not be overwritten. Choose a fresh --dest.")
    api = load_kaggle_api() if api is None else api
    try:
        with redirect_stdout(_DiscardOutput()), redirect_stderr(_DiscardOutput()):
            api.authenticate()
    except Exception:
        raise DownloadError("Kaggle authentication failed. Configure the SDK credentials and retry; "
                            "credential values and SDK details are not displayed.") from None
    reference = KAGGLE_DATASET + (f"/{dataset_version}" if dataset_version is not None else "")
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".utkface-download-", dir=dest) as temporary:
        staging = Path(temporary)
        try:
            with redirect_stdout(_DiscardOutput()), redirect_stderr(_DiscardOutput()):
                api.dataset_download_files(reference, path=str(staging), unzip=False, quiet=True)
        except Exception:
            raise DownloadError("Kaggle download failed. Check network access, authentication and dataset/version "
                                "availability; SDK exception details are not displayed.") from None
        archive = staging / ARCHIVE_NAME
        if not archive.is_file() or archive.is_symlink():
            raise DownloadError(f"Expected archive {ARCHIVE_NAME} was not produced by Kaggle.")
        extracted = staging / "validated-images"
        extract_images(archive, extracted, image_subdir)
        inspect_images(extracted)
        try:
            publish_images(extracted, target)
        except DownloadError:
            raise
        except OSError:
            raise DownloadError("Publishing the validated dataset failed; existing destination files were preserved.") from None
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or locally check UTKFace", epilog=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Parent directory; images go into <dest>/utkcropped")
    parser.add_argument("--check-only", action="store_true", help="Read-only local check; no Kaggle import/auth/download/install")
    parser.add_argument("--dataset-version", type=int, default=None, help="Optional positive Kaggle version number; omitted means latest")
    parser.add_argument("--image-subdir", type=_relative_subdir, default=None, help="Explicit relative image folder inside archive, or . for root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        if not verify_dataset(args.dest):
            raise SystemExit(1)
        return
    if (args.dest / EXPECTED_SUBFOLDER).exists() and verify_dataset(args.dest):
        return
    try:
        target = download_from_kaggle(args.dest, dataset_version=args.dataset_version, image_subdir=args.image_subdir)
    except (DownloadError, OSError, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Dataset ready at {target}. Run preparation into a fresh CSV directory:")
    print(f"  python scripts/prepare_data.py --dataset-dir '{target}' --csv-dir csv_dataset/protocol_v2 --check-decode")


if __name__ == "__main__":
    main()
