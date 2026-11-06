#!/usr/bin/env python3
"""Download UTKFace into a protected destination, or inspect it without network access.

  python scripts/download_data.py --check-only
  python scripts/download_data.py --source huggingface
  python scripts/download_data.py --source kaggle --dest dataset --dataset-version 1
  python scripts/download_data.py --image-subdir UTKFace

Hugging Face is the default, configured in configs/download.yaml; no extra SDK is needed.
For Kaggle only, install the optional dependency yourself: python -m pip install ".[data]"
Authentication is delegated to Kaggle; see https://github.com/Kaggle/kaggle-cli
The script never installs packages or inspects credential contents.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from PIL import Image

from src.configs import get_runtime_root, load_download_config
from src.data.validation import IMAGE_SUFFIXES, image_path, parse_utkface_name
from src.utils.download_config import huggingface_url, merge_download_config
from src.utils.download_config import validate_image_subdir as _relative_subdir

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


def _select_image_group(groups: dict, image_subdir: str | None) -> list:
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
    return groups[selected]


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
    return sorted(_select_image_group(groups, image_subdir), key=lambda info: info.filename)


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


def extract_tar_images(archive_path: Path, output: Path, image_subdir: str | None = None) -> None:
    """Validate every TAR member, then manually copy regular selected images only."""
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            groups, seen, total = {}, set(), 0
            for index, member in enumerate(archive, 1):
                total += member.size
                if index > MAX_MEMBERS or member.size < 0 or total > MAX_ARCHIVE_BYTES:
                    raise DownloadError("TAR exceeds the supported member-count/uncompressed-size limits.")
                name = member.name
                relative = PurePosixPath(name.rstrip("/"))
                if (not name or "\x00" in name or "\\" in name or relative.is_absolute()
                        or relative.as_posix() != name.rstrip("/")
                        or any(part in {".", ".."} or ":" in part for part in relative.parts)):
                    raise DownloadError("TAR contains an unsafe or noncanonical path.")
                if not member.isfile() and not member.isdir() or member.issparse():
                    raise DownloadError("TAR contains a link, sparse file or unsupported special file.")
                if relative in seen:
                    raise DownloadError("TAR contains duplicate member paths.")
                seen.add(relative)
                if member.isdir():
                    continue
                try:
                    parse_utkface_name(relative.name)
                except ValueError:
                    continue
                if member.size > MAX_IMAGE_BYTES:
                    raise DownloadError("A TAR image exceeds the supported uncompressed-size limit.")
                groups.setdefault(relative.parent.as_posix(), []).append(member)
            # Preserve archive offsets: sorting names would repeatedly rewind and
            # decompress the gzip stream for a large real dataset.
            members = _select_image_group(groups, image_subdir)
            output.mkdir()
            for member in members:
                with archive.extractfile(member) as source, (output / PurePosixPath(member.name).name).open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
    except DownloadError:
        raise
    except (OSError, ValueError, RuntimeError, EOFError, tarfile.TarError):
        raise DownloadError("TAR reading/extraction failed; no dataset was published.") from None


def _download_verified_archive(cfg, archive: Path) -> None:
    """Stream public HTTPS, bound its size and verify pinned bytes before extraction."""
    hf = cfg.huggingface
    for attempt in range(cfg.max_retries + 1):
        try:
            request = Request(huggingface_url(cfg), headers={"User-Agent": "facial-age-estimation/0.2"})
            with urlopen(request, timeout=cfg.timeout_seconds) as response, archive.open("wb") as output:
                if response.status != 200:
                    raise DownloadError("Expected a complete HTTP 200 archive response.")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != hf.size_bytes:
                    raise DownloadError("Hugging Face archive Content-Length disagrees with the pinned size.")
                digest, count = hashlib.sha256(), 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > hf.size_bytes:
                        raise DownloadError("Hugging Face archive exceeds the pinned size.")
                    output.write(chunk)
                    digest.update(chunk)
            if count != hf.size_bytes or digest.hexdigest() != hf.sha256:
                raise DownloadError("Hugging Face archive size/SHA-256 mismatch; nothing was extracted or published.")
            return
        except HTTPError as exc:
            retry = exc.code == 429 or 500 <= exc.code < 600
            if not retry or attempt == cfg.max_retries:
                raise DownloadError(f"Hugging Face HTTP download failed (status {exc.code}); no fallback to Kaggle.") from None
        except (URLError, TimeoutError, HTTPException, ConnectionError):
            if attempt == cfg.max_retries:
                raise DownloadError("Hugging Face network download failed; retry later. No fallback to Kaggle.") from None
        if attempt < cfg.max_retries:
            time.sleep(min(2**attempt, 4))


def download_from_huggingface(dest: Path, cfg=None) -> Path:
    """Fetch a pinned tar.gz without credentials/SDK; preserve all preexisting targets."""
    cfg = load_download_config() if cfg is None else merge_download_config(cfg)
    target = dest / EXPECTED_SUBFOLDER
    if target.exists() or target.is_symlink():
        raise DownloadError("Destination already exists; it will not be overwritten. Choose a fresh --dest.")
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".utkface-download-", dir=dest) as temporary:
        staging = Path(temporary)
        archive = staging / cfg.huggingface.filename
        print(f"Downloading {cfg.huggingface.repo_id}@{cfg.huggingface.revision} ({cfg.huggingface.size_bytes:,} bytes)...")
        _download_verified_archive(cfg, archive)
        extracted = staging / "validated-images"
        extract_tar_images(archive, extracted, cfg.huggingface.image_subdir)
        count = inspect_images(extracted)
        try:
            publish_images(extracted, target)
        except DownloadError:
            raise
        except OSError:
            raise DownloadError("Publishing the validated dataset failed; existing destination files were preserved.") from None
    print(f"Verified archive SHA-256; published {count:,} valid-name images.")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or locally check UTKFace", epilog=__doc__)
    parser.add_argument("--config", default=None, help="Download-only YAML; default: configs/download.yaml or packaged copy")
    parser.add_argument("--print-config", action="store_true", help="Print resolved download settings/URL without network or output writes")
    parser.add_argument("--source", choices=["huggingface", "kaggle"], default=None, help="Provider override (default: configured Hugging Face mirror)")
    parser.add_argument("--dest", type=Path, default=None, help="Parent directory; images go into <dest>/utkcropped")
    parser.add_argument("--check-only", action="store_true", help="Read-only local check; no provider/auth/download/install")
    parser.add_argument("--dataset-version", type=int, default=None, help="Kaggle only; positive version number, omitted means configured/latest")
    parser.add_argument("--image-subdir", type=_relative_subdir, default=None, help="Override the selected provider's relative image folder, or . for root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    updates = {}
    if args.source is not None:
        updates["source"] = args.source
    if args.dest is not None:
        updates["dest"] = str(args.dest)
    if args.dataset_version is not None:
        updates["kaggle"] = {"dataset_version": args.dataset_version}
    cfg = load_download_config(args.config, updates=updates)
    if args.image_subdir is not None:
        cfg = merge_download_config(cfg, {cfg.source: {"image_subdir": args.image_subdir}})
    if args.dataset_version is not None and cfg.source != "kaggle":
        raise ValueError("--dataset-version requires --source kaggle; Hugging Face uses its pinned revision in download.yaml")
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print(f"# Hugging Face URL: {huggingface_url(cfg)}")
        return
    dest = args.dest if args.dest is not None else get_runtime_root() / cfg.dest
    if args.check_only:
        if not verify_dataset(dest):
            raise SystemExit(1)
        return
    if (dest / EXPECTED_SUBFOLDER).exists() and verify_dataset(dest):
        print("Existing images reused; their origin/revision has not been inferred from current config.")
        return
    try:
        if cfg.source == "huggingface":
            target = download_from_huggingface(dest, cfg)
            source = f"huggingface:{cfg.huggingface.repo_id}"
            version = cfg.huggingface.revision
        else:
            target = download_from_kaggle(dest, dataset_version=cfg.kaggle.dataset_version,
                                          image_subdir=cfg.kaggle.image_subdir)
            source, version = KAGGLE_DATASET, cfg.kaggle.dataset_version or "unknown"
    except (DownloadError, OSError, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Dataset ready at {target}. Run preparation into a fresh CSV directory:")
    print(f"  python scripts/prepare_data.py --dataset-dir '{target}' --csv-dir csv_dataset/protocol_v2 "
          f"--check-decode --source '{source}' --dataset-version '{version}'")


if __name__ == "__main__":
    main()
