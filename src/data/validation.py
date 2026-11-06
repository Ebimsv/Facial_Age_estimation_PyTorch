"""Shared CSV, path, split and byte-hash contracts; no image decoding on import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

COLUMNS = ["image_name", "age", "gender", "ethnicity"]
GENDER_MAP = {0: "Male", 1: "Female"}
ETHNICITY_MAP = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_FILES = {"train": "train_set.csv", "val": "valid_set.csv", "test": "test_set.csv"}
PROTOCOL = "utkface_age_stratified_v2"


def parse_utkface_name(name: str) -> dict:
    """Parse a flat UTKFace filename without guessing unknown demographic labels."""
    parts = name.split("_")
    try:
        if Path(name).suffix.lower() not in IMAGE_SUFFIXES or len(parts) < 4 or not parts[3].split(".")[0]:
            raise ValueError
        age, gender, ethnicity = map(int, parts[:3])
        if age < 0 or gender not in GENDER_MAP or ethnicity not in ETHNICITY_MAP:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("invalid_filename_or_labels") from None
    return {"image_name": name, "age": age, "gender": GENDER_MAP[gender], "ethnicity": ETHNICITY_MAP[ethnicity]}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(root: str | Path, name: str, *, require_file: bool = True) -> Path:
    """Accept canonical relative names only; resolve symlinks before containment."""
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"Invalid image_name: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name or any(p in {".", ".."} for p in relative.parts):
        raise ValueError(f"image_name must be a canonical relative path inside root: {name!r}")
    if relative.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image extension: {name!r}; expected jpg/jpeg/png")
    base = Path(root).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Image path escapes root: {name!r}")
    if require_file and not path.is_file():
        raise ValueError(f"Image is not a regular file: {name!r} under {base}")
    return path


def validate_table(df: pd.DataFrame, *, context: str = "CSV", root=None, age_range=None) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{context}: CSV is missing columns: {sorted(missing)}")
    result = df[COLUMNS].copy()
    ages = pd.to_numeric(result["age"], errors="coerce")
    lower, upper = age_range if age_range is not None else (0, float("inf"))
    if pd.api.types.is_bool_dtype(ages) or not (np.isfinite(ages) & ages.between(lower, upper)).all():
        raise ValueError(f"{context}: age targets must be finite and inside [{lower}, {upper}]")
    result["age"] = ages
    for column, allowed in [("gender", set(GENDER_MAP.values())), ("ethnicity", set(ETHNICITY_MAP.values()))]:
        if not result[column].isin(allowed).all():
            raise ValueError(f"{context}: invalid {column}; allowed labels: {sorted(allowed)}")
    if result["image_name"].duplicated().any():
        raise ValueError(f"{context}: duplicate image_name rows are not allowed")
    # Schema/labels are checked first, even if images have not been downloaded.
    for name in result["image_name"]:
        image_path(root or Path.cwd(), name, require_file=root is not None)
    return result


def validate_partition(splits: dict[str, pd.DataFrame], master: pd.DataFrame | None = None, *, root=None) -> None:
    """Reject duplicate paths and inconsistent labels, including symlink aliases."""
    seen = set()
    all_rows = []
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"{name} split is empty; prepare nonempty split CSVs before training")
        identities = [str(image_path(root, item)) if root is not None else item for item in frame["image_name"]]
        if len(set(identities)) != len(identities) or seen.intersection(identities):
            raise ValueError(f"Split overlap or duplicate image path detected in {name}")
        seen.update(identities)
        all_rows.append(frame[COLUMNS])
    if master is not None:
        combined = pd.concat(all_rows, ignore_index=True).sort_values("image_name").reset_index(drop=True)
        expected = master[COLUMNS].sort_values("image_name").reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(combined, expected, check_dtype=False)
        except AssertionError as exc:
            raise ValueError("Split union/labels must equal the validated master CSV") from exc


def describe_splits(csv_dir: Path, root: Path, frames: dict[str, pd.DataFrame], age_range=None) -> dict:
    """Verify an optional v2 manifest, or identify unversioned CSVs without guessing history."""
    master_path = csv_dir / "utkface_dataset.csv"
    master = validate_table(pd.read_csv(master_path), context=str(master_path), root=root,
                            age_range=age_range) if master_path.exists() else None
    validate_partition(frames, master, root=root)
    files = {filename: {"sha256": sha256_file(csv_dir / filename), "count": len(frames[split])}
             for split, filename in SPLIT_FILES.items()}
    if master is not None:
        files[master_path.name] = {"sha256": sha256_file(master_path), "count": len(master)}
    manifest_path = csv_dir / "split_manifest.json"
    provenance = {"protocol": "unversioned_csv", "csv_files": files, "master_verified": master is not None,
                  "content_verified": False, "identity_leakage": "unknown", "manifest_sha256": None}
    if not manifest_path.exists():
        return provenance
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["protocol"] != PROTOCOL:
            raise ValueError("Unsupported split manifest schema/protocol")
        if manifest["csv_files"] != files or master is None:
            raise ValueError("Split manifest CSV hashes/counts disagree with current files")
        content = manifest["content_sha256"]
        if manifest["validation"]["content_hashed"] is not (content is not None):
            raise ValueError("Split manifest content-hash policy is inconsistent")
        if content is not None:
            names = set(master["image_name"])
            if not isinstance(content, dict) or set(content) != names:
                raise ValueError("Split manifest content hash inventory disagrees with master")
            if len(set(content.values())) != len(content):
                raise ValueError("Split manifest contains duplicate image content")
            for name, expected in content.items():
                if sha256_file(image_path(root, name)) != expected:
                    raise ValueError(f"Image content hash mismatch: {name}")
        provenance.update(protocol=PROTOCOL, content_verified=content is not None,
                          manifest_sha256=sha256_file(manifest_path), source=manifest["source"],
                          preparation_parameters=manifest["parameters"], validation=manifest["validation"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid split manifest: {manifest_path}") from exc
    return provenance
