"""Typed download-only settings, independent of experiment/checkpoint configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException


@dataclass
class HuggingFaceSource:
    repo_id: str = MISSING
    revision: str = MISSING
    filename: str = MISSING
    sha256: str = MISSING
    size_bytes: int = MISSING
    image_subdir: str | None = MISSING


@dataclass
class KaggleSource:
    dataset_version: int | None = MISSING
    image_subdir: str | None = MISSING


@dataclass
class DownloadConfig:
    source: str = MISSING
    dest: str = MISSING
    timeout_seconds: float = MISSING
    max_retries: int = MISSING
    huggingface: HuggingFaceSource = field(default_factory=HuggingFaceSource)
    kaggle: KaggleSource = field(default_factory=KaggleSource)


def validate_image_subdir(value: str) -> str:
    """Require a canonical relative archive directory, with . denoting archive root."""
    if value == ".":
        return value
    path = PurePosixPath(value)
    if not value or "\x00" in value or path.is_absolute() or path.as_posix() != value or any(
        item in {".", ".."} or ":" in item or "\\" in item for item in path.parts
    ):
        raise ValueError("--image-subdir must be a canonical relative archive directory (or .)")
    return value


def merge_download_config(*sources) -> DictConfig:
    """Resolve and validate settings before any network/destination I/O."""
    try:
        normalized = []
        for source in sources:
            node = OmegaConf.create(OmegaConf.to_container(source, resolve=False)) if OmegaConf.is_config(source) else OmegaConf.create(source)
            # Drop inherited read-only flags, and reject explicit ??? before merge
            # can silently inherit the previous value.
            normalized.append(OmegaConf.create(OmegaConf.to_container(node, resolve=False, throw_on_missing=True)))
        cfg = OmegaConf.merge(OmegaConf.structured(DownloadConfig), *normalized)
        OmegaConf.resolve(cfg)
        if cfg.source not in {"huggingface", "kaggle"}:
            raise ValueError("download source must be huggingface or kaggle")
        if not cfg.dest.strip() or "\x00" in cfg.dest:
            raise ValueError("download dest must be a nonempty path")
        if not 0 < cfg.timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be finite and in (0, 300]")
        if type(cfg.max_retries) is not int or not 0 <= cfg.max_retries <= 5:
            raise ValueError("max_retries must be an integer in [0, 5]")
        hf = cfg.huggingface
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", hf.repo_id):
            raise ValueError("huggingface.repo_id must be owner/dataset")
        if not re.fullmatch(r"[0-9a-f]{40}", hf.revision):
            raise ValueError("huggingface.revision must be a full lowercase 40-character commit SHA")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.tar\.gz", hf.filename):
            raise ValueError("huggingface.filename must be a plain .tar.gz filename")
        if not re.fullmatch(r"[0-9a-f]{64}", hf.sha256):
            raise ValueError("huggingface.sha256 must be a lowercase SHA-256 digest")
        if type(hf.size_bytes) is not int or not 0 < hf.size_bytes <= 8 * 1024**3:
            raise ValueError("huggingface.size_bytes must be an integer in (0, 8 GiB]")
        version = cfg.kaggle.dataset_version
        if version is not None and (type(version) is not int or version < 1):
            raise ValueError("kaggle.dataset_version must be a positive integer")
        for source in (hf, cfg.kaggle):
            if source.image_subdir is not None:
                validate_image_subdir(source.image_subdir)
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        OmegaConf.set_readonly(cfg, True)
        return cfg
    except OmegaConfBaseException as exc:
        raise ValueError(f"Invalid download configuration: {exc}") from exc


def huggingface_url(cfg: DictConfig) -> str:
    """Build the public, revision-pinned URL; never requires a token or SDK."""
    hf = cfg.huggingface
    return f"https://huggingface.co/datasets/{hf.repo_id}/resolve/{hf.revision}/{hf.filename}"
