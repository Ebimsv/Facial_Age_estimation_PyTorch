"""Load packaged defaults without depending on a repository checkout."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils.config import merge_config


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None,
    *, updates: dict | None = None,
) -> DictConfig:
    """Load a custom YAML file or the defaults distributed with the package.

    Args:
        path: Explicit config file. None selects checkout defaults when running
            from source, or packaged defaults when installed.
        overrides: CLI key=value updates applied before contract validation.
        updates: Typed CLI values (e.g. prepare_data flags), applied before overrides.

    Returns:
        Configuration available independently of the current working directory.
    """
    defaults = OmegaConf.create(files(__package__).joinpath("default.yaml").read_text(encoding="utf-8"))
    checkout = _checkout_root()
    if path is None and checkout is not None and (checkout / "configs/default.yaml").is_file():
        path = checkout / "configs/default.yaml"
    return merge_config(defaults, OmegaConf.load(path) if path is not None else {},
                        updates or {}, overrides=overrides)


def _checkout_root() -> Path | None:
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "scripts" / "train.py").is_file():
        return checkout
    return None


def get_runtime_root() -> Path:
    """Resolve relative data paths against the checkout, or the caller's directory.

    Returns:
        Repository root for source execution; current directory for installed use.
    """
    return _checkout_root() or Path.cwd()
