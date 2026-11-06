"""Small, plain experiment metadata; never records secrets or a full environment dump."""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def collect_environment() -> dict:
    packages = {}
    for package in ["torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "Pillow", "omegaconf"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    root = Path(__file__).resolve().parents[2]
    revision, dirty = None, None
    if (root / ".git").exists():
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                      text=True, check=True, timeout=5).stdout.strip()
            dirty = any(subprocess.run(["git", "diff", *args, "--quiet"], cwd=root, capture_output=True,
                                       timeout=5).returncode != 0 for args in [[], ["--cached"]])
        except (OSError, subprocess.SubprocessError):
            revision = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "git_revision": revision, "tracked_changes": dirty}
