"""Verify the distributable artifacts and installed CLI outside the checkout.

Builds use existing build dependencies and never download model weights or data.
The temporary venv inherits runtime dependencies but installs our wheel explicitly.
"""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
from packaging.requirements import Requirement

import src.configs
from src.configs import load_config

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], cwd: Path) -> str:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def distribution(tmp_path_factory):
    """Build wheel/sdist in a temporary source tree, leaving the checkout clean."""
    work = tmp_path_factory.mktemp("packaging")
    source = work / "source"
    source.mkdir()
    for name in ["src", "scripts", "demo", "configs", "docs"]:
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in [
        "pyproject.toml", "MANIFEST.in", "Readme.md", "LICENSE", "CONTRIBUTING.md",
        "requirements.txt", "requirements-dev.txt",
    ]:
        shutil.copy2(ROOT / name, source / name)
    dist = work / "dist"
    _run(
        [sys.executable, "-m", "build", "--no-isolation", "--sdist", "--wheel",
         "--outdir", str(dist), str(source)],
        work,
    )
    wheel = next(dist.glob("*.whl"))
    sdist = next(dist.glob("*.tar.gz"))
    # Also verify that a wheel can be built using only the sdist contents.
    unpacked = work / "unpacked"
    with tarfile.open(sdist) as archive:
        archive.extractall(unpacked, filter="data")
    rebuilt = work / "rebuilt"
    _run(
        [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir",
         str(rebuilt), str(next(unpacked.iterdir()))],
        work,
    )
    return work, wheel, sdist, next(rebuilt.glob("*.whl"))


@pytest.fixture(scope="module")
def installed(distribution):
    work, _, _, rebuilt = distribution
    environment = work / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    binaries = environment / ("Scripts" if os.name == "nt" else "bin")
    python = binaries / ("python.exe" if os.name == "nt" else "python")
    outside = work / "outside"
    outside.mkdir()
    _run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(rebuilt)], outside)
    return python, binaries, outside


def test_default_config_compatibility_copy():
    assert (ROOT / "configs/default.yaml").read_bytes() == (
        ROOT / "src/configs/default.yaml"
    ).read_bytes()
    assert load_config().model.name == "efficientnet_b3"


def test_explicit_config_is_loaded(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("model:\n  name: resnet50\n", encoding="utf-8")
    assert load_config(custom).model.name == "resnet50"


def test_checkout_config_remains_editable(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/default.yaml").write_text("model:\n  name: resnet50\n", encoding="utf-8")
    monkeypatch.setattr(src.configs, "_checkout_root", lambda: tmp_path)
    assert load_config().model.name == "resnet50"
    assert src.configs.get_runtime_root() == tmp_path


def test_artifacts_include_only_runtime_packages_and_resource(distribution):
    _, wheel, sdist, rebuilt = distribution
    for artifact in [wheel, rebuilt]:
        with zipfile.ZipFile(artifact) as archive:
            names = set(archive.namelist())
            assert "src/configs/default.yaml" in names
            assert {"scripts/__init__.py", "scripts/train.py", "demo/__init__.py"} <= names
            roots = {name.split("/")[0] for name in names}
            assert all(root in {"src", "scripts", "demo"} or root.endswith(".dist-info")
                       for root in roots)
            assert not any(name.endswith((".pt", ".pth", "kaggle.json")) for name in names)
            metadata = Parser().parsestr(archive.read(next(
                name for name in names if name.endswith(".dist-info/METADATA")
            )).decode())
            assert metadata["Name"] == "facial-age-estimation"
            # requirements.txt must agree with the metadata for core dependencies.
            requirements = {str(Requirement(line.split("#", 1)[0].strip()))
                            for line in (ROOT / "requirements.txt").read_text().splitlines()
                            if line.split("#", 1)[0].strip()}
            core = {str(Requirement(line)) for line in metadata.get_all("Requires-Dist")
                    if Requirement(line).marker is None}
            assert core == requirements
            entries = configparser.ConfigParser()
            entries.read_string(archive.read(next(
                name for name in names if name.endswith(".dist-info/entry_points.txt")
            )).decode())
            assert set(entries["console_scripts"]) == {"train-age", "eval-age", "export-age"}
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        assert any(name.endswith("/src/configs/default.yaml") for name in names)
        assert any(name.endswith("/configs/default.yaml") for name in names)
        assert any(name.endswith("/docs/INSTALLATION.md") for name in names)
        assert not any(name.endswith((".pt", ".pth", "kaggle.json")) for name in names)


@pytest.mark.parametrize("entrypoint", ["train-age", "eval-age", "export-age"])
def test_installed_cli_help_outside_checkout(installed, entrypoint):
    _, binaries, outside = installed
    executable = entrypoint + (".exe" if os.name == "nt" else "")
    assert "--config" in _run([str(binaries / executable), "--help"], outside)


def test_installed_defaults_custom_config_and_trainer_import(installed):
    python, _, outside = installed
    (outside / "custom.yaml").write_text("model:\n  name: resnet50\n", encoding="utf-8")
    code = """
import json
from pathlib import Path
from torch.amp import GradScaler
import src
from src.configs import get_runtime_root, load_config
from src.training import Trainer
from torch.utils.tensorboard import SummaryWriter
print(json.dumps({
    'module': src.__file__,
    'default_model': load_config().model.name,
    'custom_model': load_config('custom.yaml').model.name,
    'runtime_root': str(get_runtime_root()),
    'scaler_disabled': not GradScaler('cuda', enabled=False).is_enabled(),
    'trainer': Trainer.__name__,
}))
"""
    result = json.loads(_run([str(python), "-c", code], outside))
    assert Path(result["module"]).is_relative_to(python.parent.parent)
    assert result["default_model"] == "efficientnet_b3"
    assert result["custom_model"] == "resnet50"
    assert Path(result["runtime_root"]) == outside
    assert result["scaler_disabled"] and result["trainer"] == "Trainer"


def test_installed_train_prints_validated_custom_config(installed):
    _, binaries, outside = installed
    custom = outside / "cli.yaml"
    custom.write_text("training:\n  epochs: 2\n", encoding="utf-8")
    executable = binaries / ("train-age.exe" if os.name == "nt" else "train-age")
    output = _run([str(executable), "--config", str(custom), "--print-config",
                   "loss.type=l1", "model.output_dim=1"], outside)
    assert "epochs: 2" in output and "output_dim: 1" in output and "type: l1" in output
    assert "Using device" not in output


def test_installed_train_rejects_invalid_mode_before_data_io(installed):
    _, binaries, outside = installed
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    executable = binaries / ("train-age.exe" if os.name == "nt" else "train-age")
    result = subprocess.run([str(executable), "--print-config", "model.output_dim=1"],
                            cwd=outside, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "model.output_dim must be 101" in result.stderr
    assert "Using device" not in result.stdout


def test_installed_checkpoint_restores_inference_contract(installed):
    import torch

    from src.models import build_model
    from src.utils.checkpoint import create_checkpoint

    python, _, outside = installed
    cfg = load_config(overrides=["model.name=mobilenet_v3", "model.pretrained=false", "data.img_size=64",
                                 "model.output_dim=1", "loss.type=l1", "data.mean=[0.1,0.2,0.3]"])
    path = outside / "inference.pt"
    torch.save(create_checkpoint(build_model(cfg).eval(), cfg), path)
    code = """
import json
from PIL import Image
from src.inference import AgePredictor
p = AgePredictor.from_checkpoint('inference.pt', device='cpu')
result = p.predict(Image.new('RGB', (90, 80)))
print(json.dumps({'size': p.checkpoint.cfg.data.img_size,
                  'mean': list(p.checkpoint.cfg.data.mean),
                  'age': result.age, 'probabilities': result.probabilities}))
"""
    result = json.loads(_run([str(python), "-c", code], outside))
    assert result["size"] == 64 and result["mean"] == [.1, .2, .3]
    assert 0 <= result["age"] <= 100 and result["probabilities"] is None
