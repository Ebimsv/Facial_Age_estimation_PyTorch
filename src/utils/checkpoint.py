"""Versioned inference contracts and explicit checkpoint compatibility policies."""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
import warnings
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from src.data.transforms import get_test_transforms
from src.models import AgeEstimationModel
from src.models.legacy import LegacyResNet50
from src.training.losses import L1AgeLoss, get_loss_fn, training_objective_spec
from src.utils.config import DISTRIBUTION_LOSSES, merge_config, to_plain_config

SCHEMA_VERSION = 1
RUNTIME_KEYS = {
    "device", "data.root_dir", "data.csv_dir", "data.num_workers",
    "training.eval_batch_size", "training.use_amp",
}


class LegacyAgeDecoder(L1AgeLoss):
    """Keep historical inference raw: never silently clamp a legacy prediction."""

    @torch.no_grad()
    def predict_age(self, logits: torch.Tensor) -> torch.Tensor:
        self._validate_logits(logits)
        return logits.view(-1)


def _contract(cfg, architecture: str = "current") -> dict[str, Any]:
    plain = to_plain_config(cfg)
    return {
        "model_spec": {**plain["model"], "architecture": architecture},
        "preprocessing": {"img_size": cfg.data.img_size, "mean": list(cfg.data.mean),
                          "std": list(cfg.data.std), "color_mode": "RGB",
                          "resize": "bilinear", "antialias": True},
        "predictor": {"mode": cfg.loss.type, "start_age": cfg.loss.start_age,
                      "end_age": cfg.loss.end_age,
                      "postprocess": "raw" if architecture == "legacy_resnet50_v1" else
                      "expected_age" if cfg.loss.type in DISTRIBUTION_LOSSES else "clamp_0_100"},
    }


def create_checkpoint(model, cfg, *, epoch=None, val_mae=None, optimizer_state_dict=None,
                      training_objective=None, experiment_metadata=None, training_state=None) -> dict:
    """Build a plain versioned artifact; training/resume state is optional."""
    cfg = merge_config(cfg)
    architecture = "legacy_resnet50_v1" if isinstance(model, LegacyResNet50) else "current"
    if model.model_name != cfg.model.name or model.output_dim != cfg.model.output_dim:
        raise ValueError("Model identity/output width does not match checkpoint config")
    if model.img_size != cfg.data.img_size or model.dropout != cfg.model.dropout:
        raise ValueError("Model resolution/dropout does not match checkpoint config")
    if architecture == "legacy_resnet50_v1" and _contract(cfg, architecture) != _contract(_legacy_config(), architecture):
        raise ValueError("Legacy checkpoint creation requires the exact historical inference profile")
    artifact = {"schema_version": SCHEMA_VERSION, **_contract(cfg, architecture),
                "cfg": to_plain_config(cfg), "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_mae": val_mae}
    if training_objective is not None:
        artifact["training_objective"] = training_objective
    if experiment_metadata is not None:
        artifact["experiment_metadata"] = experiment_metadata
    if training_state is not None:
        artifact["training_state"] = training_state
    if optimizer_state_dict is not None:
        artifact["optimizer_state_dict"] = optimizer_state_dict
    return artifact


def _legacy_config():
    # Verified in the commit that introduced the tracked weight, not inferred from its filename.
    defaults = OmegaConf.create(files("src.configs").joinpath("default.yaml").read_text(encoding="utf-8"))
    return merge_config(defaults, overrides=["model.name=resnet50", "model.pretrained=false", "model.output_dim=1",
                                             "model.dropout=0.2", "data.img_size=128", "loss.type=l1",
                                             "loss.start_age=0", "loss.end_age=100", "data.max_age=85",
                                             "data.mean=[0.485,0.456,0.406]", "data.std=[0.229,0.224,0.225]"])


def _leaves(value, prefix=""):
    if not isinstance(value, dict):
        yield prefix, value
        return
    for key, item in value.items():
        yield from _leaves(item, f"{prefix}.{key}" if prefix else key)


def _runtime_config(cfg, config_path, runtime_overrides):
    updates = {}
    if config_path is not None:
        patch = OmegaConf.load(config_path)
        if not OmegaConf.is_dict(patch):
            raise ValueError("Runtime --config must be a YAML mapping")
        defaults = OmegaConf.create(files("src.configs").joinpath("default.yaml").read_text(encoding="utf-8"))
        template = OmegaConf.to_container(patch, resolve=False)
        reference = OmegaConf.to_container(defaults, resolve=False)
        for candidate_template in (template, reference):
            if isinstance(candidate_template.get("logging"), dict):
                candidate_template["logging"].setdefault("wandb_mode", "online")
            if isinstance(candidate_template.get("training"), dict):
                candidate_template["training"].setdefault("deterministic", False)
            if isinstance(candidate_template.get("loss"), dict) and candidate_template["loss"].get("type") in DISTRIBUTION_LOSSES:
                candidate_template["loss"]["type"] = "l1_mean_variance"
        if template == reference:
            # Preserve callers that explicitly supplied the default YAML before
            # checkpoints became authoritative. Custom contract conflicts still fail.
            patch = OmegaConf.create({})
            for key in RUNTIME_KEYS:
                OmegaConf.update(patch, key, OmegaConf.select(defaults, key), merge=False)
        # Resolve partial runtime YAML against the saved config, not against unrelated defaults.
        candidate = merge_config(cfg, patch)
        for key, value in _leaves(OmegaConf.to_container(patch, resolve=False)):
            actual = OmegaConf.select(candidate, key)
            if key == "loss.type" and actual in DISTRIBUTION_LOSSES and cfg.loss.type in DISTRIBUTION_LOSSES:
                continue  # compatible spelling; the artifact retains its own label
            if key not in RUNTIME_KEYS and actual != OmegaConf.select(cfg, key):
                raise ValueError(f"Checkpoint contract is immutable: cannot override {key}")
            if key in RUNTIME_KEYS:
                updates[key] = actual
    for key, value in (runtime_overrides or {}).items():
        if key not in RUNTIME_KEYS:
            raise ValueError(f"Not an allowed checkpoint runtime override: {key}")
        updates[key] = value
    # A checkpoint's training GPU is not a portable inference device default.
    updates.setdefault("device", None)
    patch = OmegaConf.create({})
    for key, value in updates.items():
        OmegaConf.update(patch, key, value, merge=False)
    return merge_config(cfg, patch)


@dataclass
class LoadedCheckpoint:
    model: AgeEstimationModel
    cfg: Any
    loss_fn: Any
    transform: Any
    device: torch.device
    metadata: dict


def load_checkpoint(
    path: str | Path, config_path: str | Path | None = None, *,
    device: str | None = None, runtime_overrides: dict | None = None,
    legacy_resnet50: bool = False, trust_legacy_pickle: bool = False,
) -> LoadedCheckpoint:
    """Restore a strict model/transform/decoder bundle without downloading weights.

    Safe deserialization is always attempted once, unless the caller explicitly
    opts into trusted legacy pickle. There is no automatic unsafe fallback.
    """
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=not trust_legacy_pickle)
    except pickle.UnpicklingError as exc:
        raise ValueError("Safe checkpoint load failed. For a trusted old pickle only, explicitly use "
                         "--trust-legacy-pickle; this may execute arbitrary pickle code.") from exc
    if not isinstance(artifact, dict) or not artifact:
        raise ValueError("Unsupported checkpoint: expected a nonempty artifact mapping")
    version = artifact.get("schema_version")
    raw_state = all(isinstance(key, str) and isinstance(value, torch.Tensor)
                    for key, value in artifact.items())
    if raw_state:
        if not legacy_resnet50:
            raise ValueError("Raw state_dict needs an explicit --legacy-resnet50 profile; no architecture is guessed")
        cfg = _legacy_config()
        architecture = "legacy_resnet50_v1"
        state = artifact
        metadata = {"schema_version": 0, "source_format": "legacy_state_dict",
                    "profile_source_commit": "432569f", **_contract(cfg, architecture)}
    else:
        if legacy_resnet50:
            raise ValueError("--legacy-resnet50 is only for the historical raw ResNet50 state_dict")
        if "schema_version" in artifact and (type(version) is not int or version != SCHEMA_VERSION):
            raise ValueError(f"Unsupported checkpoint schema_version: {version!r}")
        if "cfg" not in artifact or "model_state_dict" not in artifact:
            raise ValueError("Checkpoint metadata is incomplete: cfg and model_state_dict are required")
        # Always use bundled defaults, regardless of local checkout config edits.
        defaults = OmegaConf.create(files("src.configs").joinpath("default.yaml").read_text(encoding="utf-8"))
        saved = OmegaConf.create(artifact["cfg"])
        for key in ["model.name", "model.pretrained", "model.output_dim", "model.dropout",
                    "data.img_size", "data.mean", "data.std", "loss.type", "loss.start_age", "loss.end_age"]:
            if OmegaConf.select(saved, key) is None:
                raise ValueError(f"Checkpoint metadata is incomplete: missing cfg.{key}")
        cfg = merge_config(defaults, artifact["cfg"])
        state = artifact["model_state_dict"]
        if "model_spec" in artifact and not isinstance(artifact["model_spec"], dict):
            raise ValueError("model_spec must be a mapping")
        architecture = artifact.get("model_spec", {}).get("architecture", "current")
        if architecture not in {"current", "legacy_resnet50_v1"}:
            raise ValueError(f"Unsupported checkpoint architecture: {architecture!r}")
        expected = _contract(cfg, architecture)
        if version is not None:
            for key, value in expected.items():
                if artifact.get(key) != value:
                    raise ValueError(f"Checkpoint {key} disagrees with its saved config")
        else:
            warnings.warn("Loading an unversioned checkpoint using its saved config; resave as schema v1",
                          FutureWarning, stacklevel=2)
            for key, expected_value in [("model_name", cfg.model.name), ("output_dim", cfg.model.output_dim)]:
                if key in artifact and artifact[key] != expected_value:
                    raise ValueError(f"Unversioned checkpoint {key} disagrees with cfg")
        metadata = {"schema_version": SCHEMA_VERSION, **expected,
                    "source_format": "v1" if version is not None else "unversioned",
                    "epoch": artifact.get("epoch"), "val_mae": artifact.get("val_mae"),
                    "training_objective": artifact.get("training_objective"),
                    "experiment_metadata": artifact.get("experiment_metadata")}
    if architecture == "legacy_resnet50_v1":
        expected = _contract(_legacy_config(), architecture)
        if any(metadata[key] != expected[key] for key in expected):
            raise ValueError("Legacy profile must match historical architecture, preprocessing and raw decoder")
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("model_state_dict must be a nonempty mapping of tensors")
    overrides = dict(runtime_overrides or {})
    if device is not None:
        overrides["device"] = device
    cfg = _runtime_config(cfg, config_path, overrides)
    target = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    model = LegacyResNet50() if architecture == "legacy_resnet50_v1" else AgeEstimationModel(
        cfg.model.name, pretrained=False, output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout, img_size=cfg.data.img_size,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint weights do not match {architecture}/{cfg.model.name}: {exc}") from exc
    model = model.to(target).eval()
    loss_fn = (LegacyAgeDecoder() if architecture == "legacy_resnet50_v1" else get_loss_fn(cfg)).to(target)
    transform = get_test_transforms(cfg.data.img_size, list(cfg.data.mean), list(cfg.data.std))
    return LoadedCheckpoint(model, cfg, loss_fn, transform, target, metadata)


def add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    """Provide the same loading policy on every inference-oriented command."""
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint artifact")
    parser.add_argument("--config", default=None, help="Runtime overrides only; saved inference contract is authoritative")
    parser.add_argument("--device", default=None, help="cpu or cuda[:index]; default: auto")
    parser.add_argument("--legacy-resnet50", action="store_true", help="Explicit historical 128px/raw ResNet50 profile")
    parser.add_argument("--trust-legacy-pickle", action="store_true",
                        help="Only for trusted old pickles; permits arbitrary code execution during loading")


def load_checkpoint_args(args, **kwargs) -> LoadedCheckpoint:
    return load_checkpoint(args.checkpoint, args.config, device=args.device,
                           legacy_resnet50=args.legacy_resnet50,
                           trust_legacy_pickle=args.trust_legacy_pickle, **kwargs)


def atomic_save(artifact: dict, path: str | Path) -> None:
    """Flush a same-directory temporary file, then atomically replace one checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(artifact, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_training_checkpoint(path: str | Path) -> dict:
    """Read a restricted, explicitly resumable artifact; legacy inference files are not resumes."""
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError("Resume requires a restricted-loadable P2.1 training checkpoint") from exc
    if not isinstance(artifact, dict) or type(artifact.get("schema_version")) is not int or artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Resume requires a schema-v1 P2.1 training checkpoint")
    state = artifact.get("training_state")
    if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1:
        raise ValueError("Checkpoint has no supported complete training state; use it for inference, not --resume")
    defaults = OmegaConf.create(files("src.configs").joinpath("default.yaml").read_text(encoding="utf-8"))
    cfg = merge_config(defaults, artifact.get("cfg", {}))
    for key, expected in _contract(cfg).items():
        if artifact.get(key) != expected:
            raise ValueError(f"Resume checkpoint {key} disagrees with its saved config/current architecture")
    required = {"run_id", "next_epoch", "global_step", "successful_updates", "best_mae", "best_epoch",
                "scheduler", "scaler", "rng", "loaders", "contract", "loss", "history"}
    if required - state.keys() or {"optimizer_state_dict", "model_state_dict", "experiment_metadata", "epoch", "val_mae"} - artifact.keys():
        raise ValueError("Incomplete resume checkpoint training state")
    if not isinstance(artifact["experiment_metadata"], dict) or artifact.get("training_objective") != training_objective_spec(cfg.loss.type):
        raise ValueError("Resume objective/experiment metadata is missing or incompatible")
    return artifact
