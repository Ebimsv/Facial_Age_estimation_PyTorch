"""Round-trip contracts and explicit historical checkpoint compatibility."""

from __future__ import annotations

import copy
import pickle
from importlib.resources import files

import pytest
import torch
from omegaconf import OmegaConf

from src.configs import load_config
from src.models import build_model
from src.models.legacy import LegacyResNet50
from src.training.trainer import Trainer
from src.utils.checkpoint import _legacy_config, create_checkpoint, load_checkpoint
from src.utils.config import merge_config, to_plain_config


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("checkpoint")
    result = {}
    for mode in ["l1", "mean_variance"]:
        overrides = ["model.name=mobilenet_v3", "model.pretrained=false", "data.img_size=64",
                     "data.mean=[0.1,0.2,0.3]", "data.std=[0.5,0.6,0.7]", f"loss.type={mode}"]
        overrides += ["model.output_dim=1"] if mode == "l1" else [
            "loss.start_age=10", "loss.end_age=20", "model.output_dim=11", "data.max_age=21"]
        cfg = load_config(overrides=overrides)
        model = build_model(cfg).eval()
        artifact = create_checkpoint(model, cfg, epoch=2, val_mae=3.5)
        path = root / f"{mode}.pt"
        torch.save(artifact, path)
        result[mode] = (path, artifact, model, cfg)
    return result


@pytest.mark.parametrize("mode", ["l1", "mean_variance"])
def test_roundtrip_restores_nondefault_pipeline(artifacts, mode):
    path, artifact, model, cfg = artifacts[mode]
    loaded = load_checkpoint(path, device="cpu")
    assert loaded.cfg.data.img_size == 64
    assert list(loaded.cfg.data.mean) == [.1, .2, .3]
    assert loaded.cfg.loss.type == mode
    assert loaded.model.training is False
    assert loaded.metadata["predictor"] == artifact["predictor"]
    with torch.no_grad():
        images = torch.randn(2, 3, 64, 64)
        torch.testing.assert_close(loaded.model(images), model(images), rtol=0, atol=0)
    assert torch.load(path, weights_only=True)["cfg"] == to_plain_config(cfg)


@pytest.mark.parametrize("key,value", [
    ("schema_version", 99), ("schema_version", None), ("schema_version", True),
    ("preprocessing", {}), ("predictor", {}), ("model_spec", None),
])
def test_bad_version_or_contract_rejected(tmp_path, artifacts, key, value):
    artifact = copy.copy(artifacts["l1"][1])
    artifact[key] = value
    path = tmp_path / "bad.pt"
    torch.save(artifact, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_missing_metadata_is_not_inferred_from_defaults(tmp_path, artifacts):
    artifact = copy.deepcopy(artifacts["l1"][1])
    del artifact["cfg"]["data"]["mean"]
    path = tmp_path / "missing.pt"
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="missing cfg.data.mean"):
        load_checkpoint(path)


def test_state_shape_mismatch_is_strict(tmp_path, artifacts):
    artifact = copy.deepcopy(artifacts["l1"][1])
    artifact["model_state_dict"].pop(next(iter(artifact["model_state_dict"])))
    path = tmp_path / "wrong-weights.pt"
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="weights do not match"):
        load_checkpoint(path)


def test_runtime_overrides_and_protected_contract(tmp_path, artifacts):
    path = artifacts["l1"][0]
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("device: cpu\ndata:\n  root_dir: /new/images\n  num_workers: 0\n", encoding="utf-8")
    loaded = load_checkpoint(path, runtime)
    assert loaded.cfg.data.root_dir == "/new/images" and loaded.cfg.data.img_size == 64
    runtime.write_text("data:\n  img_size: 224\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable.*img_size"):
        load_checkpoint(path, runtime)
    with pytest.raises(ValueError, match="Not an allowed"):
        load_checkpoint(path, runtime_overrides={"model.name": "resnet50"})


def test_runtime_yaml_interpolation_resolves_against_saved_cfg(tmp_path, artifacts):
    path, _, _, cfg = artifacts["l1"]
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("data:\n  root_dir: ${data.csv_dir}\n", encoding="utf-8")
    loaded = load_checkpoint(path, runtime, device="cpu")
    assert loaded.cfg.data.root_dir == cfg.data.csv_dir


@pytest.mark.parametrize("mode", ["l1_mean_variance", "mean_variance"])
@pytest.mark.parametrize("old_training_defaults", [False, True])
def test_explicit_unmodified_default_yaml_does_not_replace_regression_contract(tmp_path, artifacts, mode, old_training_defaults):
    path, _, model, _ = artifacts["l1"]
    default_yaml = tmp_path / "external-default.yaml"
    defaults = OmegaConf.create(files("src.configs").joinpath("default.yaml").read_text())
    defaults.loss.type = mode
    if old_training_defaults:
        del defaults.training.deterministic
    default_yaml.write_text(OmegaConf.to_yaml(defaults), encoding="utf-8")
    bundle = load_checkpoint(path, default_yaml, device="cpu")
    assert bundle.cfg.loss.type == "l1" and bundle.cfg.data.img_size == 64
    assert list(bundle.cfg.data.mean) == [.1, .2, .3]
    with torch.no_grad():
        tensor = torch.randn(1, 3, 64, 64)
        torch.testing.assert_close(bundle.model(tensor), model(tensor), rtol=0, atol=0)


def test_unversioned_plain_config_is_adapted(tmp_path, artifacts):
    _, artifact, _, cfg = artifacts["l1"]
    old = {"cfg": to_plain_config(cfg), "model_state_dict": artifact["model_state_dict"],
           "model_name": cfg.model.name, "output_dim": 1}
    path = tmp_path / "unversioned.pt"
    torch.save(old, path)
    with pytest.warns(FutureWarning, match="unversioned"):
        loaded = load_checkpoint(path, device="cpu")
    assert loaded.metadata["source_format"] == "unversioned"


def test_safe_load_never_retries_unsafe(monkeypatch):
    calls = []

    def blocked(*args, **kwargs):
        calls.append(kwargs["weights_only"])
        raise pickle.UnpicklingError("unsupported global")

    monkeypatch.setattr(torch, "load", blocked)
    with pytest.raises(ValueError, match="trust-legacy-pickle"):
        load_checkpoint("old.pt")
    assert calls == [True]


def test_old_omegaconf_pickle_requires_explicit_trust(tmp_path, artifacts):
    _, artifact, _, cfg = artifacts["l1"]
    path = tmp_path / "old-pickle.pt"
    # Mimic shallow dict(cfg) checkpoints predating P1.2.
    torch.save({"cfg": dict(OmegaConf.create(to_plain_config(cfg))),
                "model_state_dict": artifact["model_state_dict"]}, path)
    with pytest.raises(ValueError, match="Safe checkpoint load failed"):
        load_checkpoint(path)
    with pytest.warns(FutureWarning):
        loaded = load_checkpoint(path, trust_legacy_pickle=True, device="cpu")
    assert loaded.cfg.loss.type == "l1"


def test_legacy_profile_is_explicit_and_keeps_raw_head(tmp_path):
    cfg = _legacy_config()
    model = LegacyResNet50().eval()
    with torch.no_grad():
        model.model.fc[2].weight.zero_()
        model.model.fc[2].bias.fill_(-7)
    path = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match="explicit --legacy-resnet50"):
        load_checkpoint(path)
    loaded = load_checkpoint(path, legacy_resnet50=True, device="cpu")
    assert loaded.cfg.data.img_size == 128 and loaded.model.model.fc[1].out_features == 256
    with torch.no_grad():
        logits = loaded.model(torch.zeros(1, 3, 128, 128))
    assert loaded.loss_fn.predict_age(logits).item() == -7
    converted = tmp_path / "versioned.pt"
    torch.save(create_checkpoint(loaded.model, cfg), converted)
    restored = load_checkpoint(converted, device="cpu")
    assert restored.metadata["predictor"]["postprocess"] == "raw"
    assert restored.metadata.get("training_objective") is None  # no invented training history


def test_trainer_writes_new_schema(tmp_path, artifacts):
    _, _, model, cfg = artifacts["l1"]
    cfg = load_config(updates=to_plain_config(cfg), overrides=[
        f"logging.save_dir={tmp_path}", "logging.use_tensorboard=false"])
    from src.training.losses import get_loss_fn
    trainer = Trainer(model, get_loss_fn(cfg), cfg, torch.device("cpu"))
    trainer._save_checkpoint(0, 4.0)
    saved = torch.load(trainer.best_ckpt_path, weights_only=True)
    assert saved["schema_version"] == 1
    assert saved["training_objective"]["name"] == "l1_raw_regression"
    assert saved["training_objective"]["revision"] == 2
    loaded = load_checkpoint(trainer.best_ckpt_path, device="cpu")
    assert loaded.cfg.loss.type == "l1"
    assert loaded.metadata["training_objective"] == saved["training_objective"]
    experiment = loaded.metadata["experiment_metadata"]
    assert experiment == saved["experiment_metadata"]
    assert experiment["data"] is None  # no loader supplied; provenance stays unknown
    assert experiment["randomness"]["configured_seed"] == cfg.training.seed
    assert isinstance(experiment["environment"]["packages"]["torch"], str)


def test_trainer_saves_loader_data_hashes_without_inventing_source(tmp_path, artifacts):
    import pandas as pd
    from PIL import Image

    from src.data.dataset import build_dataloaders
    from src.training.losses import get_loss_fn
    from src.training.trainer import Trainer

    _, _, model, cfg = artifacts["l1"]
    for split in ["train", "valid", "test"]:
        names = [f"{split}_0.png", f"{split}_1.png"]
        for name in names:
            Image.new("RGB", (2, 2)).save(tmp_path / name)
        pd.DataFrame({"image_name": names, "age": [20, 21], "gender": "Male", "ethnicity": "White"}).to_csv(
            tmp_path / f"{split}_set.csv", index=False)
    loaders = build_dataloaders(tmp_path, tmp_path, None, None, None, batch_size=2, num_workers=0, seed=cfg.training.seed)
    cfg = merge_config(cfg, {"logging": {"save_dir": str(tmp_path / "checkpoints"), "use_tensorboard": False}})
    trainer = Trainer(model, get_loss_fn(cfg), cfg, torch.device("cpu"), data_provenance=loaders[0].data_provenance)
    trainer._save_checkpoint(0, 4.)
    metadata = load_checkpoint(trainer.best_ckpt_path, device="cpu").metadata["experiment_metadata"]
    assert metadata["data"] == loaders[0].data_provenance
    assert metadata["data"]["protocol"] == "unversioned_csv"
    assert metadata["data"]["identity_leakage"] == "unknown"
    assert len(metadata["data"]["csv_files"]["train_set.csv"]["sha256"]) == 64
