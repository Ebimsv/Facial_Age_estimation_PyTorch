"""Configuration contracts, fail-fast entry points and synthetic preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ReadonlyConfigError

from scripts import prepare_data, train
from src.configs import load_config
from src.data.dataset import UTKDataset, build_dataloaders
from src.models import AgeEstimationModel, build_model
from src.training.losses import L1AgeLoss, MeanVarianceLoss, get_loss_fn
from src.utils.config import (
    merge_config,
    to_plain_config,
    validate_split_options,
    validate_training_steps,
)


@pytest.mark.parametrize(("override", "error"), [
    ("training.epohs=3", "epohs"),
    ("surprise=1", "surprise"),
    ("training.epochs=0", "training.epochs"),
    ("training.epochs=1.5", "epochs"),
    ("training.batch_size=0", "batch_size"),
    ("training.eval_batch_size=-2", "eval_batch_size"),
    ("training.seed=-1", "seed"),
    ("training.seed=4294967296", "seed"),
    ("training.grad_clip=-1", "grad_clip"),
    ("model.name=missing", "Unknown backbone"),
    ("model.output_dim=1", "output_dim"),
    ("loss.type=l1", "output_dim"),
    ("loss.type=unknown", "Unknown loss type"),
    ("loss.start_age=-1", "start_age"),
    ("loss.end_age=-1", "end_age"),
    ("loss.start_age=101", "start_age"),
    ("model.dropout=1", "dropout"),
    ("model.dropout=nan", "finite"),
    ("model.pretrained=maybe", "pretrained"),
    ("data.img_size=0", "img_size"),
    ("data.num_workers=-1", "num_workers"),
    ("data.mean=[0,1]", "three RGB"),
    ("data.mean=${model.name}", "three RGB"),
    ("data.mean=[0,nan,1]", "finite"),
    ("data.std=[1,0,1]", "std"),
    ("data.std=[1,-1,1]", "std"),
    ("data.std=[1,inf,1]", "finite"),
    ("data.train_ratio=0", "train_ratio"),
    ("data.val_ratio=-0.1", "val_ratio"),
    ("data.train_ratio=0.9", "test split"),
    ("data.max_age=0", "max_age"),
    ("data.max_age=102", "max_age"),
    ("data.root_dir=''", "root_dir"),
    ("optimizer.backbone_lr=0", "backbone_lr"),
    ("optimizer.head_lr=-0.1", "head_lr"),
    ("optimizer.weight_decay=-1", "weight_decay"),
    ("loss.lambda_mean=-1", "lambda_mean"),
    ("loss.lambda_variance=inf", "finite"),
    ("scheduler.pct_start=0", "pct_start"),
    ("scheduler.pct_start=1", "pct_start"),
    ("logging.log_every_n_steps=-1", "log_every_n_steps"),
    ("device=mps", "device"),
    ("training.epochs=???", "Missing"),
    ("training.epochs=${missing}", "missing"),
    ("training.epochs", "key=value"),
])
def test_invalid_configuration_fails(override, error):
    with pytest.raises(ValueError, match=error):
        load_config(overrides=[override])


def test_overrides_are_validated_after_merging_partial_yaml(tmp_path):
    custom = tmp_path / "partial.yaml"
    custom.write_text("model:\n  output_dim: 1\ntraining:\n  epochs: 0\n", encoding="utf-8")
    cfg = load_config(custom, overrides=["loss.type=l1", "training.epochs=2"])
    assert cfg.model.output_dim == 1 and cfg.training.epochs == 2
    assert isinstance(get_loss_fn(cfg), L1AgeLoss)


def test_config_is_resolved_plain_and_read_only():
    cfg = load_config(overrides=["optimizer.head_lr=${optimizer.backbone_lr}"])
    plain = to_plain_config(cfg)
    assert plain["optimizer"]["head_lr"] == cfg.optimizer.backbone_lr
    assert "${" not in json.dumps(plain)
    assert not OmegaConf.is_config(plain["data"])
    with pytest.raises(ReadonlyConfigError, match="read-only"):
        cfg.model.output_dim = 1


def test_structured_schema_matches_all_packaged_defaults():
    path = Path(__file__).parents[1] / "src/configs/default.yaml"
    declared = OmegaConf.to_container(OmegaConf.load(path))
    assert to_plain_config(load_config(path)) == declared


@pytest.mark.parametrize("updates", [
    {"training": {"epochs": True}},
    {"data": {"img_size": False}},
    {"training": None},
    {"data": {"std": None}},
])
def test_invalid_types_are_not_silently_accepted(updates):
    with pytest.raises(ValueError):
        load_config(updates=updates)


def test_legacy_defaults_migrate_with_visible_warning():
    with pytest.warns(FutureWarning) as messages:
        cfg = merge_config(load_config(), {
            "logging": {"save_best_only": True},
            "inference": {"checkpoint_path": None, "image_path": None, "output_path": None},
        })
    assert len(messages) == 2 and "inference" not in cfg and "save_best_only" not in cfg.logging


@pytest.mark.parametrize("updates", [
    {"logging": {"save_best_only": False}},
    {"inference": {"image_path": "photo.jpg"}},
    {"inference": {"typo": None}},
])
def test_effectful_legacy_options_require_migration(updates):
    with pytest.raises(ValueError, match="unsupported"):
        merge_config(load_config(), updates)


@pytest.mark.parametrize("size", [224, 280])
def test_vit_resolution_is_passed_to_actual_backbone(size):
    cfg = load_config(overrides=["model.name=vit_small", "model.pretrained=false", f"data.img_size={size}"])
    model = build_model(cfg).eval()
    assert model.model.patch_embed.img_size == (size, size)
    with torch.no_grad():
        assert model(torch.zeros(1, 3, size, size)).shape == (1, 101)


def test_vit_invalid_resolution_fails_before_backbone_construction(monkeypatch):
    import src.models.age_estimator as models

    def forbidden(*args, **kwargs):
        pytest.fail("Must validate before timm construction/download")

    monkeypatch.setattr(models.timm, "create_model", forbidden)
    with pytest.raises(ValueError, match="multiple of 14"):
        AgeEstimationModel("vit_small", pretrained=True, img_size=192)


def test_distribution_with_shifted_support():
    cfg = load_config(overrides=["loss.start_age=10", "loss.end_age=20", "model.output_dim=11", "data.max_age=21"])
    loss = get_loss_fn(cfg)
    assert loss.predict_age(torch.zeros(2, 11)).tolist() == pytest.approx([15, 15])
    assert torch.isfinite(loss(torch.zeros(2, 11), torch.tensor([10., 20.]))[0])


def test_model_only_factory_config_remains_compatible():
    model_spec = to_plain_config(load_config(overrides=["model.name=efficientnet_b0", "model.pretrained=false"]))["model"]
    model = build_model(OmegaConf.create({"model": model_spec}))
    assert model.model_name == "efficientnet_b0" and model.output_dim == 101


@pytest.mark.parametrize("overrides", [
    ["loss.lambda_mean=0", "loss.lambda_variance=0"],
    ["loss.start_age=20", "loss.end_age=20", "model.output_dim=1", "data.max_age=21"],
    ["loss.type=l1", "model.output_dim=1", "loss.end_age=120"],
])
def test_invalid_loss_support_and_weights(overrides):
    with pytest.raises(ValueError):
        load_config(overrides=overrides)


@pytest.mark.parametrize("loss,logits,targets", [
    (MeanVarianceLoss(), torch.zeros(2, 1), torch.ones(2)),
    (MeanVarianceLoss(), torch.zeros(2, 101), torch.ones(1)),
    (MeanVarianceLoss(), torch.zeros(2, 101), torch.tensor([-1., 20.])),
    (MeanVarianceLoss(), torch.zeros(2, 101), torch.tensor([101., 20.])),
    (MeanVarianceLoss(), torch.zeros(2, 101), torch.tensor([float("nan"), 20.])),
    (L1AgeLoss(), torch.zeros(2, 101), torch.ones(2)),
    (L1AgeLoss(), torch.zeros(2, 1), torch.ones(2, 2)),
    (L1AgeLoss(), torch.zeros(0, 1), torch.zeros(0)),
])
def test_loss_contract_rejects_invalid_shapes_and_targets(loss, logits, targets):
    with pytest.raises(ValueError):
        loss(logits, targets)


def test_distribution_decoder_rejects_regression_broadcast():
    with pytest.raises(ValueError, match="101"):
        MeanVarianceLoss().predict_age(torch.tensor([[20.]]))


def test_invalid_train_cli_fails_before_data_seed_or_model(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train-age", "model.output_dim=1"])
    monkeypatch.setattr(train, "set_seed", lambda *a: pytest.fail("Validation must run first"))
    with pytest.raises(ValueError, match="output_dim"):
        train.main()


def test_print_config_needs_no_data_or_model(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["train-age", "--print-config", "model.output_dim=1", "loss.type=l1"])
    monkeypatch.setattr(train, "set_seed", lambda *a: pytest.fail("Print-only must not initialize runtime"))
    train.main()
    assert "output_dim: 1" in capsys.readouterr().out


def _csv(path, ages):
    pd.DataFrame({"image_name": [f"{i}.jpg" for i in range(len(ages))], "age": ages,
                  "gender": "Male", "ethnicity": "White"}).to_csv(path, index=False)


@pytest.mark.parametrize("ages", [[-1], [101], [float("nan")], ["not-a-number"]])
def test_csv_age_range_rejected_before_image_io(tmp_path, ages):
    path = tmp_path / "data.csv"
    _csv(path, ages)
    with pytest.raises(ValueError, match="age targets"):
        UTKDataset(tmp_path, path, age_range=(0, 100))


@pytest.mark.parametrize(("n", "batch", "error"), [(0, 2, "empty"), (1, 2, "zero batches")])
def test_empty_and_zero_batch_loaders(tmp_path, n, batch, error):
    if n:
        (tmp_path / "0.jpg").touch()
    for split in ["train", "valid", "test"]:
        _csv(tmp_path / f"{split}_set.csv", [20] * n)
    with pytest.raises(ValueError, match=error):
        build_dataloaders(tmp_path, tmp_path, None, None, None, batch_size=batch, num_workers=0,
                          age_range=(0, 100))


def test_preparation_uses_config_and_records_effective_parameters(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(40):
        (images / f"20_0_0_{i}.jpg").touch()
    # This parameter-only test opts out of content hashes; decode checks are separate.
    custom = tmp_path / "config.yaml"
    custom.write_text("data:\n  train_ratio: 0.6\n  val_ratio: 0.25\ntraining:\n  seed: 7\n", encoding="utf-8")
    output = tmp_path / "csv"
    monkeypatch.setattr("sys.argv", ["prepare_data", "--config", str(custom), "--dataset-dir", str(images),
                                     "--csv-dir", str(output), "--train-ratio", "0.5", "--skip-content-hash"])
    prepare_data.main()
    manifest = json.loads((output / "preparation_manifest.json").read_text())
    assert manifest["parameters"]["train_ratio"] == 0.5
    assert manifest["parameters"]["val_ratio"] == 0.25
    assert manifest["parameters"]["seed"] == 7
    assert manifest["counts"] == {"master": 40, "train": 20, "val": 10, "test": 10}


def test_invalid_prepare_flags_do_not_create_output(tmp_path, monkeypatch):
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["prepare_data", "--csv-dir", str(output), "--train-ratio", "1"])
    with pytest.raises(ValueError, match="test split"):
        prepare_data.main()
    assert not output.exists()


@pytest.mark.parametrize(("train_ratio", "val_ratio"), [(0, .1), (.8, 0), (.9, .1), (float("nan"), .1)])
def test_split_validation_is_reused_by_direct_calls(train_ratio, val_ratio):
    with pytest.raises(ValueError):
        validate_split_options(train_ratio, val_ratio, 42)
    with pytest.raises(ValueError):
        prepare_data.stratified_split(pd.DataFrame(), train_ratio, val_ratio, 42)


def test_scheduler_preflight():
    with pytest.raises(ValueError, match="batches per epoch"):
        validate_training_steps(load_config(), 0)
    with pytest.raises(ValueError, match="warmup"):
        validate_training_steps(load_config(overrides=["training.epochs=2", "scheduler.pct_start=0.5"]), 1)
