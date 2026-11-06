"""Epoch-boundary restart equivalence and explicit rejection of incompatible resumes."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from scripts import train as cli
from src.training.losses import L1AgeLoss
from src.training.trainer import Trainer
from src.utils.checkpoint import atomic_save, create_checkpoint, load_training_checkpoint
from src.utils.config import merge_config
from src.utils.seed import set_seed
from tests.test_trainer import RandomDataset, TinyModel, assert_tree_equal, loaders, settings


class InterruptedError(RuntimeError):
    pass


def interrupted_run(tmp_path, monkeypatch, *, workers=0, cfg=None, data=None, mid_epoch=False, device="cpu"):
    cfg = settings(tmp_path, **{"training.deterministic": "true", "data.num_workers": workers}) if cfg is None else cfg
    set_seed(42, deterministic=True)
    trainer = Trainer(TinyModel().to(device), L1AgeLoss(), cfg, torch.device(device), data_provenance=data)
    if device == "cuda":
        trainer.scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    original = trainer.train_one_epoch

    def interrupt(loader, epoch, step):
        if epoch == 1 and not mid_epoch:
            raise InterruptedError
        return original(loader, epoch, step)

    monkeypatch.setattr(trainer, "train_one_epoch", interrupt)
    if mid_epoch:
        original_log = trainer._log_scalar

        def interrupt_log(tag, value, step):
            if step == 4:
                raise InterruptedError
            original_log(tag, value, step)

        monkeypatch.setattr(trainer, "_log_scalar", interrupt_log)
    with pytest.raises(InterruptedError):
        trainer.run(*loaders(workers))
    assert trainer.last_ckpt_path.exists()
    assert not (trainer.save_dir / ".training.lock").exists()
    return trainer, cfg


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("mid_epoch", [False, True])
def test_epoch_restart_matches_continuous_weights_optimizer_rng_and_lr(tmp_path, monkeypatch, workers, mid_epoch):
    cfg = settings(tmp_path, **{"training.deterministic": "true", "data.num_workers": workers})
    set_seed(42, deterministic=True)
    full = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    full.run(*loaders(workers))
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    broken, _ = interrupted_run(tmp_path, monkeypatch, workers=workers, cfg=cfg, mid_epoch=mid_epoch)
    for _ in range(15):
        random.random(), np.random.random(), torch.rand(10)
    resumed = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    resumed.run(*loaders(workers), resume=broken.last_ckpt_path)
    actual_rng = (random.random(), np.random.random(), torch.rand(3))
    assert_tree_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_tree_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_tree_equal(full.scheduler.state_dict(), resumed.scheduler.state_dict())
    assert_tree_equal(full.scaler.state_dict(), resumed.scaler.state_dict())
    assert_tree_equal(full.history, resumed.history)
    assert_tree_equal(expected_rng, actual_rng)
    assert (full.global_step, full.successful_updates, full.best_mae, full.best_epoch) == (
        resumed.global_step, resumed.successful_updates, resumed.best_mae, resumed.best_epoch)
    assert resumed.save_dir == broken.save_dir


@pytest.mark.parametrize("override", ["training.epochs=5", "training.seed=7", "optimizer.head_lr=0.02",
                                      "training.batch_size=8", "training.grad_clip=2", "scheduler.pct_start=0.4"])
def test_hyperparameter_and_horizon_changes_are_rejected(tmp_path, monkeypatch, override):
    broken, cfg = interrupted_run(tmp_path, monkeypatch)
    original = broken.last_ckpt_path.read_bytes()
    cfg = merge_config(cfg, overrides=[override])
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="contract changed"):
        trainer.run(*loaders(), resume=broken.last_ckpt_path)
    assert broken.last_ckpt_path.read_bytes() == original
    assert not (broken.save_dir / ".training.lock").exists()


@pytest.mark.parametrize("change", ["data", "loader", "workers", "parameter_groups"])
def test_data_and_loader_or_parameter_identity_mismatch_is_rejected(tmp_path, monkeypatch, change):
    broken, cfg = interrupted_run(tmp_path, monkeypatch, data={"csv_sha256": "same"})
    model = TinyModel()
    if change == "parameter_groups":
        model.encoder.requires_grad_(False)
    trainer = Trainer(model, L1AgeLoss(), cfg, torch.device("cpu"),
                      data_provenance={"csv_sha256": "changed" if change == "data" else "same"})
    current = loaders(2 if change == "workers" else 0, n=16 if change == "loader" else 12)
    with pytest.raises(ValueError, match="contract changed|loader contract changed"):
        trainer.run(*current, resume=broken.last_ckpt_path)


@pytest.mark.parametrize("field", ["rng", "scheduler", "scaler", "loaders", "loss", "contract"])
def test_incomplete_state_has_no_unsafe_legacy_resume(tmp_path, monkeypatch, field):
    broken, _ = interrupted_run(tmp_path, monkeypatch)
    artifact = load_training_checkpoint(broken.last_ckpt_path)
    del artifact["training_state"][field]
    target = tmp_path / "invalid.pt"
    atomic_save(artifact, target)
    with pytest.raises(ValueError, match="Incomplete"):
        load_training_checkpoint(target)


@pytest.mark.parametrize("field,value", [("version", 99), ("version", True), ("next_epoch", 100),
                                         ("global_step", 999), ("successful_updates", 999), ("run_id", "")])
def test_bad_resume_versions_and_counters_are_rejected(tmp_path, monkeypatch, field, value):
    broken, cfg = interrupted_run(tmp_path, monkeypatch)
    artifact = load_training_checkpoint(broken.last_ckpt_path)
    artifact["training_state"][field] = value
    target = broken.save_dir / "modified.pt"
    atomic_save(artifact, target)
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="state|counters|run_id"):
        trainer.run(*loaders(), resume=target)


def test_old_inference_artifact_cannot_silently_resume(tmp_path):
    path = tmp_path / "old.pt"
    atomic_save(create_checkpoint(TinyModel(), settings(tmp_path)), path)
    with pytest.raises(ValueError, match="no supported complete training state"):
        load_training_checkpoint(path)


def test_lock_refuses_second_writer_and_preserves_owner(tmp_path, monkeypatch):
    broken, cfg = interrupted_run(tmp_path, monkeypatch)
    lock = broken.save_dir / ".training.lock"
    lock.write_text("other trainer")
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="locked"):
        trainer.run(*loaders(), resume=broken.last_ckpt_path)
    assert lock.read_text() == "other trainer"
    lock.unlink()


def test_missing_or_mismatched_best_is_not_silently_overwritten(tmp_path, monkeypatch):
    broken, cfg = interrupted_run(tmp_path, monkeypatch)
    artifact = load_training_checkpoint(broken.best_ckpt_path)
    artifact["val_mae"] += 1
    atomic_save(artifact, broken.best_ckpt_path)
    before = broken.best_ckpt_path.read_bytes()
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="sibling best.pt disagree"):
        trainer.run(*loaders(), resume=broken.last_ckpt_path)
    assert broken.best_ckpt_path.read_bytes() == before
    assert not (broken.save_dir / ".training.lock").exists()


def test_last_write_failure_keeps_complete_files_and_matching_best_is_recoverable(tmp_path, monkeypatch):
    import src.training.trainer as module
    cfg = settings(tmp_path, **{"training.deterministic": "true"})
    set_seed(42, deterministic=True)
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    original = module.atomic_save

    def fail_last(artifact, path):
        if path.name == "last.pt" and artifact["epoch"] == 1:
            raise OSError("last write failed")
        return original(artifact, path)

    # Force two best epochs so best can be ahead of the last durable boundary.
    monkeypatch.setattr(trainer, "validate", lambda _: (1., 10. - len(trainer.history), .5))
    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_save", fail_last)
        with pytest.raises(OSError):
            trainer.run(*loaders())
    last, best = load_training_checkpoint(trainer.last_ckpt_path), load_training_checkpoint(trainer.best_ckpt_path)
    assert last["epoch"] == 0 and best["epoch"] == 1
    restarted = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="disagree"):
        restarted.run(*loaders(), resume=trainer.last_ckpt_path)
    recovered = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    recovered.run(*loaders(), resume=trainer.best_ckpt_path)
    assert recovered.global_step == 12


def test_cli_resume_print_config_uses_saved_config_and_rejects_changed_schedule_before_model(tmp_path, monkeypatch, capsys):
    broken, cfg = interrupted_run(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "build_model", lambda *a, **kw: pytest.fail("Print/config validation must precede model build"))
    monkeypatch.setattr("sys.argv", ["train-age", "--resume", str(broken.last_ckpt_path), "--print-config"])
    cli.main()
    assert "epochs: 4" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["train-age", "--resume", str(broken.last_ckpt_path), "training.epochs=5"])
    with pytest.raises(ValueError, match="Resume configuration changed"):
        cli.main()


def test_unsupported_sampler_or_persistent_workers_fail_before_output(tmp_path):
    cfg = settings(tmp_path)
    train = DataLoader(RandomDataset(), batch_size=4, sampler=WeightedRandomSampler(torch.ones(12), 12))
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="standard samplers"):
        trainer.run(train, loaders()[1])
    assert not (tmp_path / "checkpoints").exists()


def test_completed_resume_is_idempotent_and_logging_only_overrides_are_allowed(tmp_path, monkeypatch):
    cfg = settings(tmp_path, **{"training.deterministic": "true"})
    set_seed(42, deterministic=True)
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    trainer.run(*loaders())
    original = trainer.last_ckpt_path.read_bytes()
    changed = merge_config(cfg, {"logging": {"log_every_n_steps": 0}, "data": {"root_dir": "relocated"}})
    resumed = Trainer(TinyModel(), L1AgeLoss(), changed, torch.device("cpu"))
    resumed.run(*loaders(), resume=trainer.last_ckpt_path)
    assert resumed.start_epoch == 4 and resumed.last_ckpt_path.read_bytes() == original


@pytest.fixture(autouse=True)
def restore_determinism_flags():
    flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    torch.use_deterministic_algorithms(flags[0])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[1:]


def test_cli_resume_constructs_backbone_without_pretrained_download(tmp_path, monkeypatch):
    broken, cfg = interrupted_run(tmp_path, monkeypatch)
    observed = []
    pair = loaders()
    pair[0].data_provenance = None

    def build(cfg, *, pretrained=None):
        observed.append(pretrained)
        return TinyModel()

    monkeypatch.setattr(cli, "build_model", build)
    monkeypatch.setattr(cli, "build_dataloaders", lambda **kw: (*pair, pair[1]))
    monkeypatch.setattr("sys.argv", ["train-age", "--resume", str(broken.last_ckpt_path), "device=cpu"])
    cli.main()
    assert observed == [False]
    assert load_training_checkpoint(broken.last_ckpt_path)["training_state"]["next_epoch"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_resume_restores_adam_scaler_and_cuda_rng_before_next_epoch(tmp_path, monkeypatch):
    cfg = settings(tmp_path, **{"training.use_amp": "true", "training.deterministic": "true"})
    broken, _ = interrupted_run(tmp_path, monkeypatch, cfg=cfg, device="cuda")
    saved = load_training_checkpoint(broken.last_ckpt_path)
    state = saved["training_state"]
    assert state["successful_updates"] == 3 and saved["optimizer_state_dict"]["state"]
    resumed = Trainer(TinyModel().cuda(), L1AgeLoss(), cfg, torch.device("cuda"))
    original = resumed.train_one_epoch
    checks = []

    def verify(loader, epoch, step):
        if not checks:
            assert resumed.scaler.state_dict() == state["scaler"]
            assert_tree_equal(torch.cuda.get_rng_state_all(), state["rng"]["cuda"])
            for key, actual in resumed.optimizer.state_dict()["state"].items():
                expected = saved["optimizer_state_dict"]["state"][key]
                torch.testing.assert_close(actual["exp_avg"].cpu(), expected["exp_avg"], rtol=0, atol=0)
                assert actual["exp_avg"].is_cuda
            checks.append(True)
        return original(loader, epoch, step)

    monkeypatch.setattr(resumed, "train_one_epoch", verify)
    resumed.run(*loaders(), resume=broken.last_ckpt_path)
    assert checks and resumed.successful_updates == resumed.scheduler.last_epoch == 12
