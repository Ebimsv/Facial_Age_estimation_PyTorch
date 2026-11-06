"""Training correctness, checkpoint failure recovery, parameter identity and logger lifecycle."""

from __future__ import annotations

import copy
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.configs import load_config
from src.models import AgeEstimationModel
from src.training.losses import L1AgeLoss, get_loss_fn
from src.training.trainer import Trainer
from src.utils.checkpoint import atomic_save, load_checkpoint, load_training_checkpoint
from src.utils.seed import seed_worker


class TinyModel(nn.Module):
    """A cheap model implementing the same metadata/head contract as the production wrapper."""
    def __init__(self, *, dropout=.2):
        super().__init__()
        self.model_name, self.output_dim, self.img_size, self.dropout = "mobilenet_v3", 1, 32, dropout
        self.encoder = nn.Sequential(nn.Linear(3, 8), nn.Tanh())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(8, 1))
        with torch.no_grad():
            self.head[-1].bias.fill_(10)

    def forward(self, x):
        return self.head(self.encoder(x))

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters()) / 1e6


class RandomDataset(Dataset):
    def __init__(self, n=12, stochastic=True):
        self.n, self.stochastic = n, stochastic

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        x = torch.tensor([index / 12, 1., -.5])
        if self.stochastic:
            x += .03 * (random.random() + float(np.random.random()) + torch.rand(()))
        return x, torch.tensor([12. + index / 12]), "Male", "White"


def settings(tmp_path, **updates):
    patch = {"model": {"name": "mobilenet_v3", "pretrained": False, "output_dim": 1, "dropout": .2},
             "loss": {"type": "l1"}, "data": {"img_size": 32},
             "training": {"epochs": 4, "batch_size": 4, "eval_batch_size": 4, "use_amp": False, "grad_clip": 1.},
             "optimizer": {"backbone_lr": .01, "head_lr": .03, "weight_decay": .001},
             "scheduler": {"pct_start": .3},
             "logging": {"save_dir": str(tmp_path / "checkpoints"), "log_dir": str(tmp_path / "logs"),
                         "use_tensorboard": False, "use_wandb": False, "log_every_n_steps": 1}}
    return load_config(updates=patch, overrides=[f"{key}={value}" for key, value in updates.items()])


def loaders(workers=0, *, n=12, stochastic=True):
    return tuple(DataLoader(RandomDataset(n, stochastic=stochastic), batch_size=4, shuffle=train,
                            num_workers=workers, worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(91 + int(train))) for train in (True, False))


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("name", ["mobilenet_v3", "efficientnet_b0", "efficientnet_b3", "resnet50", "vit_small"])
def test_head_partition_uses_identity_and_preserves_state_keys(tmp_path, name):
    model = AgeEstimationModel(name, pretrained=False)
    cfg = settings(tmp_path, **{"model.name": name, "data.img_size": 224})
    trainer = Trainer(model, L1AgeLoss(), cfg, torch.device("cpu"))
    try:
        groups = trainer.optimizer.param_groups
        head_ids = {id(p) for p in model.head.parameters() if p.requires_grad}
        actual_head = {id(p) for p in groups[1]["params"]}
        backbone = {id(p) for p in groups[0]["params"]}
        assert actual_head == head_ids and not actual_head & backbone
        assert head_ids | backbone == {id(p) for p in model.parameters() if p.requires_grad}
        assert not any(key.startswith("head.") for key in model.state_dict())
        if name == "vit_small":
            assert {id(p) for p in model.model.blocks[0].mlp.fc1.parameters()} <= backbone
        if name == "mobilenet_v3":
            assert {id(p) for p in model.model.classifier[0].parameters()} <= backbone
    finally:
        trainer.close()


def test_frozen_parameters_are_not_optimized(tmp_path):
    model = TinyModel()
    model.encoder.requires_grad_(False)
    trainer = Trainer(model, L1AgeLoss(), settings(tmp_path), torch.device("cpu"))
    assert trainer.optimizer.param_groups[0]["params"] == []
    assert {id(p) for p in trainer.optimizer.param_groups[1]["params"]} == {id(p) for p in model.head.parameters()}
    trainer.close()


def test_short_training_is_finite_decreases_loss_and_saves_safe_full_states(tmp_path):
    torch.manual_seed(42)
    model = TinyModel(dropout=.2)
    trainer = Trainer(model, L1AgeLoss(), settings(tmp_path), torch.device("cpu"))
    train, val = loaders(stochastic=False)
    trainer.run(train, val)
    assert trainer.history[-1]["train/epoch_loss"] < trainer.history[0]["train/epoch_loss"]
    assert trainer.global_step == trainer.successful_updates == 12
    assert trainer.scheduler.last_epoch == 12
    assert trainer.last_ckpt_path.exists() and trainer.best_ckpt_path.exists()
    assert not (trainer.save_dir / ".training.lock").exists()
    assert sorted(path.name for path in trainer.save_dir.iterdir()) == ["best.pt", "last.pt"]
    artifact = load_training_checkpoint(trainer.last_ckpt_path)
    state = artifact["training_state"]
    assert state["next_epoch"] == 4 and state["global_step"] == 12
    assert state["rng"]["numpy"]["keys"].dtype == torch.int64
    assert_tree_equal(artifact["optimizer_state_dict"], trainer.optimizer.state_dict())


@pytest.mark.parametrize("failure", ["save", "fsync", "replace"])
def test_atomic_failure_preserves_old_bytes_and_cleans_temporary(tmp_path, monkeypatch, failure):
    import src.utils.checkpoint as checkpoints
    path = tmp_path / "best.pt"
    atomic_save({"old": torch.tensor([1.])}, path)
    original = path.read_bytes()

    def fail(*args, **kwargs):
        if failure == "save":
            args[1].write(b"partial")
        raise OSError("injected write failure")

    monkeypatch.setattr(checkpoints.torch if failure == "save" else checkpoints.os,
                        {"save": "save", "fsync": "fsync", "replace": "replace"}[failure], fail)
    with pytest.raises(OSError):
        atomic_save({"new": torch.tensor([2.])}, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_best_save_failure_preserves_metric_weights_and_run_isolation(tmp_path, monkeypatch):
    import src.utils.checkpoint as checkpoints
    cfg = settings(tmp_path)
    a, b = [Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu")) for _ in range(2)]
    a._save_checkpoint(0, 5.)
    b._save_checkpoint(0, 4.)
    assert a.save_dir != b.save_dir
    before_a, before_b = a.best_ckpt_path.read_bytes(), b.best_ckpt_path.read_bytes()
    monkeypatch.setattr(checkpoints.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("no replace")))
    with pytest.raises(OSError):
        a._save_checkpoint(1, 3.)
    assert a.best_mae == 5 and a.best_epoch == 0
    assert a.best_ckpt_path.read_bytes() == before_a and b.best_ckpt_path.read_bytes() == before_b
    a.close()
    b.close()


class SkipFirstScaler:
    def __init__(self):
        self.calls = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        self.calls += 1
        if self.calls > 1:
            optimizer.step()

    def update(self):
        pass


def test_amp_skip_does_not_advance_scheduler_or_success_counter(tmp_path):
    trainer = Trainer(TinyModel(), L1AgeLoss(), settings(tmp_path), torch.device("cpu"))
    train, _ = loaders(stochastic=False)
    trainer._build_scheduler(len(train))
    trainer.scaler = SkipFirstScaler()
    initial = [group["lr"] for group in trainer.optimizer.param_groups]
    observed = []
    original_log = trainer._log_scalar
    trainer._log_scalar = lambda tag, value, step: observed.append((step, trainer.scheduler.last_epoch,
                                                                   [g["lr"] for g in trainer.optimizer.param_groups])) if tag == "train/step_loss" else original_log(tag, value, step)
    trainer.train_one_epoch(train, 0, 0)
    assert trainer.global_step == 3 and trainer.successful_updates == trainer.scheduler.last_epoch == 2
    assert observed[0][1] == 0 and observed[0][2] == initial
    trainer.close()


@pytest.mark.parametrize("failure", ["train", "validate", "save", "wandb_init"])
def test_loggers_and_lock_are_cleaned_on_failure(tmp_path, monkeypatch, failure):
    import torch.utils.tensorboard as tb

    import src.training.trainer as training
    closed, events = [], []

    class Writer:
        def __init__(self, **kwargs):
            events.append(kwargs)

        def add_scalar(self, *args):
            pass

        def close(self):
            closed.append("tensorboard")

    run = SimpleNamespace(define_metric=lambda *a, **kw: None, log=lambda *a: None,
                          finish=lambda: closed.append("wandb"))

    def init(**kwargs):
        if failure == "wandb_init":
            raise RuntimeError("logger failed")
        return run

    monkeypatch.setattr(tb, "SummaryWriter", Writer)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    cfg = settings(tmp_path, **{"logging.use_tensorboard": "true", "logging.use_wandb": "true", "logging.wandb_mode": "offline"})
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    def fail(*args, **kwargs):
        raise RuntimeError("injected")
    if failure == "train":
        monkeypatch.setattr(trainer, "train_one_epoch", fail)
    elif failure == "validate":
        monkeypatch.setattr(trainer, "validate", fail)
    elif failure == "save":
        monkeypatch.setattr(training, "atomic_save", fail)
    with pytest.raises(RuntimeError):
        trainer.run(*loaders(stochastic=False))
    assert "tensorboard" in closed
    assert ("wandb" in closed) == (failure != "wandb_init")
    assert trainer._closed and not (trainer.save_dir / ".training.lock").exists()
    assert str(trainer.run_id) in events[0]["log_dir"]


def test_loggers_use_one_global_axis_and_separate_wandb_sessions(tmp_path, monkeypatch):
    seen, records, definitions = [], [], []

    def init(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(define_metric=lambda *a, **kw: definitions.append((a, kw)),
                               log=lambda item: records.append(item), finish=lambda: None)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    cfg = settings(tmp_path, **{"logging.use_wandb": "true", "logging.wandb_mode": "offline"})
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    trainer.run(*loaders(stochastic=False))
    assert all("trainer/global_step" in item for item in records)
    assert [item["trainer/global_step"] for item in records] == sorted(item["trainer/global_step"] for item in records)
    assert seen[0]["group"] == trainer.run_id and trainer.run_id in seen[0]["dir"]
    assert (('*',), {"step_metric": "trainer/global_step"}) in definitions


def test_tensorboard_records_real_events_under_run_id(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    cfg = settings(tmp_path, **{"logging.use_tensorboard": "true"})
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    trainer.run(*loaders(stochastic=False))
    events = EventAccumulator(str(trainer.log_dir)).Reload()
    assert [event.step for event in events.Scalars("val/mae")] == [3, 6, 9, 12]


def test_wandb_offline_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "0" * 40)
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    cfg = settings(tmp_path, **{"logging.use_wandb": "true", "logging.wandb_mode": "offline"})
    trainer = Trainer(TinyModel(), L1AgeLoss(), cfg, torch.device("cpu"))
    trainer.run(*loaders(stochastic=False))
    assert list(trainer.log_dir.glob("wandb/offline-run-*/run-*.wandb"))


def test_production_checkpoint_with_training_state_still_loads_for_inference(tmp_path):
    cfg = settings(tmp_path, **{"training.epochs": 2, "training.batch_size": 2, "training.eval_batch_size": 2})
    model = AgeEstimationModel("mobilenet_v3", pretrained=False, output_dim=1, dropout=.2, img_size=32)
    trainer = Trainer(model, get_loss_fn(cfg), cfg, torch.device("cpu"))
    data = [(torch.ones(3, 32, 32) * i, torch.tensor([20.]), "Male", "White") for i in range(4)]
    train = DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(1), shuffle=True)
    val = DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(2))
    trainer.run(train, val)
    loaded = load_checkpoint(trainer.last_ckpt_path, device="cpu")
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(torch.zeros(1, 3, 32, 32)), model(torch.zeros(1, 3, 32, 32)), rtol=0, atol=0)
    assert loaded.cfg.model.name == "mobilenet_v3"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_cuda_scaler_skip_and_checkpoint_state(tmp_path):
    cfg = settings(tmp_path, **{"training.use_amp": "true"})
    trainer = Trainer(TinyModel().cuda(), L1AgeLoss(), cfg, torch.device("cuda"))
    train, val = loaders(stochastic=False)
    trainer._build_scheduler(len(train))
    trainer.scaler = torch.amp.GradScaler("cuda", init_scale=2.**30)
    before = copy.deepcopy(trainer.model.state_dict())
    trainer.train_one_epoch(train, 0, 0)
    assert trainer.global_step == 3 and trainer.successful_updates == trainer.scheduler.last_epoch == 0
    assert_tree_equal(before, trainer.model.state_dict())
    trainer._loaders = {"train": train, "val": val}
    trainer._save_checkpoint(0, 5.)
    saved = torch.load(trainer.best_ckpt_path, weights_only=True, map_location="cpu")
    assert saved["training_state"]["rng"]["cuda"]
    assert saved["training_state"]["scaler"]["scale"] < 2.**30
    trainer.close()


@pytest.fixture(autouse=True)
def restore_determinism_flags():
    flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    torch.use_deterministic_algorithms(flags[0])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[1:]


@pytest.mark.parametrize("empty", ["train", "val"])
def test_empty_loader_fails_without_creating_run_or_reporting_success(tmp_path, empty):
    pair = list(loaders())
    pair[0 if empty == "train" else 1] = DataLoader(RandomDataset(n=0), batch_size=4)
    trainer = Trainer(TinyModel(), L1AgeLoss(), settings(tmp_path), torch.device("cpu"))
    with pytest.raises(ValueError, match="must contain batches"):
        trainer.run(*pair)
    assert trainer._closed and not (tmp_path / "checkpoints").exists()
