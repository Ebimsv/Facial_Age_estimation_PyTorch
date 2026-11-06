"""Single-device training with atomic best/last checkpoints and epoch-boundary resume."""

from __future__ import annotations

import copy
import math
import os
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import training_objective_spec
from src.utils.checkpoint import atomic_save, create_checkpoint, load_training_checkpoint
from src.utils.config import to_plain_config, validate_training_steps
from src.utils.environment import collect_environment
from src.utils.metrics import AverageMeter, compute_cumulative_score, compute_mae
from src.utils.training_state import (
    capture_loaders,
    capture_rng,
    restore_loaders,
    restore_rng,
    training_config_signature,
    validate_loaders,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Trainer:
    """Train using an explicit model.head, optional loggers and validated loader provenance.

    Fresh runs allocate unique checkpoint/log directories. Resume uses the supplied
    checkpoint's directory and refuses concurrent writers. Only exhausted epoch
    boundaries, standard samplers and nonpersistent workers are supported.
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module, cfg: DictConfig,
                 device: torch.device, data_provenance: dict | None = None) -> None:
        self.model, self.cfg, self.device = model, cfg, device
        self.loss_fn = loss_fn.to(device)
        self.experiment_metadata = {
            "data": data_provenance, "environment": collect_environment(),
            "device": {"type": str(device), "cuda_runtime": torch.version.cuda,
                       "name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            "randomness": {"configured_seed": cfg.training.seed,
                           "deterministic_requested": cfg.training.deterministic,
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "cudnn_deterministic": torch.backends.cudnn.deterministic,
                           "cudnn_benchmark": torch.backends.cudnn.benchmark,
                           "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        }
        head, backbone = self._split_param_groups()
        self.optimizer = torch.optim.AdamW([
            {"params": backbone, "lr": cfg.optimizer.backbone_lr},
            {"params": head, "lr": cfg.optimizer.head_lr},
        ], weight_decay=cfg.optimizer.weight_decay)
        self.scheduler = None
        self.use_amp = cfg.training.use_amp and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.global_step, self.successful_updates, self.start_epoch = 0, 0, 0
        self.best_mae, self.best_epoch = float("inf"), None
        self.best_ckpt_path = None
        self.last_ckpt_path = None
        self.run_id, self.save_dir, self.log_dir = None, None, None
        self.history = []
        self.writer, self.wandb_run = None, None
        self._loaders, self._lock_path, self._lock_inode = None, None, None
        self._closed = False
        self._step_hook = self.optimizer.register_step_post_hook(self._optimizer_updated)

    def _optimizer_updated(self, optimizer, args, kwargs) -> None:
        self.successful_updates += 1

    def _split_param_groups(self):
        if not isinstance(getattr(self.model, "head", None), nn.Module):
            raise ValueError("Training requires an explicit registered model.head module")
        active = {id(p) for p in self.model.parameters() if p.requires_grad}
        head = [p for p in self.model.head.parameters() if p.requires_grad]
        if not active or not {id(p) for p in head} <= active:
            raise ValueError("model.head must belong to the model, with trainable model parameters")
        head_ids = {id(p) for p in head}
        backbone = [p for p in self.model.parameters() if p.requires_grad and id(p) not in head_ids]
        return head, backbone

    def _build_scheduler(self, steps_per_epoch: int) -> None:
        validate_training_steps(self.cfg, steps_per_epoch)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=[self.cfg.optimizer.backbone_lr, self.cfg.optimizer.head_lr],
            total_steps=self.cfg.training.epochs * steps_per_epoch,
            pct_start=self.cfg.scheduler.pct_start, anneal_strategy="cos",
        )

    def _ensure_run(self) -> None:
        if self.save_dir is None:
            self.run_id = f"{self.cfg.model.name}-{uuid.uuid4().hex[:12]}"
            self.save_dir = Path(self.cfg.logging.save_dir) / self.run_id
            self.save_dir.mkdir(parents=True, exist_ok=False)
        self.log_dir = Path(self.cfg.logging.log_dir) / self.run_id

    def _acquire_lock(self) -> None:
        path = self.save_dir / ".training.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise ValueError("Run is locked. Ensure no trainer is running before manually removing .training.lock") from None
        self._lock_path, self._lock_inode = path, os.fstat(descriptor).st_ino
        try:
            os.write(descriptor, str(os.getpid()).encode())
        finally:
            os.close(descriptor)

    def _open_loggers(self, *, resumed: bool) -> None:
        if self.cfg.logging.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir),
                                        purge_step=self.global_step + 1 if resumed else None)
        if self.cfg.logging.use_wandb:
            import wandb
            session = uuid.uuid4().hex
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.wandb_run = wandb.init(
                project=self.cfg.logging.wandb_project, mode=self.cfg.logging.wandb_mode,
                id=session, name=f"{self.run_id}-{session[:8]}", group=self.run_id,
                dir=str(self.log_dir),
                config={**to_plain_config(self.cfg), "training_objective": training_objective_spec(self.cfg.loss.type),
                        "experiment_metadata": self.experiment_metadata},
            )
            self.wandb_run.define_metric("trainer/global_step")
            self.wandb_run.define_metric("*", step_metric="trainer/global_step")

    def close(self) -> None:
        """Attempt every logger cleanup and release only this trainer's lock/hook."""
        try:
            for logger, method in ((self.writer, "close"), (self.wandb_run, "finish")):
                if logger is not None:
                    try:
                        getattr(logger, method)()
                    except Exception as exc:
                        warnings.warn(f"Logger cleanup failed: {type(exc).__name__}", RuntimeWarning, stacklevel=2)
        finally:
            self.writer = self.wandb_run = None
            if self._lock_path is not None and self._lock_path.exists() and self._lock_path.stat().st_ino == self._lock_inode:
                self._lock_path.unlink()
            self._lock_path = None
            self._step_hook.remove()
            self._closed = True

    def train_one_epoch(self, loader: DataLoader, epoch: int, global_step: int) -> tuple[float, float, int]:
        self.model.train()
        loss_meter, mae_meter = AverageMeter("loss"), AverageMeter("mae")
        for images, ages, _, _ in tqdm(loader, desc=f"Epoch {epoch + 1} train", leave=False):
            images, ages = images.to(self.device, non_blocking=True), ages.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                loss, _, _ = self.loss_fn(logits, ages)
            if not torch.isfinite(loss).all():
                raise ValueError("Nonfinite training loss; previous checkpoints are preserved")
            self.scaler.scale(loss).backward()
            if self.cfg.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip,
                                         error_if_nonfinite=not self.use_amp)
            elif not self.use_amp and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                raise ValueError("Nonfinite training gradients")
            before = self.successful_updates
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.successful_updates > before and self.scheduler is not None:
                self.scheduler.step()
            pred = self.loss_fn.predict_age(logits)
            mae = compute_mae(pred, ages)
            loss_meter.update(loss.item(), n=images.size(0))
            mae_meter.update(mae, n=images.size(0))
            global_step += 1
            self.global_step = global_step
            if self.cfg.logging.log_every_n_steps and global_step % self.cfg.logging.log_every_n_steps == 0:
                self._log_scalar("train/step_loss", loss.item(), global_step)
                self._log_scalar("train/step_mae", mae, global_step)
        return loss_meter.avg, mae_meter.avg, global_step

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        loss_meter, mae_meter = AverageMeter("val_loss"), AverageMeter("val_mae")
        predictions, targets = [], []
        for images, ages, _, _ in tqdm(loader, desc="Validating", leave=False):
            images, ages = images.to(self.device, non_blocking=True), ages.to(self.device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                loss, _, _ = self.loss_fn(logits, ages)
            pred = self.loss_fn.predict_age(logits)
            if not torch.isfinite(loss).all() or not torch.isfinite(pred).all():
                raise ValueError("Nonfinite validation result; previous checkpoints are preserved")
            loss_meter.update(loss.item(), n=images.size(0))
            mae_meter.update(compute_mae(pred, ages), n=images.size(0))
            predictions.append(pred.cpu())
            targets.append(ages.cpu().view(-1))
        return loss_meter.avg, mae_meter.avg, compute_cumulative_score(torch.cat(predictions), torch.cat(targets), threshold=5)

    def _contract(self) -> dict:
        names = {id(p): name for name, p in self.model.named_parameters()}
        return {"config": training_config_signature(self.cfg), "data": self.experiment_metadata["data"], "device": str(self.device),
                "amp": self.use_amp, "randomness": self.experiment_metadata["randomness"],
                "loss_class": f"{type(self.loss_fn).__module__}.{type(self.loss_fn).__qualname__}",
                "parameters": [[(names[id(p)], list(p.shape)) for p in group["params"]] for group in self.optimizer.param_groups]}

    def _resume(self, path: Path) -> dict:
        artifact = load_training_checkpoint(path)
        state = artifact["training_state"]
        if state["contract"] != self._contract():
            raise ValueError("Resume contract changed: config/epochs, data, device, determinism or parameter groups differ")
        validate_loaders(self._loaders, state["loaders"])
        epoch, step, updates = state["next_epoch"], state["global_step"], state["successful_updates"]
        if (type(epoch) is not int or not 0 < epoch <= self.cfg.training.epochs or
                type(step) is not int or step != epoch * len(self._loaders["train"]) or
                type(updates) is not int or not 0 <= updates <= step or
                artifact["epoch"] != epoch - 1 or state["scheduler"].get("last_epoch") != updates):
            raise ValueError("Invalid resume epoch/update/scheduler counters")
        if not math.isfinite(state["best_mae"]) or not 0 <= state["best_epoch"] < epoch:
            raise ValueError("Invalid resume best metric/epoch")
        self.run_id = state["run_id"]
        if not isinstance(self.run_id, str) or Path(self.run_id).name != self.run_id or self.run_id in {".", ".."}:
            raise ValueError("Invalid resume run_id")
        self.save_dir = path.resolve().parent
        self._ensure_run()
        self._acquire_lock()
        best_path = self.save_dir / "best.pt"
        best = artifact if path.resolve() == best_path else load_training_checkpoint(best_path)
        best_state = best["training_state"]
        if (best_state["run_id"] != self.run_id or best_state["contract"] != state["contract"] or
                best["epoch"] != state["best_epoch"] or best["val_mae"] != state["best_mae"]):
            raise ValueError("Resume checkpoint and sibling best.pt disagree; use the matching best.pt or restore the matching pair")
        self.model.load_state_dict(artifact["model_state_dict"], strict=True)
        self.loss_fn.load_state_dict(state["loss"], strict=True)
        self.optimizer.load_state_dict(artifact["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.start_epoch, self.global_step, self.successful_updates = epoch, step, updates
        self.best_mae, self.best_epoch = state["best_mae"], state["best_epoch"]
        self.best_ckpt_path, self.last_ckpt_path = best_path, self.save_dir / "last.pt"
        self.history = state["history"]
        current = self.experiment_metadata
        self.experiment_metadata = copy.deepcopy(artifact["experiment_metadata"])
        self.experiment_metadata.setdefault("resumes", []).append({"from": str(path.resolve()), "environment": current["environment"],
                                                               "device": current["device"], "randomness": current["randomness"]})
        if (current["environment"]["packages"] != self.experiment_metadata["environment"]["packages"] or
                current["environment"]["python"] != self.experiment_metadata["environment"]["python"] or
                current["device"] != self.experiment_metadata["device"]):
            warnings.warn("Resume runtime/device changed; exact numerical equivalence is not guaranteed", RuntimeWarning)
        return state

    def run(self, train_loader: DataLoader, val_loader: DataLoader, *, resume: str | Path | None = None) -> None:
        if self._closed:
            raise ValueError("Trainer is closed; create a new Trainer to resume")
        try:
            if not len(train_loader) or not len(val_loader):
                raise ValueError("Training and validation loaders must contain batches")
            self._loaders = {"train": train_loader, "val": val_loader}
            capture_loaders(self._loaders)  # reject unsupported stateful/persistent loaders before creating output
            if self.experiment_metadata["data"] is None:
                self.experiment_metadata["data"] = getattr(train_loader, "data_provenance", None)
            self._build_scheduler(len(train_loader))
            state = self._resume(Path(resume)) if resume is not None else None
            self._ensure_run()
            if state is None:
                self._acquire_lock()
            self._open_loggers(resumed=state is not None)
            if state is not None:
                # Model/optimizer construction and logger setup may consume randomness.
                restore_loaders(self._loaders, state["loaders"])
                restore_rng(state["rng"])
            print(f"Run: {self.run_id} | checkpoints: {self.save_dir} | AMP: {self.use_amp}")
            for epoch in range(self.start_epoch, self.cfg.training.epochs):
                loss, mae, self.global_step = self.train_one_epoch(train_loader, epoch, self.global_step)
                val_loss, val_mae, cs5 = self.validate(val_loader)
                if not math.isfinite(val_mae) or val_mae < 0:
                    raise ValueError("Validation MAE must be finite and nonnegative")
                metrics = {"trainer/epoch": epoch, "train/epoch_loss": loss, "train/epoch_mae": mae,
                           "val/loss": val_loss, "val/mae": val_mae, "val/cs@5": cs5,
                           "lr/backbone": self.optimizer.param_groups[0]["lr"], "lr/head": self.optimizer.param_groups[1]["lr"],
                           "trainer/successful_updates": self.successful_updates,
                           "trainer/skipped_updates": self.global_step - self.successful_updates}
                for tag, value in metrics.items():
                    self._log_scalar(tag, value, self.global_step)
                self.history.append({**metrics, "global_step": self.global_step})
                if val_mae < self.best_mae:
                    self._save_checkpoint(epoch, val_mae)
                atomic_save(self._artifact(epoch, val_mae), self.save_dir / "last.pt")
                self.last_ckpt_path = self.save_dir / "last.pt"
                print(f"Epoch {epoch + 1}/{self.cfg.training.epochs}: val_mae={val_mae:.3f}; updates={self.successful_updates}/{self.global_step}")
            print(f"Training complete. Best val MAE: {self.best_mae:.3f}")
        finally:
            self.close()

    def _artifact(self, epoch: int, val_mae: float, *, best_mae=None, best_epoch=None) -> dict:
        state = None
        if self.scheduler is not None and self._loaders is not None:
            state = {"version": 1, "run_id": self.run_id, "next_epoch": epoch + 1,
                     "global_step": self.global_step, "successful_updates": self.successful_updates,
                     "best_mae": self.best_mae if best_mae is None else best_mae,
                     "best_epoch": self.best_epoch if best_epoch is None else best_epoch,
                     "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                     "rng": capture_rng(cuda=self.device.type == "cuda"), "loaders": capture_loaders(self._loaders),
                     "contract": self._contract(), "loss": self.loss_fn.state_dict(), "history": self.history}
        return create_checkpoint(self.model, self.cfg, epoch=epoch, val_mae=val_mae,
                                 optimizer_state_dict=self.optimizer.state_dict(),
                                 training_objective=training_objective_spec(self.cfg.loss.type),
                                 experiment_metadata=self.experiment_metadata, training_state=state)

    def _save_checkpoint(self, epoch: int, val_mae: float) -> None:
        """Replace one best atomically; update in-memory best only after success."""
        self._ensure_run()
        path = self.save_dir / "best.pt"
        atomic_save(self._artifact(epoch, val_mae, best_mae=val_mae, best_epoch=epoch), path)
        self.best_mae, self.best_epoch, self.best_ckpt_path = val_mae, epoch, path

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        if self.wandb_run:
            self.wandb_run.log({tag: value, "trainer/global_step": step})
