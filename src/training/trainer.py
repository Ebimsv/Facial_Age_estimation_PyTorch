"""Trainer class — encapsulates the full training and validation loop.

Features
--------
- Mixed-precision training via ``torch.amp`` (float16 forward, float32 grads).
- AdamW with differential learning rates (lower LR for backbone, higher for head).
- OneCycleLR scheduler with linear warmup.
- Gradient clipping (max-norm).
- Optional Weights & Biases logging.
- Optional TensorBoard logging.
- Saves the best checkpoint by validation MAE.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import to_plain_config
from src.utils.metrics import AverageMeter, compute_cumulative_score, compute_mae

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Trainer:
    """Manages one complete training experiment.

    Args:
        model:    Instantiated ``AgeEstimationModel`` (already on ``device``).
        loss_fn:  ``MeanVarianceLoss`` or ``L1AgeLoss`` instance.
        cfg:      OmegaConf config (loaded from ``configs/default.yaml``).
        device:   torch.device to use.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device

        # Move loss buffers (e.g., MeanVarianceLoss.ages) to the same device
        self.loss_fn = self.loss_fn.to(device)

        # ── Optimizer with differential LRs ──────────────────────────────
        # Head parameters get a higher LR; backbone gets a lower LR.
        head_params, backbone_params = self._split_param_groups()
        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.optimizer.backbone_lr},
                {"params": head_params, "lr": cfg.optimizer.head_lr},
            ],
            weight_decay=cfg.optimizer.weight_decay,
        )

        # ── Scheduler (set up after we know steps_per_epoch) ─────────────
        self.scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None

        # ── Mixed precision ───────────────────────────────────────────────
        self.use_amp = cfg.training.use_amp and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # ── Logging ───────────────────────────────────────────────────────
        self.writer = None
        if cfg.logging.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=cfg.logging.log_dir)

        self.wandb_run = None
        if cfg.logging.use_wandb:
            import wandb
            self.wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                config=to_plain_config(cfg),
                name=cfg.model.name,
            )

        # ── Checkpoint dir ────────────────────────────────────────────────
        self.save_dir = Path(cfg.logging.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_mae: float = float("inf")
        self.best_ckpt_path: Path | None = None

    # ── Parameter split ──────────────────────────────────────────────────────

    def _split_param_groups(self):
        """Return (head_params, backbone_params) based on module name heuristics."""
        head_names = {"head", "classifier", "fc"}
        head_params, backbone_params = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(h in name for h in head_names):
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        return head_params, backbone_params

    # ── Scheduler init (called once train_loader is known) ───────────────────

    def _build_scheduler(self, steps_per_epoch: int) -> None:
        total_steps = self.cfg.training.epochs * steps_per_epoch
        max_lrs = [self.cfg.optimizer.backbone_lr, self.cfg.optimizer.head_lr]
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=self.cfg.scheduler.pct_start,
            anneal_strategy="cos",
        )

    # ── Single epoch ─────────────────────────────────────────────────────────

    def train_one_epoch(
        self, loader: DataLoader, epoch: int, global_step: int
    ) -> tuple[float, float, int]:
        """Run one training epoch.

        Returns:
            (avg_loss, avg_mae, global_step)
        """
        self.model.train()
        loss_meter = AverageMeter("loss")
        mae_meter = AverageMeter("mae")

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1:>3}/{self.cfg.training.epochs}  [train]",
            leave=False,
        )

        for images, ages, _, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            ages = ages.to(self.device, non_blocking=True)  # [B, 1]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                loss, mean_loss, var_loss = self.loss_fn(logits, ages)

            self.scaler.scale(loss).backward()

            if self.cfg.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            # Age prediction for MAE tracking
            with torch.no_grad():
                pred_ages = self.loss_fn.predict_age(logits)
            mae = compute_mae(pred_ages, ages)

            n = images.size(0)
            loss_meter.update(loss.item(), n=n)
            mae_meter.update(mae, n=n)
            global_step += 1

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", mae=f"{mae_meter.avg:.2f}y")

            # Step-level logging
            if (
                self.cfg.logging.log_every_n_steps > 0
                and global_step % self.cfg.logging.log_every_n_steps == 0
            ):
                self._log_scalar("train/step_loss", loss.item(), global_step)
                self._log_scalar("train/step_mae", mae, global_step)

        return loss_meter.avg, mae_meter.avg, global_step

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> tuple[float, float, float]:
        """Run one validation pass.

        Returns:
            (avg_loss, avg_mae, cs_at_5)
        """
        self.model.eval()
        loss_meter = AverageMeter("val_loss")
        mae_meter = AverageMeter("val_mae")

        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for images, ages, _, _ in tqdm(loader, desc="  Validating          ", leave=False):
            images = images.to(self.device, non_blocking=True)
            ages = ages.to(self.device, non_blocking=True)

            with autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                loss, _, _ = self.loss_fn(logits, ages)

            pred_ages = self.loss_fn.predict_age(logits)
            mae = compute_mae(pred_ages, ages)

            n = images.size(0)
            loss_meter.update(loss.item(), n=n)
            mae_meter.update(mae, n=n)

            all_preds.append(pred_ages.cpu())
            all_targets.append(ages.cpu().view(-1))

        cs5 = compute_cumulative_score(
            torch.cat(all_preds), torch.cat(all_targets), threshold=5
        )
        return loss_meter.avg, mae_meter.avg, cs5

    # ── Main training loop ────────────────────────────────────────────────────

    def run(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Full training loop with early-stopping and checkpoint saving.

        Args:
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
        """
        self._build_scheduler(steps_per_epoch=len(train_loader))
        global_step = 0

        print(f"\n{'='*60}")
        print(f"  Model  : {self.cfg.model.name}  ({self.model.num_parameters}M params)")
        print(f"  Loss   : {self.cfg.loss.type}")
        print(f"  Device : {self.device}  |  AMP: {self.use_amp}")
        print(f"  Epochs : {self.cfg.training.epochs}  |  Batch: {self.cfg.training.batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(self.cfg.training.epochs):
            train_loss, train_mae, global_step = self.train_one_epoch(
                train_loader, epoch, global_step
            )
            val_loss, val_mae, val_cs5 = self.validate(val_loader)

            # ── Epoch-level logging ───────────────────────────────────────
            self._log_scalar("train/epoch_loss", train_loss, epoch)
            self._log_scalar("train/epoch_mae", train_mae, epoch)
            self._log_scalar("val/loss", val_loss, epoch)
            self._log_scalar("val/mae", val_mae, epoch)
            self._log_scalar("val/cs@5", val_cs5, epoch)
            self._log_scalar(
                "lr/backbone",
                self.optimizer.param_groups[0]["lr"],
                epoch,
            )
            self._log_scalar(
                "lr/head",
                self.optimizer.param_groups[1]["lr"],
                epoch,
            )

            print(
                f"Epoch {epoch + 1:>3}/{self.cfg.training.epochs}"
                f"  train_loss={train_loss:.4f}  train_mae={train_mae:.2f}y"
                f"  val_loss={val_loss:.4f}  val_mae={val_mae:.2f}y"
                f"  CS@5={val_cs5:.3f}"
            )

            # ── Save best checkpoint ──────────────────────────────────────
            if val_mae < self.best_mae:
                self.best_mae = val_mae
                self._save_checkpoint(epoch, val_mae)

        print(f"\n✓ Training complete.  Best val MAE: {self.best_mae:.2f}y")
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()

    # ── Checkpoint I/O ───────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_mae: float) -> None:
        """Save model weights; remove the previous best checkpoint."""
        if self.best_ckpt_path and self.best_ckpt_path.exists():
            os.remove(self.best_ckpt_path)

        fname = self.save_dir / f"best_{self.cfg.model.name}_epoch{epoch:03d}_mae{val_mae:.3f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_name": self.cfg.model.name,
                "output_dim": self.cfg.model.output_dim,
                "val_mae": val_mae,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "cfg": to_plain_config(self.cfg),
            },
            fname,
        )
        self.best_ckpt_path = fname
        print(f"  ✓ Checkpoint saved → {fname.name}  (val_mae={val_mae:.3f}y)")

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        if self.wandb_run:
            self.wandb_run.log({tag: value}, step=step)
