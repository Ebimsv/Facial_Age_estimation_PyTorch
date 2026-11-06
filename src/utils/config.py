"""Typed experiment configuration and fail-fast model/data contracts."""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

SUPPORTED_BACKBONES = (
    "mobilenet_v3", "efficientnet_b0", "efficientnet_b3", "resnet50", "vit_small",
)
DISTRIBUTION_LOSSES = {"l1_mean_variance", "mean_variance"}


@dataclass
class ModelConfig:
    name: str = MISSING
    pretrained: bool = MISSING
    output_dim: int = MISSING
    dropout: float = MISSING


@dataclass
class DataConfig:
    root_dir: str = MISSING
    csv_dir: str = MISSING
    img_size: int = MISSING
    num_workers: int = MISSING
    mean: list[float] = MISSING
    std: list[float] = MISSING
    max_age: int = MISSING
    train_ratio: float = MISSING
    val_ratio: float = MISSING


@dataclass
class TrainingConfig:
    epochs: int = MISSING
    batch_size: int = MISSING
    eval_batch_size: int = MISSING
    seed: int = MISSING
    use_amp: bool = MISSING
    grad_clip: float = MISSING
    deterministic: bool = False


@dataclass
class OptimizerConfig:
    backbone_lr: float = MISSING
    head_lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class SchedulerConfig:
    pct_start: float = MISSING


@dataclass
class LossConfig:
    type: str = MISSING
    lambda_mean: float = MISSING
    lambda_variance: float = MISSING
    start_age: int = MISSING
    end_age: int = MISSING


@dataclass
class LoggingConfig:
    use_wandb: bool = MISSING
    wandb_project: str = MISSING
    use_tensorboard: bool = MISSING
    log_dir: str = MISSING
    save_dir: str = MISSING
    log_every_n_steps: int = MISSING


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: str | None = None


def _integer(value: int, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")


def _finite(value: float, name: str, minimum: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if value < minimum or (positive and value == minimum):
        operator = ">" if positive else ">="
        raise ValueError(f"{name} must be {operator} {minimum}; got {value!r}")


def validate_model_options(
    name: str, pretrained: bool, output_dim: int, dropout: float, img_size: int,
) -> None:
    """Reject invalid model options before constructing a pretrained backbone."""
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {SUPPORTED_BACKBONES}")
    if type(pretrained) is not bool:
        raise ValueError("model.pretrained must be a boolean")
    _integer(output_dim, "model.output_dim", 1)
    _finite(dropout, "model.dropout", 0)
    if dropout >= 1:
        raise ValueError("model.dropout must be < 1")
    _integer(img_size, "data.img_size", 32)
    if name == "vit_small" and img_size % 14:
        raise ValueError("data.img_size must be a multiple of 14 for vit_small (e.g. 224 or 280)")


def validate_age_support(start_age: int, end_age: int) -> None:
    """Validate the inclusive integer support of the age distribution."""
    _integer(start_age, "loss.start_age", 0)
    _integer(end_age, "loss.end_age", 0)
    if start_age > end_age:
        raise ValueError("loss.start_age must be <= loss.end_age")


def validate_loss_weights(lambda_mean: float, lambda_variance: float) -> None:
    """Require finite nonnegative loss weights with at least one active term."""
    _finite(lambda_mean, "loss.lambda_mean", 0)
    _finite(lambda_variance, "loss.lambda_variance", 0)
    if lambda_mean + lambda_variance == 0:
        raise ValueError("loss.lambda_mean and loss.lambda_variance cannot both be zero")


def validate_mode(output_dim: int, loss_type: str, start_age: int, end_age: int) -> None:
    """Check that model output and age decoder describe the same mode."""
    validate_age_support(start_age, end_age)
    if loss_type not in DISTRIBUTION_LOSSES | {"l1"}:
        raise ValueError(f"Unknown loss type '{loss_type}'. Choose from: l1_mean_variance | mean_variance (alias) | l1")
    if loss_type in DISTRIBUTION_LOSSES and start_age == end_age:
        raise ValueError("l1_mean_variance requires at least two age classes")
    expected = end_age - start_age + 1 if loss_type in DISTRIBUTION_LOSSES else 1
    if output_dim != expected:
        raise ValueError(
            f"model.output_dim must be {expected} for loss.type={loss_type!r}; got {output_dim}"
        )
    if loss_type == "l1" and (start_age, end_age) != (0, 100):
        raise ValueError("l1 currently uses fixed age bounds [0, 100]; use loss.start_age=0/end_age=100")


def validate_split_options(train_ratio: float, val_ratio: float, seed: int) -> None:
    """Require three nonempty split fractions and a NumPy-compatible seed."""
    _finite(train_ratio, "data.train_ratio", 0, positive=True)
    _finite(val_ratio, "data.val_ratio", 0, positive=True)
    if train_ratio + val_ratio >= 1:
        raise ValueError("data.train_ratio + data.val_ratio must be < 1 (leave a nonempty test split)")
    _integer(seed, "training.seed", 0)
    if seed >= 2**32:
        raise ValueError("training.seed must be < 2**32")


def _migrate_legacy(raw: DictConfig) -> None:
    if "logging" in raw and isinstance(raw.logging, DictConfig) and "save_best_only" in raw.logging:
        if raw.logging.save_best_only is not True:
            raise ValueError("logging.save_best_only=false is unsupported; only the best model is saved")
        del raw.logging.save_best_only
        warnings.warn("logging.save_best_only is obsolete; best-validation saving is always enabled",
                      FutureWarning, stacklevel=3)
    if "inference" in raw:
        legacy = raw.inference
        allowed = {"checkpoint_path", "image_path", "output_path"}
        if not isinstance(legacy, DictConfig) or set(legacy) - allowed or any(
            value is not None for value in legacy.values()
        ):
            raise ValueError("inference.* YAML options are unsupported; use --checkpoint, --image, --output")
        del raw.inference
        warnings.warn("The empty inference YAML block is obsolete; use the inference CLI flags",
                      FutureWarning, stacklevel=3)


def to_plain_config(cfg: DictConfig) -> dict[str, Any]:
    """Return a recursively resolved, JSON-serializable configuration dictionary."""
    result = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a mapping")
    return result


def _reject_explicit_missing(value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_explicit_missing(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_explicit_missing(item, f"{path}[{index}]")
    elif isinstance(value, str) and value == MISSING:
        raise ValueError(f"Missing mandatory value: {path}")


def merge_config(base: DictConfig, *updates, overrides: list[str] | None = None) -> DictConfig:
    """Merge defaults, YAML/metadata updates and CLI overrides, then validate once.

    Unknown keys and invalid interpolations/types become actionable ValueErrors.
    The returned configuration is resolved and read-only to preserve validation.
    """
    try:
        sources = [OmegaConf.create(OmegaConf.to_container(base, resolve=False))]
        for update in updates:
            sources.append(OmegaConf.create(OmegaConf.to_container(update, resolve=False))
                           if OmegaConf.is_config(update) else OmegaConf.create(update))
        if overrides:
            if any("=" not in item or not item.split("=", 1)[0] for item in overrides):
                raise ValueError("CLI overrides must use key=value syntax")
            sources.append(OmegaConf.from_dotlist(overrides))
        # OmegaConf.merge otherwise inherits the base value when an update is ???.
        for source in sources:
            _reject_explicit_missing(OmegaConf.to_container(source, resolve=False))
        raw = OmegaConf.merge(*sources)
        if not isinstance(raw, DictConfig):
            raise ValueError("Configuration must be a mapping")
        _migrate_legacy(raw)
        # Some OmegaConf versions reject interpolated container types during the
        # typed merge, before _validate_values can attach the normalization key.
        # Check shape on the untyped merged mapping; leave scalar coercion to the schema.
        for name in ("mean", "std"):
            value = OmegaConf.select(raw, f"data.{name}")
            if not (OmegaConf.is_list(value) or isinstance(value, (list, tuple))) or len(value) != 3:
                raise ValueError(f"data.{name} must contain three RGB values")
        cfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), raw)
        _validate_values(cfg)  # access interpolations through their named fields for useful errors
        OmegaConf.resolve(cfg)
        to_plain_config(cfg)  # also detects mandatory missing fields
        OmegaConf.set_readonly(cfg, True)
        return cfg
    except OmegaConfBaseException as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc


def _validate_values(cfg: DictConfig) -> None:
    validate_model_options(cfg.model.name, cfg.model.pretrained, cfg.model.output_dim,
                           cfg.model.dropout, cfg.data.img_size)
    validate_mode(cfg.model.output_dim, cfg.loss.type, cfg.loss.start_age, cfg.loss.end_age)
    if cfg.loss.type in DISTRIBUTION_LOSSES:
        validate_loss_weights(cfg.loss.lambda_mean, cfg.loss.lambda_variance)
    else:
        _finite(cfg.loss.lambda_mean, "loss.lambda_mean", 0)
        _finite(cfg.loss.lambda_variance, "loss.lambda_variance", 0)
    validate_split_options(cfg.data.train_ratio, cfg.data.val_ratio, cfg.training.seed)
    _integer(cfg.data.max_age, "data.max_age", 1)
    if not cfg.loss.start_age < cfg.data.max_age <= cfg.loss.end_age + 1:
        raise ValueError("data.max_age (exclusive) must be > loss.start_age and <= loss.end_age + 1")
    _integer(cfg.data.num_workers, "data.num_workers", 0)
    for name in ["epochs", "batch_size", "eval_batch_size"]:
        _integer(cfg.training[name], f"training.{name}", 1)
    _finite(cfg.training.grad_clip, "training.grad_clip", 0)
    for name in ["backbone_lr", "head_lr"]:
        _finite(cfg.optimizer[name], f"optimizer.{name}", 0, positive=True)
    _finite(cfg.optimizer.weight_decay, "optimizer.weight_decay", 0)
    _finite(cfg.scheduler.pct_start, "scheduler.pct_start", 0, positive=True)
    if cfg.scheduler.pct_start >= 1:
        raise ValueError("scheduler.pct_start must be < 1")
    _integer(cfg.logging.log_every_n_steps, "logging.log_every_n_steps", 0)
    for name in ["mean", "std"]:
        values = cfg.data[name]
        if not isinstance(values, (list, ListConfig)) or len(values) != 3:
            raise ValueError(f"data.{name} must contain three RGB values")
        for value in values:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or (name == "std" and value <= 0):
                raise ValueError(f"data.{name} must contain finite values; std must be > 0")
    for name, value in {"data.root_dir": cfg.data.root_dir, "data.csv_dir": cfg.data.csv_dir,
                        "logging.log_dir": cfg.logging.log_dir, "logging.save_dir": cfg.logging.save_dir,
                        "logging.wandb_project": cfg.logging.wandb_project}.items():
        if not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if cfg.device is not None and not re.fullmatch(r"(?:cpu|cuda)(?::[0-9]+)?", cfg.device):
        raise ValueError("device must be null (auto), cpu, or cuda[:index]")


def validate_training_steps(cfg: DictConfig, steps_per_epoch: int) -> None:
    """Reject unusable loaders and OneCycleLR's zero-length warmup boundary."""
    _integer(steps_per_epoch, "training batches per epoch", 1)
    if math.isclose(cfg.training.epochs * steps_per_epoch * cfg.scheduler.pct_start, 1):
        raise ValueError("OneCycleLR warmup would have zero length; change epochs or scheduler.pct_start")
