"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Seed process RNGs; optionally request deterministic PyTorch operations.

    Repeatability is limited to a fixed software/hardware/worker configuration.
    PYTHONHASHSEED must be set before Python starts; changing it here has no effect.

    Args:
        seed: Integer seed value. Default is 42.
    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.use_deterministic_algorithms(deterministic)

    # Preserve the previous cuDNN policy even when strict algorithms are disabled.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed global Python/NumPy RNGs from PyTorch's per-worker seed (spawn compatible)."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
