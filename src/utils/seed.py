"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic cuDNN — slightly slower but fully reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
