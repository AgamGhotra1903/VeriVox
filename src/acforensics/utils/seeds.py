"""Reproducibility and deterministic random seed utility.

Author: Agam Ghotra (IIT2024226)
"""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Sets random seeds across Python, NumPy, PyTorch (CPU, CUDA, and Apple MPS).

    Args:
        seed: Integer seed value.
        deterministic: If True, sets torch.backends.cudnn.deterministic=True and
            torch.backends.cudnn.benchmark=False to enforce bitwise reproducibility.
            Note: Deterministic algorithms can introduce a small performance penalty
            (typically 5-15% slower convolution kernels on GPU).

    Returns:
        The integer seed applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    # Support Apple Silicon MPS reproducibility
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS uses torch.manual_seed
        pass

    return seed


def get_worker_init_fn(base_seed: int = 42):
    """Returns a worker initialization function for PyTorch DataLoader workers.

    Ensures that each DataLoader subprocess gets a unique, deterministic seed.
    """
    def _worker_init_fn(worker_id: int):
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _worker_init_fn
