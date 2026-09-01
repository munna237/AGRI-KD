"""Reproducibility helpers.

Seeding fixes initialisation and shuffling, but does not make results identical
across machines: cuDNN kernel selection, GPU architecture and library versions
all introduce small differences. See ``docs/REPRODUCIBILITY.md`` for the
variation we observe between independent runs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
