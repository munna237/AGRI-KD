"""Checkpointing with resume support.

Every epoch writes model, optimiser, scheduler and RNG state, so an interrupted
run resumes from its last completed epoch rather than restarting. A manifest
records which stages have finished.
"""

from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch


def save_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    epoch: int = 0, best: float = 0.0, **extra) -> None:
    """Write a resumable checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best": best,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        **extra,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    restore_rng: bool = True):
    """Restore a checkpoint if present.

    Returns:
        ``(start_epoch, best_metric)``; ``(0, 0.0)`` when no checkpoint exists.
    """
    if not os.path.exists(path):
        return 0, 0.0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng:
        try:
            torch.set_rng_state(ckpt["rng_torch"])
            np.random.set_state(ckpt["rng_numpy"])
            random.setstate(ckpt["rng_python"])
        except Exception:
            pass
    return ckpt.get("epoch", 0), ckpt.get("best", 0.0)


class Manifest:
    """Records which training stages have completed."""

    def __init__(self, path: str):
        self.path = path

    def _read(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path) as fh:
                return json.load(fh)
        return {}

    def is_done(self, key: str) -> bool:
        return self._read().get(key, {}).get("done", False)

    def mark_done(self, key: str, **info) -> None:
        data = self._read()
        data[key] = {"done": True,
                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(data, fh, indent=2)

    def summary(self) -> dict:
        return self._read()
