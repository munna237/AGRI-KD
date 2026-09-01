"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def normalise(x: np.ndarray) -> np.ndarray:
    """Min-max scale an array to [0, 1]."""
    x = x - x.min()
    return x / (x.max() + 1e-8)


def localization_metrics(attention: np.ndarray, ground_truth: np.ndarray,
                         threshold: float = 0.3) -> dict:
    """Localisation quality at the native attention resolution.

    Evaluating on the 7x7 grid rather than an upsampled map avoids crediting
    interpolation for spatial precision the model does not have.

    Args:
        attention: predicted map, any scale.
        ground_truth: binary target map of the same shape.
        threshold: binarisation threshold applied after min-max scaling.

    Returns:
        Dict with ``iou``, ``dice``, ``pointing`` and ``center_distance``.
    """
    pred = normalise(attention)
    pred_binary = (pred > threshold).astype(float)
    gt_binary = (ground_truth > 0.5).astype(float)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    peak = np.unravel_index(pred.argmax(), pred.shape)
    if gt_binary.sum() > 0:
        centre = np.argwhere(gt_binary > 0.5).mean(axis=0)
    else:
        centre = np.array(pred.shape) / 2

    return {
        "iou": intersection / (union + 1e-8),
        "dice": 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-8),
        "pointing": float(gt_binary[peak] > 0.5),
        "center_distance": float(
            np.sqrt(((np.array(peak) - centre) ** 2).sum()) / pred.shape[0]
        ),
    }
