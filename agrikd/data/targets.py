"""Ground-truth target construction.

Infected area percentage is normalised by the area of the assessed leaf rather
than by the whole image, so that the measure does not vary with shooting
distance. Severity grades follow from IAP through fixed thresholds.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]  # cx, cy, w, h, normalised


def parse_yolo(label_path: str) -> List[Box]:
    """Read a YOLO label file and return normalised ``(cx, cy, w, h)`` boxes."""
    boxes: List[Box] = []
    if not label_path or not os.path.exists(str(label_path)):
        return boxes
    with open(label_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5:
                boxes.append(tuple(float(v) for v in parts[1:5]))  # type: ignore
    return boxes


def box_union_mask(boxes: Sequence[Box], height: int, width: int) -> np.ndarray:
    """Rasterise boxes into a binary union mask.

    The union is used rather than the sum of box areas so that overlapping
    annotations are not counted twice.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for cx, cy, w, h in boxes:
        x1 = int(max(0, (cx - w / 2) * width))
        y1 = int(max(0, (cy - h / 2) * height))
        x2 = int(min(width, np.ceil((cx + w / 2) * width)))
        y2 = int(min(height, np.ceil((cy + h / 2) * height)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def attention_mask(boxes: Sequence[Box], size: int = 7) -> np.ndarray:
    """Project boxes onto the coarse grid used for attention supervision."""
    mask = np.zeros((size, size), dtype=np.float32)
    for cx, cy, w, h in boxes:
        x1 = int(max(0, (cx - w / 2) * size))
        y1 = int(max(0, (cy - h / 2) * size))
        x2 = int(min(size, np.ceil((cx + w / 2) * size)))
        y2 = int(min(size, np.ceil((cy + h / 2) * size)))
        mask[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)] = 1.0
    return mask


def leaf_area_iap(boxes: Sequence[Box], leaf_mask: np.ndarray) -> float:
    """Infected area percentage relative to the assessed leaf.

    Args:
        boxes: normalised lesion boxes.
        leaf_mask: binary leaf mask, any resolution.

    Returns:
        Percentage in [0, 100], or NaN when the leaf mask is empty.
    """
    leaf_area = float(leaf_mask.sum())
    if leaf_area <= 0:
        return float("nan")
    h, w = leaf_mask.shape
    lesions = box_union_mask(boxes, h, w)
    return 100.0 * float((lesions & (leaf_mask > 0)).sum()) / leaf_area


def image_area_iap(boxes: Sequence[Box], resolution: int = 1024) -> float:
    """Infected area percentage relative to the whole image.

    Retained for comparison only; leaf-area normalisation should be preferred,
    since image-area IAP varies with shooting distance.
    """
    if not boxes:
        return 0.0
    mask = box_union_mask(boxes, resolution, resolution)
    return 100.0 * float(mask.sum()) / (resolution * resolution)


def iap_to_grade(iap: float, thresholds: Sequence[float] = (5, 15, 30, 50)) -> int:
    """Map an IAP percentage to an ordinal severity grade."""
    if not np.isfinite(iap):
        return -1
    for i, t in enumerate(thresholds):
        if iap < t:
            return i
    return len(thresholds)
