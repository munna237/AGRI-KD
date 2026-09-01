"""Leaf segmentation model used to derive the IAP denominator.

Infected area percentage is normalised by the area of the assessed leaf rather
than the whole image, so a leaf mask is required for every image. This module
wraps a U-Net that is trained on a small set of hand-annotated leaf polygons and
then applied to the remaining images.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2


class LeafSegmenter:
    """U-Net leaf segmenter with letterbox-aware inference.

    The network is trained at a fixed square resolution with aspect-preserving
    padding; :meth:`predict` reverses that padding so the returned mask aligns
    with the original image.

    Args:
        encoder: ``segmentation_models_pytorch`` encoder name.
        img_size: square resolution used for training and inference.
        device: torch device.
    """

    def __init__(self, encoder: str = "resnet34", img_size: int = 512, device=None):
        self.img_size = img_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = smp.Unet(
            encoder, encoder_weights="imagenet", in_channels=3, classes=1
        ).to(self.device)
        self.transform = A.Compose([
            A.LongestMaxSize(img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(), ToTensorV2(),
        ])

    def load(self, weights_path: str) -> "LeafSegmenter":
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return a binary leaf mask at the original image resolution.

        Args:
            image_bgr: image in OpenCV BGR order.
            threshold: probability above which a pixel is leaf.

        Returns:
            ``uint8`` array of shape (H, W) with values in {0, 1}.
        """
        self.model.eval()
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        batch = self.transform(image=rgb)["image"].unsqueeze(0).to(self.device)

        logits = self.model(batch)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask = (prob > threshold).astype(np.uint8)

        # undo LongestMaxSize + PadIfNeeded
        scale = self.img_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        y0, x0 = (self.img_size - nh) // 2, (self.img_size - nw) // 2
        cropped = mask[y0:y0 + nh, x0:x0 + nw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    """Mean Dice over a batch, computed on thresholded predictions."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean().item()
