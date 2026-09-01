"""MobileNetV3-Small student with five simultaneous output heads."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class StudentModel(nn.Module):
    """Lightweight multi-task student.

    Produces five outputs from a single forward pass:

    ==========  ===========================  ==========================================
    key         shape                        meaning
    ==========  ===========================  ==========================================
    cls         (B, num_classes)             disease classification logits
    attention   (B, 7, 7)                    spatial lesion attention in [0, 1]
    iap         (B,)                         infected area percentage in [0, 100]
    severity    (B, num_severity_grades)     severity grade logits
    count       (B,)                         lesion count, non-negative
    ==========  ===========================  ==========================================

    The attention branch reads the backbone's final inverted-residual block
    (576 channels at 7x7) through a forward hook, so the spatial map is obtained
    without altering the backbone's own forward path.

    Args:
        num_classes: number of disease classes, including healthy.
        num_severity_grades: number of severity levels.
        pretrained: initialise the backbone from ImageNet weights.
    """

    SPATIAL_CHANNELS = 576

    def __init__(
        self,
        num_classes: int = 6,
        num_severity_grades: int = 5,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_small_100", pretrained=pretrained)
        self.feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes),
        )
        self.iap_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.feature_dim, 64),
            nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.count_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.feature_dim, 32),
            nn.ReLU(), nn.Linear(32, 1), nn.ReLU(),
        )
        self.severity_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.feature_dim, 32),
            nn.ReLU(), nn.Linear(32, num_severity_grades),
        )

        self.spatial_features = None
        self.backbone.blocks.register_forward_hook(
            lambda module, inp, out: setattr(self, "spatial_features", out)
        )

        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.SPATIAL_CHANNELS, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid(),
        )

    def get_attention(self):
        """Attention map for the most recent forward pass, or None before one."""
        if self.spatial_features is None:
            return None
        return self.attention_conv(self.spatial_features).squeeze(1)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        features = self.backbone(x)
        out = {
            "cls": self.classifier(features),
            "attention": self.get_attention(),
            "iap": self.iap_head(features).squeeze(-1) * 100.0,
            "severity": self.severity_head(features),
            "count": self.count_head(features).squeeze(-1),
        }
        if return_features:
            out["features"] = features
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """Inference wrapper returning interpretable values rather than logits."""
        self.eval()
        o = self.forward(x)
        return {
            "class_id": o["cls"].argmax(dim=1),
            "class_probs": o["cls"].softmax(dim=1),
            "attention": o["attention"],
            "iap_percent": o["iap"],
            "severity_grade": o["severity"].argmax(dim=1),
            "lesion_count": o["count"],
        }
