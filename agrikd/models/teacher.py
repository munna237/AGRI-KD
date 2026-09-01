"""Teacher models and the frozen ensemble used for distillation."""

from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

_SPEC = {
    "resnet50": 2048,
    "densenet121": 1024,
    "efficientnet_b0": 1280,
}


class TeacherModel(nn.Module):
    """Single teacher with a classification head and a spatial attention head.

    Intermediate activations are captured through forward hooks so that
    feature-based distillation methods can access them without modifying the
    backbone.

    Args:
        backbone_name: ``resnet50``, ``densenet121`` or ``efficientnet_b0``.
        num_classes: number of disease classes.
        pretrained: initialise from ImageNet weights.
    """

    def __init__(self, backbone_name="resnet50", num_classes=6, pretrained=True):
        super().__init__()
        if backbone_name not in _SPEC:
            raise ValueError(f"unsupported backbone: {backbone_name}")
        self.backbone_name = backbone_name
        self.spatial_channels = _SPEC[backbone_name]

        if backbone_name == "resnet50":
            weights = tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = tv.resnet50(weights=weights)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            hook_layer = self.backbone.layer4
            taps = {"layer1": self.backbone.layer1,
                    "layer2": self.backbone.layer2,
                    "layer4": self.backbone.layer4}
        elif backbone_name == "densenet121":
            weights = tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = tv.densenet121(weights=weights)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            hook_layer = self.backbone.features
            taps = {"features": self.backbone.features}
        else:
            self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            hook_layer = self.backbone.conv_head
            taps = {"conv_head": self.backbone.conv_head}

        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(self.feature_dim, num_classes)
        )

        self.intermediate = {}
        for name, layer in taps.items():
            layer.register_forward_hook(
                lambda m, i, o, n=name: self.intermediate.__setitem__(n, o)
            )

        self.spatial_features = None
        hook_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "spatial_features", o)
        )

        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.spatial_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid(),
        )

    def get_attention(self):
        if self.spatial_features is None:
            return None
        return self.attention_conv(self.spatial_features).squeeze(1)

    def forward(self, x, return_features: bool = False):
        features = self.backbone(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features, self.intermediate
        return {"cls": logits, "attention": self.get_attention()}


class TeacherEnsemble(nn.Module):
    """Frozen ensemble aggregating several teachers.

    Class predictions are averaged in probability space and returned as log
    probabilities. Attention maps are averaged after resizing to a common 7x7
    grid.

    Args:
        backbone_names: teacher backbones to instantiate.
        num_classes: number of disease classes.
    """

    ATTENTION_SIZE = (7, 7)

    def __init__(self, backbone_names, num_classes=6):
        super().__init__()
        self.backbone_names = list(backbone_names)
        self.teachers = nn.ModuleList(
            [TeacherModel(name, num_classes) for name in backbone_names]
        )

    def load_weights(self, checkpoint_dir, pattern="teacher_{}_best.pth"):
        """Load per-teacher weights, then freeze and switch to eval mode."""
        for i, name in enumerate(self.backbone_names):
            path = os.path.join(checkpoint_dir, pattern.format(name))
            self.teachers[i].load_state_dict(torch.load(path, map_location="cpu"))
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self

    def forward(self, x):
        probs, attentions, features = [], [], []
        for teacher in self.teachers:
            logits, feat, _ = teacher(x, return_features=True)
            probs.append(F.softmax(logits, dim=1))
            att = teacher.get_attention()
            if att is not None:
                attentions.append(att)
            features.append(feat)

        ensemble_logits = torch.log(torch.stack(probs, dim=0).mean(dim=0) + 1e-8)

        ensemble_attention = None
        if attentions:
            resized = [
                a if a.shape[-2:] == self.ATTENTION_SIZE else
                F.interpolate(a.unsqueeze(1), self.ATTENTION_SIZE,
                              mode="bilinear", align_corners=False).squeeze(1)
                for a in attentions
            ]
            ensemble_attention = torch.stack(resized, dim=0).mean(dim=0)

        return {"cls": ensemble_logits,
                "attention": ensemble_attention,
                "features": features}
