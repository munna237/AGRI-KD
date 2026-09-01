"""Configuration objects for AGRI-KD.

Defaults reproduce the settings reported in the paper. Override them by
constructing the dataclass directly or by loading a YAML file with
:func:`load_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class DataConfig:
    """Dataset layout and target construction."""

    root: str = "data/RiceLeafDiseaseBD"
    original_dir: str = "Original images"
    annotated_dir: str = "Annotated images ( visual with labels)"
    disease_classes: List[str] = field(default_factory=lambda: [
        "Blast", "Brown spot", "Leaf smut", "Rice Tungro", "Sheath blight"
    ])
    healthy_class: str = "Healthy"
    img_size: int = 224
    attention_size: int = 7
    #: Severity thresholds on leaf-area IAP, following Bock et al.
    severity_thresholds: List[float] = field(default_factory=lambda: [5, 15, 30, 50])
    test_size: float = 0.15
    val_size: float = 0.176
    seed: int = 42

    @property
    def all_classes(self) -> List[str]:
        return [self.healthy_class] + self.disease_classes


@dataclass
class TrainConfig:
    """Optimisation and loss weights."""

    teachers: List[str] = field(default_factory=lambda: [
        "resnet50", "densenet121", "efficientnet_b0"
    ])
    epochs_teacher: int = 25
    epochs_student: int = 35
    batch_size: int = 32
    num_workers: int = 2
    lr_teacher: float = 1e-4
    lr_student: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    label_smoothing: float = 0.1

    temperature: float = 4.0
    alpha_kd: float = 0.5
    alpha_attention_gt: float = 1.5
    alpha_attention_teacher: float = 0.2
    alpha_contrastive: float = 0.2
    weight_iap: float = 0.3
    weight_severity: float = 0.3
    weight_count: float = 0.2

    seed: int = 42
    checkpoint_dir: str = "checkpoints"


@dataclass
class LeafSegConfig:
    """Leaf segmentation training."""

    encoder: str = "resnet34"
    img_size: int = 512
    epochs: int = 40
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_split: float = 0.2
    #: Predicted masks smaller than this fraction of the frame are rejected.
    min_mask_fraction: float = 0.02


#: The six ablation configurations reported in the paper.
ABLATION = {
    "A1_Baseline":      dict(use_kd=False, use_attention=False, use_agri=False),
    "A2_AgriOnly":      dict(use_kd=False, use_attention=False, use_agri=True),
    "A3_AttentionOnly": dict(use_kd=False, use_attention=True,  use_agri=False),
    "A4_KDOnly":        dict(use_kd=True,  use_attention=False, use_agri=False),
    "A5_KD_Attention":  dict(use_kd=True,  use_attention=True,  use_agri=False),
    "A6_Full_AGRIKD":   dict(use_kd=True,  use_attention=True,  use_agri=True),
}


def load_config(path: str):
    """Load ``DataConfig``, ``TrainConfig`` and ``LeafSegConfig`` from YAML."""
    import yaml
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return (
        DataConfig(**raw.get("data", {})),
        TrainConfig(**raw.get("train", {})),
        LeafSegConfig(**raw.get("leafseg", {})),
    )


def save_config(path: str, data: DataConfig, train: TrainConfig,
                leafseg: LeafSegConfig) -> None:
    """Write the three configuration objects to a YAML file."""
    import yaml
    with open(path, "w") as fh:
        yaml.safe_dump({"data": asdict(data), "train": asdict(train),
                        "leafseg": asdict(leafseg)}, fh, sort_keys=False)
