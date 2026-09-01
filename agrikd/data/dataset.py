"""Dataset and split handling for RiceLeafDiseaseBD."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from agrikd.config import DataConfig
from agrikd.data.targets import attention_mask, iap_to_grade, parse_yolo


def _match_dir(parent: Path, name: str) -> Path | None:
    """Find a subdirectory ignoring spacing and underscore differences."""
    key = name.lower().replace(" ", "").replace("_", "")
    for p in Path(parent).iterdir():
        if p.is_dir() and p.name.lower().replace(" ", "").replace("_", "") == key:
            return p
    return None


class RiceLeafDataset(Dataset):
    """Multi-task rice leaf dataset.

    Images are read from the unannotated originals. Lesion coordinates come from
    the YOLO label files, and leaf-area IAP together with severity grades are
    read from a precomputed targets table (see ``scripts/01_build_targets.py``).

    Each item is a dict with keys ``image``, ``label``, ``attention_mask``,
    ``iap``, ``severity_grade`` and ``lesion_count``.

    Args:
        config: dataset configuration.
        targets_csv: path to the precomputed targets table.
        transform: albumentations pipeline applied to the RGB image.
        indices: optional subset of sample indices.
    """

    def __init__(
        self,
        config: DataConfig,
        targets_csv: str,
        transform=None,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.config = config
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(config.all_classes)}

        targets = pd.read_csv(targets_csv)
        self._targets: Dict[str, tuple] = {
            r.stem: (float(r.iap_leaf), int(r.grade), int(r.n_box))
            for r in targets.itertuples()
        }

        self.samples = self._index()
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def _index(self) -> List[dict]:
        root = Path(self.config.root)
        orig_root = _match_dir(root, self.config.original_dir)
        ann_root = _match_dir(root, self.config.annotated_dir)
        if orig_root is None or ann_root is None:
            raise FileNotFoundError(f"dataset folders not found under {root}")

        samples: List[dict] = []

        healthy_dir = _match_dir(orig_root, self.config.healthy_class)
        if healthy_dir is not None:
            for p in sorted(healthy_dir.glob("*")):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    samples.append(dict(
                        path=str(p), stem=p.stem, label=0,
                        class_name=self.config.healthy_class, label_path=None,
                    ))

        for disease in self.config.disease_classes:
            image_dir = _match_dir(orig_root, disease)
            class_dir = _match_dir(ann_root, disease)
            if image_dir is None or class_dir is None:
                continue
            label_dir = _match_dir(class_dir, "labels")
            if label_dir is None:
                continue
            by_stem = {p.stem: p for p in image_dir.glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
            for label_path in sorted(label_dir.glob("*.txt")):
                image_path = by_stem.get(label_path.stem)
                if image_path is None:
                    continue
                target = self._targets.get(label_path.stem)
                if target is None or not np.isfinite(target[0]):
                    continue  # no reliable leaf mask
                samples.append(dict(
                    path=str(image_path), stem=label_path.stem,
                    label=self.class_to_idx[disease], class_name=disease,
                    label_path=str(label_path),
                ))

        random.Random(self.config.seed).shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = cv2.imread(sample["path"])
        if image is None:
            image = np.zeros((self.config.img_size, self.config.img_size, 3), np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = parse_yolo(sample["label_path"]) if sample["label_path"] else []
        iap, grade, count = self._targets.get(sample["stem"], (0.0, 0, 0))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return {
            "image": image,
            "label": sample["label"],
            "stem": sample["stem"],
            "attention_mask": torch.tensor(
                attention_mask(boxes, self.config.attention_size)
            ),
            "iap": torch.tensor(iap, dtype=torch.float32),
            "severity_grade": grade,
            "lesion_count": torch.tensor(float(count)),
        }


def make_split(dataset: RiceLeafDataset, config: DataConfig,
               split_csv: str | None = None):
    """Create or reload a stratified train/validation/test split.

    Writing the split to disk is deliberate: relying on a random seed alone does
    not reproduce membership across sessions, because directory listing order is
    filesystem dependent.

    Args:
        dataset: indexed dataset.
        config: supplies split ratios and seed.
        split_csv: path to persist the split; reloaded when it already exists.

    Returns:
        ``(train_idx, val_idx, test_idx)``.
    """
    stems = [s["stem"] for s in dataset.samples]

    if split_csv and os.path.exists(split_csv):
        saved = pd.read_csv(split_csv)
        position = {stem: i for i, stem in enumerate(stems)}
        pick = lambda name: [position[s] for s in saved[saved.split == name].stem
                             if s in position]
        return pick("train"), pick("val"), pick("test")

    labels = [s["label"] for s in dataset.samples]
    train_val, test = train_test_split(
        range(len(dataset)), test_size=config.test_size,
        stratify=labels, random_state=config.seed,
    )
    train, val = train_test_split(
        train_val, test_size=config.val_size,
        stratify=[labels[i] for i in train_val], random_state=config.seed,
    )

    if split_csv:
        rows = ([{"stem": stems[i], "split": "train"} for i in train]
                + [{"stem": stems[i], "split": "val"} for i in val]
                + [{"stem": stems[i], "split": "test"} for i in test])
        os.makedirs(os.path.dirname(split_csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(split_csv, index=False)

    return train, val, test
