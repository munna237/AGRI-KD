"""Training and evaluation loops."""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from agrikd.config import TrainConfig
from agrikd.losses import AgriKDLoss
from agrikd.models import StudentModel, TeacherEnsemble, TeacherModel
from agrikd.utils import Manifest, load_checkpoint, save_checkpoint, set_seed


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_classification(model, loader, device=None) -> tuple:
    """Return ``(accuracy, macro F1)`` on a loader."""
    device = device or _device()
    model.eval()
    preds, labels = [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        logits = out["cls"] if isinstance(out, dict) else out[0]
        preds += logits.argmax(dim=1).cpu().tolist()
        labels += batch["label"].tolist()
    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro")


def train_teacher(backbone: str, train_loader: DataLoader, val_loader: DataLoader,
                  config: TrainConfig, device=None) -> str:
    """Train a single teacher, resuming from any existing checkpoint.

    Returns:
        Path to the best-performing weights.
    """
    device = device or _device()
    tag = f"teacher_{backbone}"
    manifest = Manifest(os.path.join(config.checkpoint_dir, "manifest.json"))
    best_path = os.path.join(config.checkpoint_dir, f"{tag}_best.pth")
    if manifest.is_done(tag):
        print(f"{tag}: already complete")
        return best_path

    set_seed(config.seed)
    model = TeacherModel(backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_teacher,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.epochs_teacher)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    last_path = os.path.join(config.checkpoint_dir, f"{tag}_last.pth")
    start, best = load_checkpoint(last_path, model, optimizer, scheduler)

    for epoch in range(start, config.epochs_teacher):
        t0 = time.time()
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            criterion(model(images)["cls"], labels).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        scheduler.step()

        acc, _ = evaluate_classification(model, val_loader, device)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), best_path)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch + 1, best)
        print(f"  {tag} epoch {epoch + 1:3d}/{config.epochs_teacher}  "
              f"val acc {100 * acc:5.2f}%  best {100 * best:5.2f}%  "
              f"{time.time() - t0:4.0f}s")

    manifest.mark_done(tag, best=round(best, 4))
    return best_path


def train_student(tag: str, ablation: dict, train_loader: DataLoader,
                  val_loader: DataLoader, config: TrainConfig,
                  teacher_ensemble: TeacherEnsemble | None = None,
                  device=None) -> str:
    """Train one ablation configuration of the student.

    Args:
        tag: configuration name, used for checkpoint filenames.
        ablation: ``use_kd`` / ``use_attention`` / ``use_agri`` flags.
        teacher_ensemble: required when ``use_kd`` is set.

    Returns:
        Path to the best-performing weights.
    """
    device = device or _device()
    manifest = Manifest(os.path.join(config.checkpoint_dir, "manifest.json"))
    best_path = os.path.join(config.checkpoint_dir, f"{tag}_best.pth")
    if manifest.is_done(tag):
        print(f"{tag}: already complete")
        return best_path

    if ablation["use_kd"] and teacher_ensemble is None:
        raise ValueError(f"{tag} requires a teacher ensemble")

    set_seed(config.seed)
    model = StudentModel().to(device)
    criterion = AgriKDLoss(
        use_kd=ablation["use_kd"],
        use_attention=ablation["use_attention"],
        use_agri=ablation["use_agri"],
        temperature=config.temperature,
        alpha_kd=config.alpha_kd,
        alpha_attention_gt=config.alpha_attention_gt,
        alpha_attention_teacher=config.alpha_attention_teacher,
        alpha_contrastive=config.alpha_contrastive,
        weight_iap=config.weight_iap,
        weight_severity=config.weight_severity,
        weight_count=config.weight_count,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_student,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.epochs_student)

    last_path = os.path.join(config.checkpoint_dir, f"{tag}_last.pth")
    start, best = load_checkpoint(last_path, model, optimizer, scheduler)

    for epoch in range(start, config.epochs_student):
        t0 = time.time()
        model.train()
        running = 0.0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            gpu_batch = {
                "label": batch["label"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "iap": batch["iap"].to(device),
                "severity_grade": batch["severity_grade"].to(device),
                "lesion_count": batch["lesion_count"].to(device),
            }
            optimizer.zero_grad()
            student_out = model(images, return_features=True)
            teacher_out = None
            if ablation["use_kd"]:
                with torch.no_grad():
                    teacher_out = teacher_ensemble(images)
            loss, _ = criterion(student_out, gpu_batch, teacher_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running += float(loss)
        scheduler.step()

        acc, _ = evaluate_classification(model, val_loader, device)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), best_path)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch + 1, best)
        print(f"  {tag} epoch {epoch + 1:3d}/{config.epochs_student}  "
              f"loss {running / max(len(train_loader), 1):.4f}  "
              f"val acc {100 * acc:5.2f}%  best {100 * best:5.2f}%  "
              f"{time.time() - t0:4.0f}s")

    manifest.mark_done(tag, best=round(best, 4))
    return best_path
