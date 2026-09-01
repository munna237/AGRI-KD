"""Test-set evaluation producing per-sample predictions.

Every reported metric is derived from the per-sample table written here, so that
each number in a results table traces back to a single file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, r2_score)
from torch.utils.data import DataLoader

from agrikd.utils.metrics import localization_metrics


@torch.no_grad()
def predict_per_sample(model, loader: DataLoader, device=None) -> pd.DataFrame:
    """Run the model over a loader and return one row per sample.

    Columns: ``stem``, true and predicted class, IAP, severity grade and lesion
    count, plus per-sample ``iou``, ``dice``, ``pointing`` and
    ``center_distance``.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    rows = []
    for batch in loader:
        out = model(batch["image"].to(device))
        cls = out["cls"].argmax(dim=1).cpu().numpy()
        iap = out["iap"].cpu().numpy()
        sev = out["severity"].argmax(dim=1).cpu().numpy()
        cnt = out["count"].cpu().numpy()
        att = out["attention"].cpu().numpy()
        gts = batch["attention_mask"].numpy()

        for j in range(len(cls)):
            loc = localization_metrics(att[j], gts[j])
            rows.append({
                "stem": batch["stem"][j],
                "true_class": int(batch["label"][j]),
                "pred_class": int(cls[j]),
                "true_iap": float(batch["iap"][j]),
                "pred_iap": float(iap[j]),
                "true_grade": int(batch["severity_grade"][j]),
                "pred_grade": int(sev[j]),
                "true_count": float(batch["lesion_count"][j]),
                "pred_count": float(cnt[j]),
                "has_lesion": bool(gts[j].sum() > 0),
                **loc,
            })
    return pd.DataFrame(rows)


def summarise(per_sample: pd.DataFrame, has_agri_heads: bool = True) -> dict:
    """Aggregate a per-sample table into the reported metrics.

    Localisation metrics are averaged over diseased images only, since healthy
    images have no lesion to localise.
    """
    diseased = per_sample[per_sample.has_lesion]
    result = {
        "classification_accuracy": 100 * accuracy_score(
            per_sample.true_class, per_sample.pred_class),
        "classification_f1": f1_score(
            per_sample.true_class, per_sample.pred_class, average="macro"),
        "dice": diseased.dice.mean(),
        "iou": diseased.iou.mean(),
        "pointing": 100 * diseased.pointing.mean(),
        "center_distance": diseased.center_distance.mean(),
    }
    if has_agri_heads:
        valid = per_sample.dropna(subset=["true_iap"])
        result.update({
            "iap_mae": mean_absolute_error(valid.true_iap, valid.pred_iap),
            "iap_rmse": float(np.sqrt(np.mean((valid.true_iap - valid.pred_iap) ** 2))),
            "iap_r2": r2_score(valid.true_iap, valid.pred_iap),
            "severity_accuracy": 100 * accuracy_score(valid.true_grade, valid.pred_grade),
            "severity_f1": f1_score(valid.true_grade, valid.pred_grade, average="macro"),
            "count_mae": mean_absolute_error(valid.true_count, valid.pred_count),
            "count_r2": r2_score(valid.true_count, valid.pred_count),
        })
    return result


def severity_confusion(per_sample: pd.DataFrame, num_grades: int = 5) -> np.ndarray:
    """Severity confusion matrix over samples with a valid grade."""
    valid = per_sample[per_sample.true_grade >= 0]
    return confusion_matrix(valid.true_grade, valid.pred_grade,
                            labels=list(range(num_grades)))
