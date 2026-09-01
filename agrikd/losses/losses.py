"""Loss functions for AGRI-KD training.

The composite objective combines a classification term with distillation,
attention supervision, contrastive alignment, and the three agricultural
regression and classification terms. Weights are set in
:class:`agrikd.config.TrainConfig`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLoss(nn.Module):
    """Dice plus binary cross-entropy for attention map supervision.

    Dice handles the severe foreground/background imbalance of small lesions;
    BCE provides a pixelwise gradient. Predictions are resized to the target
    resolution when the two differ.

    Args:
        dice_weight: weight on the Dice term.
        bce_weight: weight on the BCE term.
    """

    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1), target.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        target_binary = (target > 0.5).float()
        p = pred.reshape(pred.size(0), -1)
        t = target_binary.reshape(target_binary.size(0), -1)
        inter = (p * t).sum(dim=1)
        dice = 1 - (2 * inter + 1) / (p.sum(dim=1) + t.sum(dim=1) + 1)
        bce = F.binary_cross_entropy(
            pred.clamp(1e-6, 1 - 1e-6), target_binary, reduction="mean"
        )
        return self.dice_weight * dice.mean() + self.bce_weight * bce


class KDLoss(nn.Module):
    """Temperature-scaled Kullback-Leibler distillation loss.

    Args:
        temperature: softening temperature applied to both distributions.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor) -> torch.Tensor:
        t = self.temperature
        return F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.softmax(teacher_logits / t, dim=1),
            reduction="batchmean",
        ) * (t ** 2)


class ContrastiveLoss(nn.Module):
    """Supervised contrastive loss over pooled student features.

    Pulls together representations sharing a class label and pushes apart those
    that do not, encouraging class-discriminative structure in the compressed
    feature space.

    Args:
        temperature: scaling applied to the similarity matrix.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        positive = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        off_diagonal = 1 - torch.eye(positive.size(0), device=features.device)
        positive = positive * off_diagonal
        exp_sim = torch.exp(sim) * off_diagonal
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        mean_log_prob = (positive * log_prob).sum(dim=1) / (positive.sum(dim=1) + 1e-8)
        return -mean_log_prob.mean()


class AgriKDLoss(nn.Module):
    """Composite AGRI-KD objective.

    Terms are enabled independently so that the ablation configurations of the
    paper can be reproduced from a single class.

    Args:
        use_kd: enable soft-label distillation and contrastive alignment.
        use_attention: enable ground-truth attention supervision.
        use_agri: enable the IAP, severity and lesion-count terms.
        temperature: distillation temperature.
        alpha_kd: weight on soft-label distillation.
        alpha_attention_gt: weight on ground-truth attention supervision.
        alpha_attention_teacher: weight on teacher attention transfer.
        alpha_contrastive: weight on contrastive alignment.
        weight_iap: weight on the IAP regression term.
        weight_severity: weight on the severity classification term.
        weight_count: weight on the lesion-count regression term.
        label_smoothing: label smoothing for the classification term.
    """

    def __init__(
        self,
        use_kd: bool = True,
        use_attention: bool = True,
        use_agri: bool = True,
        temperature: float = 4.0,
        alpha_kd: float = 0.5,
        alpha_attention_gt: float = 1.5,
        alpha_attention_teacher: float = 0.2,
        alpha_contrastive: float = 0.2,
        weight_iap: float = 0.3,
        weight_severity: float = 0.3,
        weight_count: float = 0.2,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.use_kd = use_kd
        self.use_attention = use_attention
        self.use_agri = use_agri
        self.alpha_kd = alpha_kd
        self.alpha_attention_gt = alpha_attention_gt
        self.alpha_attention_teacher = alpha_attention_teacher
        self.alpha_contrastive = alpha_contrastive
        self.weight_iap = weight_iap
        self.weight_severity = weight_severity
        self.weight_count = weight_count

        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kd_loss = KDLoss(temperature)
        self.attention_loss = AttentionLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.regression_loss = nn.SmoothL1Loss()
        self.severity_loss = nn.CrossEntropyLoss()

    def forward(self, student_out: dict, batch: dict,
                teacher_out: dict | None = None) -> tuple:
        """Compute the total loss and a breakdown of its components.

        Args:
            student_out: output dict from :class:`StudentModel`.
            batch: batch dict with ``label``, ``attention_mask``, ``iap``,
                ``severity_grade`` and ``lesion_count``.
            teacher_out: output dict from :class:`TeacherEnsemble`, required
                when ``use_kd`` is set.

        Returns:
            ``(total_loss, components)`` where ``components`` maps term names to
            detached float values for logging.
        """
        labels = batch["label"]
        total = self.cls_loss(student_out["cls"], labels)
        parts = {"cls": float(total)}

        if self.use_kd and teacher_out is not None:
            kd = self.kd_loss(student_out["cls"], teacher_out["cls"])
            total = total + self.alpha_kd * kd
            parts["kd"] = float(kd)

            contrast = self.contrastive_loss(student_out["features"], labels)
            total = total + self.alpha_contrastive * contrast
            parts["contrastive"] = float(contrast)

            if self.use_attention and teacher_out.get("attention") is not None:
                att_t = self.attention_loss(
                    student_out["attention"], teacher_out["attention"]
                )
                total = total + self.alpha_attention_teacher * att_t
                parts["attention_teacher"] = float(att_t)

        if self.use_attention and student_out["attention"] is not None:
            att_gt = self.attention_loss(
                student_out["attention"], batch["attention_mask"]
            )
            total = total + self.alpha_attention_gt * att_gt
            parts["attention_gt"] = float(att_gt)

        if self.use_agri:
            iap = self.regression_loss(student_out["iap"] / 100, batch["iap"] / 100)
            sev = self.severity_loss(student_out["severity"], batch["severity_grade"])
            cnt = self.regression_loss(student_out["count"], batch["lesion_count"])
            total = (total + self.weight_iap * iap
                     + self.weight_severity * sev
                     + self.weight_count * cnt)
            parts.update(iap=float(iap), severity=float(sev), count=float(cnt))

        return total, parts
