"""Segmentation metrics with explicit empty-mask handling."""

from __future__ import annotations

import torch


def dice_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> tuple[float, float]:
    """Return thresholded Dice and IoU scores for binary segmentation tensors."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in the range [0, 1]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")

    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    truth = targets >= 0.5
    intersection = (predictions & truth).sum().item()
    predicted_area = predictions.sum().item()
    truth_area = truth.sum().item()
    union = predicted_area + truth_area - intersection
    dice = (2 * intersection + eps) / (predicted_area + truth_area + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)
