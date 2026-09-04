"""Audited weighted multilabel losses used in the GCN ablation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def positive_class_weights(targets: torch.Tensor) -> torch.Tensor:
    """Return the locked per-label positive weight ``N_l^- / N_l^+``."""
    if not targets.is_floating_point():
        targets = targets.float()
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    return negatives / positives.clamp_min(1.0)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross entropy with the positive term weighted by ``weights``."""
    return F.binary_cross_entropy_with_logits(
        logits,
        targets.to(dtype=logits.dtype),
        pos_weight=weights.to(dtype=logits.dtype),
        reduction=reduction,
    )


def weighted_asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Weighted ASL with class weights applied only to the positive term."""
    targets = targets.to(dtype=logits.dtype)
    weights = weights.to(dtype=logits.dtype)
    probability = torch.sigmoid(logits)
    positive_probability = probability
    negative_probability = 1.0 - probability
    if clip > 0:
        negative_probability = (negative_probability + clip).clamp(max=1.0)
    positive_term = targets * torch.log(positive_probability.clamp_min(eps))
    negative_term = (1.0 - targets) * torch.log(negative_probability.clamp_min(eps))
    if gamma_pos > 0:
        positive_term = positive_term * (1.0 - positive_probability).pow(gamma_pos)
    if gamma_neg > 0:
        negative_term = negative_term * probability.pow(gamma_neg)
    loss = -(weights * positive_term + negative_term)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return loss.mean()
