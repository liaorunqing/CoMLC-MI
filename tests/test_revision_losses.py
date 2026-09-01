from __future__ import annotations

import torch
import torch.nn.functional as F

from src.revision_losses import (
    positive_class_weights,
    weighted_asymmetric_loss_with_logits,
    weighted_bce_with_logits,
)


def test_positive_class_weights_equal_negative_over_positive() -> None:
    targets = torch.tensor([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=torch.float32)
    assert torch.allclose(positive_class_weights(targets), torch.tensor([3.0, 3.0]))


def test_weighted_bce_matches_hand_calculation() -> None:
    logits = torch.tensor([[0.0, 1.0], [-1.0, 0.5]], dtype=torch.float64)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    weights = torch.tensor([3.0, 2.0], dtype=torch.float64)
    observed = weighted_bce_with_logits(logits, targets, weights, reduction="none")
    expected = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weights, reduction="none"
    )
    assert torch.allclose(observed, expected)


def test_weighted_asl_reduces_to_weighted_bce_without_focusing_or_clip() -> None:
    logits = torch.tensor([[0.2, -0.7], [1.2, 0.1]], dtype=torch.float64)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    weights = torch.tensor([4.0, 2.0], dtype=torch.float64)
    asl = weighted_asymmetric_loss_with_logits(
        logits,
        targets,
        weights,
        gamma_pos=0.0,
        gamma_neg=0.0,
        clip=0.0,
        reduction="none",
    )
    bce = weighted_bce_with_logits(logits, targets, weights, reduction="none")
    assert torch.allclose(asl, bce, atol=1e-10)

