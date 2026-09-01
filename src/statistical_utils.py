"""Validated statistical utilities for paired model comparisons."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def _placement_values(positive_scores: np.ndarray, negative_scores: np.ndarray):
    comparison = (
        (positive_scores[:, None] > negative_scores[None, :]).astype(float)
        + 0.5 * (positive_scores[:, None] == negative_scores[None, :])
    )
    return comparison.mean(axis=1), comparison.mean(axis=0)


def delong_roc_test(y_true, pred_a, pred_b):
    """Paired two-sided DeLong test for the difference between two ROC AUCs.

    The covariance is the sum of the positive and negative structural-component
    covariance matrices divided by their respective sample counts, following
    DeLong et al. (1988). Returns ``z, p, auc_a, auc_b``.
    """
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    positive = y_true == 1
    negative = y_true == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive < 2 or n_negative < 2:
        return np.nan, np.nan, np.nan, np.nan

    v10_a, v01_a = _placement_values(pred_a[positive], pred_a[negative])
    v10_b, v01_b = _placement_values(pred_b[positive], pred_b[negative])
    auc_a = float(v10_a.mean())
    auc_b = float(v10_b.mean())

    covariance_positive = np.cov(np.vstack((v10_a, v10_b)), ddof=1) / n_positive
    covariance_negative = np.cov(np.vstack((v01_a, v01_b)), ddof=1) / n_negative
    covariance = covariance_positive + covariance_negative
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    if not np.isfinite(variance) or variance <= 0:
        return np.nan, np.nan, auc_a, auc_b
    z_value = (auc_a - auc_b) / np.sqrt(variance)
    p_value = 2.0 * norm.sf(abs(z_value))
    return float(z_value), float(p_value), auc_a, auc_b


def permutation_auc_test(y_true, pred_a, pred_b, n_permutations=10_000, random_state=42):
    """Paired randomization test that swaps model scores within each patient."""
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    if np.unique(y_true).size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    observed = roc_auc_score(y_true, pred_a) - roc_auc_score(y_true, pred_b)
    extreme = 0
    for _ in range(n_permutations):
        swap = rng.random(len(y_true)) < 0.5
        permuted_a = np.where(swap, pred_b, pred_a)
        permuted_b = np.where(swap, pred_a, pred_b)
        difference = roc_auc_score(y_true, permuted_a) - roc_auc_score(y_true, permuted_b)
        extreme += abs(difference) >= abs(observed)
    return float(observed), float((extreme + 1) / (n_permutations + 1))


def paired_stratified_auc_difference_ci(
    y_true, pred_a, pred_b, n_bootstrap=2_000, random_state=42, confidence=0.95
):
    """Percentile CI for paired AUC difference, stratified by outcome class."""
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)
    if len(positive_indices) < 2 or len(negative_indices) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    differences = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled = np.concatenate((
            rng.choice(positive_indices, len(positive_indices), replace=True),
            rng.choice(negative_indices, len(negative_indices), replace=True),
        ))
        differences[iteration] = (
            roc_auc_score(y_true[sampled], pred_a[sampled])
            - roc_auc_score(y_true[sampled], pred_b[sampled])
        )
    tail = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(differences, [tail, 1.0 - tail]))


def benjamini_hochberg(p_values, alpha=0.05):
    """Return BH step-up rejections and monotone adjusted p values."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or np.isnan(p_values).any():
        raise ValueError("p_values must be a one-dimensional array without NaNs.")
    n_tests = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    ranks = np.arange(1, n_tests + 1)

    passing = ranked <= alpha * ranks / n_tests
    reject_sorted = np.zeros(n_tests, dtype=bool)
    if passing.any():
        reject_sorted[: np.flatnonzero(passing)[-1] + 1] = True

    adjusted_sorted = ranked * n_tests / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    reject = np.empty(n_tests, dtype=bool)
    adjusted = np.empty(n_tests, dtype=float)
    reject[order] = reject_sorted
    adjusted[order] = adjusted_sorted
    return reject, adjusted
