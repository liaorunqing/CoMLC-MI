"""Label-space diagnostics used by preprocessing and synthetic experiments.

The primary dependency summary is the mean directed conditional co-occurrence
P(Y_j=1 | Y_i=1) over all ordered off-diagonal label pairs.  The diagonal is
excluded because P(Y_i=1 | Y_i=1)=1 by definition and would inflate the score.
"""

from __future__ import annotations

import numpy as np


def _as_binary_matrix(y: np.ndarray) -> np.ndarray:
    matrix = np.asarray(y)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D label matrix, received shape {matrix.shape}.")
    if not np.isin(matrix, (0, 1)).all():
        values = np.unique(matrix)
        raise ValueError(f"Labels must be binary before metric calculation; found {values}.")
    # int64 prevents overflow in matrix multiplication for cohorts with more
    # than 127 co-positive observations.
    return matrix.astype(np.int64, copy=False)


def conditional_cooccurrence(y: np.ndarray) -> np.ndarray:
    """Return directed conditional probabilities C[i,j] = P(Y_j=1 | Y_i=1).

    Rows corresponding to labels with no positive observations remain zero.
    The diagonal is set to zero so summaries can use the full off-diagonal mask.
    """
    matrix = _as_binary_matrix(y)
    n_labels = matrix.shape[1]
    counts = matrix.sum(axis=0)
    joint = matrix.T @ matrix
    conditional = np.divide(
        joint,
        counts[:, None],
        out=np.zeros((n_labels, n_labels), dtype=float),
        where=counts[:, None] > 0,
    )
    np.fill_diagonal(conditional, 0.0)
    return conditional


def label_space_metrics(y: np.ndarray) -> dict:
    """Compute prevalence, cardinality, density, and dependency diagnostics."""
    matrix = _as_binary_matrix(y)
    n_samples, n_labels = matrix.shape
    conditional = conditional_cooccurrence(matrix)
    off_diagonal = ~np.eye(n_labels, dtype=bool)
    off_values = conditional[off_diagonal]

    # Mean pairwise phi correlation is reported as a complementary, prevalence-
    # adjusted dependence measure. Undefined constant-label pairs are omitted.
    phi_values = []
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            if matrix[:, i].std() == 0 or matrix[:, j].std() == 0:
                continue
            phi_values.append(float(np.corrcoef(matrix[:, i], matrix[:, j])[0, 1]))

    cardinality = float(matrix.sum(axis=1).mean())
    return {
        "n_samples": int(n_samples),
        "n_labels": int(n_labels),
        "positive_counts": matrix.sum(axis=0).astype(int).tolist(),
        "prevalence": matrix.mean(axis=0).astype(float).tolist(),
        "label_cardinality": cardinality,
        "label_density": cardinality / n_labels,
        "label_dependency_score": float(off_values.mean()),
        "median_conditional_cooccurrence": float(np.median(off_values)),
        "mean_pairwise_phi": float(np.mean(phi_values)) if phi_values else None,
        "directed_pairs_above_0_20": int(np.sum(off_values > 0.20)),
        "conditional_cooccurrence": conditional.tolist(),
    }
