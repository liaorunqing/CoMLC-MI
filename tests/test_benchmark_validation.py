from __future__ import annotations

import numpy as np

from src.benchmark_validation import assignment_matrix, repeated_outer_folds


def test_repeated_outer_folds_cover_each_patient_once_per_repeat() -> None:
    rng = np.random.default_rng(42)
    y = (rng.random((120, 12)) < np.linspace(0.05, 0.35, 12)).astype(int)
    folds = repeated_outer_folds(y, n_repeats=5, n_splits=5, base_seed=42)
    assignment = assignment_matrix(folds, n_samples=len(y), n_repeats=5)
    assert assignment.shape == (5, 120)
    assert np.all((assignment >= 0) & (assignment < 5))
    assert len(folds) == 25
    for repeat in range(5):
        counts = np.bincount(assignment[repeat], minlength=5)
        assert counts.sum() == len(y)


def test_split_generation_is_reproducible() -> None:
    rng = np.random.default_rng(7)
    y = (rng.random((80, 4)) < 0.2).astype(int)
    first = assignment_matrix(repeated_outer_folds(y, 2, 4, 99), len(y), 2)
    second = assignment_matrix(repeated_outer_folds(y, 2, 4, 99), len(y), 2)
    assert np.array_equal(first, second)


