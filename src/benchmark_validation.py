"""Deterministic repeated multilabel outer/inner split construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

try:
    from skmultilearn.model_selection import IterativeStratification
except ImportError:  # pragma: no cover - dependency is part of requirements.txt
    IterativeStratification = None


@dataclass(frozen=True)
class FoldIndices:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def _iterative_split(y: np.ndarray, n_splits: int, seed: int):
    if IterativeStratification is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(np.zeros(len(y)))
        return
    # The implementation itself has no random_state.  A seeded permutation
    # makes repeats deterministic while preserving iterative stratification.
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(y))
    splitter = IterativeStratification(
        n_splits=n_splits,
        order=2,
        sample_distribution_per_fold=[1.0 / n_splits] * n_splits,
    )
    # skmultilearn uses NumPy's legacy global RNG internally.  Preserve the
    # caller's state while making the split exactly reproducible.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        materialized = list(
            splitter.split(np.zeros((len(y), 1)), y[permutation])
        )
    finally:
        np.random.set_state(state)
    for train_permuted, test_permuted in materialized:
        yield permutation[train_permuted], permutation[test_permuted]


def repeated_outer_folds(
    y: np.ndarray,
    n_repeats: int = 5,
    n_splits: int = 5,
    base_seed: int = 42,
) -> list[FoldIndices]:
    y = np.asarray(y, dtype=np.int8)
    folds: list[FoldIndices] = []
    for repeat in range(n_repeats):
        seen = np.zeros(len(y), dtype=np.int8)
        for fold, (train, test) in enumerate(
            _iterative_split(y, n_splits=n_splits, seed=base_seed + repeat)
        ):
            seen[test] += 1
            folds.append(FoldIndices(repeat, fold, np.asarray(train), np.asarray(test)))
        if not np.all(seen == 1):
            raise AssertionError("Every patient must occur in exactly one test fold per repeat.")
    return folds


def inner_folds(y: np.ndarray, n_splits: int, seed: int):
    yield from _iterative_split(np.asarray(y, dtype=np.int8), n_splits, seed)


def assignment_matrix(folds: list[FoldIndices], n_samples: int, n_repeats: int) -> np.ndarray:
    assignment = np.full((n_repeats, n_samples), -1, dtype=np.int16)
    for item in folds:
        if np.any(assignment[item.repeat, item.test] != -1):
            raise AssertionError("Outer test assignments overlap within a repeat.")
        assignment[item.repeat, item.test] = item.fold
    if np.any(assignment < 0):
        raise AssertionError("Outer test assignments do not cover every patient.")
    return assignment
