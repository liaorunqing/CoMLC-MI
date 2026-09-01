from __future__ import annotations

import numpy as np

from src.synthetic_experiments import SyntheticDesign, generate_synthetic_multilabel


def test_formal_synthetic_defaults_are_locked() -> None:
    design = SyntheticDesign()
    assert (design.n_samples, design.n_features, design.n_labels) == (1000, 50, 10)


def test_changing_correlation_preserves_features_and_label_prevalence() -> None:
    design = SyntheticDesign(n_samples=300, n_features=12, n_labels=5, n_informative=6)
    x_low, y_low, _, _ = generate_synthetic_multilabel(design, 0.0, random_state=17)
    x_high, y_high, _, _ = generate_synthetic_multilabel(design, 0.9, random_state=17)
    assert np.array_equal(x_low, x_high)
    assert np.array_equal(y_low.sum(axis=0), y_high.sum(axis=0))

