from __future__ import annotations

import numpy as np

from src.revision_models import (
    fit_predict_model,
    select_top_features,
    validation_auc_weights,
)


def _data(seed: int = 42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(100, 20)).astype(np.float32)
    y = np.zeros((100, 12), dtype=np.int8)
    for label in range(12):
        y[:, label] = (x[:, label % 20] + rng.normal(size=100) > 0).astype(np.int8)
    return x[:80], y[:80], x[80:], y[80:]


def test_training_fold_feature_selection_is_deterministic() -> None:
    x_train, y_train, _, _ = _data()
    first = select_top_features(x_train, y_train, n_features=10, seed=11, n_estimators=5)
    second = select_top_features(x_train, y_train, n_features=10, seed=11, n_estimators=5)
    assert np.array_equal(first, second)
    assert len(first) == 10
    assert len(np.unique(first)) == 10


def test_br_lr_and_lp_rf_return_finite_multilabel_probabilities() -> None:
    x_train, y_train, x_test, _ = _data()
    lr = fit_predict_model("BR-LR", x_train, y_train, x_test, 5, {"max_iter": 500})
    lp = fit_predict_model(
        "LP-RF",
        x_train,
        y_train,
        x_test,
        5,
        {"n_estimators": 10, "max_depth": 4, "n_jobs": 1},
    )
    assert lr.shape == lp.shape == (20, 12)
    assert np.isfinite(lr).all() and np.isfinite(lp).all()
    assert np.all((lr > 0) & (lr < 1))
    assert np.all((lp > 0) & (lp < 1))


def test_inner_validation_weights_sum_to_one_per_label() -> None:
    _, _, _, y = _data()
    rng = np.random.default_rng(2)
    lp = rng.random(y.shape)
    tab = rng.random(y.shape)
    weights = validation_auc_weights(y, lp, tab)
    assert weights.shape == (2, 12)
    assert np.allclose(weights.sum(axis=0), 1.0)

