from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.external_transportability import (
    _weighted_auc,
    _weighted_average_precision,
    calibration_bins,
    harmonize_strict,
    summarize_with_bootstrap,
    weighted_metrics,
    weighted_metrics_batch,
)


def _source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AGE": [60, 70, 80],
            "SEX": [1, 0, 1],
            "INF_ANAM": [0, 2, np.nan],
            "ZSN_A": [1, 0, np.nan],
            "LET_IS": [0, 2, 0],
        }
    )


def _registry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Study ID": [10, 10, 11, 12],
            "Event number": [1, 2, 1, 1],
            "Age at admission": [61, 62, 71, 81],
            "Gender": ["Man", "Man", "Woman", "Man"],
            "History of myocardial infarction": ["Not", "Yes", "Yes", "Unknown"],
            "History of heart failure": ["Yes", "Not", "Not", "Yes"],
            "First hospital admission of an event": [
                "2019-01-01",
                "2019-02-01",
                "2019-03-01",
                "2019-04-01",
            ],
            "Date of death": ["2019-01-08", pd.NaT, "2019-03-31", "2019-05-02"],
        }
    )


def test_harmonization_uses_first_events_and_preserves_unknown_history() -> None:
    data = harmonize_strict(_source_frame(), _registry_frame())
    assert len(data.external_x) == 3
    assert data.external_x.index.tolist() == [0, 2, 3]
    assert data.external_x.loc[3, "prior_mi"] != data.external_x.loc[3, "prior_mi"]
    assert data.external_complete_case.tolist() == [True, True, False]
    assert data.external_endpoints["death_7d"].tolist() == [1, 0, 0]
    assert data.external_endpoints["death_30d"].tolist() == [1, 1, 0]
    assert data.source_y.tolist() == [0, 1, 0]


def test_weighted_rank_metrics_match_sklearn_for_unit_weights() -> None:
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.7, 0.4, 0.6, 0.9, 0.2])
    weights = np.ones(len(y))
    assert np.isclose(_weighted_auc(y, p, weights), roc_auc_score(y, p))
    assert np.isclose(_weighted_average_precision(y, p, weights), average_precision_score(y, p))


def test_bootstrap_is_reproducible_and_paired() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    predictions = {
        "TabPFN-3": np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.25, 0.65]),
        "Logistic regression": np.array([0.15, 0.75, 0.3, 0.6, 0.35, 0.85, 0.2, 0.7]),
    }
    first = summarize_with_bootstrap(y, predictions, n_boot=40, seed=42)
    second = summarize_with_bootstrap(y, predictions, n_boot=40, seed=42)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[1].shape[0] == 1
    assert calibration_bins(predictions["TabPFN-3"]).shape == y.shape


def test_vectorized_bootstrap_metrics_match_scalar_metrics() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.7, 0.6, 0.3, 0.9, 0.8, 0.2])
    bins = calibration_bins(p)
    auc_order = np.argsort(p, kind="mergesort")
    ap_order = np.argsort(-p, kind="mergesort")
    eps = 1e-6
    cache = {
        "auc_order": auc_order,
        "auc_starts": np.r_[0, np.flatnonzero(np.diff(p[auc_order])) + 1],
        "ap_order": ap_order,
        "ap_starts": np.r_[0, np.flatnonzero(np.diff(p[ap_order])) + 1],
        "eta": np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)),
    }
    weights = np.array([[1, 2, 0, 1, 1, 1, 2, 0], [0, 1, 2, 1, 1, 2, 0, 1]], dtype=float)
    batch = weighted_metrics_batch(y, p, weights, bins, cache)
    names = [
        "auc",
        "auprc",
        "brier",
        "ece",
        "calibration_intercept",
        "calibration_slope",
        "mean_predicted",
        "observed_rate",
        "oe_ratio",
    ]
    for index, row_weights in enumerate(weights):
        scalar = weighted_metrics(y, p, row_weights, bins, cache)
        assert np.allclose(batch[index], [scalar[name] for name in names], atol=1e-10)


def test_real_registry_counts_when_available() -> None:
    path = os.getenv("HUNGARIAN_MI_XLSX")
    if path is None:
        path = Path.home() / "Downloads" / "Hungarian Myocardial Infarction Registry extracted database.xlsx"
    try:
        registry = pd.read_excel(path)
    except FileNotFoundError:
        return
    source = pd.read_csv("dataset/Myocardial infarction complications Database.csv")
    data = harmonize_strict(source, registry)
    assert len(data.external_x) == 29_596
    assert int(data.external_endpoints["death_30d"].sum()) == 3_888
    assert int(data.external_endpoints["death_7d"].sum()) == 2_408
    assert int(data.external_complete_case.sum()) == 28_477
