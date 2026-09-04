from __future__ import annotations

import numpy as np

from src.benchmark_metrics import (
    aggregate_metrics,
    calibration_bootstrap_intervals,
    expected_calibration_error,
    paired_label_auc_tests,
    paired_patient_bootstrap,
)


def _multilabel_example(seed: int = 42):
    rng = np.random.default_rng(seed)
    y = (rng.random((200, 12)) < np.linspace(0.08, 0.35, 12)).astype(int)
    base = np.clip(0.1 + 0.7 * y + rng.normal(0, 0.12, y.shape), 0.001, 0.999)
    other = np.clip(base + rng.normal(0, 0.04, y.shape), 0.001, 0.999)
    return y, base, other


def test_primary_metrics_are_finite_and_threshold_is_fixed() -> None:
    y, probability, _ = _multilabel_example()
    metrics = aggregate_metrics(y, probability)
    assert set(metrics) == {
        "macro_auroc", "micro_auroc", "macro_auprc", "brier", "ece_10",
        "macro_f1_0_5",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0 <= expected_calibration_error(y[:, 0], probability[:, 0]) <= 1


def test_patient_bootstrap_is_paired_and_reproducible() -> None:
    y, model_a, model_b = _multilabel_example()
    predictions = {"A": model_a, "B": model_b}
    summary_a, arrays_a = paired_patient_bootstrap(y, predictions, n_bootstrap=30, seed=9)
    summary_b, arrays_b = paired_patient_bootstrap(y, predictions, n_bootstrap=30, seed=9, n_jobs=2)
    assert summary_a.equals(summary_b)
    assert np.array_equal(arrays_a["A"], arrays_b["A"])
    assert np.array_equal(arrays_a["B"], arrays_b["B"])
    assert arrays_a["A"].shape == arrays_a["B"].shape == (30, 6)
    ece_rows = summary_a[summary_a["metric"] == "ece_10"]
    assert ece_rows["interval_method"].eq("bootstrap standard-error normal").all()
    assert ((ece_rows["ci_low"] <= ece_rows["estimate"]) & (ece_rows["estimate"] <= ece_rows["ci_high"])).all()
    assert summary_a.loc[summary_a["metric"] != "ece_10", "interval_method"].eq(
        "bootstrap percentile"
    ).all()


def test_calibration_bootstrap_is_finite_and_marks_rare_labels() -> None:
    y, model_a, _ = _multilabel_example()
    frame = calibration_bootstrap_intervals(y, model_a, n_bootstrap=10, seed=5)
    parallel = calibration_bootstrap_intervals(y, model_a, n_bootstrap=10, seed=5, n_jobs=2)
    assert frame.equals(parallel)
    assert len(frame) == 12 * 4
    assert frame["n_bootstrap"].eq(10).all()
    assert np.isfinite(frame[["estimate", "ci_low", "ci_high"]].to_numpy()).all()
    expected = np.where(frame["positive_n"] < 30, "unstable", "stable")
    assert np.array_equal(frame["stability"].to_numpy(), expected)


def test_per_label_test_routes_rare_outcomes_to_permutation_and_applies_bh() -> None:
    y, model_a, model_b = _multilabel_example()
    frame = paired_label_auc_tests(
        y, model_a, model_b, low_event_threshold=30, n_permutations=50, seed=11
    )
    assert len(frame) == 12
    estimable = frame["test"].ne("not estimable")
    assert set(frame.loc[estimable & (frame["positive_n"] < 30), "test"]) <= {"paired permutation"}
    assert set(frame.loc[frame["positive_n"] >= 30, "test"]) <= {"paired DeLong"}
    assert frame.loc[~estimable, "p_value"].eq(1.0).all()
    assert frame.loc[~estimable, "stability"].eq("not estimable").all()
    assert frame["p_value_bh"].between(0, 1).all()

