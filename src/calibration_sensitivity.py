"""Unweighted BR-logistic calibration sensitivity on the locked outer folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .revision_features import FoldPreprocessor, LABEL_COLS, prepare_outcomes
from .revision_metrics import (
    aggregate_metrics,
    calibration_bootstrap_intervals,
    per_label_metrics,
    reliability_table,
)


def _fit_fold(
    raw: pd.DataFrame,
    outcomes: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
    config: dict,
) -> np.ndarray:
    prep = FoldPreprocessor(
        feature_contract=config["feature_contract"],
        random_state=seed,
        rf_estimators=config["preprocessing"]["iterative_rf_estimators"],
        iterative_max_iter=config["preprocessing"]["iterative_max_iter"],
    ).fit(raw.iloc[train])
    if set(prep.audit_.fitted_row_ids) & set(test):
        raise AssertionError("Outer test rows reached calibration-sensitivity preprocessing.")
    x_train = prep.transform(raw.iloc[train]).to_numpy()
    x_test = prep.transform(raw.iloc[test]).to_numpy()
    result = np.empty((len(test), outcomes.shape[1]), dtype=np.float64)
    for label_index in range(outcomes.shape[1]):
        y_train = outcomes[train, label_index]
        if np.unique(y_train).size < 2:
            result[:, label_index] = float(y_train[0])
            continue
        estimator = LogisticRegression(
            max_iter=5000,
            class_weight=None,
            random_state=seed + label_index,
            solver="liblinear",
        )
        estimator.fit(x_train, y_train)
        result[:, label_index] = estimator.predict_proba(x_test)[:, 1]
    return np.clip(result, 1e-6, 1 - 1e-6)


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(config["dataset"])
    outcomes = prepare_outcomes(raw).to_numpy(dtype=np.int8)
    assignments = pd.read_csv(output_dir / "outer_fold_assignments.csv")
    repeats = config["validation"]["outer_repeats"]
    folds = config["validation"]["outer_folds"]
    repeated = np.full((repeats, len(raw), len(LABEL_COLS)), np.nan, dtype=np.float64)
    for repeat in range(repeats):
        column = f"repeat_{repeat + 1}_outer_fold"
        for fold in range(folds):
            test = np.flatnonzero(assignments[column].to_numpy() == fold)
            train = np.flatnonzero(assignments[column].to_numpy() != fold)
            seed = config["seed"] + repeat * 1000 + fold * 100
            repeated[repeat, test] = _fit_fold(raw, outcomes, train, test, seed, config)
    if not np.isfinite(repeated).all():
        raise AssertionError("Incomplete unweighted logistic OOF predictions.")
    mean_probability = repeated.mean(axis=0)
    np.savez_compressed(
        output_dir / "calibration_sensitivity_unweighted_lr_oof.npz",
        y_true=outcomes,
        repeated_oof=repeated,
        mean_oof=mean_probability,
    )
    pd.DataFrame([{"model": "BR-LR-unweighted", **aggregate_metrics(outcomes, mean_probability)}]).to_csv(
        output_dir / "calibration_sensitivity_unweighted_lr_metrics.csv", index=False
    )
    per_label = per_label_metrics(outcomes, mean_probability)
    per_label.insert(0, "model", "BR-LR-unweighted")
    per_label.to_csv(output_dir / "calibration_sensitivity_unweighted_lr_per_label.csv", index=False)
    reliability_table(outcomes, mean_probability, "BR-LR-unweighted").to_csv(
        output_dir / "calibration_sensitivity_unweighted_lr_bins.csv", index=False
    )
    intervals = calibration_bootstrap_intervals(
        outcomes,
        mean_probability,
        n_bootstrap=config["statistics"]["patient_bootstrap"],
        seed=config["seed"] + 777_000,
        n_jobs=config["statistics"].get("calibration_bootstrap_jobs", 1),
    )
    intervals.insert(0, "model", "BR-LR-unweighted")
    intervals.to_csv(output_dir / "calibration_sensitivity_unweighted_lr_intervals.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/comlc_mi_benchmark.json"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
