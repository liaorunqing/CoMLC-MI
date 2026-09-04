"""Per-label calibration analysis for saved test-set probabilities."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss

from config import LABEL_COLS, LABEL_SHORT, ORIG_PROCESSED_DIR, OUTPUT_DIR


def adaptive_ece(y_true, probabilities, n_bins=10):
    """Expected calibration error using approximately equal-count bins."""
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(probabilities)
    bins = np.array_split(order, min(n_bins, len(order)))
    ece = 0.0
    records = []
    for bin_index, indices in enumerate(bins):
        if len(indices) == 0:
            continue
        confidence = float(probabilities[indices].mean())
        observed = float(y_true[indices].mean())
        weight = len(indices) / len(y_true)
        ece += weight * abs(observed - confidence)
        records.append({
            "bin": bin_index + 1,
            "n": int(len(indices)),
            "mean_probability": confidence,
            "observed_frequency": observed,
        })
    return float(ece), records


def calibration_intercept_slope(y_true, probabilities):
    """Joint logistic recalibration intercept and slope on the logit scale."""
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if np.unique(y_true).size < 2:
        return np.nan, np.nan
    logits = np.log(probabilities / (1.0 - probabilities))

    def objective(parameters):
        recalibrated = 1.0 / (1.0 + np.exp(-(parameters[0] + parameters[1] * logits)))
        return log_loss(y_true, recalibrated, labels=[0, 1])

    fit = minimize(objective, x0=np.array([0.0, 1.0]), method="BFGS")
    if not fit.success and not np.isfinite(fit.fun):
        return np.nan, np.nan
    return float(fit.x[0]), float(fit.x[1])


def evaluate_calibration(y_true, probabilities, model_name):
    if y_true.shape != probabilities.shape:
        raise ValueError(f"Shape mismatch for {model_name}: {y_true.shape} vs {probabilities.shape}.")
    rows = []
    reliability = {}
    for label_index, label in enumerate(LABEL_COLS):
        observed = y_true[:, label_index]
        predicted = probabilities[:, label_index]
        intercept, slope = calibration_intercept_slope(observed, predicted)
        ece, bins = adaptive_ece(observed, predicted)
        rows.append({
            "model": model_name,
            "label": label,
            "short_label": LABEL_SHORT[label_index],
            "n": len(observed),
            "n_positive": int(observed.sum()),
            "prevalence": float(observed.mean()),
            "mean_predicted_probability": float(predicted.mean()),
            "brier_score": float(brier_score_loss(observed, predicted)),
            "adaptive_ece_10": ece,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        })
        reliability[f"{model_name}:{label}"] = bins
    return rows, reliability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["tabpfn", "catboost", "ens2_wt"],
        help="Prediction stems corresponding to PUB_<stem>_preds.npy.",
    )
    args = parser.parse_args()

    y_test = pd.read_csv(os.path.join(ORIG_PROCESSED_DIR, "y_test.csv")).to_numpy(dtype=int)
    all_rows = []
    all_reliability = {}
    missing = []
    for model_name in args.models:
        path = os.path.join(OUTPUT_DIR, f"PUB_{model_name}_preds.npy")
        if not os.path.exists(path):
            missing.append(path)
            continue
        probabilities = np.load(path)
        rows, reliability = evaluate_calibration(y_test, probabilities, model_name)
        all_rows.extend(rows)
        all_reliability.update(reliability)

    if missing:
        raise FileNotFoundError(
            "Calibration requires saved out-of-sample probabilities. Missing:\n- "
            + "\n- ".join(missing)
            + "\nRun src/run_all.py before calibration_analysis.py."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(os.path.join(OUTPUT_DIR, "CALIBRATION_RESULTS.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "CALIBRATION_RELIABILITY_BINS.json"), "w") as handle:
        json.dump(all_reliability, handle, indent=2)
    print(pd.DataFrame(all_rows).to_string(index=False))


if __name__ == "__main__":
    main()
