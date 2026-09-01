"""Metrics, calibration, bootstrap intervals, and paired label tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from .revision_features import LABEL_COLS
from .statistical_utils import (
    benjamini_hochberg,
    delong_roc_test,
    permutation_auc_test,
)


EPSILON = 1e-6


def quantile_bins(probability: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Assign equal-frequency bins with deterministic tie handling."""
    probability = np.asarray(probability, dtype=float)
    order = np.argsort(probability, kind="mergesort")
    bins = np.empty(len(probability), dtype=np.int16)
    for bin_id, indices in enumerate(np.array_split(order, n_bins)):
        bins[indices] = bin_id
    return bins


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10
) -> float:
    bins = quantile_bins(probability, n_bins)
    return expected_calibration_error_from_bins(y_true, probability, bins, n_bins)


def expected_calibration_error_from_bins(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Evaluate ECE for a fixed bin assignment.

    Fixed assignments are used inside the bootstrap so that resampling
    estimates uncertainty for the original decile-based statistic instead of
    redefining risk deciles after duplicate observations are introduced.
    """
    y_true = np.asarray(y_true)
    probability = np.asarray(probability, dtype=float)
    bins = np.asarray(bins, dtype=np.int16)
    if not (len(y_true) == len(probability) == len(bins)):
        raise ValueError("y_true, probability, and bins must have the same length.")
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bins == bin_id
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(ece)


def calibration_intercept_slope(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Fit outcome ~ logit(probability) without recalibrating predictions."""
    y_true = np.asarray(y_true, dtype=int)
    if np.unique(y_true).size < 2:
        return np.nan, np.nan
    eta = logit(np.clip(np.asarray(probability, dtype=float), EPSILON, 1 - EPSILON))
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
        fit_intercept=True,
    )
    try:
        model.fit(eta.reshape(-1, 1), y_true)
    except Exception:
        return np.nan, np.nan
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def per_label_metrics(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    records = []
    for label_index, label in enumerate(LABEL_COLS):
        truth = y_true[:, label_index].astype(int)
        pred = np.clip(probability[:, label_index].astype(float), EPSILON, 1 - EPSILON)
        intercept, slope = calibration_intercept_slope(truth, pred)
        record = {
            "label": label,
            "n": len(truth),
            "positive_n": int(truth.sum()),
            "positive_rate": float(truth.mean()),
            "auroc": roc_auc_score(truth, pred) if np.unique(truth).size == 2 else np.nan,
            "auprc": average_precision_score(truth, pred) if truth.sum() else np.nan,
            "brier": brier_score_loss(truth, pred),
            "ece_10": expected_calibration_error(truth, pred, n_bins=10),
            "macro_f1_component": f1_score(truth, pred >= 0.5, zero_division=0),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "observed_expected_ratio": float(truth.mean() / pred.mean()),
            "calibration_stability": "unstable" if truth.sum() < 30 else "stable",
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    label_auc = []
    label_auprc = []
    label_ece = []
    label_f1 = []
    for label_index in range(y_true.shape[1]):
        truth = y_true[:, label_index]
        pred = probability[:, label_index]
        label_auc.append(roc_auc_score(truth, pred))
        label_auprc.append(average_precision_score(truth, pred))
        label_ece.append(expected_calibration_error(truth, pred, n_bins=10))
        label_f1.append(f1_score(truth, pred >= 0.5, zero_division=0))
    return {
        "macro_auroc": float(np.mean(label_auc)),
        "micro_auroc": float(roc_auc_score(y_true.ravel(), probability.ravel())),
        "macro_auprc": float(np.mean(label_auprc)),
        "brier": float(np.mean((y_true - probability) ** 2)),
        "ece_10": float(np.mean(label_ece)),
        "macro_f1_0_5": float(np.mean(label_f1)),
    }


def reliability_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    model_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    records = []
    for label_index, label in enumerate(LABEL_COLS):
        truth = y_true[:, label_index]
        pred = probability[:, label_index]
        bins = quantile_bins(pred, n_bins)
        for bin_id in range(n_bins):
            mask = bins == bin_id
            records.append(
                {
                    "model": model_name,
                    "label": label,
                    "bin": bin_id + 1,
                    "n": int(mask.sum()),
                    "mean_predicted": float(pred[mask].mean()),
                    "observed_rate": float(truth[mask].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def paired_patient_bootstrap(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    n_bootstrap: int = 2_000,
    seed: int = 42,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Ordinary patient-level bootstrap using one resample for every model."""
    y_true = np.asarray(y_true, dtype=np.int8)
    n_patients = len(y_true)
    rng = np.random.default_rng(seed)
    model_names = list(predictions)
    metric_names = list(aggregate_metrics(y_true, predictions[model_names[0]]))
    valid_samples = []
    attempts = 0
    max_attempts = n_bootstrap * 20
    while len(valid_samples) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sampled = rng.integers(0, n_patients, size=n_patients)
        sampled_y = y_true[sampled]
        # Retain a common, fully evaluable draw for all paired estimates.
        if np.any(sampled_y.sum(axis=0) == 0) or np.any(sampled_y.sum(axis=0) == n_patients):
            continue
        valid_samples.append(sampled)
    if len(valid_samples) != n_bootstrap:
        raise RuntimeError(f"Only {len(valid_samples)}/{n_bootstrap} valid paired bootstrap draws were obtained.")
    sampled_indices = np.asarray(valid_samples, dtype=np.int32)

    reference_bins = {
        name: np.stack(
            [quantile_bins(predictions[name][:, label_index], 10) for label_index in range(y_true.shape[1])]
        )
        for name in model_names
    }

    def evaluate_model(name: str, probability: np.ndarray) -> np.ndarray:
        result = np.empty((n_bootstrap, len(metric_names)), dtype=np.float64)
        for bootstrap_index, sampled in enumerate(sampled_indices):
            metrics = aggregate_metrics(y_true[sampled], probability[sampled])
            metrics["ece_10"] = float(
                np.mean(
                    [
                        expected_calibration_error_from_bins(
                            y_true[sampled, label_index],
                            probability[sampled, label_index],
                            reference_bins[name][label_index, sampled],
                            10,
                        )
                        for label_index in range(y_true.shape[1])
                    ]
                )
            )
            result[bootstrap_index] = [metrics[key] for key in metric_names]
        return result

    evaluated = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_model)(name, predictions[name]) for name in model_names
    )
    arrays = dict(zip(model_names, evaluated))

    rows = []
    for name in model_names:
        point = aggregate_metrics(y_true, predictions[name])
        for column, metric in enumerate(metric_names):
            if metric == "ece_10":
                # The absolute bin error is upward biased after resampling even
                # with fixed bins.  A bootstrap-SE interval targets uncertainty
                # around the reported plug-in ECE and avoids presenting a
                # percentile interval wholly above its own point estimate.
                standard_error = arrays[name][:, column].std(ddof=1)
                low = max(0.0, point[metric] - 1.959963984540054 * standard_error)
                high = point[metric] + 1.959963984540054 * standard_error
                interval_method = "bootstrap standard-error normal"
            else:
                low, high = np.quantile(arrays[name][:, column], [0.025, 0.975])
                interval_method = "bootstrap percentile"
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_bootstrap": n_bootstrap,
                    "interval_method": interval_method,
                }
            )
    return pd.DataFrame(rows), arrays


def paired_metric_difference(
    arrays: dict[str, np.ndarray],
    metric_names: list[str],
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    difference = arrays[model_a] - arrays[model_b]
    rows = []
    for column, metric in enumerate(metric_names):
        low, high = np.quantile(difference[:, column], [0.025, 0.975])
        rows.append(
            {
                "contrast": f"{model_a} - {model_b}",
                "metric": metric,
                "mean_bootstrap_difference": float(difference[:, column].mean()),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def paired_label_auc_tests(
    y_true: np.ndarray,
    model_a_probability: np.ndarray,
    model_b_probability: np.ndarray,
    low_event_threshold: int = 30,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """DeLong or randomization test per label, then BH across 12 labels."""
    rows = []
    for label_index, label in enumerate(LABEL_COLS):
        truth = y_true[:, label_index].astype(int)
        pred_a = model_a_probability[:, label_index]
        pred_b = model_b_probability[:, label_index]
        positive_n = int(truth.sum())
        if np.unique(truth).size < 2:
            # AUC is undefined when a resampled/evaluation subset contains only
            # one class.  Retain the label in the 12-outcome multiplicity family
            # with a conservative p value of one and expose the estimand as NaN.
            auc_a = np.nan
            auc_b = np.nan
            difference = np.nan
            p_value = 1.0
            test = "not estimable"
            stability = "not estimable"
        elif positive_n < low_event_threshold:
            auc_a = roc_auc_score(truth, pred_a)
            auc_b = roc_auc_score(truth, pred_b)
            difference, p_value = permutation_auc_test(
                truth,
                pred_a,
                pred_b,
                n_permutations=n_permutations,
                random_state=seed + label_index,
            )
            test = "paired permutation"
            stability = "unstable"
        else:
            _, p_value, auc_a, auc_b = delong_roc_test(truth, pred_a, pred_b)
            difference = auc_a - auc_b
            test = "paired DeLong"
            stability = "stable"
        if not np.isfinite(p_value):
            # Zero paired-score variance can make the asymptotic statistic
            # undefined even when both classes are present.  A p value of one
            # is the conservative no-evidence result for multiplicity control.
            p_value = 1.0
        rows.append(
            {
                "label": label,
                "positive_n": positive_n,
                "test": test,
                "auc_model_a": auc_a,
                "auc_model_b": auc_b,
                "difference": difference,
                "p_value": p_value,
                "stability": stability,
            }
        )
    frame = pd.DataFrame(rows)
    reject, adjusted = benjamini_hochberg(frame["p_value"].to_numpy())
    frame["p_value_bh"] = adjusted
    frame["reject_bh_0_05"] = reject
    return frame


def per_label_auc_difference_bootstrap(
    y_true: np.ndarray,
    model_a_probability: np.ndarray,
    model_b_probability: np.ndarray,
    n_bootstrap: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Ordinary patient-level paired bootstrap for all 12 AUROC contrasts."""
    y_true = np.asarray(y_true, dtype=np.int8)
    rng = np.random.default_rng(seed)
    differences = np.full((n_bootstrap, y_true.shape[1]), np.nan, dtype=np.float64)
    completed = 0
    attempts = 0
    while completed < n_bootstrap and attempts < n_bootstrap * 20:
        attempts += 1
        sampled = rng.integers(0, len(y_true), size=len(y_true))
        sampled_y = y_true[sampled]
        if np.any(sampled_y.sum(axis=0) == 0) or np.any(sampled_y.sum(axis=0) == len(y_true)):
            continue
        for label_index in range(y_true.shape[1]):
            truth = sampled_y[:, label_index]
            differences[completed, label_index] = (
                roc_auc_score(truth, model_a_probability[sampled, label_index])
                - roc_auc_score(truth, model_b_probability[sampled, label_index])
            )
        completed += 1
    if completed != n_bootstrap:
        raise RuntimeError("Unable to obtain the requested valid paired label bootstraps.")
    rows = []
    for label_index, label in enumerate(LABEL_COLS):
        point = (
            roc_auc_score(y_true[:, label_index], model_a_probability[:, label_index])
            - roc_auc_score(y_true[:, label_index], model_b_probability[:, label_index])
        )
        low, high = np.quantile(differences[:, label_index], [0.025, 0.975])
        rows.append(
            {
                "label": label,
                "positive_n": int(y_true[:, label_index].sum()),
                "difference": float(point),
                "ci_low": float(low),
                "ci_high": float(high),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def _calibration_label_bootstrap(
    truth: np.ndarray,
    pred: np.ndarray,
    label: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    positive_n = int(truth.sum())
    point_intercept, point_slope = calibration_intercept_slope(truth, pred)
    reference_bins = quantile_bins(pred, 10)
    point = {
        "brier": brier_score_loss(truth, pred),
        "ece_10": expected_calibration_error(truth, pred),
        "calibration_intercept": point_intercept,
        "calibration_slope": point_slope,
    }
    samples = {name: [] for name in point}
    attempts = 0
    while len(samples["brier"]) < n_bootstrap and attempts < n_bootstrap * 20:
        attempts += 1
        selected = rng.integers(0, len(truth), size=len(truth))
        sampled_truth = truth[selected]
        if np.unique(sampled_truth).size < 2:
            continue
        sampled_pred = pred[selected]
        intercept, slope = calibration_intercept_slope(sampled_truth, sampled_pred)
        values = {
            "brier": brier_score_loss(sampled_truth, sampled_pred),
            "ece_10": expected_calibration_error_from_bins(
                sampled_truth, sampled_pred, reference_bins[selected], 10
            ),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }
        if not all(np.isfinite(value) for value in values.values()):
            continue
        for name, value in values.items():
            samples[name].append(value)
    rows = []
    for metric, estimate in point.items():
        if len(samples[metric]) != n_bootstrap:
            low = high = np.nan
        else:
            low, high = np.quantile(samples[metric], [0.025, 0.975])
        rows.append(
            {
                "label": label,
                "positive_n": positive_n,
                "stability": "unstable" if positive_n < 30 else "stable",
                "metric": metric,
                "estimate": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
                "n_bootstrap": n_bootstrap,
            }
        )
    return rows


def calibration_bootstrap_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bootstrap: int = 2_000,
    seed: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Patient-bootstrap intervals for per-label calibration quantities."""
    jobs = [
        delayed(_calibration_label_bootstrap)(
            y_true[:, label_index],
            probability[:, label_index],
            label,
            n_bootstrap,
            seed + 100_003 * label_index,
        )
        for label_index, label in enumerate(LABEL_COLS)
    ]
    nested = Parallel(n_jobs=n_jobs)(jobs)
    return pd.DataFrame([row for label_rows in nested for row in label_rows])
