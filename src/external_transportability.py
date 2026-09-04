"""Cross-cohort mortality transportability stress test.

This module evaluates source-fitted reduced common-feature counterparts of the
TabPFN binary-relevance component and logistic regression.  It does not perform
external validation of the complete 12-label benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier


SEED = 42
STRICT_FEATURES = ["age_years", "male", "prior_mi", "prior_hf"]
ENDPOINTS = ("death_30d", "death_7d")
MODEL_ORDER = ("TabPFN-3", "Logistic regression")


@dataclass(frozen=True)
class HarmonizedData:
    source_x: pd.DataFrame
    source_y: pd.Series
    external_x: pd.DataFrame
    external_endpoints: dict[str, pd.Series]
    external_complete_case: pd.Series


def _binary_map(series: pd.Series) -> pd.Series:
    """Map registry Yes/No values while preserving unknown entries."""
    return series.map({"Yes": 1.0, "Not": 0.0})


def harmonize_strict(uci: pd.DataFrame, registry: pd.DataFrame) -> HarmonizedData:
    """Create the revision-locked strict four-feature source/external datasets."""
    registry = registry.copy()
    registry.columns = registry.columns.astype(str).str.strip()
    first = registry.loc[registry["Event number"].eq(1)].copy()

    source_x = pd.DataFrame(
        {
            "age_years": pd.to_numeric(uci["AGE"], errors="coerce"),
            "male": pd.to_numeric(uci["SEX"], errors="coerce"),
            "prior_mi": uci["INF_ANAM"].gt(0).where(uci["INF_ANAM"].notna()),
            "prior_hf": uci["ZSN_A"].gt(0).where(uci["ZSN_A"].notna()),
        }
    ).astype(float)
    source_y = uci["LET_IS"].gt(0).astype(int)

    external_x = pd.DataFrame(
        {
            "age_years": pd.to_numeric(first["Age at admission"], errors="coerce"),
            "male": first["Gender"].map({"Man": 1.0, "Woman": 0.0}),
            "prior_mi": _binary_map(first["History of myocardial infarction"]),
            "prior_hf": _binary_map(first["History of heart failure"]),
        },
        index=first.index,
    ).astype(float)

    admission = pd.to_datetime(first["First hospital admission of an event"], errors="coerce")
    death = pd.to_datetime(first["Date of death"], errors="coerce")
    days = (death.dt.normalize() - admission.dt.normalize()).dt.days
    if admission.isna().any():
        raise ValueError("First-event cohort contains missing admission dates.")
    if days.dropna().lt(0).any():
        raise ValueError("Death date precedes admission in the first-event cohort.")

    endpoints = {
        "death_7d": days.between(0, 7).astype(int),
        "death_30d": days.between(0, 30).astype(int),
    }
    complete = external_x.notna().all(axis=1)
    return HarmonizedData(source_x, source_y, external_x, endpoints, complete)


def _tabpfn(n_estimators: int = 8, n_preprocessing_jobs: int = 1) -> TabPFNClassifier:
    return TabPFNClassifier(
        device="cpu",
        random_state=SEED,
        ignore_pretraining_limits=True,
        n_estimators=n_estimators,
        n_preprocessing_jobs=n_preprocessing_jobs,
        show_progress_bar=False,
    )


def fit_predictions(
    data: HarmonizedData,
    *,
    source_cv_folds: int = 5,
    logistic_max_iter: int = 5000,
    tabpfn_n_estimators: int = 8,
    tabpfn_n_preprocessing_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """Fit only on Krasnoyarsk data and return OOF and external predictions."""
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    x_source = data.source_x[STRICT_FEATURES]
    x_external = data.external_x[STRICT_FEATURES]
    y = data.source_y.to_numpy(dtype=int)
    cv = StratifiedKFold(source_cv_folds, shuffle=True, random_state=SEED)

    logistic_oof = np.zeros(len(y), dtype=float)
    tabpfn_oof = np.zeros(len(y), dtype=float)
    for train, validation in cv.split(x_source, y):
        logistic = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=logistic_max_iter, random_state=SEED),
        )
        logistic.fit(x_source.iloc[train], y[train])
        logistic_oof[validation] = logistic.predict_proba(x_source.iloc[validation])[:, 1]

        imputer = SimpleImputer(strategy="median")
        train_x = imputer.fit_transform(x_source.iloc[train])
        validation_x = imputer.transform(x_source.iloc[validation])
        tabpfn = _tabpfn(tabpfn_n_estimators, tabpfn_n_preprocessing_jobs)
        tabpfn.fit(train_x, y[train])
        tabpfn_oof[validation] = tabpfn.predict_proba(validation_x)[:, 1]

    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=logistic_max_iter, random_state=SEED),
    )
    logistic.fit(x_source, y)
    logistic_external = logistic.predict_proba(x_external)[:, 1]

    imputer = SimpleImputer(strategy="median")
    source_imputed = imputer.fit_transform(x_source)
    external_imputed = imputer.transform(x_external)
    tabpfn = _tabpfn(tabpfn_n_estimators, tabpfn_n_preprocessing_jobs)
    tabpfn.fit(source_imputed, y)
    tabpfn_external = tabpfn.predict_proba(external_imputed)[:, 1]

    return {
        "source_logistic": logistic_oof,
        "source_tabpfn": tabpfn_oof,
        "external_logistic": logistic_external,
        "external_tabpfn": tabpfn_external,
    }


def calibration_bins(p: np.ndarray, n_bins: int = 10) -> np.ndarray:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.zeros(len(p), dtype=int)
    return np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)


def _weighted_auc(
    y: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
    order: np.ndarray | None = None,
    starts: np.ndarray | None = None,
) -> float:
    if order is None:
        order = np.argsort(p, kind="mergesort")
    ys, ps, ws = y[order], p[order], weights[order]
    positive = ws * ys
    negative = ws * (1 - ys)
    total_positive, total_negative = positive.sum(), negative.sum()
    if total_positive == 0 or total_negative == 0:
        return np.nan
    if starts is None:
        starts = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
    group_positive = np.add.reduceat(positive, starts)
    group_negative = np.add.reduceat(negative, starts)
    negative_before = np.cumsum(group_negative) - group_negative
    concordant = np.sum(group_positive * (negative_before + 0.5 * group_negative))
    return float(concordant / (total_positive * total_negative))


def _weighted_average_precision(
    y: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
    order: np.ndarray | None = None,
    starts: np.ndarray | None = None,
) -> float:
    if order is None:
        order = np.argsort(-p, kind="mergesort")
    ys, ps, ws = y[order], p[order], weights[order]
    positive = ws * ys
    total_positive = positive.sum()
    if total_positive == 0:
        return np.nan
    if starts is None:
        starts = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
    group_positive = np.add.reduceat(positive, starts)
    group_total = np.add.reduceat(ws, starts)
    retained = group_total > 0
    group_positive = group_positive[retained]
    group_total = group_total[retained]
    cumulative_positive = np.cumsum(group_positive)
    cumulative_total = np.cumsum(group_total)
    precision = cumulative_positive / cumulative_total
    return float(np.sum(precision * group_positive) / total_positive)


def _weighted_calibration(
    y: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
    eta: np.ndarray | None = None,
) -> tuple[float, float]:
    if eta is None:
        eps = 1e-6
        eta = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(30):
        fitted = expit(beta[0] + beta[1] * eta)
        variance = np.maximum(fitted * (1 - fitted), 1e-12)
        residual = weights * (y - fitted)
        gradient = np.array([residual.sum(), np.sum(residual * eta)])
        h00 = np.sum(weights * variance)
        h01 = np.sum(weights * variance * eta)
        h11 = np.sum(weights * variance * eta * eta)
        information = np.array([[h00, h01], [h01, h11]])
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def _weighted_ece(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray, bins: np.ndarray
) -> float:
    total = weights.sum()
    value = 0.0
    for bin_id in np.unique(bins):
        mask = bins == bin_id
        bin_weight = weights[mask].sum()
        if bin_weight == 0:
            continue
        observed = np.sum(weights[mask] * y[mask]) / bin_weight
        predicted = np.sum(weights[mask] * p[mask]) / bin_weight
        value += (bin_weight / total) * abs(observed - predicted)
    return float(value)


def weighted_metrics(
    y: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    cache: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    cache = cache or {}
    total = weights.sum()
    observed = float(np.sum(weights * y) / total)
    predicted = float(np.sum(weights * p) / total)
    intercept, slope = _weighted_calibration(y, p, weights, cache.get("eta"))
    return {
        "auc": _weighted_auc(y, p, weights, cache.get("auc_order"), cache.get("auc_starts")),
        "auprc": _weighted_average_precision(
            y, p, weights, cache.get("ap_order"), cache.get("ap_starts")
        ),
        "brier": float(np.sum(weights * (p - y) ** 2) / total),
        "ece": _weighted_ece(y, p, weights, bins),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_predicted": predicted,
        "observed_rate": observed,
        "oe_ratio": observed / predicted,
    }


def weighted_metrics_batch(
    y: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Vectorized metrics for a batch of bootstrap multinomial weights."""
    total = weights.sum(axis=1)
    observed = weights @ y / total
    predicted = weights @ p / total
    brier = weights @ ((p - y) ** 2) / total

    ece = np.zeros(len(weights), dtype=float)
    for bin_id in np.unique(bins):
        mask = bins == bin_id
        bin_weight = weights[:, mask].sum(axis=1)
        observed_numerator = weights[:, mask] @ y[mask]
        predicted_numerator = weights[:, mask] @ p[mask]
        valid = bin_weight > 0
        ece[valid] += (bin_weight[valid] / total[valid]) * np.abs(
            observed_numerator[valid] / bin_weight[valid]
            - predicted_numerator[valid] / bin_weight[valid]
        )

    auc_order = cache["auc_order"]
    auc_weights = weights[:, auc_order]
    auc_y = y[auc_order]
    positive = auc_weights * auc_y
    negative = auc_weights * (1 - auc_y)
    group_positive = np.add.reduceat(positive, cache["auc_starts"], axis=1)
    group_negative = np.add.reduceat(negative, cache["auc_starts"], axis=1)
    negative_before = np.cumsum(group_negative, axis=1) - group_negative
    total_positive = group_positive.sum(axis=1)
    total_negative = group_negative.sum(axis=1)
    auc_denominator = total_positive * total_negative
    auc = np.divide(
        np.sum(group_positive * (negative_before + 0.5 * group_negative), axis=1),
        auc_denominator,
        out=np.full(len(weights), np.nan),
        where=auc_denominator > 0,
    )

    ap_order = cache["ap_order"]
    ap_weights = weights[:, ap_order]
    ap_y = y[ap_order]
    ap_positive = ap_weights * ap_y
    group_positive = np.add.reduceat(ap_positive, cache["ap_starts"], axis=1)
    group_total = np.add.reduceat(ap_weights, cache["ap_starts"], axis=1)
    cumulative_positive = np.cumsum(group_positive, axis=1)
    cumulative_total = np.cumsum(group_total, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive, dtype=float),
        where=cumulative_total > 0,
    )
    ap_denominator = group_positive.sum(axis=1)
    auprc = np.divide(
        np.sum(precision * group_positive, axis=1),
        ap_denominator,
        out=np.full(len(weights), np.nan),
        where=ap_denominator > 0,
    )

    eta = cache["eta"]
    beta0 = np.zeros(len(weights), dtype=float)
    beta1 = np.ones(len(weights), dtype=float)
    for _ in range(30):
        fitted = expit(beta0[:, None] + beta1[:, None] * eta[None, :])
        variance = np.maximum(fitted * (1 - fitted), 1e-12)
        residual = weights * (y[None, :] - fitted)
        gradient0 = residual.sum(axis=1)
        gradient1 = (residual * eta[None, :]).sum(axis=1)
        working = weights * variance
        h00 = working.sum(axis=1)
        h01 = (working * eta[None, :]).sum(axis=1)
        h11 = (working * eta[None, :] ** 2).sum(axis=1)
        determinant = h00 * h11 - h01 * h01
        valid = np.abs(determinant) > 1e-14
        step0 = np.zeros(len(weights), dtype=float)
        step1 = np.zeros(len(weights), dtype=float)
        step0[valid] = (gradient0[valid] * h11[valid] - gradient1[valid] * h01[valid]) / determinant[valid]
        step1[valid] = (h00[valid] * gradient1[valid] - h01[valid] * gradient0[valid]) / determinant[valid]
        beta0 += step0
        beta1 += step1
        if max(np.max(np.abs(step0)), np.max(np.abs(step1))) < 1e-8:
            break

    return np.column_stack(
        [auc, auprc, brier, ece, beta0, beta1, predicted, observed, observed / predicted]
    )


def summarize_with_bootstrap(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Return model summaries, paired differences, and bootstrap arrays."""
    y = np.asarray(y, dtype=int)
    predictions = {key: np.asarray(value, dtype=float) for key, value in predictions.items()}
    bins = {key: calibration_bins(value) for key, value in predictions.items()}
    caches = {}
    for model, pred in predictions.items():
        auc_order = np.argsort(pred, kind="mergesort")
        ap_order = np.argsort(-pred, kind="mergesort")
        eps = 1e-6
        caches[model] = {
            "auc_order": auc_order,
            "auc_starts": np.r_[0, np.flatnonzero(np.diff(pred[auc_order])) + 1],
            "ap_order": ap_order,
            "ap_starts": np.r_[0, np.flatnonzero(np.diff(pred[ap_order])) + 1],
            "eta": np.log(np.clip(pred, eps, 1 - eps) / np.clip(1 - pred, eps, 1 - eps)),
        }
    unit_weights = np.ones(len(y), dtype=float)
    point = {
        model: weighted_metrics(y, pred, unit_weights, bins[model], caches[model])
        for model, pred in predictions.items()
    }
    metric_names = list(next(iter(point.values())).keys())
    boot = {model: np.empty((n_boot, len(metric_names)), dtype=float) for model in predictions}
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(y), 1 / len(y), dtype=float)
    batch_size = 20
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        weights = rng.multinomial(len(y), probabilities, size=stop - start).astype(float)
        for model, pred in predictions.items():
            boot[model][start:stop] = weighted_metrics_batch(
                y, pred, weights, bins[model], caches[model]
            )

    rows = []
    for model in predictions:
        row = {"model": model, "n": len(y), "events": int(y.sum())}
        for index, metric in enumerate(metric_names):
            valid = boot[model][:, index]
            valid = valid[np.isfinite(valid)]
            row[metric] = point[model][metric]
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.quantile(valid, [0.025, 0.975])
        rows.append(row)

    difference_rows = []
    if set(MODEL_ORDER).issubset(predictions):
        primary, comparator = MODEL_ORDER
        row = {"contrast": f"{primary} minus {comparator}", "n": len(y), "events": int(y.sum())}
        for index, metric in enumerate(metric_names):
            delta = boot[primary][:, index] - boot[comparator][:, index]
            delta = delta[np.isfinite(delta)]
            row[f"delta_{metric}"] = point[primary][metric] - point[comparator][metric]
            row[f"delta_{metric}_ci_low"], row[f"delta_{metric}_ci_high"] = np.quantile(
                delta, [0.025, 0.975]
            )
        difference_rows.append(row)
    arrays = {f"{model}__{metric}": values[:, index] for model, values in boot.items() for index, metric in enumerate(metric_names)}
    return pd.DataFrame(rows), pd.DataFrame(difference_rows), arrays


def reliability_table(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    cohort: str,
    endpoint: str,
    subset: str,
) -> pd.DataFrame:
    rows = []
    for model, p in predictions.items():
        bins = calibration_bins(p)
        for bin_id in np.unique(bins):
            mask = bins == bin_id
            events = int(y[mask].sum())
            rows.append(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "subset": subset,
                    "model": model,
                    "bin": int(bin_id + 1),
                    "n": int(mask.sum()),
                    "events": events,
                    "mean_predicted": float(p[mask].mean()),
                    "observed_rate": float(y[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def cohort_summary(data: HarmonizedData) -> pd.DataFrame:
    rows = []
    for feature in STRICT_FEATURES:
        source = data.source_x[feature]
        external = data.external_x[feature]
        pooled = np.sqrt((source.var(ddof=1) + external.var(ddof=1)) / 2)
        rows.append(
            {
                "feature": feature,
                "source_mean": float(source.mean()),
                "source_missing": int(source.isna().sum()),
                "external_mean": float(external.mean()),
                "external_missing": int(external.isna().sum()),
                "standardized_mean_difference": float((external.mean() - source.mean()) / pooled),
            }
        )
    return pd.DataFrame(rows)


def run(
    input_path: Path,
    uci_path: Path,
    output_dir: Path,
    n_boot: int,
    reuse_predictions: bool = False,
    model_config: dict | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = pd.read_excel(input_path)
    uci = pd.read_csv(uci_path)
    data = harmonize_strict(uci, registry)
    if len(data.external_x) != 29_596:
        raise ValueError(f"Expected 29,596 first events, found {len(data.external_x):,}.")
    if int(data.external_endpoints["death_30d"].sum()) != 3_888:
        raise ValueError("Unexpected 30-day death count.")
    if int(data.external_endpoints["death_7d"].sum()) != 2_408:
        raise ValueError("Unexpected 7-day death count.")
    if int(data.external_complete_case.sum()) != 28_477:
        raise ValueError("Unexpected complete-case count.")

    started = time.time()
    prediction_path = output_dir / "external_transportability_predictions.npz"
    metadata_path = output_dir / "external_transportability_metadata.json"
    external_config = model_config or {}
    model_specification = {
        "source_cv_folds": int(external_config.get("source_cv_folds", 5)),
        "logistic_max_iter": int(external_config.get("logistic_max_iter", 5000)),
        "tabpfn_n_estimators": int(external_config.get("tabpfn_n_estimators", 8)),
        "tabpfn_n_preprocessing_jobs": int(external_config.get("tabpfn_n_preprocessing_jobs", 1)),
    }
    reusable = False
    if reuse_predictions and prediction_path.exists() and metadata_path.exists():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = all(previous_metadata.get(key) == value for key, value in model_specification.items())
    if reusable:
        stored = np.load(prediction_path)
        fitted = {
            "source_logistic": stored["source_logistic"],
            "source_tabpfn": stored["source_tabpfn"],
            "external_logistic": stored["external_logistic"],
            "external_tabpfn": stored["external_tabpfn"],
        }
    else:
        fitted = fit_predictions(
            data,
            **model_specification,
        )
    np.savez_compressed(
        prediction_path,
        source_y=data.source_y.to_numpy(dtype=int),
        source_logistic=fitted["source_logistic"],
        source_tabpfn=fitted["source_tabpfn"],
        external_death_30d=data.external_endpoints["death_30d"].to_numpy(dtype=int),
        external_death_7d=data.external_endpoints["death_7d"].to_numpy(dtype=int),
        external_complete_case=data.external_complete_case.to_numpy(dtype=bool),
        external_logistic=fitted["external_logistic"],
        external_tabpfn=fitted["external_tabpfn"],
    )

    all_summaries = []
    all_differences = []
    all_reliability = []
    bootstrap_arrays = {}
    analyses = [
        (
            "Krasnoyarsk",
            "in-hospital fatal outcome",
            "5-fold out-of-fold",
            data.source_y.to_numpy(dtype=int),
            np.ones(len(data.source_y), dtype=bool),
            {"TabPFN-3": fitted["source_tabpfn"], "Logistic regression": fitted["source_logistic"]},
        )
    ]
    for endpoint in ENDPOINTS:
        outcome = data.external_endpoints[endpoint].to_numpy(dtype=int)
        predictions = {"TabPFN-3": fitted["external_tabpfn"], "Logistic regression": fitted["external_logistic"]}
        analyses.extend(
            [
                ("Hungarian AMI registry", endpoint, "all first events", outcome, np.ones(len(outcome), dtype=bool), predictions),
                ("Hungarian AMI registry", endpoint, "complete case", outcome, data.external_complete_case.to_numpy(dtype=bool), predictions),
            ]
        )

    for analysis_index, (cohort, endpoint, subset, outcome, mask, predictions) in enumerate(analyses):
        selected_y = outcome[mask]
        selected_predictions = {model: values[mask] for model, values in predictions.items()}
        summary, differences, arrays = summarize_with_bootstrap(
            selected_y, selected_predictions, n_boot=n_boot, seed=SEED + analysis_index * 1000
        )
        for frame in (summary, differences):
            frame.insert(0, "subset", subset)
            frame.insert(0, "endpoint", endpoint)
            frame.insert(0, "cohort", cohort)
        all_summaries.append(summary)
        all_differences.append(differences)
        all_reliability.append(reliability_table(selected_y, selected_predictions, cohort, endpoint, subset))
        prefix = f"analysis_{analysis_index}"
        for key, values in arrays.items():
            bootstrap_arrays[f"{prefix}__{key}"] = values

    results = pd.concat(all_summaries, ignore_index=True)
    differences = pd.concat(all_differences, ignore_index=True)
    reliability = pd.concat(all_reliability, ignore_index=True)
    results.to_csv(output_dir / "external_transportability_results.csv", index=False)
    differences.to_csv(output_dir / "external_transportability_paired_differences.csv", index=False)
    reliability.to_csv(output_dir / "external_transportability_calibration_bins.csv", index=False)
    cohort_summary(data).to_csv(output_dir / "external_transportability_cohort_summary.csv", index=False)
    np.savez_compressed(output_dir / "external_transportability_bootstrap.npz", **bootstrap_arrays)

    metadata = {
        "name": "cross-cohort mortality transportability stress test",
        "scope": "Partial transportability analysis of mortality using a reduced common-feature model.",
        "specification_timing": "Revision-stage specification locked before external model fitting and evaluation.",
        "source_endpoint": "In-hospital fatal outcome (LET_IS > 0).",
        "external_primary_endpoint": "30-day all-cause mortality derived from admission and death dates.",
        "external_sensitivity_endpoint": "7-day all-cause mortality.",
        "features": STRICT_FEATURES,
        "independent_unit": "First AMI event selected by Event number = 1; Study ID was not used.",
        "models": list(MODEL_ORDER),
        **model_specification,
        "bootstrap": f"{n_boot} ordinary patient-level paired multinomial bootstrap resamples.",
        "seed": SEED,
        "runtime_seconds": time.time() - started,
        "dataset_doi": "10.17632/2v7n2r3xch.1",
        "associated_article_doi": "10.3390/ijerph23081010",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "external_transportability_results.json").open("w", encoding="utf-8") as stream:
        json.dump(results.to_dict(orient="records"), stream, indent=2, ensure_ascii=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hungarian-xlsx", required=True, type=Path)
    parser.add_argument("--uci-csv", default=Path("dataset/Myocardial infarction complications Database.csv"), type=Path)
    parser.add_argument("--output-dir", default=Path("output/external_validation"), type=Path)
    parser.add_argument("--bootstrap", default=2000, type=int)
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()
    result = run(
        args.hungarian_xlsx,
        args.uci_csv,
        args.output_dir,
        args.bootstrap,
        reuse_predictions=args.reuse_predictions,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
