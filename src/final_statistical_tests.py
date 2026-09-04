"""Per-label paired model comparisons with multiplicity correction.

Large-sample labels use the paired DeLong test. Labels with fewer than 15
positive test observations use a paired score-swap permutation test as the
primary analysis. The threshold is a prespecified operational safeguard, not a
claim that DeLong is universally valid above 15 positives.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from config import LABEL_COLS, LABEL_SHORT, ORIG_PROCESSED_DIR, OUTPUT_DIR, RANDOM_SEED
from statistical_utils import (
    benjamini_hochberg,
    delong_roc_test,
    paired_stratified_auc_difference_ci,
    permutation_auc_test,
)


def _finite_or_none(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _load_predictions(stem):
    path = os.path.join(OUTPUT_DIR, f"PUB_{stem}_preds.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run src/run_all.py first.")
    return np.load(path)


def compare_models(y_test, primary, comparator, comparator_name, n_permutations=10_000):
    results = []
    for label_index, label in enumerate(LABEL_COLS):
        observed = y_test[:, label_index]
        n_positive = int(observed.sum())
        n_negative = int(len(observed) - n_positive)
        z_value, delong_p, auc_primary, auc_comparator = delong_roc_test(
            observed, primary[:, label_index], comparator[:, label_index]
        )
        ci_low, ci_high = paired_stratified_auc_difference_ci(
            observed,
            primary[:, label_index],
            comparator[:, label_index],
            random_state=RANDOM_SEED + 100 + label_index,
        )
        permutation_p = None
        if n_positive < 15:
            _, permutation_p = permutation_auc_test(
                observed,
                primary[:, label_index],
                comparator[:, label_index],
                n_permutations=n_permutations,
                random_state=RANDOM_SEED + label_index,
            )
            primary_test = "paired permutation"
            primary_p = permutation_p
        else:
            primary_test = "paired DeLong"
            primary_p = delong_p

        results.append({
            "label": label,
            "short_label": LABEL_SHORT[label_index],
            "n_positive": n_positive,
            "n_negative": n_negative,
            "auc_primary_ensemble": _finite_or_none(auc_primary),
            f"auc_{comparator_name}": _finite_or_none(auc_comparator),
            "auc_difference": _finite_or_none(auc_primary - auc_comparator),
            "auc_difference_ci_low": _finite_or_none(ci_low),
            "auc_difference_ci_high": _finite_or_none(ci_high),
            "delong_z": _finite_or_none(z_value),
            "delong_p": _finite_or_none(delong_p),
            "permutation_p": _finite_or_none(permutation_p),
            "primary_test": primary_test,
            "primary_p": _finite_or_none(primary_p),
        })

    valid = [
        row for row in results
        if row["primary_p"] is not None and np.isfinite(row["primary_p"])
    ]
    reject, adjusted = benjamini_hochberg([row["primary_p"] for row in valid])
    for row, rejected, q_value in zip(valid, reject, adjusted):
        row["bh_reject_0_05"] = bool(rejected)
        row["bh_adjusted_p"] = float(q_value)
    return results


def main():
    y_test = pd.read_csv(os.path.join(ORIG_PROCESSED_DIR, "y_test.csv")).to_numpy(dtype=int)
    primary = _load_predictions("ens2_wt")
    tabpfn = _load_predictions("tabpfn")
    lp_rf = _load_predictions("lp_rf")

    if not (primary.shape == tabpfn.shape == lp_rf.shape == y_test.shape):
        raise ValueError(
            f"Prediction/label shapes differ: y={y_test.shape}, ensemble={primary.shape}, "
            f"TabPFN={tabpfn.shape}, LP-RF={lp_rf.shape}."
        )

    ensemble_vs_tabpfn = compare_models(y_test, primary, tabpfn, "tabpfn")
    ensemble_vs_lp_rf = compare_models(y_test, primary, lp_rf, "lp_rf")
    results = {
        "metadata": {
            "primary_model": "validation-AUC-weighted TabPFN-3 + CatBoost ensemble",
            "large_sample_test": "paired DeLong",
            "small_sample_test": "paired score-swap permutation (10,000 permutations)",
            "small_sample_rule": "fewer than 15 positive test observations",
            "multiplicity": "Benjamini-Hochberg across 12 labels within each comparator family",
            "interval": "2,000 class-stratified paired bootstrap resamples per label",
            "random_seed": RANDOM_SEED,
        },
        "ensemble_vs_tabpfn": ensemble_vs_tabpfn,
        "ensemble_vs_lp_rf": ensemble_vs_lp_rf,
    }
    output_json = os.path.join(OUTPUT_DIR, "FINAL_STATISTICAL_RESULTS.json")
    with open(output_json, "w") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)

    rows = []
    for family, family_results in (
        ("ensemble_vs_tabpfn", ensemble_vs_tabpfn),
        ("ensemble_vs_lp_rf", ensemble_vs_lp_rf),
    ):
        rows.extend({"comparison": family, **row} for row in family_results)
    output_csv = os.path.join(OUTPUT_DIR, "FINAL_STATISTICAL_RESULTS.csv")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
