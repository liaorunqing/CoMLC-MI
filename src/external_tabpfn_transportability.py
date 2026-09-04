"""Evaluate the manuscript's TabPFN binary component on the strict common feature set.

The result is a single-fatal-outcome transportability sensitivity analysis and
does not validate the dependency-aware 12-label framework.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

from external_mortality_transportability import harmonize, metrics, style_workbook


SEED = 42
STRICT4 = ["age_years", "male", "prior_mi", "prior_hf"]


def classifier() -> TabPFNClassifier:
    return TabPFNClassifier(device="cpu", random_state=SEED, ignore_pretraining_limits=True)


def run(input_path: Path, uci_path: Path, output_dir: Path, n_boot: int) -> pd.DataFrame:
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = pd.read_excel(input_path)
    registry.columns = registry.columns.astype(str).str.strip()
    uci = pd.read_csv(uci_path)
    x_source, y_source, x_external, endpoints = harmonize(uci, registry)

    imputer = SimpleImputer(strategy="median")
    source = imputer.fit_transform(x_source[STRICT4])
    external = imputer.transform(x_external[STRICT4])

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    internal_p = np.zeros(len(y_source), dtype=float)
    cv_start = time.time()
    for train_index, validation_index in cv.split(source, y_source):
        fitted = classifier()
        fitted.fit(source[train_index], y_source.iloc[train_index].to_numpy())
        internal_p[validation_index] = fitted.predict_proba(source[validation_index])[:, 1]
    cv_seconds = time.time() - cv_start

    rows = []
    internal = metrics(y_source, internal_p, n_boot)
    internal.update(
        {
            "cohort": "Krasnoyarsk 5-fold out-of-fold",
            "endpoint": "in-hospital lethal outcome",
            "model": "TabPFN-3 binary relevance component",
            "feature_set": "strict4",
            "analysis_subset": "all with source-fitted median imputation",
            "runtime_seconds": cv_seconds,
        }
    )
    rows.append(internal)

    fitted = classifier()
    fit_start = time.time()
    fitted.fit(source, y_source.to_numpy())
    fit_seconds = time.time() - fit_start
    predict_start = time.time()
    external_p = fitted.predict_proba(external)[:, 1]
    predict_seconds = time.time() - predict_start

    for endpoint, y_external in endpoints.items():
        result = metrics(y_external, external_p, n_boot)
        result.update(
            {
                "cohort": "Hungarian first-event cohort",
                "endpoint": endpoint,
                "model": "TabPFN-3 binary relevance component",
                "feature_set": "strict4",
                "analysis_subset": "all with source-fitted median imputation",
                "runtime_seconds": fit_seconds + predict_seconds,
            }
        )
        rows.append(result)

        complete = x_external[STRICT4].notna().all(axis=1)
        complete_result = metrics(y_external.loc[complete], external_p[complete.to_numpy()], n_boot)
        complete_result.update(
            {
                "cohort": "Hungarian first-event cohort",
                "endpoint": endpoint,
                "model": "TabPFN-3 binary relevance component",
                "feature_set": "strict4",
                "analysis_subset": "complete case",
                "runtime_seconds": fit_seconds + predict_seconds,
            }
        )
        rows.append(complete_result)

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "tabpfn_mortality_transportability_results.csv", index=False)
    with (output_dir / "tabpfn_mortality_transportability_results.json").open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(orient="records"), stream, indent=2, ensure_ascii=False)
    workbook_path = output_dir / "tabpfn_mortality_transportability_results.xlsx"
    notes = pd.DataFrame(
        [
            ("scope", "Single fatal-outcome sensitivity analysis; not external validation of the 12-label dependency-aware framework."),
            ("features", "Age, sex, prior myocardial infarction and prior heart failure only."),
            ("outcome mismatch", "Source is in-hospital lethal outcome; external endpoints are date-derived all-cause mortality."),
            ("bootstrap", f"Exploratory class-stratified bootstrap with {n_boot} resamples."),
        ],
        columns=["item", "note"],
    )
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Metrics", index=False)
        notes.to_excel(writer, sheet_name="Interpretation", index=False)
    style_workbook(workbook_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hungarian-xlsx", required=True, type=Path)
    parser.add_argument("--uci-csv", default=Path("dataset/Myocardial infarction complications Database.csv"), type=Path)
    parser.add_argument("--output-dir", default=Path("output/external_validation"), type=Path)
    parser.add_argument("--bootstrap", default=200, type=int)
    args = parser.parse_args()
    result = run(args.hungarian_xlsx, args.uci_csv, args.output_dir, args.bootstrap)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
