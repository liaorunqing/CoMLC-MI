"""Generate appendix LaTeX tables from the benchmark result files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .generate_r1_latex import DISPLAY_NAMES
from .revision_features import LABEL_COLS, prepare_outcomes


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "benchmark"
OUTPUT = ROOT / "paper" / "generated_r1"


def esc(value: object) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return r"\NA"
    text = str(value)
    if text == r"\NA":
        return text
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def number(value: object, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if not np.isfinite(value):
        return r"\NA"
    formatted = f"{abs(value):.{digits}f}"
    return f"$-${formatted}" if value < 0 else formatted


def interval(row: pd.Series, estimate: str, low: str, high: str, digits: int = 3) -> str:
    return f"{number(row[estimate], digits)} ({number(row[low], digits)}--{number(row[high], digits)})"


def longtable(path: Path, caption: str, label: str, columns: list[str], rows: list[list[object]], spec: str | None = None, font: str = r"\scriptsize") -> None:
    spec = spec or ("l" + "r" * (len(columns) - 1))
    header = " & ".join(columns) + r" \\" 
    lines = [
        font,
        rf"\begin{{longtable}}{{{spec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{" + str(len(columns)) + r"}{c}{\tablename\ \thetable\ (continued)}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(len(columns)) + r"}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    lines.extend(" & ".join(esc(cell) for cell in row) + r" \\" for row in rows)
    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scaled_tabular(
    path: Path,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[object]],
    spec: str,
    width: str = r"\textwidth",
) -> None:
    """Write a compact non-floating table that is guaranteed to fit the page width."""
    lines = [
        r"\begin{center}",
        rf"\captionof{{table}}{{{caption}}}\label{{{label}}}",
        rf"\resizebox{{{width}}}{{!}}{{%",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(esc(cell) for cell in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{center}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def feature_contract() -> None:
    contracts = json.loads((RESULTS / "feature_contract.json").read_text(encoding="utf-8"))["feature_sets"]
    ordered = contracts["retrospective_full_111"]
    rows = []
    keys = ["admission_safe_v1", "prospective_24h", "prospective_48h", "prospective_72h", "retrospective_full_111"]
    for index, variable in enumerate(ordered, start=1):
        rows.append([index, variable, *("Yes" if variable in contracts[key] else "No" for key in keys)])
    longtable(
        OUTPUT / "supp_feature_contract.tex",
        "Complete feature-contract membership. The two variables ending in MISSING are row-wise indicators whose definition is fixed but whose imputation/scaling objects are fitted only on training rows.",
        "tab:supp_features",
        ["No.", "Variable", "Admission", "24 h", "48 h", "72 h", "Retrospective"],
        rows,
        "rlccccc",
    )


def outcome_definitions() -> None:
    definitions = [
        ("FIBR_PREDS", "Atrial fibrillation", "1", "Recorded hospitalization complication", "No onset timestamp; prevalent and incident events cannot be separated"),
        ("PREDS_TAH", "Supraventricular tachycardia", "1", "Recorded hospitalization complication", "Rare endpoint; no onset timestamp"),
        ("JELUD_TAH", "Ventricular tachycardia", "1", "Recorded hospitalization complication", "No onset timestamp"),
        ("FIBR_JELUD", "Ventricular fibrillation", "1", "Recorded hospitalization complication", "No onset timestamp"),
        ("A_V_BLOK", "Third-degree AV block", "1", "Recorded hospitalization complication", "No onset timestamp"),
        ("OTEK_LANC", "Pulmonary edema", "1", "Recorded hospitalization complication", "No onset timestamp; may overlap a lethal-cause code"),
        ("RAZRIV", "Myocardial rupture", "1", "Recorded hospitalization complication", "No onset timestamp; may overlap a lethal-cause code"),
        ("DRESSLER", "Dressler syndrome", "1", "Recorded hospitalization complication", "May manifest after the early inpatient window"),
        ("ZSN", "Chronic heart failure outcome", "1", "Recorded hospitalization complication", "Distinct field from baseline HF history but clinically overlapping; no onset timestamp"),
        ("REC_IM", "Recurrent myocardial infarction", "1", "Recorded hospitalization complication", "No onset timestamp"),
        ("P_IM_STEN", "Post-infarction angina", "1", "Recorded hospitalization complication", "May manifest beyond the earliest observation window"),
        ("LET_IS", "Fatal outcome", "1--7", "Any nonzero coded lethal cause", "Cause code, not a time-to-event field; zero means no coded lethal cause"),
    ]
    pd.DataFrame(
        definitions,
        columns=["source_field", "clinical_label", "positive_code", "operational_target", "timing_interpretation_caveat"],
    ).to_csv(RESULTS / "outcome_definitions.csv", index=False)
    longtable(
        OUTPUT / "supp_outcome_definitions.tex",
        "Operational outcome definitions and interpretation boundaries from the source data dictionary.",
        "tab:supp_outcomes",
        ["Source field", "Clinical label", "Positive code", "Operational target", "Timing/interpretation caveat"],
        [list(row) for row in definitions],
        r"lllp{0.22\textwidth}p{0.32\textwidth}",
        r"\tiny",
    )


def fold_summary() -> None:
    assignments = pd.read_csv(RESULTS / "outer_fold_assignments.csv")
    config = json.loads((ROOT / "configs" / "comlc_mi_benchmark.json").read_text(encoding="utf-8"))
    outcomes = prepare_outcomes(pd.read_csv(ROOT / config["dataset"]))
    rows = []
    repeat_columns = [column for column in assignments if column.startswith("repeat_")]
    for repeat_index, column in enumerate(repeat_columns, start=1):
        for fold in range(1, 6):
            mask = assignments[column].to_numpy() == fold - 1
            counts = outcomes.loc[mask].sum(axis=0)
            rows.append([repeat_index, fold, int(mask.sum()), int(counts.min()), int(counts.max()), int(counts[LABEL_COLS[-1]])])
    longtable(
        OUTPUT / "supp_fold_summary.tex",
        "Outer-fold size and event-count audit. Full per-label membership is provided in the machine-readable assignment file.",
        "tab:supp_folds",
        ["Repeat", "Fold", "$n$", "Minimum label events", "Maximum label events", "Deaths"],
        rows,
        "rrrrrr",
    )


def internal_models() -> None:
    frame = pd.read_csv(RESULTS / "internal_bootstrap_summary.csv").set_index(["model", "metric"])
    metrics = ["macro_auroc", "micro_auroc", "macro_auprc", "brier", "ece_10", "macro_f1_0_5"]
    rows = []
    for model, display in DISPLAY_NAMES.items():
        cells = [display]
        for metric in metrics:
            item = frame.loc[(model, metric)]
            cells.append(interval(item, "estimate", "ci_low", "ci_high", 4 if metric in {"brier", "ece_10"} else 3))
        rows.append(cells)
    scaled_tabular(
        OUTPUT / "supp_internal_models.tex",
        "Complete internal aggregate performance: estimate (95\\% patient-bootstrap interval).",
        "tab:supp_internal_models",
        ["Model", "Macro-AUROC", "Micro-AUROC", "Macro-AUPRC", "Brier", "ECE$_{10}$", "Macro-F1"],
        rows,
        "lcccccc",
    )
    repeats = pd.read_csv(RESULTS / "internal_repeat_metrics.csv")
    rows = []
    for model, group in repeats.groupby("model", sort=False):
        rows.append([
            DISPLAY_NAMES.get(model, model),
            number(group["macro_auroc"].mean(), 4), number(group["macro_auroc"].std(ddof=1), 4),
            f"{number(group['macro_auroc'].min(), 4)}--{number(group['macro_auroc'].max(), 4)}",
            number(group["macro_auprc"].mean(), 4), number(group["macro_auprc"].std(ddof=1), 4),
            number(group["brier"].mean(), 4), number(group["brier"].std(ddof=1), 4),
        ])
    longtable(
        OUTPUT / "supp_repeat_stability.tex",
        "Sensitivity to the five outer-fold repetitions. SD is descriptive across the five complete repeated OOF prediction sets and is not the primary confidence interval.",
        "tab:supp_repeat_stability",
        ["Model", "Macro-AUROC mean", "SD", "Range", "Macro-AUPRC mean", "SD", "Brier mean", "SD"],
        rows,
        "lrrrrrrr",
        r"\tiny",
    )


def internal_per_label() -> None:
    frame = pd.read_csv(RESULTS / "internal_per_label_metrics.csv")
    rows = []
    for row in frame.itertuples(index=False):
        rows.append([
            DISPLAY_NAMES.get(row.model, row.model), row.label, row.positive_n,
            number(row.auroc), number(row.auprc), number(row.brier), number(row.ece_10),
            number(row.macro_f1_component), number(row.observed_expected_ratio), row.calibration_stability,
        ])
    longtable(
        OUTPUT / "supp_internal_per_label.tex",
        "Per-label metrics for all internal models from mean repeated OOF predictions.",
        "tab:supp_internal_labels",
        ["Model", "Label", "Events", "AUROC", "AUPRC", "Brier", "ECE$_{10}$", "F1", "O:E", "Calibration"],
        rows,
        "llrrrrrrrl",
        r"\tiny",
    )


def primary_inference() -> None:
    tests = pd.read_csv(RESULTS / "primary_per_label_tests.csv")
    boots = pd.read_csv(RESULTS / "primary_per_label_bootstrap.csv").rename(columns={"difference": "bootstrap_point"})
    frame = tests.merge(boots[["label", "ci_low", "ci_high"]], on="label", how="left")
    rows = []
    for row in frame.itertuples(index=False):
        rows.append([
            row.label, row.positive_n, row.test, number(row.difference, 4),
            f"{number(row.ci_low, 4)}--{number(row.ci_high, 4)}", number(row.p_value, 4),
            number(row.p_value_bh, 4), "Yes" if row.reject_bh_0_05 else "No", row.stability,
        ])
    longtable(
        OUTPUT / "supp_primary_inference.tex",
        "Per-label primary paired comparison: weighted ensemble minus TabPFN-3.",
        "tab:supp_primary_inference",
        ["Label", "Events", "Test", "$\\Delta$AUROC", "95\\% CI", "$p$", "BH $q$", "Reject", "Stability"],
        rows,
        "lrlrrrrll",
        r"\tiny",
    )


def calibration_intervals() -> None:
    frame = pd.read_csv(RESULTS / "primary_calibration_intervals.csv")
    rows = []
    for row in frame.itertuples(index=False):
        rows.append([
            DISPLAY_NAMES.get(row.model, row.model), row.label, row.positive_n, row.metric,
            number(row.estimate, 4), number(row.ci_low, 4), number(row.ci_high, 4), row.stability,
        ])
    longtable(
        OUTPUT / "supp_calibration_intervals.tex",
        "Per-label calibration estimates and 95\\% patient-bootstrap intervals for the primary models.",
        "tab:supp_calibration",
        ["Model", "Label", "Events", "Metric", "Estimate", "CI low", "CI high", "Stability"],
        rows,
        "llrlrrrl",
        r"\tiny",
    )
    sensitivity = pd.read_csv(RESULTS / "calibration_sensitivity_unweighted_lr_metrics.csv")
    per_label = pd.read_csv(RESULTS / "calibration_sensitivity_unweighted_lr_per_label.csv")
    row = sensitivity.iloc[0]
    stable = per_label.query("calibration_stability == 'stable'")
    scaled_tabular(
        OUTPUT / "appendix_unweighted_lr_calibration.tex",
        "Unweighted BR-logistic calibration sensitivity from the same mean repeated OOF predictions. This appendix analysis does not alter the primary model ranking.",
        "tab:app_unweighted_lr",
        ["Model", "Macro-AUROC", "Macro-AUPRC", "Brier", "ECE$_{10}$", "Macro-F1", "Median intercept", "Median slope"],
        [["BR--LR (unweighted)", number(row["macro_auroc"], 4), number(row["macro_auprc"], 4), number(row["brier"], 4), number(row["ece_10"], 4), number(row["macro_f1_0_5"], 4), number(stable["calibration_intercept"].median(), 3), number(stable["calibration_slope"].median(), 3)]],
        "lrrrrrrr",
    )


def gcn_tables() -> None:
    raw = pd.read_csv(RESULTS / "gcn_ablation" / "gcn_ablation_10_seed_full.csv")
    rows = [[row.seed, row.adjacency, row.head, row.loss, row.best_epoch, number(row.validation_macro_auroc, 4), number(row.test_macro_auroc, 4), number(row.test_micro_auroc, 4)] for row in raw.itertuples(index=False)]
    longtable(
        OUTPUT / "supp_gcn_full.tex",
        "Complete 18-configuration GCN ablation for seeds 42--51.",
        "tab:supp_gcn_full",
        ["Seed", "Graph", "Head", "Loss", "Epoch", "Validation macro", "Test macro", "Test micro"],
        rows,
        "rllllrrr",
        r"\tiny",
    )
    paired = pd.read_csv(RESULTS / "gcn_ablation" / "gcn_ablation_paired_differences.csv")
    rows = [[row.head, row.loss, row.contrast, row.n_paired_seeds, number(row.mean_difference, 4), number(row.ci_low, 4), number(row.ci_high, 4)] for row in paired.itertuples(index=False)]
    longtable(
        OUTPUT / "supp_gcn_paired.tex",
        "Paired graph-minus-no-graph Macro-AUROC differences across ten seeds.",
        "tab:supp_gcn_paired",
        ["Head", "Loss", "Contrast", "$n$", "Mean difference", "CI low", "CI high"],
        rows,
        "lllrrrr",
    )


def synthetic_table() -> None:
    frame = pd.read_csv(RESULTS / "synthetic" / "controlled_dependency_summary.csv")
    rows = [[number(row.rho, 2), row.model, number(row.realized_lds_mean), number(row.realized_lds_sd), number(row.label_density_mean), number(row.true_score_auc_mean), number(row.macro_auc_mean), number(row.macro_auc_sd)] for row in frame.itertuples(index=False)]
    longtable(
        OUTPUT / "supp_synthetic.tex",
        "Controlled synthetic experiment summary over ten paired runs.",
        "tab:supp_synthetic",
        ["Residual $\\rho$", "Model", "Realized LDS", "LDS SD", "LD", "True-score AUC", "Macro-AUROC", "AUC SD"],
        rows,
        "rlrrrrrr",
        r"\tiny",
    )


def temporal_table() -> None:
    frame = pd.read_csv(RESULTS / "temporal" / "temporal_model_metrics.csv")
    rows = [[row.horizon, row.feature_count, number(row.macro_auroc, 4), number(row.micro_auroc, 4), number(row.macro_auprc, 4), number(row.brier, 4), number(row.ece_10, 4), number(row.macro_f1_0_5, 4)] for row in frame.itertuples(index=False)]
    longtable(
        OUTPUT / "supp_temporal.tex",
        "Four-horizon shared-MLP sensitivity analysis.",
        "tab:supp_temporal",
        ["Horizon", "Variables", "Macro-AUROC", "Micro-AUROC", "Macro-AUPRC", "Brier", "ECE$_{10}$", "Macro-F1"],
        rows,
        "lrrrrrrr",
        r"\tiny",
    )


def external_table() -> None:
    folder = RESULTS / "external_transportability"
    cohort = pd.read_csv(folder / "external_transportability_cohort_summary.csv").set_index("feature")
    mapping = {
        "age_years": ("Age (years)", "AGE; numeric years", "Age at admission; numeric years"),
        "male": ("Male sex", "SEX; 1=male, 0=female", "Gender; Man=1, Woman=0"),
        "prior_mi": ("Previous MI", "INF_ANAM > 0", "History of myocardial infarction; Yes=1, Not=0"),
        "prior_hf": ("Previous heart failure", "ZSN_A > 0", "History of heart failure; Yes=1, Not=0"),
    }
    mapping_rows = []
    for feature, (display, source_definition, external_definition) in mapping.items():
        item = cohort.loc[feature]
        mapping_rows.append([
            display,
            source_definition,
            external_definition,
            int(item["source_missing"]),
            int(item["external_missing"]),
            number(item["source_mean"], 3),
            number(item["external_mean"], 3),
            number(item["standardized_mean_difference"], 3),
        ])
    scaled_tabular(
        OUTPUT / "supp_external_mapping.tex",
        "Locked four-variable harmonization, missingness, and case-mix shift. Means are proportions for binary variables; SMD denotes standardized mean difference (Hungarian minus source).",
        "tab:supp_external_mapping",
        ["Aligned feature", "Krasnoyarsk definition", "Hungarian definition", "Source missing", "Hungarian missing", "Source mean", "Hungarian mean", "SMD"],
        mapping_rows,
        "lllrrrrr",
    )

    frame = pd.read_csv(folder / "external_transportability_results.csv")
    indexed = frame.set_index(["cohort", "endpoint", "subset", "model"])
    primary = indexed.loc[("Hungarian AMI registry", "death_30d", "all first events", "TabPFN-3")]
    complete_30 = indexed.loc[("Hungarian AMI registry", "death_30d", "complete case", "TabPFN-3")]
    seven = indexed.loc[("Hungarian AMI registry", "death_7d", "all first events", "TabPFN-3")]
    complete_7 = indexed.loc[("Hungarian AMI registry", "death_7d", "complete case", "TabPFN-3")]
    first_n = int(primary["n"])
    complete_n = int(complete_30["n"])
    selection_rows = [
        ["All AMI event records in released registry", 30_883, r"\NA", "Starting event-level extract"],
        ["Independent index-event cohort", first_n, int(primary["events"]), "Event number = 1; Study ID not used"],
        ["Excluded non-index event rows", 30_883 - first_n, r"\NA", "Event number not equal to 1"],
        ["30-day primary endpoint, all index events", first_n, int(primary["events"]), "All-cause death on days 0--30"],
        ["30-day complete-case sensitivity", complete_n, int(complete_30["events"]), f"{first_n - complete_n} rows excluded for any mapped-feature missingness"],
        ["7-day endpoint, all index events", int(seven["n"]), int(seven["events"]), "All-cause death on days 0--7"],
        ["7-day complete-case sensitivity", int(complete_7["n"]), int(complete_7["events"]), "Same complete-case mask as 30-day analysis"],
    ]
    scaled_tabular(
        OUTPUT / "supp_external_selection.tex",
        "Hungarian registry case selection and endpoint accounting.",
        "tab:supp_external_selection",
        ["Cohort/analysis set", "$n$", "Deaths", "Rule"],
        selection_rows,
        "lrrl",
    )

    performance_rows = []
    calibration_rows = []
    for _, row in frame.iterrows():
        identifiers = [
            row["cohort"], row["endpoint"], row["subset"], row["model"], int(row["n"]), int(row["events"]),
        ]
        performance_rows.append(identifiers + [
            interval(row, "auc", "auc_ci_low", "auc_ci_high"), interval(row, "auprc", "auprc_ci_low", "auprc_ci_high"),
            interval(row, "brier", "brier_ci_low", "brier_ci_high", 4), interval(row, "ece", "ece_ci_low", "ece_ci_high", 4),
        ])
        calibration_rows.append(identifiers + [
            interval(row, "calibration_intercept", "calibration_intercept_ci_low", "calibration_intercept_ci_high"),
            interval(row, "calibration_slope", "calibration_slope_ci_low", "calibration_slope_ci_high"),
            interval(row, "oe_ratio", "oe_ratio_ci_low", "oe_ratio_ci_high"),
        ])
    scaled_tabular(
        OUTPUT / "supp_external_performance.tex",
        "Source and external mortality discrimination and probability error: estimate (95\\% patient-bootstrap interval).",
        "tab:supp_external_performance",
        ["Cohort", "Endpoint", "Subset", "Model", "$n$", "Events", "AUROC", "AUPRC", "Brier", "ECE"],
        performance_rows,
        "llllrrcccc",
    )
    scaled_tabular(
        OUTPUT / "supp_external_calibration.tex",
        "Source and external mortality calibration: estimate (95\\% patient-bootstrap interval).",
        "tab:supp_external_calibration",
        ["Cohort", "Endpoint", "Subset", "Model", "$n$", "Events", "Intercept", "Slope", "O:E"],
        calibration_rows,
        "llllrrccc",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    feature_contract()
    outcome_definitions()
    fold_summary()
    internal_models()
    internal_per_label()
    primary_inference()
    calibration_intervals()
    gcn_tables()
    synthetic_table()
    temporal_table()
    external_table()
    print(f"Appendix table fragments written to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
