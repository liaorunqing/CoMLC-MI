"""Generate LaTeX result fragments directly from benchmark CSV outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .benchmark_models import MODEL_NAMES


DISPLAY_NAMES = {
    "BR-LR": "BR--LR",
    "BR-XGBoost": "BR--XGBoost",
    "BR-LightGBM": "BR--LightGBM",
    "BR-CatBoost": "BR--CatBoost",
    "ECC-LightGBM": "ECC--LightGBM",
    "LP-RF": "LP--RF",
    "RAkELd-RF": "RAkELd--RF",
    "Shared-MLP": "Shared MLP",
    "MultiTask-DNN": "Multitask DNN",
    "TabPFN": "TabPFN-3",
    "Ensemble-LP-RF-TabPFN-Equal": "LP--RF--TabPFN-3 (equal)",
    "Ensemble-LP-RF-TabPFN-Weighted": "LP--RF--TabPFN-3 (inner-OOF weighted)",
}

LABEL_DISPLAY = {
    "FIBR_PREDS": "AF",
    "PREDS_TAH": "SVT",
    "JELUD_TAH": "VT",
    "FIBR_JELUD": "VF",
    "A_V_BLOK": "AVB",
    "OTEK_LANC": "PulEd",
    "RAZRIV": "Rupt",
    "DRESSLER": "Dress",
    "ZSN": "CHF",
    "REC_IM": "ReMI",
    "P_IM_STEN": "PIA",
    "LET_IS": "Leth",
}


def _escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _interval(row, digits: int = 3) -> str:
    return (
        f"${row.estimate:.{digits}f}$ "
        f"(${row.ci_low:.{digits}f}$--${row.ci_high:.{digits}f}$)"
    )


def write_model_table(results_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(results_dir / "internal_bootstrap_summary.csv")
    indexed = frame.set_index(["model", "metric"])
    order = list(DISPLAY_NAMES)
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Internal performance from the mean of five outer-fold OOF predictions per patient. Values are estimates (95\% patient-bootstrap CI); Macro-F1 uses the fixed 0.5 threshold.}",
        r"\label{tab:main_results}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Macro-AUROC & Micro-AUROC & Macro-AUPRC & Brier & ECE$_{10}$ / Macro-F1 \\",
        r"\midrule",
    ]
    for model in order:
        values = {metric: indexed.loc[(model, metric)] for metric in ["macro_auroc", "micro_auroc", "macro_auprc", "brier", "ece_10", "macro_f1_0_5"]}
        lines.append(
            f"{DISPLAY_NAMES[model]} & {_interval(values['macro_auroc'])} & {_interval(values['micro_auroc'])} & "
            f"{_interval(values['macro_auprc'])} & {_interval(values['brier'], 4)} & "
            f"{_interval(values['ece_10'], 4)} / {_interval(values['macro_f1_0_5'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (output_dir / "table_internal_models.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_primary_macros(results_dir: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(results_dir / "internal_model_metrics.csv").set_index("model")
    contrast = pd.read_csv(results_dir / "primary_paired_difference.csv").set_index("metric")
    tests = pd.read_csv(results_dir / "primary_per_label_tests.csv")
    best_single = metrics.loc["TabPFN"]
    ensemble = metrics.loc["Ensemble-LP-RF-TabPFN-Weighted"]
    macro_difference = contrast.loc["macro_auroc"]
    significant = int(tests["reject_bh_0_05"].sum())
    lines = [
        f"\\newcommand{{\\RTabMacroAUC}}{{{best_single.macro_auroc:.4f}}}",
        f"\\newcommand{{\\REnsMacroAUC}}{{{ensemble.macro_auroc:.4f}}}",
        f"\\newcommand{{\\REnsMacroAUPRC}}{{{ensemble.macro_auprc:.4f}}}",
        f"\\newcommand{{\\RMacroDiff}}{{{macro_difference.estimate:.4f}}}",
        f"\\newcommand{{\\RMacroDiffLow}}{{{macro_difference.ci_low:.4f}}}",
        f"\\newcommand{{\\RMacroDiffHigh}}{{{macro_difference.ci_high:.4f}}}",
        f"\\newcommand{{\\RBHSignificant}}{{{significant}}}",
    ]
    (output_dir / "result_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gcn_table(results_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(results_dir / "gcn_ablation" / "gcn_ablation_paired_differences.csv")
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Paired graph-minus-no-graph Macro-AUROC differences across seeds 42--51. Intervals are two-sided 95\% $t$ intervals over ten paired seed differences.}",
        r"\label{tab:gcn_paired}",
        r"\small",
        r"\begin{tabular}{lll r}",
        r"\toprule",
        r"Head & Loss & Graph contrast & Mean difference (95\% CI) \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        lines.append(
            f"{_escape(row.head)} & {_escape(row.loss)} & {_escape(row.contrast.replace(' - ', ' vs. '))} & "
            f"${row.mean_difference:.4f}$ (${row.ci_low:.4f}$--${row.ci_high:.4f}$) \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (output_dir / "table_gcn_paired.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_temporal_table(results_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(results_dir / "temporal" / "temporal_model_metrics.csv")
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Prospective feature-horizon sensitivity analysis using the shared MLP baseline and identical repeated outer folds.}",
        r"\label{tab:temporal}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Horizon & Features & Macro-AUROC & Macro-AUPRC & Brier \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        lines.append(
            f"{row.horizon} & {row.feature_count} & {row.macro_auroc:.4f} & {row.macro_auprc:.4f} & {row.brier:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output_dir / "table_temporal.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_synthetic_table(results_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(results_dir / "synthetic" / "controlled_dependency_summary.csv")
    models = ["BR", "CC", "BR-kNN (distance-weighted, k=10)", "GCN"]
    rows = []
    for rho, group in frame.groupby("rho", sort=True):
        indexed = group.set_index("model")
        reference = group.iloc[0]
        rows.append(
            f"{rho:.2f} & {reference.realized_lds_mean:.3f} & {reference.label_density_mean:.3f} & "
            f"{reference.true_score_auc_mean:.3f} & "
            + " & ".join(f"{indexed.loc[model, 'macro_auc_mean']:.3f}" for model in models)
            + r" \\"
        )
    lines = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{Controlled dependency experiment over ten paired datasets per setting. LDS, label density, true-score AUC, and model Macro-AUROC are means across repetitions.}",
        r"\label{tab:synthetic}", r"\small",
        r"\begin{tabular}{rrrrrrrr}", r"\toprule",
        r"Residual $\rho$ & Realized LDS & LD & True-score AUC & BR & CC & BR-kNN & GCN \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    ]
    (output_dir / "table_synthetic.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_primary_label_table(results_dir: Path, output_dir: Path) -> None:
    tests = pd.read_csv(results_dir / "primary_per_label_tests.csv")
    intervals = pd.read_csv(results_dir / "primary_per_label_bootstrap.csv").set_index("label")
    lines = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{Primary per-label comparison: inner-OOF-weighted LP--RF--TabPFN-3 ensemble minus TabPFN-3. Intervals use 2,000 paired patient bootstraps; $q$ is Benjamini--Hochberg adjusted across 12 labels.}",
        r"\label{tab:primary_labels}", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrlrrl@{}}", r"\toprule",
        r"Label & Events & $\Delta$AUROC & 95\% CI & Test & BH $q$ & Stability \\", r"\midrule",
    ]
    for row in tests.itertuples(index=False):
        interval_row = intervals.loc[row.label]
        lines.append(
            f"{LABEL_DISPLAY[row.label]} & {row.positive_n} & ${row.difference:.4f}$ & "
            f"${interval_row.ci_low:.4f}$--${interval_row.ci_high:.4f}$ & {_escape(row.test)} & "
            f"${row.p_value_bh:.4f}$ & {_escape(row.stability)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}"])
    (output_dir / "table_primary_labels.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_table(results_dir: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(results_dir / "internal_model_metrics.csv").set_index("model")
    per_label = pd.read_csv(results_dir / "internal_per_label_metrics.csv")
    models = ["TabPFN", "Ensemble-LP-RF-TabPFN-Weighted"]
    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Internal calibration summary from mean repeated OOF predictions. Intercept and slope are fitted simultaneously and summarized as medians over labels with at least 30 events; complete per-label bootstrap intervals are in Appendix C.}",
        r"\label{tab:calibration}", r"\small",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Model & Brier & ECE$_{10}$ & Median intercept & Median slope \\", r"\midrule",
    ]
    for model in models:
        stable = per_label[(per_label["model"] == model) & (per_label["calibration_stability"] == "stable")]
        lines.append(
            f"{'Weighted ensemble' if model.endswith('Weighted') else DISPLAY_NAMES[model]} & {metrics.loc[model, 'brier']:.4f} & {metrics.loc[model, 'ece_10']:.4f} & "
            f"{stable['calibration_intercept'].median():.3f} & {stable['calibration_slope'].median():.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    (output_dir / "table_calibration.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_external_table(results_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(results_dir / "external_transportability" / "external_transportability_results.csv")
    frame = frame.query("cohort == 'Hungarian AMI registry' and endpoint == 'death_30d' and subset == 'all first events'")
    lines = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{Cross-cohort 30-day mortality transportability stress test in 29,596 Hungarian index events (3,888 deaths). Values are estimates (95\% patient-bootstrap intervals).}",
        r"\label{tab:external}", r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Model & AUROC & AUPRC & Brier & ECE$_{10}$ & Intercept & Slope & O:E \\", r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{_escape(row.model)} & ${row.auc:.3f}$ (${row.auc_ci_low:.3f}$--${row.auc_ci_high:.3f}$) & "
            f"${row.auprc:.3f}$ (${row.auprc_ci_low:.3f}$--${row.auprc_ci_high:.3f}$) & "
            f"${row.brier:.4f}$ (${row.brier_ci_low:.4f}$--${row.brier_ci_high:.4f}$) & "
            f"${row.ece:.4f}$ (${row.ece_ci_low:.4f}$--${row.ece_ci_high:.4f}$) & "
            f"${row.calibration_intercept:.3f}$ & ${row.calibration_slope:.3f}$ & ${row.oe_ratio:.3f}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    (output_dir / "table_external.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manuscript_facts(results_dir: Path, output_dir: Path) -> None:
    """Export a compact evidence ledger used for prose and cross-file QA."""
    metrics = pd.read_csv(results_dir / "internal_model_metrics.csv").set_index("model")
    bootstrap = pd.read_csv(results_dir / "internal_bootstrap_summary.csv")
    primary = pd.read_csv(results_dir / "primary_paired_difference.csv").set_index("metric")
    secondary = pd.read_csv(results_dir / "secondary_paired_difference_lp_rf.csv").set_index("metric")
    tests = pd.read_csv(results_dir / "primary_per_label_tests.csv")
    best_single_name = metrics.loc[MODEL_NAMES, "macro_auroc"].idxmax()
    best_all_name = metrics["macro_auroc"].idxmax()
    gcn = pd.read_csv(results_dir / "gcn_ablation" / "gcn_ablation_10_seed_full.csv")
    gcn_means = (
        gcn.groupby(["adjacency", "head", "loss"])["test_macro_auroc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    gcn_best = gcn_means.loc[gcn_means["mean"].idxmax()]
    gcn_paired = pd.read_csv(results_dir / "gcn_ablation" / "gcn_ablation_paired_differences.csv")
    synthetic = pd.read_csv(results_dir / "synthetic" / "controlled_dependency_summary.csv")
    temporal = pd.read_csv(results_dir / "temporal" / "temporal_model_metrics.csv")
    external = pd.read_csv(results_dir / "external_transportability" / "external_transportability_results.csv")
    external_primary = external.query("cohort == 'Hungarian AMI registry' and endpoint == 'death_30d' and subset == 'all first events'")

    def metric_payload(model: str) -> dict[str, float]:
        return {key: float(metrics.loc[model, key]) for key in metrics.columns}

    fact = {
        "internal": {
            "best_single": {"model": best_single_name, **metric_payload(best_single_name)},
            "best_overall": {"model": best_all_name, **metric_payload(best_all_name)},
            "tabpfn": metric_payload("TabPFN"),
            "weighted_ensemble": metric_payload("Ensemble-LP-RF-TabPFN-Weighted"),
            "primary_difference": {
                metric: {
                    "estimate": float(row.estimate),
                    "bootstrap_mean": float(row.mean_bootstrap_difference),
                    "ci_low": float(row.ci_low),
                    "ci_high": float(row.ci_high),
                }
                for metric, row in primary.iterrows()
            },
            "secondary_difference_vs_lp_rf": {
                metric: {
                    "estimate": float(row.estimate),
                    "bootstrap_mean": float(row.mean_bootstrap_difference),
                    "ci_low": float(row.ci_low),
                    "ci_high": float(row.ci_high),
                }
                for metric, row in secondary.iterrows()
            },
            "bh_significant_labels": int(tests["reject_bh_0_05"].sum()),
            "bootstrap_rows": int(len(bootstrap)),
        },
        "gcn_ablation": {
            "best_mean_configuration": {
                "adjacency": str(gcn_best["adjacency"]),
                "head": str(gcn_best["head"]),
                "loss": str(gcn_best["loss"]),
                "mean_macro_auroc": float(gcn_best["mean"]),
                "sd": float(gcn_best["std"]),
            },
            "paired_differences": gcn_paired.to_dict(orient="records"),
        },
        "synthetic": {
            "realized_lds_min": float(synthetic["realized_lds_mean"].min()),
            "realized_lds_max": float(synthetic["realized_lds_mean"].max()),
            "label_density_min": float(synthetic["label_density_mean"].min()),
            "label_density_max": float(synthetic["label_density_mean"].max()),
            "true_score_auc_min": float(synthetic["true_score_auc_mean"].min()),
            "true_score_auc_max": float(synthetic["true_score_auc_mean"].max()),
            "summary": synthetic.to_dict(orient="records"),
        },
        "temporal": temporal.to_dict(orient="records"),
        "external_primary": external_primary.to_dict(orient="records"),
    }
    (output_dir / "manuscript_facts.json").write_text(json.dumps(fact, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/generated"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_model_table(args.results_dir, args.output_dir)
    write_primary_macros(args.results_dir, args.output_dir)
    write_gcn_table(args.results_dir, args.output_dir)
    write_temporal_table(args.results_dir, args.output_dir)
    write_synthetic_table(args.results_dir, args.output_dir)
    write_primary_label_table(args.results_dir, args.output_dir)
    write_calibration_table(args.results_dir, args.output_dir)
    write_external_table(args.results_dir, args.output_dir)
    write_manuscript_facts(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()

