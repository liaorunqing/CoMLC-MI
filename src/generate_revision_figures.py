"""Generate revised submission figures from auditable result files."""

from __future__ import annotations

import os
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures", "revised")
SYNTHETIC_DIR = os.path.join(PROJECT_DIR, "output", "synthetic_controlled")
RESULT_DIR = os.path.join(PROJECT_DIR, "output", "improvements_v2")
TEMPORAL_DIR = os.path.join(PROJECT_DIR, "output", "temporal_corrected")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

COLORS = {
    "BR": "#274C77",
    "CC": "#4F7CAC",
    "BR-kNN (distance-weighted, k=10)": "#8FB9C3",
    "GCN": "#D76C4B",
    "neutral": "#66717E",
    "dark": "#243447",
    "pale": "#EDF2F6",
    "accent": "#2A9D8F",
    "gold": "#D9A441",
}


def save_publication_figure(fig, stem):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    base = os.path.join(FIGURE_DIR, stem)
    fig.savefig(base + ".svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".tiff", dpi=600, bbox_inches="tight")


def synthetic_control_figure():
    """Hero comparison plus two design-balance audit panels."""
    raw = pd.read_csv(os.path.join(SYNTHETIC_DIR, "controlled_dependency_raw.csv"))
    summary = pd.read_csv(os.path.join(SYNTHETIC_DIR, "controlled_dependency_summary.csv"))
    fig = plt.figure(figsize=(7.16, 3.75))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.35, 1.35, 1.0), hspace=0.40, wspace=0.48)
    ax_main = fig.add_subplot(grid[:, :2])
    ax_density = fig.add_subplot(grid[0, 2])
    ax_signal = fig.add_subplot(grid[1, 2])

    for model in ("BR", "CC", "BR-kNN (distance-weighted, k=10)", "GCN"):
        model_raw = raw[raw["model"] == model]
        for _, run_data in model_raw.groupby("run"):
            ax_main.plot(
                run_data["realized_lds"], run_data["macro_auc"],
                color=COLORS[model], alpha=0.08, linewidth=0.55,
            )
        model_summary = summary[summary["model"] == model]
        ax_main.errorbar(
            model_summary["realized_lds_mean"], model_summary["macro_auc_mean"],
            xerr=model_summary["realized_lds_sd"], yerr=model_summary["macro_auc_sd"],
            color=COLORS[model], marker="o", markersize=4, linewidth=1.6,
            capsize=2, label=model,
        )

    ax_main.set_xlabel("Realized label dependency score (LDS)")
    ax_main.set_ylabel("Test Macro-AUC")
    ax_main.set_title("Macro-AUC across controlled residual dependence", loc="left", weight="bold", pad=6)
    ax_main.grid(color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax_main.legend(loc="lower left", ncol=2, title="Model", fontsize=6.5, title_fontsize=6.5)
    ax_main.text(-0.13, 1.04, "a", transform=ax_main.transAxes, fontsize=9, weight="bold")

    balance = summary[summary["model"] == "BR"]
    ax_density.errorbar(
        balance["realized_lds_mean"], balance["label_density_mean"],
        yerr=balance["label_density_sd"], color=COLORS["accent"], marker="o",
        linewidth=1.4, capsize=2,
    )
    ax_density.axhline(0.10, color=COLORS["neutral"], linestyle="--", linewidth=0.9)
    ax_density.set_ylim(0.095, 0.105)
    ax_density.set_ylabel("Label density")
    ax_density.set_title("Label density held fixed", loc="left", weight="bold", pad=6)
    ax_density.grid(color="#E5E5E5", linewidth=0.5)
    ax_density.text(-0.30, 1.08, "b", transform=ax_density.transAxes, fontsize=9, weight="bold")

    ax_signal.errorbar(
        balance["realized_lds_mean"], balance["true_score_auc_mean"],
        yerr=balance["true_score_auc_sd"], color=COLORS["neutral"], marker="s",
        linewidth=1.4, capsize=2,
    )
    ax_signal.set_ylim(0.775, 0.810)
    ax_signal.set_xlabel("Realized LDS")
    ax_signal.set_ylabel("True-score AUC")
    ax_signal.set_title("Single-label signal held stable", loc="left", weight="bold", pad=6)
    ax_signal.grid(color="#E5E5E5", linewidth=0.5)
    ax_signal.text(-0.30, 1.08, "c", transform=ax_signal.transAxes, fontsize=9, weight="bold")
    fig.text(0.055, 0.99, "Controlled dependency sensitivity analysis", ha="left", va="top",
             fontsize=8.6, weight="bold", color=COLORS["dark"])
    save_publication_figure(fig, "fig_synthetic_controlled")
    plt.close(fig)


def _box(ax, xy, width, height, number, title, body, color):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor="white", edgecolor="#A8B2BD", linewidth=0.75,
    )
    ax.add_patch(patch)
    ax.add_patch(FancyBboxPatch(
        (xy[0], xy[1] + height - 0.095), width, 0.095,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor=color, edgecolor=color, linewidth=0,
    ))
    ax.text(xy[0] + 0.014, xy[1] + height - 0.047, number, color="white",
            weight="bold", va="center", ha="left", fontsize=7.2)
    ax.text(
        xy[0] + 0.043, xy[1] + height - 0.047, title,
        weight="bold", va="center", fontsize=6.5, color="white",
    )
    ax.text(
        xy[0] + 0.014, xy[1] + height - 0.125, body,
        va="top", fontsize=5.35, linespacing=1.34, color=COLORS["dark"],
    )


def workflow_figure():
    """Auditable two-lane algorithm diagram with explicit data-use boundaries."""
    fig, ax = plt.subplots(figsize=(7.16, 3.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.015, "1", "Cohort", "1,700 patients\n12 binary outcomes\nLET_IS > 0:\nin-hospital death · 70/10/20 split", "#3D6D91"),
        (0.212, "2", "Training only", "Fit imputer and scaler\nAdd two missingness flags\nCompute LC, LD and LDS\nEstimate P(Yj | Yi)", "#287C78"),
        (0.409, "3", "Candidate models", "LP-RF: 200 trees; depth 10\nECC: 50 chains\nTabPFN: top 80 features\nCatBoost: 500; depth 6", "#8A6A35"),
        (0.606, "4", "Validation only", "CatBoost early stop: 50\nDerive per-label AUC weights\nFreeze all specifications", "#B55F48"),
        (0.803, "5", "Locked test", "AUROC and AUPRC\nBrier, ECE and calibration\nPaired tests and 95% CIs\nBH correction", "#665A8C"),
    ]
    width, height, y = 0.18, 0.33, 0.54
    for x, number, title, body, color in boxes:
        _box(ax, (x, y), width, height, number, title, body, color)
    for index in range(len(boxes) - 1):
        start = boxes[index][0] + width
        end = boxes[index + 1][0]
        ax.add_patch(FancyArrowPatch(
            (start + 0.004, 0.705), (end - 0.004, 0.705),
            arrowstyle="-|>", mutation_scale=8, linewidth=0.85, color="#65717D",
        ))
    ax.text(0.015, 0.945, "Internal benchmark", fontsize=7.7, weight="bold", color=COLORS["dark"])
    ax.plot([0.015, 0.985], [0.92, 0.92], color="#D6DCE2", linewidth=0.8)
    ax.text(0.015, 0.455, "Partial external transportability (fatal outcome only)",
            fontsize=7.7, weight="bold", color=COLORS["dark"])

    external_cards = [
        (0.015, 0.12, 0.275, "Frozen source fit", "Four shared admission variables\nage · sex · prior MI · prior HF", "#E7F0F6"),
        (0.355, 0.12, 0.275, "Independent target cohort", "29,596 first Hungarian AMI events\n30-day death primary · 7-day sensitivity", "#E8F4F1"),
        (0.695, 0.12, 0.290, "Locked transportability audit", "TabPFN vs logistic · 2,000 paired bootstraps\nNo external tuning or recalibration", "#F4ECE8"),
    ]
    for x, yy, ww, title, body, face in external_cards:
        card = FancyBboxPatch((x, yy), ww, 0.19, boxstyle="round,pad=0.009,rounding_size=0.012",
                              facecolor=face, edgecolor="#A8B2BD", linewidth=0.7)
        ax.add_patch(card)
        ax.text(x + 0.014, yy + 0.133, title, fontsize=6.35, weight="bold", color=COLORS["dark"])
        ax.text(x + 0.014, yy + 0.090, body, fontsize=5.25, va="top", linespacing=1.24,
                color=COLORS["dark"])
    for x0, x1 in ((0.290, 0.355), (0.630, 0.695)):
        ax.add_patch(FancyArrowPatch((x0 + 0.005, 0.215), (x1 - 0.005, 0.215), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.85, color="#65717D"))
    save_publication_figure(fig, "fig_workflow_revised")
    plt.close(fig)


def cooccurrence_figure():
    """Training-only directed conditional label matrix with an explicit diagonal mask."""
    with open(os.path.join(PROJECT_DIR, "processed_data", "label_space_metrics.json")) as handle:
        payload = json.load(handle)
    matrix = np.asarray(payload["train"]["conditional_cooccurrence"], dtype=float)
    labels = ["AF", "SVT", "VT", "VF", "AVB", "PulEd", "Rupt", "Dress", "CHF", "ReMI", "PIA", "Leth"]
    masked = np.ma.array(matrix, mask=np.eye(len(labels), dtype=bool))
    cmap = mpl.colormaps["Blues"].copy()
    cmap.set_bad("#F0F2F4")

    fig, ax = plt.subplots(figsize=(7.16, 4.25))
    image = ax.imshow(masked, cmap=cmap, norm=Normalize(0, 1), aspect="equal")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Target label (column)")
    ax.set_ylabel("Conditioning label (row)")
    ax.set_title(r"Training-set directed conditional probabilities  $P(Y_j=1\mid Y_i=1)$",
                 loc="left", weight="bold", pad=7)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row == col:
                ax.text(col, row, "—", ha="center", va="center", color="#7B8792", fontsize=6.2)
                continue
            value = matrix[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if value >= 0.48 else "#1F2D3D", fontsize=5.3)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Conditional probability")
    save_publication_figure(fig, "fig_cooccurrence")
    plt.close(fig)


def statistical_forest_figure():
    results = pd.read_csv(os.path.join(RESULT_DIR, "FINAL_STATISTICAL_RESULTS.csv"))
    results = results[results["comparison"] == "ensemble_vs_tabpfn"].copy()
    results = results.iloc[::-1]
    y_positions = np.arange(len(results))
    differences = results["auc_difference"].to_numpy()
    low = results["auc_difference_ci_low"].to_numpy()
    high = results["auc_difference_ci_high"].to_numpy()
    errors = np.vstack((differences - low, high - differences))

    # A compact single-column canvas keeps the 12-label forest plot readable
    # without giving it the visually stretched, portrait-heavy appearance of
    # the earlier 3.5 x 4.6 in export.
    fig, ax = plt.subplots(figsize=(3.5, 3.75))
    colors = np.where(differences >= 0, "#457B9D", "#E76F51")
    for position, difference, error, color in zip(y_positions, differences, errors.T, colors):
        ax.errorbar(
            difference, position, xerr=error[:, None], fmt="o", color=color,
            ecolor=color, markersize=4, capsize=2, linewidth=1.2,
        )
    ax.axvline(0, color="#495057", linestyle="--", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(results["short_label"])
    ax.set_xlabel("AUC difference (weighted ensemble − TabPFN-3)")
    ax.set_title("Per-label AUC difference with 95% intervals", loc="left", weight="bold", pad=6)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.5)
    save_publication_figure(fig, "fig_forest_corrected")
    plt.close(fig)


def calibration_figure():
    calibration = pd.read_csv(os.path.join(RESULT_DIR, "CALIBRATION_RESULTS.csv"))
    with open(os.path.join(RESULT_DIR, "CALIBRATION_RELIABILITY_BINS.json")) as handle:
        bins = json.load(handle)
    model_order = ["lp_rf", "tabpfn", "catboost", "ens2_wt"]
    model_labels = ["LP-RF", "TabPFN-3", "CatBoost", "Weighted ens."]
    label_order = calibration[calibration["model"] == model_order[0]]["short_label"].tolist()

    fig = plt.figure(figsize=(7.16, 5.25))
    grid = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.40)
    ax_brier = fig.add_subplot(grid[0, 0])
    ax_ece = fig.add_subplot(grid[0, 1])
    ax_reliability = fig.add_subplot(grid[1, 0])
    ax_parameters = fig.add_subplot(grid[1, 1])

    def metric_matrix(metric):
        return np.vstack([
            calibration[calibration["model"] == model].set_index("short_label").loc[label_order, metric]
            for model in model_order
        ])

    for ax, metric, title, vmax in (
        (ax_brier, "brier_score", "Probability error (Brier score)", 0.35),
        (ax_ece, "adaptive_ece_10", "Adaptive 10-bin ECE", 0.55),
    ):
        matrix = metric_matrix(metric)
        image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(label_order)))
        ax.set_xticklabels(label_order, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(model_order)))
        ax.set_yticklabels(model_labels)
        ax.set_title(title, loc="left", weight="bold")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    ax_ece.set_yticklabels([])
    ax_ece.set_ylabel("")

    common_labels = ["FIBR_PREDS", "OTEK_LANC", "ZSN", "LET_IS"]
    common_names = ["AF", "PulEd", "CHF", "Leth"]
    common_colors = ["#457B9D", "#2A9D8F", "#E9C46A", "#E76F51"]
    ax_reliability.plot([0, 1], [0, 1], linestyle="--", color="#6C757D", linewidth=0.9)
    for label, short, color in zip(common_labels, common_names, common_colors):
        frame = pd.DataFrame(bins[f"ens2_wt:{label}"])
        ax_reliability.plot(
            frame["mean_probability"], frame["observed_frequency"],
            marker="o", markersize=3, linewidth=1.1, color=color, label=short,
        )
    ax_reliability.set_xlim(0, 0.75)
    ax_reliability.set_ylim(0, 0.75)
    ax_reliability.set_aspect("equal", adjustable="box")
    ax_reliability.set_xlabel("Mean predicted probability")
    ax_reliability.set_ylabel("Observed frequency")
    ax_reliability.set_title("Weighted-ensemble reliability", loc="left", weight="bold")
    ax_reliability.legend(ncol=2, title="Common labels", loc="upper left", fontsize=5.8,
                          title_fontsize=5.8, handlelength=1.6, borderpad=0.3, labelspacing=0.25)
    ax_reliability.grid(color="#E5E5E5", linewidth=0.5)

    ensemble = calibration[calibration["model"] == "ens2_wt"].set_index("short_label").loc[label_order]
    ax_parameters.scatter(
        ensemble["calibration_slope"], ensemble["calibration_intercept"],
        color="#6C757D", s=20,
    )
    label_offsets = {
        "AF": (4, -7), "SVT": (4, 4), "VT": (4, 4), "VF": (-18, 4),
        "AVB": (4, -7), "PulEd": (4, 5), "Rupt": (4, -8), "Dress": (4, 4),
        "CHF": (4, -8), "ReMI": (4, 5), "PIA": (4, -7), "Leth": (4, -8),
    }
    for label, row in ensemble.iterrows():
        ax_parameters.annotate(label, (row["calibration_slope"], row["calibration_intercept"]),
                               xytext=label_offsets.get(label, (4, 3)), textcoords="offset points", fontsize=5.0)
    ax_parameters.scatter([1], [0], marker="*", s=70, color="#2A9D8F", label="Ideal (1, 0)")
    ax_parameters.axvline(1, color="#B8B8B8", linewidth=0.7)
    ax_parameters.axhline(0, color="#B8B8B8", linewidth=0.7)
    ax_parameters.set_xlabel("Calibration slope")
    ax_parameters.set_ylabel("Calibration intercept")
    ax_parameters.set_title("Per-label recalibration parameters", loc="left", weight="bold")
    ax_parameters.legend(loc="lower right")
    ax_parameters.grid(color="#E5E5E5", linewidth=0.5)

    for label, ax in zip("abcd", (ax_brier, ax_ece, ax_reliability, ax_parameters)):
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontsize=9, weight="bold")
    fig.text(0.055, 0.995, "Calibration audit across models and complications",
             ha="left", va="top", fontsize=8.6, weight="bold", color=COLORS["dark"])
    save_publication_figure(fig, "fig_calibration")
    plt.close(fig)


def temporal_figure():
    frame = pd.read_csv(os.path.join(TEMPORAL_DIR, "temporal_auc_table.csv"))
    frame = frame.rename(columns={frame.columns[0]: "label"})
    horizons = ["Admission (t0)", "24 hours (t1)", "48 hours (t2)", "72 hours (t3)"]
    short = {
        "FIBR_PREDS": "AF", "PREDS_TAH": "SVT", "JELUD_TAH": "VT",
        "FIBR_JELUD": "VF", "A_V_BLOK": "AVB", "OTEK_LANC": "PulEd",
        "RAZRIV": "Rupt", "DRESSLER": "Dress", "ZSN": "CHF",
        "REC_IM": "ReMI", "P_IM_STEN": "PIA", "LET_IS": "Leth",
    }
    matrix = frame[horizons].to_numpy(float)
    macro = matrix.mean(axis=0)
    # The figure spans both IEEE columns.  A wider, shallower canvas avoids
    # vertically elongating the heat-map cells and the companion line panel.
    fig, (ax_heat, ax_macro) = plt.subplots(
        1, 2, figsize=(7.16, 3.55), gridspec_kw={"width_ratios": (1.62, 1), "wspace": 0.38},
    )
    image = ax_heat.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.55, vmax=0.85)
    ax_heat.set_yticks(np.arange(len(frame)))
    ax_heat.set_yticklabels([short[value] for value in frame["label"]])
    ax_heat.set_xticks(np.arange(4))
    ax_heat.set_xticklabels(["Admission", "24 h", "48 h", "72 h"])
    ax_heat.set_title("Per-label AUC by feature horizon", loc="left", weight="bold", pad=6)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] < 0.68 else "black"
            ax_heat.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center",
                         fontsize=5.4, color=color)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.04, pad=0.02)
    colorbar.ax.set_title("AUC", fontsize=6, pad=3)

    ax_macro.plot(np.arange(4), macro, marker="o", color="#2A9D8F", linewidth=1.6)
    ax_macro.set_xticks(np.arange(4))
    ax_macro.set_xticklabels(["Admission", "24 h", "48 h", "72 h"], rotation=15, ha="right")
    ax_macro.set_ylim(0.67, 0.74)
    ax_macro.set_ylabel("Macro-AUC")
    ax_macro.set_title("Macro-AUC sensitivity analysis", loc="left", weight="bold", pad=6)
    ax_macro.grid(color="#E5E5E5", linewidth=0.5)
    for position, value in enumerate(macro):
        x_offset = 5 if position == 0 else (-2 if position == len(macro) - 1 else 0)
        alignment = "left" if position == 0 else "center"
        ax_macro.annotate(f"{value:.4f}", (position, value), xytext=(x_offset, 7),
                          textcoords="offset points", ha=alignment, fontsize=6)
    ax_heat.text(-0.16, 1.04, "a", transform=ax_heat.transAxes, fontsize=9, weight="bold")
    ax_macro.text(-0.16, 1.04, "b", transform=ax_macro.transAxes, fontsize=9, weight="bold")
    fig.text(0.055, 0.99, "Four-horizon neural sensitivity analysis", ha="left", va="top",
             fontsize=8.6, weight="bold", color=COLORS["dark"])
    save_publication_figure(fig, "fig_temporal_corrected")
    plt.close(fig)


def main():
    synthetic_control_figure()
    workflow_figure()
    cooccurrence_figure()
    statistical_forest_figure()
    calibration_figure()
    temporal_figure()
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
