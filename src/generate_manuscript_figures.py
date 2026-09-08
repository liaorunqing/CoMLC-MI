"""Generate manuscript figures from machine-readable benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .benchmark_features import LABEL_COLS, prepare_outcomes


COLORS = {
    "navy": "#24465F",
    "blue": "#4C78A8",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#D55E5E",
    "gray": "#6B7280",
    "light_blue": "#E7F0F6",
    "light_teal": "#E4F3F0",
    "light_gold": "#FBF3D5",
    "light_gray": "#F2F4F6",
}
ENSEMBLE_WEIGHTED = "Ensemble-LP-RF-TabPFN-Weighted"
FIGURE_NAMES = {
    "BR-LR": "BR–LR",
    "BR-XGBoost": "BR–XGBoost",
    "BR-LightGBM": "BR–LightGBM",
    "BR-CatBoost": "BR–CatBoost",
    "ECC-LightGBM": "ECC–LightGBM",
    "LP-RF": "LP–RF",
    "RAkELd-RF": "RAkELd–RF",
    "Shared-MLP": "Shared MLP",
    "MultiTask-DNN": "Multitask DNN",
    "TabPFN": "TabPFN-3",
    "Ensemble-LP-RF-TabPFN-Equal": "Ensemble (equal)",
    "Ensemble-LP-RF-TabPFN-Weighted": "Ensemble (inner-OOF weighted)",
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


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def save_publication_figure(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _box(ax, x, y, w, h, text, face, edge=None, fontsize=6.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.9,
        edgecolor=edge or COLORS["navy"],
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, start, end, color=None, style="-", connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            linestyle=style,
            color=color or COLORS["navy"],
            connectionstyle=connectionstyle,
        )
    )


def workflow_figure(output_dir: Path) -> None:
    """Nested-validation and data-use boundary schematic."""
    fig, ax = plt.subplots(figsize=(7.15, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.97, "a", fontweight="bold", fontsize=9, va="top")
    ax.text(0.05, 0.97, "Internal 12-label benchmark: every decision stays inside the outer training fold", fontweight="bold", fontsize=8, va="top")

    _box(ax, 0.03, 0.76, 0.14, 0.11, "1,700 patients\n91 admission-stage\nfeatures; 12 labels", COLORS["light_blue"])
    _box(ax, 0.23, 0.76, 0.14, 0.11, "5 repeats ×\n5 outer folds", COLORS["light_blue"])
    _box(ax, 0.43, 0.76, 0.20, 0.11, "Outer training patients\nPreprocessing fit here only", COLORS["light_teal"])
    _box(ax, 0.72, 0.76, 0.20, 0.11, "Outer test patients\nPredictions only; never used\nfor fitting or decisions", "#FCE8E6", COLORS["red"])
    _arrow(ax, (0.17, 0.815), (0.23, 0.815))
    _arrow(ax, (0.37, 0.815), (0.43, 0.815))
    _arrow(
        ax,
        (0.36, 0.86),
        (0.73, 0.86),
        COLORS["red"],
        "--",
        "arc3,rad=-0.22",
    )

    _box(ax, 0.10, 0.52, 0.20, 0.13, "4 inner folds\nFold-local imputation, scaling,\nTabPFN top-80 selection", COLORS["light_gold"])
    _box(ax, 0.40, 0.52, 0.20, 0.13, "Inner OOF decisions\nLP-RF–TabPFN weights;\nCatBoost/NN training rounds", COLORS["light_gold"])
    _box(ax, 0.70, 0.52, 0.20, 0.13, "Refit on complete\nouter training fold\n(all 10 model families)", COLORS["light_teal"])
    _arrow(ax, (0.53, 0.76), (0.20, 0.65))
    _arrow(ax, (0.30, 0.585), (0.40, 0.585))
    _arrow(ax, (0.60, 0.585), (0.70, 0.585))
    _arrow(ax, (0.80, 0.65), (0.82, 0.76))

    _box(ax, 0.74, 0.31, 0.22, 0.11, "Each patient receives\none OOF prediction\nper repeat", COLORS["light_blue"])
    _box(ax, 0.43, 0.31, 0.22, 0.11, "Average five OOF\npredictions per patient", COLORS["light_blue"])
    _box(ax, 0.10, 0.31, 0.22, 0.11, "Paired patient bootstrap,\nDeLong/permutation, BH,\ncalibration intervals", COLORS["light_blue"])
    # Route the held-out-test connection outside the refit box.  The previous
    # curved edge crossed the refit text and obscured the data-use boundary.
    ax.plot(
        [0.92, 0.965, 0.965],
        [0.815, 0.70, 0.445],
        color=COLORS["navy"],
        linewidth=0.9,
        solid_capstyle="round",
    )
    _arrow(ax, (0.965, 0.445), (0.94, 0.42), connectionstyle="arc3,rad=0")
    _arrow(ax, (0.74, 0.365), (0.65, 0.365))
    _arrow(ax, (0.43, 0.365), (0.32, 0.365))
    ax.text(
        0.50,
        0.275,
        "Key locked settings are shown; complete model hyperparameters are reported in Methods and Appendix B.",
        ha="center",
        va="center",
        fontsize=5.2,
        color=COLORS["gray"],
    )

    ax.plot([0.02, 0.98], [0.245, 0.245], color="#CDD4DA", lw=0.8)
    ax.text(0.01, 0.22, "b", fontweight="bold", fontsize=9, va="top")
    ax.text(0.05, 0.22, "Separate cross-cohort mortality transportability stress test", fontweight="bold", fontsize=8, va="top")
    _box(ax, 0.07, 0.045, 0.21, 0.11, "Krasnoyarsk source\n4 shared admission variables;\nin-hospital death", COLORS["light_teal"])
    _box(ax, 0.39, 0.045, 0.21, 0.11, "Fit source-only TabPFN-3\nand logistic regression\n(no external tuning)", COLORS["light_gold"])
    _box(ax, 0.71, 0.045, 0.23, 0.11, "29,596 Hungarian index events\n30-day mortality primary;\n7-day/complete-case sensitivity", COLORS["light_blue"])
    _arrow(ax, (0.28, 0.10), (0.39, 0.10))
    _arrow(ax, (0.60, 0.10), (0.71, 0.10))
    save_publication_figure(fig, output_dir / "fig1_nested_workflow")


def cooccurrence_figure(results_dir: Path, output_dir: Path) -> None:
    """Square full-cohort descriptive conditional co-occurrence matrix."""
    config_path = Path("configs/comlc_mi_benchmark.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = pd.read_csv(config["dataset"])
    outcomes = prepare_outcomes(data).to_numpy(dtype=np.int8)
    counts = outcomes.sum(axis=0)
    conditional = np.zeros((len(LABEL_COLS), len(LABEL_COLS)), dtype=float)
    for source in range(len(LABEL_COLS)):
        if counts[source] > 0:
            conditional[source] = (outcomes[:, source, None] * outcomes).sum(axis=0) / counts[source]
    np.fill_diagonal(conditional, 1.0)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    image = ax.imshow(conditional, cmap="YlGnBu", vmin=0, vmax=1, aspect="equal")
    display_labels = [LABEL_DISPLAY[label] for label in LABEL_COLS]
    ax.set_xticks(np.arange(len(LABEL_COLS)), display_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(LABEL_COLS)), display_labels)
    ax.set_xlabel("Conditioned outcome (column)")
    ax.set_ylabel("Conditioning outcome (row)")
    ax.set_title("Directed conditional co-occurrence", fontweight="bold")
    for row in range(len(LABEL_COLS)):
        for column in range(len(LABEL_COLS)):
            value = conditional[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=5.0,
                    color="white" if value > 0.58 else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(r"$P(Y_{column}=1\mid Y_{row}=1)$")
    save_publication_figure(fig, output_dir / "fig_cooccurrence")


def model_performance_figure(results_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(results_dir / "internal_bootstrap_summary.csv")
    metrics = ["macro_auroc", "macro_auprc", "macro_f1_0_5", "brier"]
    titles = ["Macro-AUROC", "Macro-AUPRC", "Macro-F1 at 0.5", "Brier score"]
    names = summary["model"].drop_duplicates().tolist()
    order = (
        summary[summary["metric"] == "macro_auroc"]
        .sort_values("estimate", ascending=True)["model"]
        .tolist()
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 4.4), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(order))
    for panel, (ax, metric, title) in enumerate(zip(axes, metrics, titles)):
        subset = summary[summary["metric"] == metric].set_index("model").loc[order]
        color = [COLORS["teal"] if name == ENSEMBLE_WEIGHTED else COLORS["blue"] if name == "TabPFN" else "#9AA6B2" for name in order]
        estimate = subset["estimate"].to_numpy()
        errors = np.vstack([estimate - subset["ci_low"].to_numpy(), subset["ci_high"].to_numpy() - estimate])
        for row, (point, low_error, high_error, row_color) in enumerate(
            zip(estimate, errors[0], errors[1], color)
        ):
            ax.errorbar(
                point,
                row,
                xerr=np.array([[low_error], [high_error]]),
                fmt="none",
                ecolor=row_color,
                elinewidth=1.0,
                capsize=2,
            )
        ax.scatter(estimate, y, c=color, s=18, zorder=3)
        ax.set_title(title)
        ax.grid(axis="x", color="#E5E7EB", lw=0.6)
        ax.text(-0.18, 1.02, chr(ord("a") + panel), transform=ax.transAxes, fontweight="bold", fontsize=9)
        if panel == 0:
            ax.set_yticks(y, [FIGURE_NAMES.get(name, name) for name in order])
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("Repeated outer-fold performance with 95% patient-bootstrap intervals", y=1.01, fontsize=9, fontweight="bold")
    save_publication_figure(fig, output_dir / "fig_model_performance")


def forest_figure(results_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(results_dir / "primary_per_label_bootstrap.csv")
    tests = pd.read_csv(results_dir / "primary_per_label_tests.csv")[["label", "p_value_bh"]]
    frame = frame.merge(tests, on="label", how="left", validate="one_to_one")
    frame = frame.sort_values("difference")
    y = np.arange(len(frame))
    colors = np.where(
        frame["p_value_bh"] < 0.05,
        COLORS["teal"],
        np.where(frame["ci_low"] > 0, COLORS["orange"], COLORS["blue"]),
    )
    # A label-wise forest plot is intrinsically vertical.  Keep it at the IEEE
    # single-column width so intervals are not visually inflated by a shallow,
    # full-width canvas.
    fig, ax = plt.subplots(figsize=(3.45, 4.25))
    estimate = frame["difference"].to_numpy()
    low_errors = estimate - frame["ci_low"].to_numpy()
    high_errors = frame["ci_high"].to_numpy() - estimate
    for row, (point, low_error, high_error, row_color) in enumerate(
        zip(estimate, low_errors, high_errors, colors)
    ):
        ax.errorbar(
            point,
            row,
            xerr=np.array([[low_error], [high_error]]),
            fmt="none",
            ecolor=row_color,
            elinewidth=1.2,
            capsize=2,
        )
    ax.scatter(estimate, y, c=colors, s=20, zorder=3)
    ax.axvline(0, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.set_yticks(y, frame["label"].map(LABEL_DISPLAY))
    ax.set_xlabel("AUROC difference: weighted ensemble − TabPFN")
    ax.set_title("Per-label paired differences (95% CI)", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=COLORS["teal"], lw=1.2, label="BH-adjusted $q<0.05$"),
            Line2D([0], [0], marker="o", color=COLORS["orange"], lw=1.2, label="CI excludes 0; BH non-significant"),
            Line2D([0], [0], marker="o", color=COLORS["blue"], lw=1.2, label="CI includes 0"),
        ],
        loc="lower right",
        ncol=1,
        frameon=False,
        fontsize=5.3,
    )
    save_publication_figure(fig, output_dir / "fig_primary_forest")


def calibration_figure(results_dir: Path, output_dir: Path) -> None:
    """Per-label error scores plus aggregate label-specific decile reliability."""
    per_label = pd.read_csv(results_dir / "internal_per_label_metrics.csv")
    bins = pd.read_csv(results_dir / "internal_calibration_bins.csv")
    models = ["TabPFN", ENSEMBLE_WEIGHTED]
    labels = per_label["label"].drop_duplicates().tolist()
    short_names = {"TabPFN": "TabPFN-3", ENSEMBLE_WEIGHTED: "Weighted ensemble"}
    palette = {"TabPFN": COLORS["blue"], ENSEMBLE_WEIGHTED: COLORS["teal"]}
    fig = plt.figure(figsize=(7.15, 5.15))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.25],
        hspace=0.48,
        wspace=0.30,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]
    x = np.arange(len(labels))
    for model in models:
        subset = per_label[per_label["model"] == model].set_index("label").loc[labels]
        axes[0].plot(x, subset["brier"], marker="o", ms=3, lw=1.0, color=palette[model], label=short_names[model])
        axes[1].plot(x, subset["ece_10"], marker="o", ms=3, lw=1.0, color=palette[model], label=short_names[model])
    for ax, title, ylabel in zip(axes[:2], ["a  Per-label Brier score", "b  Per-label ECE"], ["Brier score", "ECE$_{10}$"]):
        ax.set_xticks(x, [LABEL_DISPLAY[label] for label in labels], rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    for model in models:
        subset = bins[bins["model"] == model]
        pooled = pd.DataFrame(
            [
                {
                    "bin": bin_id,
                "mean_predicted": np.average(frame["mean_predicted"], weights=frame["n"]),
                "observed_rate": np.average(frame["observed_rate"], weights=frame["n"]),
                "n": int(frame["n"].sum()),
                }
                for bin_id, frame in subset.groupby("bin", sort=True)
            ]
        )
        axes[2].plot(pooled["mean_predicted"], pooled["observed_rate"], marker="o", ms=4, lw=1.2,
                     color=palette[model], label=short_names[model])
    maximum = max(0.35, float(axes[2].get_xlim()[1]), float(axes[2].get_ylim()[1]))
    axes[2].plot([0, maximum], [0, maximum], linestyle="--", color=COLORS["gray"], lw=0.8, label="Ideal")
    axes[2].set_xlim(0, maximum)
    axes[2].set_ylim(0, maximum)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed frequency")
    axes[2].set_title("c  Mean of label-specific deciles", loc="left", fontweight="bold")
    axes[2].grid(color="#E5E7EB", lw=0.6)
    axes[2].legend(loc="upper left", frameon=False, fontsize=5.5)
    fig.suptitle("Internal calibration audit (n = 1,700 patients; 12 outcomes)", y=0.995, fontsize=9, fontweight="bold")
    save_publication_figure(fig, output_dir / "fig_internal_calibration")


def calibration_small_multiples(results_dir: Path, output_dir: Path) -> None:
    """Twelve label-specific reliability panels for the three calibration comparators."""
    primary = pd.read_csv(results_dir / "internal_calibration_bins.csv")
    unweighted = pd.read_csv(results_dir / "calibration_sensitivity_unweighted_lr_bins.csv")
    bins = pd.concat([primary, unweighted], ignore_index=True)
    models = ["TabPFN", ENSEMBLE_WEIGHTED, "BR-LR-unweighted"]
    names = {
        "TabPFN": "TabPFN-3",
        ENSEMBLE_WEIGHTED: "Weighted ensemble",
        "BR-LR-unweighted": "Unweighted BR-LR",
    }
    palette = {
        "TabPFN": COLORS["blue"],
        ENSEMBLE_WEIGHTED: COLORS["teal"],
        "BR-LR-unweighted": COLORS["orange"],
    }
    per_label = pd.read_csv(results_dir / "internal_per_label_metrics.csv")
    event_counts = per_label.query("model == 'TabPFN'").set_index("label")["positive_n"].to_dict()
    fig, axes = plt.subplots(3, 4, figsize=(7.15, 6.1), constrained_layout=True)
    for ax, label in zip(axes.ravel(), LABEL_COLS):
        upper = 0.0
        for model in models:
            subset = bins[(bins["model"] == model) & (bins["label"] == label)].sort_values("bin")
            ax.plot(
                subset["mean_predicted"], subset["observed_rate"],
                marker="o", markersize=2.3, linewidth=0.85,
                color=palette[model], label=names[model],
            )
            if len(subset):
                upper = max(upper, float(subset[["mean_predicted", "observed_rate"]].to_numpy().max()))
        upper = max(upper * 1.08, 0.08)
        ax.plot([0, upper], [0, upper], "--", color=COLORS["gray"], linewidth=0.65)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#E5E7EB", linewidth=0.45)
        ax.set_title(f"{LABEL_DISPLAY[label]} ($n_+$={event_counts[label]})", fontsize=6.5, fontweight="bold")
    for ax in axes[-1, :]:
        ax.set_xlabel("Predicted", fontsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed", fontsize=6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.025), ncol=3, frameon=False)
    fig.suptitle("Outcome-specific calibration from equal-frequency deciles (n = 1,700)", fontsize=9, fontweight="bold", y=1.055)
    save_publication_figure(fig, output_dir / "fig_calibration_small_multiples")


def gcn_figure(results_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(results_dir / "gcn_ablation" / "gcn_ablation_paired_differences.csv")
    frame["configuration"] = frame["head"] + " / " + frame["loss"] + " / " + frame["contrast"]
    frame = frame.sort_values("mean_difference")
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    estimate = frame["mean_difference"].to_numpy()
    colors = np.where(
        frame["ci_low"].to_numpy() > 0,
        COLORS["teal"],
        np.where(frame["ci_high"].to_numpy() < 0, COLORS["red"], COLORS["gray"]),
    )
    low_errors = estimate - frame["ci_low"].to_numpy()
    high_errors = frame["ci_high"].to_numpy() - estimate
    for row, (point, low_error, high_error, row_color) in enumerate(
        zip(estimate, low_errors, high_errors, colors)
    ):
        ax.errorbar(
            point,
            row,
            xerr=np.array([[low_error], [high_error]]),
            fmt="none",
            ecolor=row_color,
            capsize=2,
            elinewidth=1.1,
        )
    ax.scatter(estimate, y, c=colors, s=20, zorder=3)
    ax.axvline(0, color=COLORS["gray"], linestyle="--", lw=0.8)
    ax.set_yticks(y, frame["configuration"])
    ax.set_xlabel("Paired Macro-AUROC difference across 10 seeds")
    ax.set_title("Graph variants relative to matched no-graph controls", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    save_publication_figure(fig, output_dir / "fig_gcn_paired")


def temporal_figure(results_dir: Path, output_dir: Path) -> None:
    labels = pd.read_csv(results_dir / "temporal" / "temporal_per_label_metrics.csv")
    metrics = pd.read_csv(results_dir / "temporal" / "temporal_model_metrics.csv")
    pivot = labels.pivot(index="label", columns="horizon", values="auroc").loc[:, ["Admission", "24 h", "48 h", "72 h"]]
    # The heatmap and the horizon trend answer different questions.  Stack
    # them at single-column width instead of compressing both into one row.
    fig, (ax_heat, ax_line) = plt.subplots(
        2,
        1,
        figsize=(3.45, 5.95),
        gridspec_kw={"height_ratios": [3.25, 1.35], "hspace": 0.58},
    )
    image = ax_heat.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0.5, vmax=0.9)
    ax_heat.set_xticks(np.arange(4), pivot.columns)
    ax_heat.set_yticks(np.arange(len(pivot)), [LABEL_DISPLAY[label] for label in pivot.index])
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            ax_heat.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.72 else "black", fontsize=6)
    ax_heat.set_title("a  Per-label AUROC", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.03)
    colorbar.set_label("AUROC")
    ax_line.plot(metrics["horizon"], metrics["macro_auroc"], marker="o", color=COLORS["teal"], lw=1.7)
    annotation_offsets = {
        "Admission": (0, 7),
        "24 h": (0, 7),
        "48 h": (0, -18),
        "72 h": (0, 7),
    }
    for x, y, count in zip(metrics["horizon"], metrics["macro_auroc"], metrics["feature_count"]):
        dx, dy = annotation_offsets[str(x)]
        ax_line.annotate(
            f"{y:.3f}\n({count} features)",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="top" if dy < 0 else "bottom",
            fontsize=6,
        )
    ax_line.set_ylabel("Macro-AUROC")
    ax_line.set_title("b  Accumulated-information sensitivity", loc="left", fontweight="bold", pad=8)
    values = metrics["macro_auroc"].to_numpy()
    ax_line.set_ylim(values.min() - 0.0010, values.max() + 0.0023)
    ax_line.tick_params(axis="x", labelrotation=18)
    ax_line.margins(x=0.10)
    ax_line.grid(color="#E5E7EB", lw=0.6)
    save_publication_figure(fig, output_dir / "fig_temporal")


def synthetic_figure(results_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(results_dir / "synthetic" / "controlled_dependency_summary.csv")
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    palette = {"BR": COLORS["blue"], "CC": COLORS["orange"], "BR-kNN (distance-weighted, k=10)": COLORS["gray"], "GCN": COLORS["teal"]}
    for model, subset in frame.groupby("model"):
        subset = subset.sort_values("realized_lds_mean")
        ax.errorbar(
            subset["realized_lds_mean"],
            subset["macro_auc_mean"],
            xerr=subset["realized_lds_sd"],
            yerr=subset["macro_auc_sd"],
            marker="o",
            markersize=4,
            linewidth=1.3,
            capsize=2,
            color=palette[model],
            label=model,
        )
    ax.set_xlabel("Realized label dependency score")
    ax.set_ylabel("Macro-AUROC, mean ± s.d. (10 datasets)")
    ax.set_title("Controlled dependency sensitivity", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(color="#E5E7EB", lw=0.6)
    save_publication_figure(fig, output_dir / "fig_synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/manuscript"))
    parser.add_argument("--workflow-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow_figure(args.output_dir)
    if args.workflow_only:
        return
    cooccurrence_figure(args.results_dir, args.output_dir)
    model_performance_figure(args.results_dir, args.output_dir)
    forest_figure(args.results_dir, args.output_dir)
    calibration_figure(args.results_dir, args.output_dir)
    calibration_small_multiples(args.results_dir, args.output_dir)
    gcn_figure(args.results_dir, args.output_dir)
    temporal_figure(args.results_dir, args.output_dir)
    synthetic_figure(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()

