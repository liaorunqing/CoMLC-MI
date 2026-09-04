"""Generate the manuscript figure for partial external transportability.

Figure contract
---------------
Conclusion: source-fitted reduced common-feature models retain modest ranking
performance in the contemporary Hungarian cohort, but both overestimate
30-day mortality. Logistic regression discriminates slightly better, whereas
TabPFN has lower Brier score and ECE.
Evidence: 2,000 paired patient-level bootstrap replicates and ten adaptive
(equal-frequency) calibration bins among 29,596 first AMI events.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "output" / "benchmark" / "external_transportability"
FIGURE_DIR = ROOT / "figures" / "manuscript"

TABPFN = "#2166AC"
LOGISTIC = "#4D4D4D"
ACCENT = "#D55E00"
GRID = "#D9D9D9"
MODEL_COLORS = {"TabPFN-3": TABPFN, "Logistic regression": LOGISTIC}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def wilson_interval(events: np.ndarray, n: np.ndarray, z: float = 1.959963984540054) -> tuple[np.ndarray, np.ndarray]:
    """Return two-sided 95% Wilson intervals for binomial proportions."""
    p = events / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denominator
    return np.maximum(0.0, centre - half_width), np.minimum(1.0, centre + half_width)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.10, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def make_figure() -> plt.Figure:
    metrics = pd.read_csv(RESULT_DIR / "external_transportability_results.csv")
    bins = pd.read_csv(RESULT_DIR / "external_transportability_calibration_bins.csv")
    external = metrics.query(
        "cohort == 'Hungarian AMI registry' and endpoint == 'death_30d' and subset == 'all first events'"
    ).set_index("model")
    reliability = bins.query(
        "cohort == 'Hungarian AMI registry' and endpoint == 'death_30d' and subset == 'all first events'"
    )

    # Give the reliability curve a square hero panel and stack the two compact
    # summaries beside it.  This avoids compressing three unrelated axes into
    # a shallow single row.
    fig = plt.figure(figsize=(7.16, 4.85), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.42],
        height_ratios=[1.0, 1.0],
        wspace=0.40,
        hspace=0.52,
    )
    ax_discrimination = fig.add_subplot(grid[0, 0])
    ax_error = fig.add_subplot(grid[1, 0])
    ax_calibration = fig.add_subplot(grid[:, 1])

    # a, discrimination with percentile bootstrap intervals.
    y_positions = {("AUROC", "TabPFN-3"): 3.15, ("AUROC", "Logistic regression"): 2.75,
                   ("AUPRC", "TabPFN-3"): 1.45, ("AUPRC", "Logistic regression"): 1.05}
    metric_fields = {
        "AUROC": ("auc", "auc_ci_low", "auc_ci_high"),
        "AUPRC": ("auprc", "auprc_ci_low", "auprc_ci_high"),
    }
    for label, fields in metric_fields.items():
        for model in ("TabPFN-3", "Logistic regression"):
            estimate, low, high = (float(external.loc[model, field]) for field in fields)
            y = y_positions[(label, model)]
            ax_discrimination.errorbar(
                estimate,
                y,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=MODEL_COLORS[model],
                markersize=4.4,
                capsize=2.2,
                linewidth=1.1,
            )
            ax_discrimination.text(estimate + 0.015, y, f"{estimate:.3f}", va="center", fontsize=6.8)
    prevalence = 3888 / 29596
    ax_discrimination.vlines(0.5, 2.50, 3.40, color="#888888", linestyle="--", linewidth=0.8)
    ax_discrimination.vlines(prevalence, 0.80, 1.70, color="#888888", linestyle="--", linewidth=0.8)
    ax_discrimination.text(0.5, 3.43, "0.50", ha="center", va="bottom", color="#666666", fontsize=5.5)
    ax_discrimination.text(prevalence, 1.73, f"prev. {prevalence:.2f}", ha="left", va="bottom",
                           color="#666666", fontsize=5.5)
    ax_discrimination.set_xlim(0.10, 0.76)
    ax_discrimination.set_ylim(0.55, 3.55)
    ax_discrimination.set_yticks([1.25, 2.95], ["AUPRC", "AUROC"])
    ax_discrimination.set_xlabel("Estimate (95% CI)")
    ax_discrimination.set_title("Discrimination", loc="left", fontweight="bold", pad=6)
    ax_discrimination.grid(axis="x", color=GRID, linewidth=0.5)
    panel_label(ax_discrimination, "a")

    # b, reliability curve. Each point is one equal-frequency risk decile.
    ax_calibration.plot([0, 0.40], [0, 0.40], color="#7F7F7F", linestyle="--", linewidth=1.0, label="Ideal")
    for model in ("TabPFN-3", "Logistic regression"):
        model_bins = reliability.loc[reliability["model"].eq(model)].sort_values("bin")
        low, high = wilson_interval(model_bins["events"].to_numpy(), model_bins["n"].to_numpy())
        observed = model_bins["observed_rate"].to_numpy()
        ax_calibration.errorbar(
            model_bins["mean_predicted"],
            observed,
            yerr=np.vstack([observed - low, high - observed]),
            color=MODEL_COLORS[model],
            marker="o" if model == "TabPFN-3" else "s",
            markersize=3.5,
            linewidth=1.25,
            capsize=1.7,
            label=model.replace(" regression", ""),
        )
    ax_calibration.set_xlim(0, 0.40)
    ax_calibration.set_ylim(0, 0.40)
    ax_calibration.set_aspect("equal", adjustable="box")
    ax_calibration.set_xlabel("Mean predicted risk")
    ax_calibration.set_ylabel("Observed 30-day mortality")
    ax_calibration.set_title("Calibration by risk decile", loc="left", fontweight="bold", pad=6)
    ax_calibration.grid(color=GRID, linewidth=0.5)
    ax_calibration.legend(loc="upper left", fontsize=6.7)
    panel_label(ax_calibration, "b")

    # c, proper scoring and calibration error; lower values are better.
    x = np.array([0, 1], dtype=float)
    width = 0.31
    for offset, model in zip((-width / 2, width / 2), ("TabPFN-3", "Logistic regression")):
        values = [external.loc[model, "brier"], external.loc[model, "ece"]]
        bars = ax_error.bar(
            x + offset,
            values,
            width=width,
            color=MODEL_COLORS[model],
            label=model.replace(" regression", ""),
        )
        for bar, value in zip(bars, values):
            ax_error.text(bar.get_x() + bar.get_width() / 2, value - 0.004, f"{value:.3f}",
                          ha="center", va="top", fontsize=5.8, color="white", fontweight="bold")
    ax_error.set_xticks(x, ["Brier\nscore", "10-bin\nECE"])
    ax_error.set_ylim(0, 0.13)
    ax_error.set_ylabel("Error (lower is better)")
    ax_error.set_title("Probability error", loc="left", fontweight="bold", pad=6)
    ax_error.grid(axis="y", color=GRID, linewidth=0.5)
    ax_error.legend(loc="upper right", fontsize=6.7)
    panel_label(ax_error, "c")

    fig.text(0.055, 0.995, "Cross-cohort mortality transportability stress test",
             ha="left", va="top", fontsize=8.6, fontweight="bold", color="#243447")
    return fig


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    base = FIGURE_DIR / "fig_external_transportability"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
