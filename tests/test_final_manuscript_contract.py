from pathlib import Path

import pandas as pd


def test_manuscript_uses_integrated_appendices_and_current_model_names():
    text = Path("paper/gai_revised_clean.tex").read_text(encoding="utf-8")
    banned = [
        "Supplementary Material",
        "Supplementary Section",
        "access-2026-35668-r1",
        "TabPFN v2",
        "ML-KNN",
        "prespecified external validation",
    ]
    assert all(term not in text for term in banned)
    assert "\\input{paper/appendices.tex}" in text
    assert "TabPFN-3" in text
    assert "RAkELd--RF" in text
    assert "https://github.com/liaorunqing/CoMLC-MI" in text


def test_primary_plugin_estimates_match_saved_predictions_summary():
    frame = pd.read_csv("output/benchmark/primary_paired_difference.csv").set_index("metric")
    expected = {
        "macro_auroc": 0.0246,
        "macro_auprc": 0.0087,
        "ece_10": 0.0056,
        "macro_f1_0_5": -0.0152,
    }
    for metric, rounded in expected.items():
        assert round(float(frame.loc[metric, "estimate"]), 4) == rounded


def test_unweighted_lr_calibration_sensitivity_is_present_and_finite():
    frame = pd.read_csv("output/benchmark/calibration_sensitivity_unweighted_lr_metrics.csv")
    assert not frame.empty
    assert frame.select_dtypes("number").notna().all().all()
