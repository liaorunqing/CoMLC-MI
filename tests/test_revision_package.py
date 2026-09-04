from pathlib import Path

from src.package_submission import _code_entries, _is_safe_public, _paper_entries


def test_public_code_entries_exclude_external_patient_arrays_and_workbooks():
    archive_names = [name.lower() for _, name in _code_entries()]
    assert not any("external_transportability_predictions" in name for name in archive_names)
    assert not any("external_transportability_bootstrap" in name for name in archive_names)
    assert not any(name.endswith((".xlsx", ".xls", ".xlsm")) for name in archive_names)
    assert not any(name.endswith((".ckpt", ".pt", ".pth")) for name in archive_names)


def test_code_package_contains_neutral_interface_and_machine_readable_results():
    archive_names = {name for _, name in _code_entries()}
    assert "configs/comlc_mi_benchmark.json" in archive_names
    assert "src/run_benchmark.py" in archive_names
    assert "output/benchmark/internal_model_metrics.csv" in archive_names
    assert "output/benchmark/gcn_ablation/gcn_ablation_10_seed_full.csv" in archive_names
    assert "output/benchmark/external_transportability/external_transportability_results.csv" in archive_names


def test_latex_packages_include_appendices_and_highlighted_source():
    clean = {name for _, name in _paper_entries("gai_revised_clean.tex")}
    highlighted = {name for _, name in _paper_entries("gai_revised_highlighted.tex")}
    assert "paper/appendices.tex" in clean
    assert "paper/gai_revised_clean.tex" in clean
    assert "paper/gai_revised_highlighted.tex" in highlighted
    assert not any("supplementary_r1" in name for name in clean | highlighted)


def test_public_safety_filter_rejects_sensitive_artifacts():
    assert not _is_safe_public(Path("x.xlsx"), "x.xlsx")
    assert not _is_safe_public(Path("x.npz"), "external_transportability_predictions.npz")
    assert _is_safe_public(Path("internal_oof_predictions.npz"), "internal_oof_predictions.npz")
