from pathlib import Path

from src.package_r1_submission import SUPPLEMENT_FILES


def test_supplement_whitelist_excludes_patient_level_arrays_and_raw_workbooks():
    archive_names = [name.lower() for _, name in SUPPLEMENT_FILES]
    source_names = [Path(source).name.lower() for source, _ in SUPPLEMENT_FILES]
    assert not any("prediction" in name for name in archive_names)
    assert not any(name.endswith(".npz") for name in archive_names)
    assert not any(name.endswith((".xlsx", ".xls")) for name in archive_names)
    assert not any("hungarian myocardial infarction registry" in name for name in source_names)


def test_supplement_whitelist_contains_required_machine_readable_domains():
    archive_names = {name for _, name in SUPPLEMENT_FILES}
    assert "machine_readable/internal_model_metrics.csv" in archive_names
    assert "machine_readable/gcn_ablation_10_seed_full.csv" in archive_names
    assert "machine_readable/controlled_dependency_raw.csv" in archive_names
    assert "machine_readable/external_transportability_results.csv" in archive_names
