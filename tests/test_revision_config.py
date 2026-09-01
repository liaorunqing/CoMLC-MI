from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from src.run_revision import _internal_analysis_fingerprint


def test_formal_config_locks_requested_protocol() -> None:
    config = json.loads(Path("configs/access_2026_35668_r1.json").read_text(encoding="utf-8"))
    assert config["feature_contract"] == "admission_safe_v1"
    assert config["validation"] == {
        "outer_repeats": 5,
        "outer_folds": 5,
        "inner_folds": 4,
        "multilabel_stratification_order": 2,
    }
    assert config["gcn_ablation"]["seeds"] == list(range(42, 52))
    assert config["gcn_ablation"]["encoder_hidden"] == [256, 128]
    assert config["gcn_ablation"]["embedding_dim"] == 64
    assert config["gcn_ablation"]["asymmetric_cutoff"] == 0.15
    assert config["gcn_ablation"]["symmetric_cutoff"] == 0.08
    assert config["gcn_ablation"]["learning_rate"] == 0.001
    assert config["statistics"]["patient_bootstrap"] == 2000
    assert config["statistics"]["patient_bootstrap_jobs"] == 12
    assert config["statistics"]["calibration_bootstrap_jobs"] == 12
    assert config["statistics"]["paired_permutations"] == 10000
    assert config["models"]["TabPFN"]["inner_n_estimators"] == 2
    assert config["models"]["TabPFN"]["n_estimators"] == 8
    assert config["synthetic"] == {
        "samples": 1000,
        "features": 50,
        "labels": 10,
        "correlation_levels": [0.0, 0.25, 0.5, 0.75, 0.9],
        "repeats": 10,
        "gcn_epochs": 80,
    }
    assert config["external"]["expected_first_events"] == 29596
    assert config["external"]["expected_complete_cases"] == 28477
    assert config["external"]["source_cv_folds"] == 5
    assert config["external"]["logistic_max_iter"] == 5000
    assert config["external"]["tabpfn_n_estimators"] == 8
    assert config["external"]["tabpfn_n_preprocessing_jobs"] == 1
    assert config["release"]["tag"] == "access-2026-35668-r1"


def test_internal_checkpoint_fingerprint_ignores_unrelated_release_fields() -> None:
    config = json.loads(Path("configs/access_2026_35668_r1.json").read_text(encoding="utf-8"))
    unrelated = deepcopy(config)
    unrelated["external"]["bootstrap"] = 999
    assert _internal_analysis_fingerprint(config) == _internal_analysis_fingerprint(unrelated)
    changed_model = deepcopy(config)
    changed_model["models"]["LP-RF"]["n_estimators"] += 1
    assert _internal_analysis_fingerprint(config) != _internal_analysis_fingerprint(changed_model)
