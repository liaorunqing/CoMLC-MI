"""Single formal entry point for the IEEE Access R1 reconstruction.

Formal command
--------------
python -m src.run_revision --config configs/access_2026_35668_r1.json

The run is checkpointed by outer fold.  Optional ``--stage`` is provided for
auditing and recovery; omitting it executes the full locked workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .revision_features import FoldPreprocessor, LABEL_COLS, feature_sets, prepare_outcomes
from .revision_metrics import (
    aggregate_metrics,
    calibration_bootstrap_intervals,
    paired_label_auc_tests,
    per_label_auc_difference_bootstrap,
    paired_metric_difference,
    paired_patient_bootstrap,
    per_label_metrics,
    reliability_table,
)
from .revision_models import (
    MODEL_NAMES,
    catboost_best_iterations,
    fit_predict_model,
    neural_best_epoch,
    select_top_features,
    validation_auc_weights,
)
from .revision_validation import assignment_matrix, inner_folds, repeated_outer_folds


ENSEMBLE_EQUAL = "Ensemble-LP-RF-TabPFN-Equal"
ENSEMBLE_WEIGHTED = "Ensemble-LP-RF-TabPFN-Weighted"
ALL_PREDICTION_NAMES = MODEL_NAMES + [ENSEMBLE_EQUAL, ENSEMBLE_WEIGHTED]

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ["dataset", "output_dir", "feature_contract", "validation", "models", "statistics"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}")
    if config["feature_contract"] != "admission_safe_v1":
        raise ValueError("The formal primary analysis must use admission_safe_v1.")
    return config


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _internal_analysis_fingerprint(config: dict) -> str:
    specification = {
        "seed": config["seed"],
        "dataset": config["dataset"],
        "feature_contract": config["feature_contract"],
        "validation": config["validation"],
        "preprocessing": config["preprocessing"],
        "models": config["models"],
        "ensembles": config["ensembles"],
    }
    return hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _fold_preprocessor(config: dict, seed: int) -> FoldPreprocessor:
    preprocessing = config["preprocessing"]
    return FoldPreprocessor(
        feature_contract=config["feature_contract"],
        random_state=seed,
        rf_estimators=preprocessing["iterative_rf_estimators"],
        iterative_max_iter=preprocessing["iterative_max_iter"],
    )


def _package_versions() -> dict[str, str]:
    packages = [
        "numpy", "pandas", "scipy", "scikit-learn", "scikit-multilearn",
        "xgboost", "lightgbm", "catboost", "tabpfn", "torch", "torch-geometric",
        "matplotlib", "seaborn", "joblib",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            # Some Windows PyTorch installations retain an invalid dist-info
            # directory while the importable module remains intact. Record the
            # module version rather than incorrectly reporting it as absent.
            module_name = {"torch-geometric": "torch_geometric"}.get(
                package, package.replace("-", "_")
            )
            try:
                module = __import__(module_name)
                versions[package] = str(module.__version__)
            except (ImportError, AttributeError):
                versions[package] = "not installed"
    return versions


def _write_environment(output_dir: Path, config_path: Path) -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "config": str(config_path.as_posix()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (output_dir / "runtime_environment.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _save_splits(data: pd.DataFrame, outcomes: np.ndarray, config: dict, output_dir: Path):
    validation = config["validation"]
    folds = repeated_outer_folds(
        outcomes,
        n_repeats=validation["outer_repeats"],
        n_splits=validation["outer_folds"],
        base_seed=config["seed"],
    )
    assignment = assignment_matrix(
        folds,
        n_samples=len(data),
        n_repeats=validation["outer_repeats"],
    )
    split_frame = pd.DataFrame({"patient_row": data.index})
    for repeat in range(validation["outer_repeats"]):
        split_frame[f"repeat_{repeat + 1}_outer_fold"] = assignment[repeat]
    split_frame.to_csv(output_dir / "outer_fold_assignments.csv", index=False)
    return folds, assignment


def _inner_decisions(
    raw_data: pd.DataFrame,
    outcomes: np.ndarray,
    outer_train: np.ndarray,
    config: dict,
    seed: int,
) -> tuple[np.ndarray, list[int], dict[str, int], list[dict]]:
    y_outer = outcomes[outer_train]
    inner_lp = np.full_like(y_outer, np.nan, dtype=np.float64)
    inner_tabpfn = np.full_like(y_outer, np.nan, dtype=np.float64)
    catboost_rounds = []
    neural_rounds = {"Shared-MLP": [], "MultiTask-DNN": []}
    audits = []
    for inner_index, (inner_train_rel, inner_validation_rel) in enumerate(
        inner_folds(y_outer, config["validation"]["inner_folds"], seed)
    ):
        print(
            f"  [inner] fold {inner_index + 1}/{config['validation']['inner_folds']}",
            flush=True,
        )
        inner_seed = seed + 100 + inner_index
        inner_train = outer_train[inner_train_rel]
        inner_validation = outer_train[inner_validation_rel]
        processor = _fold_preprocessor(config, inner_seed).fit(raw_data.iloc[inner_train])
        if set(processor.audit_.fitted_row_ids) & set(inner_validation):
            raise AssertionError("Inner validation rows reached the preprocessing fit.")
        train_x = processor.transform(raw_data.iloc[inner_train]).to_numpy()
        validation_x = processor.transform(raw_data.iloc[inner_validation]).to_numpy()
        train_y = outcomes[inner_train]
        validation_y = outcomes[inner_validation]
        selected = select_top_features(
            train_x,
            train_y,
            n_features=config["models"]["TabPFN"]["top_features"],
            seed=inner_seed,
            n_estimators=config["models"]["TabPFN"]["selector_estimators"],
        )
        inner_lp[inner_validation_rel] = fit_predict_model(
            "LP-RF", train_x, train_y, validation_x, inner_seed,
            config["models"]["LP-RF"],
        )
        inner_tabpfn_params = dict(config["models"]["TabPFN"])
        inner_tabpfn_params["n_estimators"] = inner_tabpfn_params["inner_n_estimators"]
        inner_tabpfn[inner_validation_rel] = fit_predict_model(
            "TabPFN", train_x, train_y, validation_x, inner_seed,
            inner_tabpfn_params, selected_features=selected,
        )
        catboost_rounds.append(
            catboost_best_iterations(
                train_x,
                train_y,
                validation_x,
                validation_y,
                inner_seed,
                config["models"]["BR-CatBoost"],
            )
        )
        for neural_name in neural_rounds:
            neural_rounds[neural_name].append(
                neural_best_epoch(
                    neural_name,
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                    inner_seed,
                    config["models"][neural_name],
                )
            )
        audits.append(
            {
                "inner_fold": inner_index,
                "fit_rows": list(map(int, processor.audit_.fitted_row_ids)),
                "validation_rows": list(map(int, inner_validation)),
                "selected_tabpfn_features": selected.tolist(),
            }
        )
    if not np.isfinite(inner_lp).all() or not np.isfinite(inner_tabpfn).all():
        raise AssertionError("Four-fold inner OOF predictions are incomplete.")
    weights = validation_auc_weights(y_outer, inner_lp, inner_tabpfn)
    catboost_iterations = np.maximum(
        1, np.rint(np.median(np.asarray(catboost_rounds), axis=0)).astype(int)
    ).tolist()
    neural_epochs = {
        name: max(1, int(round(float(np.median(values)))))
        for name, values in neural_rounds.items()
    }
    return weights, catboost_iterations, neural_epochs, audits


def _fit_outer_fold(
    raw_data: pd.DataFrame,
    outcomes: np.ndarray,
    fold,
    config: dict,
    checkpoint_dir: Path,
) -> dict:
    checkpoint = checkpoint_dir / f"repeat_{fold.repeat + 1}_fold_{fold.fold + 1}.npz"
    metadata_path = checkpoint.with_suffix(".json")
    if checkpoint.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fingerprint = _internal_analysis_fingerprint(config)
        stored_fingerprint = metadata.get("internal_analysis_sha256")
        if stored_fingerprint is not None and stored_fingerprint != fingerprint:
            raise RuntimeError(f"Checkpoint specification mismatch: {metadata_path.name}")
        expected_seed = config["seed"] + fold.repeat * 1000 + fold.fold * 100
        if (
            int(metadata.get("repeat", -1)) != fold.repeat + 1
            or int(metadata.get("outer_fold", -1)) != fold.fold + 1
            or int(metadata.get("seed", -1)) != expected_seed
            or int(metadata.get("feature_count", -1)) != 91
            or len(metadata.get("tabpfn_selected_feature_indices", [])) != 80
        ):
            raise RuntimeError(f"Checkpoint metadata audit failed: {metadata_path.name}")
        if stored_fingerprint is None:
            metadata["internal_analysis_sha256"] = fingerprint
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        stored = np.load(checkpoint)
        result = {"test_indices": stored["test_indices"]}
        for name in ALL_PREDICTION_NAMES:
            result[name] = stored[_slug(name)]
        return result

    seed = config["seed"] + fold.repeat * 1000 + fold.fold * 100
    weights, catboost_iterations, neural_epochs, inner_audits = _inner_decisions(
        raw_data, outcomes, fold.train, config, seed
    )
    processor = _fold_preprocessor(config, seed).fit(raw_data.iloc[fold.train])
    if set(processor.audit_.fitted_row_ids) & set(fold.test):
        raise AssertionError("Outer test rows reached the preprocessing fit.")
    train_x_frame = processor.transform(raw_data.iloc[fold.train])
    test_x_frame = processor.transform(raw_data.iloc[fold.test])
    train_x = train_x_frame.to_numpy()
    test_x = test_x_frame.to_numpy()
    train_y = outcomes[fold.train]
    selected = select_top_features(
        train_x,
        train_y,
        n_features=config["models"]["TabPFN"]["top_features"],
        seed=seed,
        n_estimators=config["models"]["TabPFN"]["selector_estimators"],
    )

    predictions = {}
    for model_name in MODEL_NAMES:
        print(f"  [outer] fitting {model_name}", flush=True)
        predictions[model_name] = fit_predict_model(
            model_name,
            train_x,
            train_y,
            test_x,
            seed,
            config["models"][model_name],
            selected_features=selected if model_name == "TabPFN" else None,
            catboost_iterations=catboost_iterations if model_name == "BR-CatBoost" else None,
            neural_epochs=neural_epochs.get(model_name),
        )
    predictions[ENSEMBLE_EQUAL] = 0.5 * (
        predictions["LP-RF"] + predictions["TabPFN"]
    )
    predictions[ENSEMBLE_WEIGHTED] = (
        predictions["LP-RF"] * weights[0][None, :]
        + predictions["TabPFN"] * weights[1][None, :]
    )
    arrays = {"test_indices": fold.test}
    arrays.update({_slug(name): value for name, value in predictions.items()})
    np.savez_compressed(checkpoint, **arrays)
    metadata = {
        "repeat": fold.repeat + 1,
        "outer_fold": fold.fold + 1,
        "seed": seed,
        "outer_fit_rows": list(map(int, processor.audit_.fitted_row_ids)),
        "outer_test_rows": list(map(int, fold.test)),
        "feature_names": list(processor.get_feature_names_out()),
        "feature_count": train_x.shape[1],
        "tabpfn_selected_feature_indices": selected.tolist(),
        "ensemble_weights_lp_tabpfn": weights.tolist(),
        "catboost_iterations": catboost_iterations,
        "neural_epochs": neural_epochs,
        "inner_folds": inner_audits,
        "internal_analysis_sha256": _internal_analysis_fingerprint(config),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = {"test_indices": fold.test, **predictions}
    return result


def run_internal(
    config: dict,
    output_dir: Path,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict[str, np.ndarray]:
    raw_data = pd.read_csv(config["dataset"])
    outcomes = prepare_outcomes(raw_data).to_numpy(dtype=np.int8)
    contracts = feature_sets(raw_data.columns)
    feature_contract = {
        "primary": config["feature_contract"],
        "feature_sets": {key: values for key, values in contracts.items()},
        "excluded_engineered_interactions": 8,
    }
    (output_dir / "feature_contract.json").write_text(
        json.dumps(feature_contract, indent=2), encoding="utf-8"
    )
    folds, _ = _save_splits(raw_data, outcomes, config, output_dir)
    repeats = config["validation"]["outer_repeats"]
    predictions = {
        name: np.full((repeats, len(raw_data), len(LABEL_COLS)), np.nan, dtype=np.float64)
        for name in ALL_PREDICTION_NAMES
    }
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    selected_folds = folds
    if shard_count is not None:
        if shard_index is None or not (0 <= shard_index < shard_count):
            raise ValueError("fold shard index must be in [0, fold_shards).")
        selected_folds = [fold for ordinal, fold in enumerate(folds) if ordinal % shard_count == shard_index]
    for fold in selected_folds:
        print(f"[internal] repeat {fold.repeat + 1}/5, outer fold {fold.fold + 1}/5", flush=True)
        result = _fit_outer_fold(raw_data, outcomes, fold, config, checkpoint_dir)
        test_indices = result["test_indices"].astype(int)
        for name in ALL_PREDICTION_NAMES:
            predictions[name][fold.repeat, test_indices] = result[name]
    if shard_count is not None:
        return {}
    for name, value in predictions.items():
        if not np.isfinite(value).all():
            raise AssertionError(f"Incomplete OOF predictions for {name}.")
    np.savez_compressed(
        output_dir / "internal_oof_predictions.npz",
        y_true=outcomes,
        **{_slug(name): value for name, value in predictions.items()},
    )
    return predictions


def load_internal_predictions(config: dict, output_dir: Path):
    stored = np.load(output_dir / "internal_oof_predictions.npz")
    outcomes = stored["y_true"]
    predictions = {name: stored[_slug(name)] for name in ALL_PREDICTION_NAMES}
    return outcomes, predictions


def summarize_internal(config: dict, output_dir: Path) -> None:
    outcomes, repeated_predictions = load_internal_predictions(config, output_dir)
    predictions = {name: values.mean(axis=0) for name, values in repeated_predictions.items()}
    metric_rows = []
    repeat_metric_rows = []
    label_rows = []
    reliability_rows = []
    for name, probability in predictions.items():
        metric_rows.append({"model": name, **aggregate_metrics(outcomes, probability)})
        for repeat_index, repeat_probability in enumerate(repeated_predictions[name], start=1):
            repeat_metric_rows.append(
                {"model": name, "repeat": repeat_index, **aggregate_metrics(outcomes, repeat_probability)}
            )
        label_frame = per_label_metrics(outcomes, probability)
        label_frame.insert(0, "model", name)
        label_rows.append(label_frame)
        reliability_rows.append(reliability_table(outcomes, probability, name))
    pd.DataFrame(metric_rows).to_csv(output_dir / "internal_model_metrics.csv", index=False)
    pd.DataFrame(repeat_metric_rows).to_csv(
        output_dir / "internal_repeat_metrics.csv", index=False
    )
    pd.concat(label_rows, ignore_index=True).to_csv(
        output_dir / "internal_per_label_metrics.csv", index=False
    )
    pd.concat(reliability_rows, ignore_index=True).to_csv(
        output_dir / "internal_calibration_bins.csv", index=False
    )

    bootstrap_summary, bootstrap_arrays = paired_patient_bootstrap(
        outcomes,
        predictions,
        n_bootstrap=config["statistics"]["patient_bootstrap"],
        seed=config["seed"],
        n_jobs=config["statistics"].get("patient_bootstrap_jobs", 1),
    )
    bootstrap_summary.to_csv(output_dir / "internal_bootstrap_summary.csv", index=False)
    metric_names = list(aggregate_metrics(outcomes, predictions[MODEL_NAMES[0]]))
    paired_metric_difference(
        bootstrap_arrays,
        metric_names,
        ENSEMBLE_WEIGHTED,
        "TabPFN",
    ).to_csv(output_dir / "primary_paired_difference.csv", index=False)
    np.savez_compressed(
        output_dir / "internal_bootstrap_arrays.npz",
        **{_slug(name): value for name, value in bootstrap_arrays.items()},
    )
    paired_label_auc_tests(
        outcomes,
        predictions[ENSEMBLE_WEIGHTED],
        predictions["TabPFN"],
        low_event_threshold=config["statistics"]["low_event_threshold"],
        n_permutations=config["statistics"]["paired_permutations"],
        seed=config["seed"],
    ).to_csv(output_dir / "primary_per_label_tests.csv", index=False)
    per_label_auc_difference_bootstrap(
        outcomes,
        predictions[ENSEMBLE_WEIGHTED],
        predictions["TabPFN"],
        n_bootstrap=config["statistics"]["patient_bootstrap"],
        seed=config["seed"],
    ).to_csv(output_dir / "primary_per_label_bootstrap.csv", index=False)
    calibration_frames = []
    for offset, name in enumerate(config["statistics"]["calibration_bootstrap_models"]):
        frame = calibration_bootstrap_intervals(
            outcomes,
            predictions[name],
            n_bootstrap=config["statistics"]["patient_bootstrap"],
            seed=config["seed"] + 10_000 * offset,
            n_jobs=config["statistics"].get("calibration_bootstrap_jobs", 1),
        )
        frame.insert(0, "model", name)
        calibration_frames.append(frame)
    pd.concat(calibration_frames, ignore_index=True).to_csv(
        output_dir / "primary_calibration_intervals.csv", index=False
    )


def run_external(config: dict, output_dir: Path) -> None:
    from .external_transportability import run

    configured = config["external"].get("hungarian_xlsx")
    environment = os.getenv("HUNGARIAN_MI_XLSX")
    candidate = configured or environment
    if candidate:
        registry_path = Path(candidate)
    else:
        registry_path = Path.home() / "Downloads" / "Hungarian Myocardial Infarction Registry extracted database.xlsx"
    if not registry_path.exists():
        raise FileNotFoundError(
            "Hungarian registry file not found. Set HUNGARIAN_MI_XLSX to its local path."
        )
    run(
        registry_path,
        Path(config["dataset"]),
        output_dir / "external_transportability",
        n_boot=config["external"]["bootstrap"],
        reuse_predictions=True,
        model_config=config["external"],
    )


def run_gcn(
    config: dict,
    output_dir: Path,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> None:
    from .revision_gcn import run_gcn_ablation

    seeds = None
    assemble = True
    if shard_count is not None:
        if shard_index is None or not (0 <= shard_index < shard_count):
            raise ValueError("GCN shard index must be in [0, gcn_shards).")
        seeds = [
            seed for ordinal, seed in enumerate(config["gcn_ablation"]["seeds"])
            if ordinal % shard_count == shard_index
        ]
        assemble = False
    run_gcn_ablation(config, output_dir / "gcn_ablation", seeds=seeds, assemble=assemble)


def run_temporal_analysis(
    config: dict,
    output_dir: Path,
    shard_index: int | None = None,
    shard_count: int | None = None,
    fold_shard_index: int | None = None,
    fold_shard_count: int | None = None,
) -> None:
    from .revision_temporal import HORIZONS, run_temporal

    horizons = None
    fold_indices = None
    assemble = True
    if shard_count is not None:
        if shard_index is None or not (0 <= shard_index < shard_count):
            raise ValueError("Temporal shard index must be in [0, temporal_shards).")
        horizons = [
            contract for ordinal, (_, contract) in enumerate(HORIZONS)
            if ordinal % shard_count == shard_index
        ]
        assemble = False
    if fold_shard_count is not None:
        if fold_shard_index is None or not (0 <= fold_shard_index < fold_shard_count):
            raise ValueError("Temporal fold shard index must be in [0, temporal_fold_shards).")
        total_folds = config["validation"]["outer_repeats"] * config["validation"]["outer_folds"]
        fold_indices = [
            ordinal for ordinal in range(total_folds)
            if ordinal % fold_shard_count == fold_shard_index
        ]
        assemble = False
    run_temporal(
        config,
        output_dir / "temporal",
        horizons=horizons,
        fold_indices=fold_indices,
        assemble=assemble,
    )


def run_synthetic_analysis(
    config: dict,
    output_dir: Path,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> None:
    from .synthetic_experiments import SyntheticDesign, run_controlled_experiment

    specification = config["synthetic"]
    design = SyntheticDesign(
        n_samples=specification["samples"],
        n_features=specification["features"],
        n_labels=specification["labels"],
    )
    run_indices = None
    assemble = True
    if shard_count is not None:
        if shard_index is None or not (0 <= shard_index < shard_count):
            raise ValueError("Synthetic shard index must be in [0, synthetic_shards).")
        run_indices = [
            run for run in range(specification["repeats"])
            if run % shard_count == shard_index
        ]
        assemble = False
    run_controlled_experiment(
        design,
        tuple(specification["correlation_levels"]),
        specification["repeats"],
        specification["gcn_epochs"],
        output_dir=output_dir / "synthetic",
        run_indices=run_indices,
        assemble=assemble,
    )


def write_sha256_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.json":
            continue
        rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    (output_dir / "SHA256SUMS.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=["all", "splits", "internal", "summarize", "gcn", "temporal", "synthetic", "external"],
        default="all",
        help="Recovery/audit stage; the formal command uses the default 'all'.",
    )
    parser.add_argument("--fold-shard-index", type=int)
    parser.add_argument("--fold-shards", type=int)
    parser.add_argument("--gcn-shard-index", type=int)
    parser.add_argument("--gcn-shards", type=int)
    parser.add_argument("--synthetic-shard-index", type=int)
    parser.add_argument("--synthetic-shards", type=int)
    parser.add_argument("--temporal-shard-index", type=int)
    parser.add_argument("--temporal-shards", type=int)
    parser.add_argument("--temporal-fold-shard-index", type=int)
    parser.add_argument("--temporal-fold-shards", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_environment(output_dir, args.config)
    started = time.time()
    if args.stage == "splits":
        raw_data = pd.read_csv(config["dataset"])
        outcomes = prepare_outcomes(raw_data).to_numpy(dtype=np.int8)
        _save_splits(raw_data, outcomes, config, output_dir)
    if (args.fold_shard_index is None) != (args.fold_shards is None):
        raise ValueError("--fold-shard-index and --fold-shards must be supplied together.")
    if args.stage not in {"internal"} and args.fold_shards is not None:
        raise ValueError("Fold sharding is available only for --stage internal.")
    if (args.gcn_shard_index is None) != (args.gcn_shards is None):
        raise ValueError("--gcn-shard-index and --gcn-shards must be supplied together.")
    if args.stage != "gcn" and args.gcn_shards is not None:
        raise ValueError("GCN sharding is available only for --stage gcn.")
    if (args.synthetic_shard_index is None) != (args.synthetic_shards is None):
        raise ValueError("--synthetic-shard-index and --synthetic-shards must be supplied together.")
    if args.stage != "synthetic" and args.synthetic_shards is not None:
        raise ValueError("Synthetic sharding is available only for --stage synthetic.")
    if (args.temporal_shard_index is None) != (args.temporal_shards is None):
        raise ValueError("--temporal-shard-index and --temporal-shards must be supplied together.")
    if args.stage != "temporal" and args.temporal_shards is not None:
        raise ValueError("Temporal sharding is available only for --stage temporal.")
    if (args.temporal_fold_shard_index is None) != (args.temporal_fold_shards is None):
        raise ValueError(
            "--temporal-fold-shard-index and --temporal-fold-shards must be supplied together."
        )
    if args.stage != "temporal" and args.temporal_fold_shards is not None:
        raise ValueError("Temporal fold sharding is available only for --stage temporal.")
    if args.stage in {"all", "internal"}:
        run_internal(config, output_dir, args.fold_shard_index, args.fold_shards)
    if args.stage in {"all", "internal", "summarize"} and args.fold_shards is None:
        summarize_internal(config, output_dir)
    if args.stage in {"all", "gcn"}:
        run_gcn(config, output_dir, args.gcn_shard_index, args.gcn_shards)
    if args.stage in {"all", "temporal"}:
        run_temporal_analysis(
            config,
            output_dir,
            args.temporal_shard_index,
            args.temporal_shards,
            args.temporal_fold_shard_index,
            args.temporal_fold_shards,
        )
    if args.stage in {"all", "synthetic"}:
        run_synthetic_analysis(
            config,
            output_dir,
            args.synthetic_shard_index,
            args.synthetic_shards,
        )
    if args.stage in {"all", "external"}:
        run_external(config, output_dir)
    sharded = (
        args.fold_shards is not None
        or args.gcn_shards is not None
        or args.synthetic_shards is not None
        or args.temporal_shards is not None
        or args.temporal_fold_shards is not None
    )
    (output_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "shard_complete" if sharded else "complete",
                "stage": args.stage,
                "fold_shard_index": args.fold_shard_index,
                "fold_shards": args.fold_shards,
                "gcn_shard_index": args.gcn_shard_index,
                "gcn_shards": args.gcn_shards,
                "synthetic_shard_index": args.synthetic_shard_index,
                "synthetic_shards": args.synthetic_shards,
                "temporal_shard_index": args.temporal_shard_index,
                "temporal_shards": args.temporal_shards,
                "elapsed_seconds": time.time() - started,
                "formal_command": "python -m src.run_revision --config configs/access_2026_35668_r1.json",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_sha256_manifest(output_dir)


if __name__ == "__main__":
    main()
