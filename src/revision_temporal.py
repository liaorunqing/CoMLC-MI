"""Prospective four-horizon sensitivity analysis using one neural baseline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .revision_features import FoldPreprocessor, feature_sets, prepare_outcomes
from .revision_metrics import aggregate_metrics, per_label_metrics
from .revision_models import fit_predict_model, neural_best_epoch
from .revision_validation import inner_folds, repeated_outer_folds


HORIZONS = [
    ("Admission", "admission_safe_v1"),
    ("24 h", "prospective_24h"),
    ("48 h", "prospective_48h"),
    ("72 h", "prospective_72h"),
]


def _processor(config: dict, contract: str, seed: int) -> FoldPreprocessor:
    return FoldPreprocessor(
        feature_contract=contract,
        random_state=seed,
        rf_estimators=config["preprocessing"]["iterative_rf_estimators"],
        iterative_max_iter=config["preprocessing"]["iterative_max_iter"],
    )


def run_temporal(
    config: dict,
    output_dir: Path,
    horizons: list[str] | None = None,
    fold_indices: list[int] | None = None,
    assemble: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(config["dataset"])
    outcomes = prepare_outcomes(data).to_numpy(dtype=np.int8)
    folds = repeated_outer_folds(
        outcomes,
        config["validation"]["outer_repeats"],
        config["validation"]["outer_folds"],
        config["seed"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_specification = {
        "horizons": HORIZONS,
        "validation": config["validation"],
        "preprocessing": config["preprocessing"],
        "model": config["models"]["Shared-MLP"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(checkpoint_specification, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = output_dir / "temporal_checkpoint_specification.json"
    if not manifest.exists():
        temporary = manifest.with_name(f"{manifest.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps({"sha256": fingerprint, **checkpoint_specification}, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, manifest)
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    if previous.get("sha256") != fingerprint:
        raise RuntimeError("Existing temporal checkpoints use a different specification.")

    selected = HORIZONS if horizons is None else [item for item in HORIZONS if item[1] in horizons]
    if horizons is not None and len(selected) != len(horizons):
        raise ValueError("Unknown temporal feature contract requested.")
    for horizon_index, (horizon, contract) in enumerate(HORIZONS):
        if (horizon, contract) not in selected:
            continue
        requested_folds = None if fold_indices is None else set(fold_indices)
        for fold_ordinal, fold in enumerate(folds):
            if requested_folds is not None and fold_ordinal not in requested_folds:
                continue
            checkpoint = output_dir / f"temporal_{contract}_r{fold.repeat + 1}_f{fold.fold + 1}.npz"
            if checkpoint.exists():
                continue
            outer_y = outcomes[fold.train]
            epoch_candidates = []
            for inner_index, (inner_train_rel, inner_validation_rel) in enumerate(
                inner_folds(outer_y, config["validation"]["inner_folds"], config["seed"] + horizon_index * 10_000 + fold.repeat * 1000 + fold.fold * 100)
            ):
                train_indices = fold.train[inner_train_rel]
                validation_indices = fold.train[inner_validation_rel]
                seed = config["seed"] + horizon_index * 10_000 + fold.repeat * 1000 + fold.fold * 100 + inner_index
                processor = _processor(config, contract, seed).fit(data.iloc[train_indices])
                train_x = processor.transform(data.iloc[train_indices]).to_numpy()
                validation_x = processor.transform(data.iloc[validation_indices]).to_numpy()
                epoch_candidates.append(
                    neural_best_epoch(
                        "Shared-MLP",
                        train_x,
                        outcomes[train_indices],
                        validation_x,
                        outcomes[validation_indices],
                        seed,
                        config["models"]["Shared-MLP"],
                    )
                )
            epochs = max(1, int(round(float(np.median(epoch_candidates)))))
            seed = config["seed"] + horizon_index * 10_000 + fold.repeat * 1000 + fold.fold * 100
            processor = _processor(config, contract, seed).fit(data.iloc[fold.train])
            train_x = processor.transform(data.iloc[fold.train]).to_numpy()
            test_x = processor.transform(data.iloc[fold.test]).to_numpy()
            probability = fit_predict_model(
                "Shared-MLP",
                train_x,
                outcomes[fold.train],
                test_x,
                seed,
                config["models"]["Shared-MLP"],
                neural_epochs=epochs,
            )
            temporary_checkpoint = checkpoint.with_name(
                f"{checkpoint.name}.{os.getpid()}.tmp.npz"
            )
            np.savez_compressed(
                temporary_checkpoint,
                test_indices=fold.test,
                probability=probability,
                selected_epochs=np.asarray(epoch_candidates),
            )
            os.replace(temporary_checkpoint, checkpoint)

    if not assemble:
        return pd.DataFrame(), pd.DataFrame()

    all_predictions = {}
    metrics_rows = []
    label_rows = []
    for horizon, contract in HORIZONS:
        repeated = np.full(
            (config["validation"]["outer_repeats"], len(data), outcomes.shape[1]),
            np.nan,
            dtype=np.float64,
        )
        for fold in folds:
            checkpoint = output_dir / f"temporal_{contract}_r{fold.repeat + 1}_f{fold.fold + 1}.npz"
            if not checkpoint.exists():
                raise RuntimeError(f"Missing temporal checkpoint: {checkpoint.name}")
            stored = np.load(checkpoint)
            repeated[fold.repeat, stored["test_indices"].astype(int)] = stored["probability"]
        if not np.isfinite(repeated).all():
            raise AssertionError(f"Incomplete temporal OOF predictions for {contract}.")
        probability = repeated.mean(axis=0)
        feature_count = len(feature_sets(data.columns)[contract])
        metrics_rows.append(
            {"horizon": horizon, "feature_contract": contract, "feature_count": feature_count, **aggregate_metrics(outcomes, probability)}
        )
        label_frame = per_label_metrics(outcomes, probability)
        label_frame.insert(0, "feature_count", feature_count)
        label_frame.insert(0, "feature_contract", contract)
        label_frame.insert(0, "horizon", horizon)
        label_rows.append(label_frame)
        all_predictions[contract] = repeated
    metrics = pd.DataFrame(metrics_rows)
    labels = pd.concat(label_rows, ignore_index=True)
    metrics.to_csv(output_dir / "temporal_model_metrics.csv", index=False)
    labels.to_csv(output_dir / "temporal_per_label_metrics.csv", index=False)
    np.savez_compressed(output_dir / "temporal_oof_predictions.npz", y_true=outcomes, **all_predictions)
    (output_dir / "temporal_metadata.json").write_text(
        json.dumps(
            {
                "scope": "Prospective feature-horizon sensitivity analysis of one shared MLP baseline.",
                "feature_counts": {horizon: int(count) for horizon, count in zip(metrics["horizon"], metrics["feature_count"])},
                "validation": "5 repeats x 5 outer folds; four inner folds select epochs.",
                "warning": "Later horizons are accumulated-information risk stratification and may contain downstream care information.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics, labels
