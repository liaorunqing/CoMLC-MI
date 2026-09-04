#!/usr/bin/env python3
"""Controlled synthetic experiments for label-dependency sensitivity.

The generator changes only the correlation of Gaussian residuals across labels.
For a given run, the feature matrix, label-specific feature weights, residual
draws, marginal prevalences, and train/validation/test indices are reused at
every dependency setting. The input ``rho`` is a data-generating parameter,
not an LDS value; reported dependency is always measured from binary labels.

Usage:
    python src/synthetic_experiments.py --runs 10
    python src/synthetic_experiments.py --quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

try:
    from .label_metrics import label_space_metrics
except ImportError:  # direct-script compatibility
    from label_metrics import label_space_metrics

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "synthetic_controlled")
RANDOM_SEED = 42
MODEL_NAMES = ("BR", "CC", "BR-kNN (distance-weighted, k=10)", "GCN")


@dataclass(frozen=True)
class SyntheticDesign:
    n_samples: int = 1000
    n_features: int = 50
    n_labels: int = 10
    n_informative: int = 15
    prevalence: float = 0.10
    feature_signal: float = 0.65


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    return centered / np.where(scale == 0, 1.0, scale)


def _fixed_split_indices(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def generate_synthetic_multilabel(
    design: SyntheticDesign,
    rho: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate labels with fixed marginals/SNR and controlled residual correlation.

    ``rho`` is the equicorrelation among label-specific Gaussian residuals.
    Marginal residual variance remains one for every rho, so the univariate
    signal-to-noise ratio is unchanged by design. Ranking each label at a fixed
    quantile makes its realized prevalence identical across settings.
    """
    if not 0 <= rho < 1:
        raise ValueError("rho must lie in [0, 1).")

    rng = np.random.default_rng(random_state)
    x = rng.normal(size=(design.n_samples, design.n_features))
    weights = rng.normal(size=(design.n_informative, design.n_labels))
    weights /= np.linalg.norm(weights, axis=0, keepdims=True)
    feature_scores = _standardize_columns(x[:, :design.n_informative] @ weights)

    # The same draws are reused at every rho within a run.
    shared_error = rng.normal(size=(design.n_samples, 1))
    independent_error = rng.normal(size=(design.n_samples, design.n_labels))
    residual = np.sqrt(rho) * shared_error + np.sqrt(1.0 - rho) * independent_error
    latent = design.feature_signal * feature_scores + residual

    n_positive = max(2, int(round(design.prevalence * design.n_samples)))
    y = np.zeros((design.n_samples, design.n_labels), dtype=np.int64)
    for label_index in range(design.n_labels):
        positive_indices = np.argpartition(latent[:, label_index], -n_positive)[-n_positive:]
        y[positive_indices, label_index] = 1

    metrics = label_space_metrics(y)
    metrics["mean_true_score_auc"] = float(np.mean([
        roc_auc_score(y[:, label_index], feature_scores[:, label_index])
        for label_index in range(design.n_labels)
    ]))
    metrics["rho"] = float(rho)
    metrics["target_prevalence"] = float(design.prevalence)
    return x, y, feature_scores, metrics


class SimpleGCN(nn.Module):
    """Two-layer feature encoder with one label-graph convolution."""

    def __init__(self, n_features: int, n_labels: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.label_embeddings = nn.Parameter(torch.randn(n_labels, hidden_dim) * 0.1)
        self.graph_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        patient_embedding = self.encoder(x)
        degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1e-8)
        label_embedding = torch.relu(self.graph_layer((adjacency / degree) @ self.label_embeddings))
        return patient_embedding @ label_embedding.T


def compute_adjacency(y_train: np.ndarray) -> torch.Tensor:
    conditional = np.asarray(label_space_metrics(y_train)["conditional_cooccurrence"])
    prevalence = y_train.mean(axis=0)
    excess = np.clip(conditional - prevalence[None, :], 0.0, None)
    return torch.tensor(excess + np.eye(y_train.shape[1]), dtype=torch.float32)


def _macro_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    aucs = [
        roc_auc_score(y_true[:, label_index], probabilities[:, label_index])
        for label_index in range(y_true.shape[1])
        if np.unique(y_true[:, label_index]).size == 2
    ]
    return float(np.mean(aucs))


def _lgbm(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=24,
        verbosity=-1,
        random_state=seed,
        n_jobs=1,
    )


def evaluate_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    gcn_epochs: int,
) -> tuple[float, float]:
    """Fit one prespecified model and return test Macro-AUC and wall time."""
    start = time.perf_counter()
    n_labels = y_train.shape[1]
    probabilities = np.zeros((len(x_test), n_labels), dtype=float)

    if model_name == "BR":
        for label_index in range(n_labels):
            classifier = _lgbm(seed + label_index)
            classifier.fit(x_train, y_train[:, label_index])
            probabilities[:, label_index] = classifier.predict_proba(x_test)[:, 1]

    elif model_name == "CC":
        chain = np.random.default_rng(seed).permutation(n_labels)
        train_history = np.empty((len(x_train), 0))
        test_history = np.empty((len(x_test), 0))
        for position, label_index in enumerate(chain):
            classifier = _lgbm(seed + position)
            classifier.fit(np.column_stack((x_train, train_history)), y_train[:, label_index])
            current = classifier.predict_proba(np.column_stack((x_test, test_history)))[:, 1]
            probabilities[:, label_index] = current
            train_history = np.column_stack((train_history, y_train[:, label_index]))
            test_history = np.column_stack((test_history, current))

    elif model_name == "BR-kNN (distance-weighted, k=10)":
        for label_index in range(n_labels):
            classifier = KNeighborsClassifier(n_neighbors=10, weights="distance")
            classifier.fit(x_train, y_train[:, label_index])
            probabilities[:, label_index] = classifier.predict_proba(x_test)[:, 1]

    elif model_name == "GCN":
        torch.manual_seed(seed)
        scaler = StandardScaler()
        x_train_t = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
        x_val_t = torch.tensor(scaler.transform(x_val), dtype=torch.float32)
        x_test_t = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        adjacency = compute_adjacency(y_train)

        model = SimpleGCN(x_train.shape[1], n_labels)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        prevalence = y_train.mean(axis=0)
        positive_weight = torch.tensor(
            (1.0 - prevalence) / np.maximum(prevalence, 1e-6), dtype=torch.float32
        )
        loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

        best_state = None
        best_validation_auc = -np.inf
        stale_epochs = 0
        for epoch in range(gcn_epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_function(model(x_train_t, adjacency), y_train_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    validation_probabilities = torch.sigmoid(model(x_val_t, adjacency)).numpy()
                validation_auc = _macro_auc(y_val, validation_probabilities)
                if validation_auc > best_validation_auc + 1e-4:
                    best_validation_auc = validation_auc
                    best_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 5
                if stale_epochs >= 20:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(x_test_t, adjacency)).numpy()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return _macro_auc(y_test, probabilities), time.perf_counter() - start


def run_controlled_experiment(
    design: SyntheticDesign,
    rho_levels: tuple[float, ...],
    n_runs: int,
    gcn_epochs: int,
    output_dir: str | os.PathLike | None = None,
    run_indices: list[int] | None = None,
    assemble: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired repeated simulations and export raw and summarized results."""
    destination = Path(output_dir or OUTPUT_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_specification = {
        "design": design.__dict__,
        "rho_levels": list(rho_levels),
        "n_runs": n_runs,
        "gcn_epochs": gcn_epochs,
    }
    fingerprint = hashlib.sha256(
        json.dumps(checkpoint_specification, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint_manifest = destination / "controlled_dependency_checkpoint_specification.json"
    if not checkpoint_manifest.exists():
        temporary_manifest = checkpoint_manifest.with_name(
            f"{checkpoint_manifest.name}.{os.getpid()}.tmp"
        )
        temporary_manifest.write_text(
            json.dumps({"sha256": fingerprint, **checkpoint_specification}, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_manifest, checkpoint_manifest)
    previous = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    if previous.get("sha256") != fingerprint:
        raise RuntimeError("Existing synthetic checkpoints use a different specification.")
    selected_runs = list(range(n_runs) if run_indices is None else run_indices)
    for run in selected_runs:
        checkpoint = destination / f"controlled_dependency_run_{run + 1}.csv"
        if checkpoint.exists():
            continue
        raw_rows: list[dict] = []
        seed = RANDOM_SEED + 1000 * run
        train_index, val_index, test_index = _fixed_split_indices(design.n_samples, seed + 19)
        for rho in rho_levels:
            x, y, _, metrics = generate_synthetic_multilabel(design, rho, seed)
            for model_name in MODEL_NAMES:
                macro_auc, elapsed = evaluate_model(
                    model_name,
                    x[train_index], y[train_index],
                    x[val_index], y[val_index],
                    x[test_index], y[test_index],
                    seed + 100 * MODEL_NAMES.index(model_name),
                    gcn_epochs,
                )
                raw_rows.append({
                    "run": run,
                    "seed": seed,
                    "rho": rho,
                    "model": model_name,
                    "macro_auc": macro_auc,
                    "elapsed_seconds": elapsed,
                    "label_cardinality": metrics["label_cardinality"],
                    "label_density": metrics["label_density"],
                    "realized_lds": metrics["label_dependency_score"],
                    "median_conditional_cooccurrence": metrics["median_conditional_cooccurrence"],
                    "mean_pairwise_phi": metrics["mean_pairwise_phi"],
                    "mean_true_score_auc": metrics["mean_true_score_auc"],
                })
                print(
                    f"run={run + 1:02d}/{n_runs} rho={rho:.2f} {model_name:6s} "
                    f"LDS={metrics['label_dependency_score']:.3f} AUC={macro_auc:.4f}"
                )
        pd.DataFrame(raw_rows).to_csv(checkpoint, index=False)

    if not assemble:
        available = [destination / f"controlled_dependency_run_{run + 1}.csv" for run in selected_runs]
        raw = pd.concat([pd.read_csv(path) for path in available if path.exists()], ignore_index=True)
        return raw, pd.DataFrame()

    checkpoints = [destination / f"controlled_dependency_run_{run + 1}.csv" for run in range(n_runs)]
    missing = [path.name for path in checkpoints if not path.exists()]
    if missing:
        raise RuntimeError(f"Synthetic run checkpoints are incomplete: {missing}")
    raw = pd.concat([pd.read_csv(path) for path in checkpoints], ignore_index=True)
    summary = (
        raw.groupby(["rho", "model"], as_index=False)
        .agg(
            macro_auc_mean=("macro_auc", "mean"),
            macro_auc_sd=("macro_auc", "std"),
            realized_lds_mean=("realized_lds", "mean"),
            realized_lds_sd=("realized_lds", "std"),
            label_density_mean=("label_density", "mean"),
            label_density_sd=("label_density", "std"),
            true_score_auc_mean=("mean_true_score_auc", "mean"),
            true_score_auc_sd=("mean_true_score_auc", "std"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
        )
    )

    raw.to_csv(destination / "controlled_dependency_raw.csv", index=False)
    summary.to_csv(destination / "controlled_dependency_summary.csv", index=False)
    metadata = {
        "design": design.__dict__,
        "rho_levels": list(rho_levels),
        "n_runs": n_runs,
        "gcn_epochs_max": gcn_epochs,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "models": list(MODEL_NAMES),
        "interpretation": (
            "rho is the residual-correlation parameter. realized_lds is the observed "
            "binary-label metric and is the only dependency value used in reporting."
        ),
    }
    with open(
        destination / "controlled_dependency_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)
    return raw, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="Independent repeated data sets.")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=80, help="Maximum GCN epochs.")
    parser.add_argument("--prevalence", type=float, default=0.10)
    parser.add_argument("--quick", action="store_true", help="Two-run smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        design = SyntheticDesign(n_samples=500, prevalence=args.prevalence)
        n_runs, epochs = 2, 30
        rho_levels = (0.0, 0.5, 0.9)
    else:
        design = SyntheticDesign(n_samples=args.samples, prevalence=args.prevalence)
        n_runs, epochs = args.runs, args.epochs
        rho_levels = (0.0, 0.25, 0.50, 0.75, 0.90)

    _, summary = run_controlled_experiment(design, rho_levels, n_runs, epochs)
    print("\nControlled synthetic experiment summary")
    print(summary.to_string(index=False))
    print(f"\nArtifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
