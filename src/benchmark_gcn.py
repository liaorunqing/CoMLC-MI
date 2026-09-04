"""Paired 18-configuration, 10-seed GCN ablation for the benchmark."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv

from .benchmark_features import FoldPreprocessor, LABEL_COLS, prepare_outcomes
from .benchmark_losses import (
    positive_class_weights,
    weighted_asymmetric_loss_with_logits,
    weighted_bce_with_logits,
)
from .benchmark_models import seed_everything
from .benchmark_validation import inner_folds, repeated_outer_folds


ADJACENCIES = ("none", "asymmetric", "symmetric")
HEADS = ("dot", "concat")
LOSSES = ("W-BCE", "W-ASL-g2", "W-ASL-g3")


def build_adjacency(
    y: np.ndarray,
    kind: str,
    asymmetric_cutoff: float = 0.15,
    symmetric_cutoff: float = 0.08,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if kind == "none":
        return None, None
    marginal = y.sum(axis=0)
    conditional = np.eye(y.shape[1], dtype=np.float32)
    for source in range(y.shape[1]):
        for target in range(y.shape[1]):
            if source != target:
                conditional[source, target] = np.sum((y[:, source] == 1) & (y[:, target] == 1)) / max(marginal[source], 1)
    if kind == "asymmetric":
        adjacency = conditional
        threshold = asymmetric_cutoff
    elif kind == "symmetric":
        adjacency = np.maximum(conditional, conditional.T)
        threshold = symmetric_cutoff
    else:
        raise KeyError(kind)
    adjacency[adjacency < threshold] = 0
    np.fill_diagonal(adjacency, 1.0)
    edges = np.argwhere(adjacency > 0)
    weights = adjacency[edges[:, 0], edges[:, 1]]
    return torch.tensor(edges.T, dtype=torch.long), torch.tensor(weights, dtype=torch.float32)


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: tuple[int, int] = (256, 128),
        embedding_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden[0]), nn.BatchNorm1d(hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.BatchNorm1d(hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, value):
        return self.network(value)


class PairedGCNModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_labels: int,
        head: str,
        embedding_dim: int = 64,
        encoder_hidden: tuple[int, int] = (256, 128),
        graph_hidden: int = 128,
        concat_head_width: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head_kind = head
        self.encoder = FeatureEncoder(n_features, encoder_hidden, embedding_dim, dropout)
        self.label_embeddings = nn.Parameter(torch.randn(n_labels, embedding_dim) * 0.1)
        self.graph_1 = GCNConv(embedding_dim, graph_hidden)
        self.graph_2 = GCNConv(graph_hidden, embedding_dim)
        self.graph_dropout = nn.Dropout(dropout)
        if head == "concat":
            self.heads = nn.ModuleList(
                [nn.Sequential(nn.Linear(2 * embedding_dim, concat_head_width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(concat_head_width, 1)) for _ in range(n_labels)]
            )

    def forward(self, x, edge_index=None, edge_weight=None):
        patient = self.encoder(x)
        label = self.label_embeddings
        if edge_index is not None:
            label = F.relu(self.graph_1(label, edge_index, edge_weight))
            label = self.graph_dropout(label)
            label = self.graph_2(label, edge_index, edge_weight)
        if self.head_kind == "dot":
            return patient @ label.T
        return torch.cat(
            [head(torch.cat([patient, label[index].expand(len(patient), -1)], dim=1)) for index, head in enumerate(self.heads)],
            dim=1,
        )


def _batch_schedule(n_samples: int, epochs: int, batch_size: int, seed: int) -> list[list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(epochs):
        order = rng.permutation(n_samples)
        schedule.append(list(np.array_split(order, np.ceil(n_samples / batch_size).astype(int))))
    return schedule


def _macro_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean([
        roc_auc_score(y[:, label], probability[:, label])
        for label in range(y.shape[1])
        if np.unique(y[:, label]).size == 2
    ]))


def _train_one(
    initial_state: dict,
    head: str,
    loss_name: str,
    adjacency: tuple[torch.Tensor | None, torch.Tensor | None],
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    schedule: list[list[np.ndarray]],
    seed: int,
    architecture: dict | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 25,
    min_delta: float = 1e-4,
    gamma_positive: float = 0.0,
    gamma_negative: tuple[float, float] = (2.0, 3.0),
    probability_clip: float = 0.05,
) -> dict:
    seed_everything(seed)
    architecture = architecture or {}
    model = PairedGCNModel(train_x.shape[1], train_y.shape[1], head, **architecture)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weights = positive_class_weights(torch.tensor(train_y, dtype=torch.float32))
    edge_index, edge_weight = adjacency
    x_tensor = torch.tensor(train_x, dtype=torch.float32)
    y_tensor = torch.tensor(train_y, dtype=torch.float32)
    validation_tensor = torch.tensor(validation_x, dtype=torch.float32)
    test_tensor = torch.tensor(test_x, dtype=torch.float32)
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch, batches in enumerate(schedule, start=1):
        model.train()
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[indices], edge_index, edge_weight)
            if loss_name == "W-BCE":
                loss = weighted_bce_with_logits(logits, y_tensor[indices], weights)
            else:
                gamma = gamma_negative[0] if loss_name == "W-ASL-g2" else gamma_negative[1]
                loss = weighted_asymmetric_loss_with_logits(
                    logits,
                    y_tensor[indices],
                    weights,
                    gamma_pos=gamma_positive,
                    gamma_neg=gamma,
                    clip=probability_clip,
                )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_probability = torch.sigmoid(
                model(validation_tensor, edge_index, edge_weight)
            ).numpy()
        score = _macro_auc(validation_y, validation_probability)
        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("GCN training did not produce a valid validation estimate.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probability = torch.sigmoid(model(test_tensor, edge_index, edge_weight)).numpy()
    return {
        "best_epoch": best_epoch,
        "validation_macro_auroc": best_score,
        "test_macro_auroc": _macro_auc(test_y, test_probability),
        "test_micro_auroc": float(roc_auc_score(test_y.ravel(), test_probability.ravel())),
    }


def run_gcn_ablation(
    config: dict,
    output_dir: Path,
    seeds: list[int] | None = None,
    assemble: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the paired-seed ablation on a locked outer/inner partition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_specification = {
        "feature_contract": config["feature_contract"],
        "validation": config["validation"],
        "gcn_ablation": config["gcn_ablation"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(checkpoint_specification, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint_manifest = output_dir / "gcn_checkpoint_specification.json"
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
        raise RuntimeError("Existing GCN checkpoints were created under a different specification.")
    data = pd.read_csv(config["dataset"])
    outcomes = prepare_outcomes(data).to_numpy(dtype=np.int8)
    outer = repeated_outer_folds(outcomes, 1, config["validation"]["outer_folds"], config["seed"])[0]
    first_inner_train, first_inner_validation = next(
        inner_folds(outcomes[outer.train], config["validation"]["inner_folds"], config["seed"])
    )
    train_indices = outer.train[first_inner_train]
    validation_indices = outer.train[first_inner_validation]
    test_indices = outer.test
    processor = FoldPreprocessor(
        feature_contract=config["feature_contract"],
        random_state=config["seed"],
        rf_estimators=config["preprocessing"]["iterative_rf_estimators"],
        iterative_max_iter=config["preprocessing"]["iterative_max_iter"],
    ).fit(data.iloc[train_indices])
    train_x = processor.transform(data.iloc[train_indices]).to_numpy()
    validation_x = processor.transform(data.iloc[validation_indices]).to_numpy()
    test_x = processor.transform(data.iloc[test_indices]).to_numpy()
    train_y = outcomes[train_indices]
    validation_y = outcomes[validation_indices]
    test_y = outcomes[test_indices]
    max_epochs = config["gcn_ablation"].get("max_epochs", 200)
    batch_size = config["gcn_ablation"].get("batch_size", 64)
    gcn_config = config["gcn_ablation"]
    architecture = {
        "embedding_dim": int(gcn_config["embedding_dim"]),
        "encoder_hidden": tuple(gcn_config["encoder_hidden"]),
        "graph_hidden": int(gcn_config["graph_hidden"]),
        "concat_head_width": int(gcn_config["concat_head_width"]),
        "dropout": float(gcn_config["dropout"]),
    }
    requested_seeds = list(config["gcn_ablation"]["seeds"] if seeds is None else seeds)
    for seed in requested_seeds:
        checkpoint = output_dir / f"gcn_seed_{seed}.csv"
        if checkpoint.exists():
            continue
        seed_records = []
        schedule = _batch_schedule(len(train_x), max_epochs, batch_size, seed)
        initial_states = {}
        for head in HEADS:
            seed_everything(seed)
            initial_states[head] = copy.deepcopy(
                PairedGCNModel(train_x.shape[1], train_y.shape[1], head, **architecture).state_dict()
            )
        adjacencies = {
            kind: build_adjacency(
                train_y,
                kind,
                asymmetric_cutoff=float(gcn_config["asymmetric_cutoff"]),
                symmetric_cutoff=float(gcn_config["symmetric_cutoff"]),
            )
            for kind in ADJACENCIES
        }
        for head in HEADS:
            for loss_name in LOSSES:
                for adjacency_name in ADJACENCIES:
                    result = _train_one(
                        initial_states[head],
                        head,
                        loss_name,
                        adjacencies[adjacency_name],
                        train_x,
                        train_y,
                        validation_x,
                        validation_y,
                        test_x,
                        test_y,
                        schedule,
                        seed,
                        architecture=architecture,
                        learning_rate=float(gcn_config["learning_rate"]),
                        weight_decay=float(gcn_config["weight_decay"]),
                        patience=config["gcn_ablation"].get("patience", 25),
                        min_delta=float(gcn_config["min_delta"]),
                        gamma_positive=float(gcn_config["asymmetric_loss_gamma_positive"]),
                        gamma_negative=tuple(gcn_config["asymmetric_loss_gamma_negative"]),
                        probability_clip=float(gcn_config["asymmetric_loss_clip"]),
                    )
                    seed_records.append(
                        {
                            "seed": seed,
                            "adjacency": adjacency_name,
                            "head": head,
                            "loss": loss_name,
                            **result,
                        }
                    )
        pd.DataFrame(seed_records).to_csv(checkpoint, index=False)
    if not assemble:
        available = [output_dir / f"gcn_seed_{seed}.csv" for seed in requested_seeds]
        raw = pd.concat([pd.read_csv(path) for path in available if path.exists()], ignore_index=True)
        return raw, pd.DataFrame()

    checkpoint_paths = [output_dir / f"gcn_seed_{seed}.csv" for seed in config["gcn_ablation"]["seeds"]]
    missing = [path.name for path in checkpoint_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"GCN seed checkpoints are incomplete: {missing}")
    raw = pd.concat([pd.read_csv(path) for path in checkpoint_paths], ignore_index=True)
    paired_rows = []
    for head in HEADS:
        for loss_name in LOSSES:
            subset = raw[(raw["head"] == head) & (raw["loss"] == loss_name)]
            baseline = subset[subset["adjacency"] == "none"].set_index("seed")["test_macro_auroc"]
            for graph in ("asymmetric", "symmetric"):
                graph_values = subset[subset["adjacency"] == graph].set_index("seed")["test_macro_auroc"]
                difference = (graph_values - baseline).sort_index().to_numpy()
                half_width = t.ppf(0.975, df=len(difference) - 1) * difference.std(ddof=1) / np.sqrt(len(difference))
                paired_rows.append(
                    {
                        "head": head,
                        "loss": loss_name,
                        "contrast": f"{graph} - none",
                        "n_paired_seeds": len(difference),
                        "mean_difference": float(difference.mean()),
                        "ci_low": float(difference.mean() - half_width),
                        "ci_high": float(difference.mean() + half_width),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    raw.to_csv(output_dir / "gcn_ablation_10_seed_full.csv", index=False)
    paired.to_csv(output_dir / "gcn_ablation_paired_differences.csv", index=False)
    (output_dir / "gcn_ablation_metadata.json").write_text(
        json.dumps(
            {
                "feature_contract": config["feature_contract"],
                "train_rows": train_indices.tolist(),
                "validation_rows": validation_indices.tolist(),
                "test_rows": test_indices.tolist(),
                "seeds": config["gcn_ablation"]["seeds"],
                "hyperparameters": config["gcn_ablation"],
                "pairing": "Within each seed, matching graph variants share initialization and epoch-wise batch order.",
                "positive_weight": "N_negative / N_positive",
                "ci": "Two-sided 95% t interval over 10 paired seed differences.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw, paired

