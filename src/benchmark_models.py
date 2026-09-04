"""Uniform model adapters for the locked benchmark."""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import roc_auc_score

from .benchmark_features import LABEL_COLS


MODEL_NAMES = [
    "BR-LR",
    "BR-XGBoost",
    "BR-LightGBM",
    "BR-CatBoost",
    "ECC-LightGBM",
    "LP-RF",
    "RAkELd-RF",
    "Shared-MLP",
    "MultiTask-DNN",
    "TabPFN",
]

warnings.filterwarnings("ignore", message="X does not have valid feature names")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _dense_probability(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if isinstance(value, list):
        columns = []
        for item in value:
            if hasattr(item, "toarray"):
                item = item.toarray()
            item = np.asarray(item)
            columns.append(item[:, -1] if item.ndim == 2 else item.ravel())
        value = np.column_stack(columns)
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    return np.clip(result, 1e-6, 1 - 1e-6)


def _constant_probability(y: np.ndarray, n_test: int) -> np.ndarray | None:
    unique = np.unique(y)
    if unique.size == 1:
        return np.full(n_test, float(unique[0]), dtype=np.float64)
    return None


def select_top_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int = 80,
    seed: int = 42,
    n_estimators: int = 50,
) -> np.ndarray:
    """Aggregate training-fold-only RF importance over all 12 outcomes."""
    if x_train.shape[1] <= n_features:
        return np.arange(x_train.shape[1])
    importance = np.zeros(x_train.shape[1], dtype=np.float64)
    for label_index in range(y_train.shape[1]):
        if np.unique(y_train[:, label_index]).size < 2:
            continue
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            random_state=seed + label_index,
            n_jobs=1,
        )
        estimator.fit(x_train, y_train[:, label_index])
        importance += estimator.feature_importances_
    return np.argsort(importance, kind="stable")[::-1][:n_features]


def _fit_binary_relevance(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    params: dict,
    iterations: list[int] | None = None,
) -> np.ndarray:
    predictions = np.empty((len(x_test), y_train.shape[1]), dtype=np.float64)
    for label_index in range(y_train.shape[1]):
        target = y_train[:, label_index].astype(int)
        constant = _constant_probability(target, len(x_test))
        if constant is not None:
            predictions[:, label_index] = constant
            continue
        positive = max(int(target.sum()), 1)
        negative = len(target) - positive
        weight = negative / positive
        local_seed = seed + label_index
        if model_name == "BR-LR":
            estimator = LogisticRegression(
                max_iter=params.get("max_iter", 5000),
                class_weight=params.get("class_weight", "balanced"),
                random_state=local_seed,
                solver="liblinear",
            )
        elif model_name == "BR-XGBoost":
            import xgboost as xgb

            estimator = xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 1.0),
                colsample_bytree=params.get("colsample_bytree", 1.0),
                scale_pos_weight=weight,
                eval_metric="logloss",
                random_state=local_seed,
                n_jobs=params.get("n_jobs", 1),
                verbosity=0,
            )
        elif model_name == "BR-LightGBM":
            import lightgbm as lgb

            estimator = lgb.LGBMClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.03),
                scale_pos_weight=weight,
                random_state=local_seed,
                n_jobs=params.get("n_jobs", 1),
                verbose=-1,
                force_col_wise=True,
                deterministic=True,
            )
        elif model_name == "BR-CatBoost":
            from catboost import CatBoostClassifier

            estimator = CatBoostClassifier(
                iterations=(iterations[label_index] if iterations else params.get("iterations", 500)),
                depth=params.get("depth", 6),
                learning_rate=params.get("learning_rate", 0.03),
                class_weights=[1.0, weight],
                random_seed=local_seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=params.get("thread_count", 1),
                loss_function="Logloss",
                eval_metric="AUC",
            )
        else:  # pragma: no cover - guarded by public adapter
            raise KeyError(model_name)
        estimator.fit(x_train, target)
        predictions[:, label_index] = estimator.predict_proba(x_test)[:, 1]
    return np.clip(predictions, 1e-6, 1 - 1e-6)


def catboost_best_iterations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    params: dict,
) -> list[int]:
    """Estimate per-label training rounds without using an outer test fold."""
    from catboost import CatBoostClassifier

    rounds = []
    for label_index in range(y_train.shape[1]):
        target = y_train[:, label_index].astype(int)
        if np.unique(target).size < 2 or np.unique(y_validation[:, label_index]).size < 2:
            rounds.append(params.get("iterations", 500))
            continue
        positive = max(int(target.sum()), 1)
        negative = len(target) - positive
        estimator = CatBoostClassifier(
            iterations=params.get("iterations", 500),
            depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.03),
            class_weights=[1.0, negative / positive],
            random_seed=seed + label_index,
            verbose=False,
            allow_writing_files=False,
            thread_count=params.get("thread_count", 1),
            loss_function="Logloss",
            eval_metric="AUC",
            od_type="Iter",
            od_wait=params.get("early_stopping_rounds", 50),
            use_best_model=True,
        )
        estimator.fit(
            x_train,
            target,
            eval_set=(x_validation, y_validation[:, label_index].astype(int)),
        )
        best = estimator.get_best_iteration()
        rounds.append(int(best + 1 if best is not None and best >= 0 else params.get("iterations", 500)))
    return rounds


def _fit_ecc_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    params: dict,
) -> np.ndarray:
    import lightgbm as lgb

    n_chains = params.get("n_chains", 50)

    def train_chain(chain_index: int):
        local_seed = seed + chain_index
        order = np.random.default_rng(local_seed).permutation(y_train.shape[1])
        base = lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.03),
            n_jobs=1,
            random_state=local_seed,
            verbose=-1,
            force_col_wise=True,
            deterministic=True,
        )
        chain = ClassifierChain(base, order=order, random_state=local_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chain.fit(x_train, y_train)
            return chain.predict_proba(x_test)

    values = Parallel(n_jobs=params.get("parallel_jobs", -1))(
        delayed(train_chain)(index) for index in range(n_chains)
    )
    return np.clip(np.mean(values, axis=0), 1e-6, 1 - 1e-6)


def _fit_problem_transform(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    params: dict,
) -> np.ndarray:
    from skmultilearn.ensemble import RakelD
    from skmultilearn.problem_transform import LabelPowerset

    forest = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 10),
        random_state=seed,
        n_jobs=params.get("n_jobs", -1),
        class_weight=params.get("class_weight"),
    )
    if model_name == "LP-RF":
        estimator = LabelPowerset(classifier=forest)
    else:
        estimator = RakelD(
            base_classifier=forest,
            labelset_size=params.get("labelset_size", 4),
            base_classifier_require_dense=[True, True],
        )
    estimator.fit(x_train, y_train)
    return _dense_probability(estimator.predict_proba(x_test))


def _fit_tabpfn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    params: dict,
    selected_features: np.ndarray | None,
) -> np.ndarray:
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    from tabpfn import TabPFNClassifier

    if selected_features is None:
        selected_features = select_top_features(
            x_train,
            y_train,
            n_features=params.get("top_features", 80),
            seed=seed,
            n_estimators=params.get("selector_estimators", 50),
        )
    predictions = np.empty((len(x_test), y_train.shape[1]), dtype=np.float64)
    for label_index in range(y_train.shape[1]):
        target = y_train[:, label_index].astype(int)
        constant = _constant_probability(target, len(x_test))
        if constant is not None:
            predictions[:, label_index] = constant
            continue
        estimator = TabPFNClassifier(
            device=params.get("device", "cpu"),
            random_state=seed + label_index,
            ignore_pretraining_limits=True,
            n_estimators=params.get("n_estimators", 8),
            n_preprocessing_jobs=params.get("n_preprocessing_jobs", 1),
            show_progress_bar=False,
        )
        estimator.fit(x_train[:, selected_features], target)
        predictions[:, label_index] = estimator.predict_proba(
            x_test[:, selected_features]
        )[:, 1]
    return np.clip(predictions, 1e-6, 1 - 1e-6)


class _SharedMLP:
    @staticmethod
    def build(n_features: int, n_labels: int, params: dict):
        import torch.nn as nn

        hidden = params.get("hidden", [128, 64])
        layers = []
        previous = n_features
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(params.get("dropout", 0.2))])
            previous = width
        layers.append(nn.Linear(previous, n_labels))
        return nn.Sequential(*layers)


class _MultiTaskDNN:
    @staticmethod
    def build(n_features: int, n_labels: int, params: dict):
        import torch
        import torch.nn as nn

        shared_width = params.get("shared_width", 128)
        head_width = params.get("head_width", 32)

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(n_features, shared_width),
                    nn.ReLU(),
                    nn.Dropout(params.get("dropout", 0.2)),
                )
                self.heads = nn.ModuleList(
                    [nn.Sequential(nn.Linear(shared_width, head_width), nn.ReLU(), nn.Linear(head_width, 1)) for _ in range(n_labels)]
                )

            def forward(self, value):
                encoded = self.shared(value)
                return torch.cat([head(encoded) for head in self.heads], dim=1)

        return Network()


def _fit_neural(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    params: dict,
    epochs: int | None,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed)
    network = (
        _SharedMLP.build(x_train.shape[1], y_train.shape[1], params)
        if model_name == "Shared-MLP"
        else _MultiTaskDNN.build(x_train.shape[1], y_train.shape[1], params)
    )
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=params.get("learning_rate", 1e-3),
        weight_decay=params.get("weight_decay", 1e-5),
    )
    target_tensor = torch.tensor(y_train, dtype=torch.float32)
    positives = target_tensor.sum(dim=0)
    weights = (len(target_tensor) - positives) / positives.clamp_min(1.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), target_tensor),
        batch_size=params.get("batch_size", 64),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    network.train()
    for _ in range(epochs or params.get("epochs", 100)):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    network.eval()
    with torch.no_grad():
        probability = torch.sigmoid(network(torch.tensor(x_test, dtype=torch.float32))).numpy()
    return np.clip(probability, 1e-6, 1 - 1e-6)


def neural_best_epoch(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    params: dict,
) -> int:
    """Select the epoch on an inner validation fold, then discard the model."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed)
    network = (
        _SharedMLP.build(x_train.shape[1], y_train.shape[1], params)
        if model_name == "Shared-MLP"
        else _MultiTaskDNN.build(x_train.shape[1], y_train.shape[1], params)
    )
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=params.get("learning_rate", 1e-3),
        weight_decay=params.get("weight_decay", 1e-5),
    )
    train_target = torch.tensor(y_train, dtype=torch.float32)
    positive = train_target.sum(dim=0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=(len(train_target) - positive) / positive.clamp_min(1.0))
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), train_target),
        batch_size=params.get("batch_size", 64),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    validation_x = torch.tensor(x_validation, dtype=torch.float32)
    best_auc = -np.inf
    best_epoch = 1
    patience = 0
    for epoch in range(1, params.get("max_epochs", 200) + 1):
        network.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            probability = torch.sigmoid(network(validation_x)).numpy()
        aucs = [
            roc_auc_score(y_validation[:, index], probability[:, index])
            for index in range(y_validation.shape[1])
            if np.unique(y_validation[:, index]).size == 2
        ]
        score = float(np.mean(aucs))
        if score > best_auc + params.get("min_delta", 1e-4):
            best_auc = score
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= params.get("patience", 20):
            break
    return best_epoch


def fit_predict_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    params: dict,
    *,
    selected_features: np.ndarray | None = None,
    catboost_iterations: list[int] | None = None,
    neural_epochs: int | None = None,
) -> np.ndarray:
    """Fit one named benchmark model and return 12-label probabilities."""
    seed_everything(seed)
    if model_name in {"BR-LR", "BR-XGBoost", "BR-LightGBM", "BR-CatBoost"}:
        return _fit_binary_relevance(
            model_name, x_train, y_train, x_test, seed, params, catboost_iterations
        )
    if model_name == "ECC-LightGBM":
        return _fit_ecc_lightgbm(x_train, y_train, x_test, seed, params)
    if model_name in {"LP-RF", "RAkELd-RF"}:
        return _fit_problem_transform(model_name, x_train, y_train, x_test, seed, params)
    if model_name == "TabPFN":
        return _fit_tabpfn(
            x_train, y_train, x_test, seed, params, selected_features
        )
    if model_name in {"Shared-MLP", "MultiTask-DNN"}:
        return _fit_neural(
            model_name, x_train, y_train, x_test, seed, params, neural_epochs
        )
    raise KeyError(f"Unknown model: {model_name}")


def validation_auc_weights(
    y_validation: np.ndarray,
    lp_probability: np.ndarray,
    tabpfn_probability: np.ndarray,
) -> np.ndarray:
    """Softmax of per-label inner OOF AUROCs; shape is (2, 12)."""
    aucs = np.empty((2, y_validation.shape[1]), dtype=np.float64)
    for label_index in range(y_validation.shape[1]):
        target = y_validation[:, label_index]
        if np.unique(target).size < 2:
            aucs[:, label_index] = 0.5
        else:
            aucs[0, label_index] = roc_auc_score(target, lp_probability[:, label_index])
            aucs[1, label_index] = roc_auc_score(target, tabpfn_probability[:, label_index])
    shifted = aucs - aucs.max(axis=0, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=0, keepdims=True)

