from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import revision_temporal


class _IdentityProcessor:
    def fit(self, frame):
        return self

    def transform(self, frame):
        return pd.DataFrame({"x": np.arange(len(frame), dtype=float)})


def test_temporal_shard_writes_checkpoint_without_assembly(monkeypatch, tmp_path) -> None:
    data = pd.DataFrame({"placeholder": np.arange(6)})
    outcomes = np.tile(np.array([[0, 1] * 6], dtype=np.int8), (6, 1))
    fold = SimpleNamespace(
        repeat=0,
        fold=0,
        train=np.array([0, 1, 2, 3]),
        test=np.array([4, 5]),
    )
    monkeypatch.setattr(revision_temporal.pd, "read_csv", lambda _: data)
    monkeypatch.setattr(revision_temporal, "prepare_outcomes", lambda _: pd.DataFrame(outcomes))
    monkeypatch.setattr(revision_temporal, "repeated_outer_folds", lambda *args: [fold])
    monkeypatch.setattr(
        revision_temporal,
        "inner_folds",
        lambda *args: iter([(np.array([0, 1]), np.array([2, 3]))]),
    )
    monkeypatch.setattr(revision_temporal, "_processor", lambda *args: _IdentityProcessor())
    monkeypatch.setattr(revision_temporal, "neural_best_epoch", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        revision_temporal,
        "fit_predict_model",
        lambda *args, **kwargs: np.full((2, outcomes.shape[1]), 0.25),
    )
    config = {
        "dataset": "unused.csv",
        "seed": 42,
        "validation": {"outer_repeats": 1, "outer_folds": 2, "inner_folds": 2},
        "preprocessing": {},
        "models": {"Shared-MLP": {}},
    }
    metrics, labels = revision_temporal.run_temporal(
        config,
        tmp_path,
        horizons=["admission_safe_v1"],
        fold_indices=[0],
        assemble=False,
    )
    assert metrics.empty and labels.empty
    checkpoint = tmp_path / "temporal_admission_safe_v1_r1_f1.npz"
    assert checkpoint.exists()
    stored = np.load(checkpoint)
    assert np.array_equal(stored["test_indices"], fold.test)
    assert stored["probability"].shape == (2, outcomes.shape[1])
    assert not list(tmp_path.glob("*.tmp.npz"))
