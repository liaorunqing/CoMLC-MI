from __future__ import annotations

import copy

import numpy as np

from src.revision_gcn import (
    PairedGCNModel,
    _batch_schedule,
    _train_one,
    build_adjacency,
)
from src.revision_models import seed_everything


def test_batch_schedule_is_shared_and_reproducible() -> None:
    first = _batch_schedule(103, epochs=4, batch_size=16, seed=42)
    second = _batch_schedule(103, epochs=4, batch_size=16, seed=42)
    assert len(first) == len(second) == 4
    for epoch_a, epoch_b in zip(first, second):
        assert all(np.array_equal(a, b) for a, b in zip(epoch_a, epoch_b))
        assert np.array_equal(np.sort(np.concatenate(epoch_a)), np.arange(103))


def test_matching_configuration_repeats_exactly() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(72, 8)).astype(np.float32)
    y = (rng.random((72, 3)) < 0.3).astype(np.float32)
    # Force both classes in every split and label.
    y[:6] = np.eye(3, dtype=np.float32).repeat(2, axis=0)
    train_x, validation_x, test_x = x[:48], x[48:60], x[60:]
    train_y, validation_y, test_y = y[:48], y[48:60], y[60:]
    seed_everything(13)
    initial = copy.deepcopy(PairedGCNModel(8, 3, "dot").state_dict())
    schedule = _batch_schedule(48, epochs=3, batch_size=16, seed=13)
    adjacency = build_adjacency(train_y, "symmetric")
    first = _train_one(
        initial, "dot", "W-BCE", adjacency,
        train_x, train_y, validation_x, validation_y, test_x, test_y,
        schedule, 13, patience=3,
    )
    second = _train_one(
        initial, "dot", "W-BCE", adjacency,
        train_x, train_y, validation_x, validation_y, test_x, test_y,
        schedule, 13, patience=3,
    )
    assert first == second

