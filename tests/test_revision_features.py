from __future__ import annotations

import numpy as np
import pandas as pd

from src.revision_features import (
    AMBIGUOUS_INPATIENT_FEATURES,
    DISALLOWED_ENGINEERED_FEATURES,
    FoldPreprocessor,
    INTERVAL_COUNT_FEATURES,
    feature_sets,
    prepare_outcomes,
)


DATA_PATH = "dataset/Myocardial infarction complications Database.csv"


def test_locked_feature_counts_and_temporal_boundaries() -> None:
    data = pd.read_csv(DATA_PATH)
    sets = feature_sets(data.columns)
    assert {name: len(values) for name, values in sets.items()} == {
        "admission_safe_v1": 91,
        "prospective_24h": 94,
        "prospective_48h": 97,
        "prospective_72h": 100,
        "retrospective_full_111": 111,
    }
    primary = set(sets["admission_safe_v1"])
    assert primary.isdisjoint(INTERVAL_COUNT_FEATURES)
    assert primary.isdisjoint(AMBIGUOUS_INPATIENT_FEATURES)
    assert primary.isdisjoint(DISALLOWED_ENGINEERED_FEATURES)


def test_outcomes_are_binary() -> None:
    outcomes = prepare_outcomes(pd.read_csv(DATA_PATH))
    assert outcomes.shape == (1700, 12)
    assert np.isin(outcomes.to_numpy(), (0, 1)).all()


def test_fold_preprocessor_never_fits_on_test_rows() -> None:
    data = pd.read_csv(DATA_PATH)
    train = data.iloc[:120].copy()
    test = data.iloc[120:160].copy()
    processor = FoldPreprocessor(
        feature_contract="admission_safe_v1",
        random_state=7,
        rf_estimators=3,
        iterative_max_iter=2,
    ).fit(train)
    transformed_train = processor.transform(train)
    transformed_test = processor.transform(test)
    assert transformed_train.shape == (120, 91)
    assert transformed_test.shape == (40, 91)
    assert set(processor.audit_.fitted_row_ids) == set(train.index)
    assert set(processor.audit_.fitted_row_ids).isdisjoint(test.index)
    assert np.isfinite(transformed_train.to_numpy()).all()
    assert np.isfinite(transformed_test.to_numpy()).all()

