"""Leakage-safe feature definitions and fold-local preprocessing.

The formal revision analysis has one admission-time feature contract.  All
derived state in :class:`FoldPreprocessor` is learned from the supplied
training rows only and the fitted row identifiers are retained for automated
leakage audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler


LABEL_COLS = [
    "FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK",
    "OTEK_LANC", "RAZRIV", "DRESSLER", "ZSN", "REC_IM", "P_IM_STEN",
    "LET_IS",
]

EXCLUDED_SOURCE_FEATURES = ["ID", "KFK_BLOOD", "IBS_NASL"]

# Interval-count variables are not available at the admission prediction time.
INTERVAL_COUNT_FEATURES = [
    "R_AB_1_n", "NA_R_1_n", "NOT_NA_1_n",
    "R_AB_2_n", "NA_R_2_n", "NOT_NA_2_n",
    "R_AB_3_n", "NA_R_3_n", "NOT_NA_3_n",
]

# Timing of these hospital-treatment variables is insufficiently defined in
# the source dictionary for a strict admission-time analysis.
AMBIGUOUS_INPATIENT_FEATURES = [
    "NA_KB", "NOT_NA_KB", "LID_KB", "NITR_S", "LID_S_n",
    "B_BLOK_S_n", "ANT_CA_S_n", "GEPAR_S_n", "ASP_S_n", "TIKL_S_n",
    "TRENT_S_n",
]

MISSING_INDICATOR_BASES = ["S_AD_KBRIG", "D_AD_KBRIG"]
MISSING_INDICATOR_FEATURES = [f"{name}_MISSING" for name in MISSING_INDICATOR_BASES]

ITERATIVE_RF_CANDIDATES = [
    "GIPO_K", "K_BLOOD", "GIPER_NA", "NA_BLOOD", "ALT_BLOOD",
    "AST_BLOOD", "L_BLOOD", "ROE", "DLIT_AG", "D_AD_ORIT",
    "S_AD_ORIT", "S_AD_KBRIG", "D_AD_KBRIG",
]

DISALLOWED_ENGINEERED_FEATURES = [
    "hemodynamic_severity", "ecg_extent_score", "arrhythmia_burden",
    "metabolic_stress", "age_fc_interaction", "antithrombotic_adequacy",
    "time_delay_severe", "age_time_risk",
]


def source_feature_names(columns: Sequence[str]) -> list[str]:
    """Return the 109 source-dictionary predictors in their original order."""
    excluded = set(LABEL_COLS + EXCLUDED_SOURCE_FEATURES)
    names = [name for name in columns if name not in excluded]
    if len(names) != 109:
        raise ValueError(f"Expected 109 source predictors, found {len(names)}.")
    return names


def feature_sets(columns: Sequence[str]) -> dict[str, list[str]]:
    """Build the locked 91/94/97/100/111 revision feature sets."""
    full_measured = source_feature_names(columns)
    excluded_primary = set(INTERVAL_COUNT_FEATURES + AMBIGUOUS_INPATIENT_FEATURES)
    admission_measured = [name for name in full_measured if name not in excluded_primary]
    interval_groups = [
        ["R_AB_1_n", "NA_R_1_n", "NOT_NA_1_n"],
        ["R_AB_2_n", "NA_R_2_n", "NOT_NA_2_n"],
        ["R_AB_3_n", "NA_R_3_n", "NOT_NA_3_n"],
    ]
    sets = {
        "admission_safe_v1": admission_measured + MISSING_INDICATOR_FEATURES,
        "prospective_24h": admission_measured + interval_groups[0] + MISSING_INDICATOR_FEATURES,
        "prospective_48h": admission_measured + interval_groups[0] + interval_groups[1] + MISSING_INDICATOR_FEATURES,
        "prospective_72h": admission_measured + sum(interval_groups, []) + MISSING_INDICATOR_FEATURES,
        "retrospective_full_111": full_measured + MISSING_INDICATOR_FEATURES,
    }
    expected = {
        "admission_safe_v1": 91,
        "prospective_24h": 94,
        "prospective_48h": 97,
        "prospective_72h": 100,
        "retrospective_full_111": 111,
    }
    for key, count in expected.items():
        if len(sets[key]) != count or len(set(sets[key])) != count:
            raise ValueError(f"Feature contract {key} must contain {count} unique variables.")
    return sets


def prepare_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the 12 binary outcomes, including binary all-cause in-hospital death."""
    outcome = frame[LABEL_COLS].copy()
    for name in LABEL_COLS:
        if name == "LET_IS":
            outcome[name] = (outcome[name].fillna(0).astype(int) > 0).astype(np.int8)
        else:
            outcome[name] = outcome[name].fillna(0).astype(np.int8)
    values = outcome.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError("All formal outcomes must be binary.")
    return outcome


def raw_features_for_contract(frame: pd.DataFrame, contract: str) -> pd.DataFrame:
    """Select measured columns for a contract; indicators are fold-generated."""
    sets = feature_sets(frame.columns)
    if contract not in sets:
        raise KeyError(f"Unknown feature contract: {contract}")
    measured = [name for name in sets[contract] if name not in MISSING_INDICATOR_FEATURES]
    return frame.loc[:, measured].copy()


@dataclass(frozen=True)
class PreprocessingAudit:
    feature_contract: str
    fitted_row_ids: tuple
    output_features: tuple[str, ...]
    iterative_features: tuple[str, ...]
    median_features: tuple[str, ...]
    mode_features: tuple[str, ...]
    scaled_features: tuple[str, ...]


class FoldPreprocessor(BaseEstimator, TransformerMixin):
    """Two-tier imputation and scaling fitted within one training fold.

    Parameters are intentionally explicit so the JSON configuration completely
    describes the formal run.  Row identifiers are recorded only for QA and
    are never passed to an estimator.
    """

    def __init__(
        self,
        feature_contract: str = "admission_safe_v1",
        random_state: int = 42,
        rf_estimators: int = 50,
        iterative_max_iter: int = 10,
    ) -> None:
        self.feature_contract = feature_contract
        self.random_state = random_state
        self.rf_estimators = rf_estimators
        self.iterative_max_iter = iterative_max_iter

    def _with_indicators(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected = raw_features_for_contract(frame, self.feature_contract)
        for base in MISSING_INDICATOR_BASES:
            selected[f"{base}_MISSING"] = frame[base].isna().astype(np.int8)
        ordered = feature_sets(frame.columns)[self.feature_contract]
        return selected.loc[:, ordered]

    def fit(self, x: pd.DataFrame, y=None):
        if not isinstance(x, pd.DataFrame):
            raise TypeError("FoldPreprocessor requires a pandas DataFrame with stable row indices.")
        work = self._with_indicators(x)
        self.fitted_row_ids_ = tuple(x.index.tolist())
        self.feature_names_out_ = list(work.columns)
        self.iterative_features_ = [name for name in ITERATIVE_RF_CANDIDATES if name in work]
        remaining = [name for name in work if name not in self.iterative_features_]
        self.mode_features_ = [name for name in remaining if work[name].dropna().nunique() <= 2]
        self.median_features_ = [name for name in remaining if name not in self.mode_features_]

        if self.iterative_features_:
            self.iterative_imputer_ = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=self.rf_estimators,
                    random_state=self.random_state,
                    n_jobs=1,
                ),
                max_iter=self.iterative_max_iter,
                random_state=self.random_state,
                initial_strategy="median",
                skip_complete=True,
            )
            self.iterative_imputer_.fit(work[self.iterative_features_])
        if self.median_features_:
            self.median_imputer_ = SimpleImputer(strategy="median")
            self.median_imputer_.fit(work[self.median_features_])
        if self.mode_features_:
            self.mode_imputer_ = SimpleImputer(strategy="most_frequent")
            self.mode_imputer_.fit(work[self.mode_features_])

        imputed = self._impute(work)
        self.binary_features_ = [name for name in imputed if imputed[name].nunique() <= 2]
        self.scaled_features_ = [name for name in imputed if name not in self.binary_features_]
        self.scaler_ = StandardScaler()
        if self.scaled_features_:
            self.scaler_.fit(imputed[self.scaled_features_])
        self.audit_ = PreprocessingAudit(
            feature_contract=self.feature_contract,
            fitted_row_ids=self.fitted_row_ids_,
            output_features=tuple(self.feature_names_out_),
            iterative_features=tuple(self.iterative_features_),
            median_features=tuple(self.median_features_),
            mode_features=tuple(self.mode_features_),
            scaled_features=tuple(self.scaled_features_),
        )
        return self

    def _impute(self, work: pd.DataFrame) -> pd.DataFrame:
        result = work.copy()
        if self.iterative_features_:
            result.loc[:, self.iterative_features_] = self.iterative_imputer_.transform(
                work[self.iterative_features_]
            )
        if self.median_features_:
            result.loc[:, self.median_features_] = self.median_imputer_.transform(
                work[self.median_features_]
            )
        if self.mode_features_:
            result.loc[:, self.mode_features_] = self.mode_imputer_.transform(
                work[self.mode_features_]
            )
        return result

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("FoldPreprocessor must be fitted before transform.")
        work = self._with_indicators(x)
        if list(work.columns) != self.feature_names_out_:
            raise ValueError("Feature contract changed between fit and transform.")
        result = self._impute(work)
        if self.scaled_features_:
            result.loc[:, self.scaled_features_] = self.scaler_.transform(
                result[self.scaled_features_]
            )
        values = result.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Preprocessing produced non-finite values.")
        return pd.DataFrame(values, index=x.index, columns=self.feature_names_out_)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        del input_features
        return np.asarray(self.feature_names_out_, dtype=object)

