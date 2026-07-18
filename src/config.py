"""
================================================================================
CoMLC-MI v2: Centralized Configuration
================================================================================
Single source of truth for all paths, feature definitions, and label metadata.
Imported by all other modules to eliminate duplication and inconsistency.
================================================================================
"""
import os
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')
RAW_CSV = os.path.join(DATASET_DIR, 'Myocardial infarction complications Database.csv')

# Preprocessed data directory (project-relative path for reproducibility)
ORIG_PROCESSED_DIR = os.path.join(PROJECT_DIR, 'processed_data')

# v2 output directories
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output', 'improvements_v2')
BASE_DIR = PROJECT_DIR  # backward compat
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Random Seed ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Label Definitions ────────────────────────────────────────────────────
LABEL_COLS = [
    'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK',
    'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS'
]
N_LABELS = len(LABEL_COLS)

LABEL_SHORT = ['AF', 'SVT', 'VT', 'VF', 'AVB', 'PulEd', 'Rupt', 'Dress', 'CHF', 'ReMI', 'PIA', 'Leth']
LABEL_FULL = [
    'Atrial Fibrillation', 'Supraventricular Tachycardia', 'Ventricular Tachycardia',
    'Ventricular Fibrillation', '3rd-degree AV Block', 'Pulmonary Edema',
    'Myocardial Rupture', 'Dressler Syndrome', 'Chronic Heart Failure',
    'Recurrent MI', 'Post-infarction Angina', 'Lethal Outcome',
]

# RARE_LABEL_INDICES: labels with prevalence < 5% — need special handling
# Computed from training data; hardcoded here based on EDA findings
RARE_LABEL_INDICES = [1, 2, 3, 4, 6, 7]  # SVT, VT, VF, AVB, Rupture, Dressler

# ── Feature Definitions ──────────────────────────────────────────────────
EXCLUDE_FEATURES = ['KFK_BLOOD', 'IBS_NASL']  # >95% missing

HIGH_MISSING_FEATURES = ['S_AD_KBRIG', 'D_AD_KBRIG']  # ~63% missing — keep with indicator

ALL_INPUT_FEATURES = [
    # Demographics
    'AGE', 'SEX',
    # Medical history / comorbidities (~40 features)
    'INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST',
    'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A',
    'nr_11', 'nr_01', 'nr_02', 'nr_03', 'nr_04', 'nr_07', 'nr_08',
    'np_01', 'np_04', 'np_05', 'np_07', 'np_08', 'np_09', 'np_10',
    'endocr_01', 'endocr_02', 'endocr_03',
    'zab_leg_01', 'zab_leg_02', 'zab_leg_03', 'zab_leg_04', 'zab_leg_06',
    # Admission physiology
    'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT',
    'O_L_POST', 'K_SH_POST', 'MP_TP_POST', 'SVT_POST', 'GT_POST', 'FIB_G_POST',
    # ECG findings (~25 features)
    'ant_im', 'lat_im', 'inf_im', 'post_im', 'IM_PG_P',
    'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04', 'ritm_ecg_p_06',
    'ritm_ecg_p_07', 'ritm_ecg_p_08',
    'n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04',
    'n_r_ecg_p_05', 'n_r_ecg_p_06', 'n_r_ecg_p_08', 'n_r_ecg_p_09', 'n_r_ecg_p_10',
    'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04', 'n_p_ecg_p_05',
    'n_p_ecg_p_06', 'n_p_ecg_p_07', 'n_p_ecg_p_08', 'n_p_ecg_p_09',
    'n_p_ecg_p_10', 'n_p_ecg_p_11', 'n_p_ecg_p_12',
    # Treatment: fibrinolytic therapy (~7 features)
    'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05',
    'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08',
    # Blood labs (~9 features)
    'GIPO_K', 'K_BLOOD', 'GIPER_NA', 'NA_BLOOD',
    'ALT_BLOOD', 'AST_BLOOD', 'L_BLOOD', 'ROE',
    # Temporal: admission
    'TIME_B_S',
    # ICU Day 1
    'R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_KB', 'NOT_NA_KB', 'LID_KB', 'NITR_S',
    # ICU Day 2
    'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n',
    'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n',
    # ICU Day 3
    'LID_S_n', 'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n',
    'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n',
]

# ── Hyperparameter Defaults (from ablation study) ────────────────────────
DEFAULT_HP = {
    'hidden_dim': 64,
    'dropout': 0.3,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 64,
    'n_estimators_lgb': 200,
    'max_depth_lgb': 6,
    'learning_rate_lgb': 0.03,
    'n_chains': 50,
    'n_cv_outer': 5,
    'n_cv_inner': 2,
}

# ── TabPFN Settings ──────────────────────────────────────────────────────
TABPFN_N_FEATURES = 80
TABPFN_MAX_SAMPLES = None  # use ALL training data (fixed from 1000)


def get_label_prevalence(y):
    """Compute per-label prevalence from label matrix."""
    return y.mean(axis=0)


def get_pos_weights(y):
    """Compute inverse prevalence weights for BCE loss."""
    prev = get_label_prevalence(y)
    return 1.0 / (prev + 1e-6)


# Verification
if __name__ == '__main__':
    print(f"Config loaded: {len(ALL_INPUT_FEATURES)} features, {N_LABELS} labels")
    print(f"Labels: {LABEL_COLS}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
