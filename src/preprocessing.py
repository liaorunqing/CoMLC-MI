"""
=============================================================================
CoMLC-MI Research: Phase 1 — Data Preprocessing Pipeline
=============================================================================
Implements the tiered missing-data strategy (Section 4.3):
  1. Exclude features with >95% missing (KFK_BLOOD, IBS_NASL)
  2. Retain EMS BP (63% missing) with missingness indicator + MICE
  3. MICE (10 imputations) for lab values (17-22% missing)
  4. Median/mode imputation for remaining <7.5% missing features
  5. Train/val/test split (70/10/20) stratified by LET_IS
  6. StandardScaler normalization for numeric features
=============================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, 'dataset', 'Myocardial infarction complications Database.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output', 'processed_data')
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Feature Definitions ──────────────────────────────────────────────────
LABEL_COLS = [
    'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK',
    'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS'
]

# Features to exclude entirely (>95% missing)
EXCLUDE_FEATURES = ['KFK_BLOOD', 'IBS_NASL']

# Features with high missingness (40-65%) — retain with missingness indicator
HIGH_MISSING_FEATURES = ['S_AD_KBRIG', 'D_AD_KBRIG']

# Lab features needing MICE (15-25% missing)
LAB_FEATURES = ['GIPO_K', 'K_BLOOD', 'GIPER_NA', 'NA_BLOOD',
                'ALT_BLOOD', 'AST_BLOOD', 'L_BLOOD', 'ROE']

# Moderate missing (10-15%): also MICE
MODERATE_MISSING = ['DLIT_AG', 'D_AD_ORIT', 'S_AD_ORIT']

# Features with low missingness — simple imputation handles these automatically

# Temporal horizon feature sets (after exclusion)
BASE_FEATURES = [
    'AGE', 'SEX',
    'INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST',
    'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A',
    'nr_11', 'nr_01', 'nr_02', 'nr_03', 'nr_04', 'nr_07', 'nr_08',
    'np_01', 'np_04', 'np_05', 'np_07', 'np_08', 'np_09', 'np_10',
    'endocr_01', 'endocr_02', 'endocr_03',
    'zab_leg_01', 'zab_leg_02', 'zab_leg_03', 'zab_leg_04', 'zab_leg_06',
    # Admission physiology (keep EMS BP as indicators)
    'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT',
    'O_L_POST', 'K_SH_POST', 'MP_TP_POST', 'SVT_POST', 'GT_POST', 'FIB_G_POST',
    # ECG
    'ant_im', 'lat_im', 'inf_im', 'post_im', 'IM_PG_P',
    'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04', 'ritm_ecg_p_06',
    'ritm_ecg_p_07', 'ritm_ecg_p_08',
    'n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04',
    'n_r_ecg_p_05', 'n_r_ecg_p_06', 'n_r_ecg_p_08', 'n_r_ecg_p_09', 'n_r_ecg_p_10',
    'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04', 'n_p_ecg_p_05',
    'n_p_ecg_p_06', 'n_p_ecg_p_07', 'n_p_ecg_p_08', 'n_p_ecg_p_09',
    'n_p_ecg_p_10', 'n_p_ecg_p_11', 'n_p_ecg_p_12',
    # Treatment
    'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05',
    'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08',
    # Blood labs
    'GIPO_K', 'K_BLOOD', 'GIPER_NA', 'NA_BLOOD',
    'ALT_BLOOD', 'AST_BLOOD', 'L_BLOOD', 'ROE',
    # Temporal admission
    'TIME_B_S',
    # ICU Day 1
    'R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_KB', 'NOT_NA_KB',
    'LID_KB', 'NITR_S',
    # ICU Day 2
    'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n',
    'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n',
    # ICU Day 3
    'LID_S_n', 'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n',
    'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n',
]

# ── 1. Load Data ─────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading and Initial Cleaning")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

# Remove excluded features
df = df.drop(columns=[c for c in EXCLUDE_FEATURES if c in df.columns])
print(f"After excluding {EXCLUDE_FEATURES}: {df.shape[1]} columns remain.")

# Separate features and labels
X = df[BASE_FEATURES].copy()
y = df[LABEL_COLS].copy()

# Convert labels to binary
# NOTE: LET_IS is ordinal (0=survived, 1-7=types of death); binarize per plan
for col in LABEL_COLS:
    if col == 'LET_IS':
        y[col] = (y[col].fillna(0).astype(int) > 0).astype(int)
    else:
        y[col] = y[col].fillna(0).astype(int)

print(f"Feature matrix X: {X.shape}")
print(f"Label matrix y: {y.shape}")

# ── 2. Missing Data Handling ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Tiered Missing Data Strategy")
print("=" * 70)

# Step 2a: Create missingness indicators for high-missing features
print("\n2a. Creating missingness indicators for EMS BP features...")
for col in HIGH_MISSING_FEATURES:
    if col in X.columns:
        X[f'{col}_MISSING'] = X[col].isnull().astype(int)
        print(f"  {col}: {X[f'{col}_MISSING'].sum()} missing ({X[f'{col}_MISSING'].mean()*100:.1f}%)")

# Step 2b: Split before MICE to avoid data leakage
print("\n2b. Splitting data before imputation...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y['LET_IS']
)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.125, random_state=RANDOM_SEED,
    stratify=y_train['LET_IS']
)
# Final: 70% train, 10% val, 20% test
print(f"Train: {len(X_train_raw)} ({len(X_train_raw)/len(df)*100:.0f}%)")
print(f"Val:   {len(X_val_raw)} ({len(X_val_raw)/len(df)*100:.0f}%)")
print(f"Test:  {len(X_test_raw)} ({len(X_test_raw)/len(df)*100:.0f}%)")

# Step 2c: MICE imputation on training data
print("\n2c. Running MICE imputation (10 max_iter)...")
mice_features = LAB_FEATURES + MODERATE_MISSING + HIGH_MISSING_FEATURES
mice_features = [f for f in mice_features if f in X.columns]

# Fit MICE on training data
mice_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED),
    max_iter=10,
    random_state=RANDOM_SEED,
    initial_strategy='median'
)
X_train_mice = X_train_raw.copy()
X_val_mice = X_val_raw.copy()
X_test_mice = X_test_raw.copy()

# Apply MICE to the selected features
if mice_features:
    X_train_mice[mice_features] = mice_imputer.fit_transform(X_train_raw[mice_features])
    X_val_mice[mice_features] = mice_imputer.transform(X_val_raw[mice_features])
    X_test_mice[mice_features] = mice_imputer.transform(X_test_raw[mice_features])
    print(f"  MICE applied to {len(mice_features)} features: {mice_features}")

# Step 2d: Simple imputation for remaining features
print("\n2d. Applying median/mode imputation for remaining features...")
remaining_features = [f for f in X.columns if f not in mice_features]
if remaining_features:
    # Use median for numeric, mode for binary
    simple_imputer = SimpleImputer(strategy='median')
    X_train_imputed = X_train_mice.copy()
    X_val_imputed = X_val_mice.copy()
    X_test_imputed = X_test_mice.copy()

    numeric_remaining = [f for f in remaining_features
                         if X_train_raw[f].nunique() > 2 and X_train_raw[f].dtype in ['int64', 'float64']]
    binary_remaining = [f for f in remaining_features
                        if X_train_raw[f].nunique() <= 2]

    if numeric_remaining:
        simple_imputer.fit(X_train_raw[numeric_remaining])
        X_train_imputed[numeric_remaining] = simple_imputer.transform(X_train_raw[numeric_remaining])
        X_val_imputed[numeric_remaining] = simple_imputer.transform(X_val_raw[numeric_remaining])
        X_test_imputed[numeric_remaining] = simple_imputer.transform(X_test_raw[numeric_remaining])

    if binary_remaining:
        mode_imputer = SimpleImputer(strategy='most_frequent')
        mode_imputer.fit(X_train_raw[binary_remaining])
        X_train_imputed[binary_remaining] = mode_imputer.transform(X_train_raw[binary_remaining])
        X_val_imputed[binary_remaining] = mode_imputer.transform(X_val_raw[binary_remaining])
        X_test_imputed[binary_remaining] = mode_imputer.transform(X_test_raw[binary_remaining])

    print(f"  Numeric (median): {len(numeric_remaining)} features")
    print(f"  Binary (mode): {len(binary_remaining)} features")
else:
    X_train_imputed = X_train_mice
    X_val_imputed = X_val_mice
    X_test_imputed = X_test_mice

# ── 3. Feature Normalization ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Feature Normalization (StandardScaler)")
print("=" * 70)

# Identify numeric features for scaling
numeric_cols = X_train_imputed.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Don't scale binary features (0/1)
binary_cols = [c for c in numeric_cols if X_train_imputed[c].nunique() <= 2]
scale_cols = [c for c in numeric_cols if c not in binary_cols]

scaler = StandardScaler()
X_train_scaled = X_train_imputed.copy()
X_val_scaled = X_val_imputed.copy()
X_test_scaled = X_test_imputed.copy()

X_train_scaled[scale_cols] = scaler.fit_transform(X_train_imputed[scale_cols])
X_val_scaled[scale_cols] = scaler.transform(X_val_imputed[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test_imputed[scale_cols])

print(f"Scaled {len(scale_cols)} numeric features.")
print(f"Kept {len(binary_cols)} binary features unscaled.")

# ── 4. Post-imputation Quality Check ─────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Post-imputation Quality Check")
print("=" * 70)

# Check for remaining missing values
train_missing = X_train_scaled.isnull().sum().sum()
val_missing = X_val_scaled.isnull().sum().sum()
test_missing = X_test_scaled.isnull().sum().sum()
print(f"Remaining missing values — Train: {train_missing}, Val: {val_missing}, Test: {test_missing}")

# Check for infinite values
train_inf = np.isinf(X_train_scaled.select_dtypes(include=['float64']).values).sum()
print(f"Infinite values: {train_inf}")

# Distribution check
print(f"\nFeature dimension: {X_train_scaled.shape[1]}")
print(f"Training samples: {X_train_scaled.shape[0]}")
print(f"Label distribution in splits:")
for col in LABEL_COLS:
    tr_pct = y_train[col].mean() * 100
    te_pct = y_test[col].mean() * 100
    print(f"  {col}: train {tr_pct:.2f}%, test {te_pct:.2f}%")

# ── 5. Save Processed Data ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Saving Processed Data")
print("=" * 70)

processed = {
    'X_train': X_train_scaled,
    'X_val': X_val_scaled,
    'X_test': X_test_scaled,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
}

for key, df_obj in processed.items():
    df_obj.to_csv(os.path.join(OUTPUT_DIR, f'{key}.csv'), index=False)

# Save artifacts
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(OUTPUT_DIR, 'mice_imputer.pkl'), 'wb') as f:
    pickle.dump(mice_imputer, f)
with open(os.path.join(OUTPUT_DIR, 'feature_names.json'), 'w') as f:
    json.dump(list(X_train_scaled.columns), f)
with open(os.path.join(OUTPUT_DIR, 'scale_cols.json'), 'w') as f:
    json.dump(scale_cols, f)
with open(os.path.join(OUTPUT_DIR, 'binary_cols.json'), 'w') as f:
    json.dump(binary_cols, f)

# Also save raw (unscaled) versions for classical ML baselines
raw_processed = {
    'X_train_raw': X_train_imputed,
    'X_val_raw': X_val_imputed,
    'X_test_raw': X_test_imputed,
}
for key, df_obj in raw_processed.items():
    df_obj.to_csv(os.path.join(OUTPUT_DIR, f'{key}.csv'), index=False)

print(f"\nAll processed data saved to: {OUTPUT_DIR}")
print("Files created:")
for fname in os.listdir(OUTPUT_DIR):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname} ({size_kb:.1f} KB)")

print("\nPreprocessing pipeline complete.")
