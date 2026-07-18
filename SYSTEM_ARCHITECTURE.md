# CoMLC-MI System Architecture & Dataset Characteristics

> Completed 2026-06-21, covering the full research journey across 126 experiments

---

## Part 1: Dataset Characteristics

### 1.1 Basic Information

| Property | Value |
|----------|-------|
| Source | Krasnoyarsk Interdistrict Clinical Hospital No. 20, Russia |
| Time Window | 1992–1995 |
| Total Patients | 1,700 |
| Input Features | 111 (after excluding CPK and IBS_NASL) |
| Output Labels | 12 binary complications |
| Overall Missing Rate | 8.47% |
| Complete Cases | 0 (all patients have at least one missing feature) |
| Avg. Label Cardinality | 1.271 (average 1.27 complications per patient) |

### 1.2 Feature Grouping (111 Input Features)

#### Demographic Features (2)
`AGE`, `SEX`

#### Medical History & Comorbidities (~18)
`INF_ANAM` (number of prior MIs), `STENOK_AN` (angina functional class), `FK_STENOK` (angina functional grade), `IBS_POST` (prior CAD diagnosis), `GB` (hypertension grade), `SIM_GIPERT` (symptomatic hypertension), `DLIT_AG` (duration of arterial hypertension), `ZSN_A` (history of chronic heart failure),

Comorbidity markers: `nr_01`–`nr_11` (various concomitant diagnoses), `np_01`–`np_10` (prior complications), `endocr_01`–`endocr_03` (endocrine diseases: DM, obesity, thyrotoxicosis, etc.), `zab_leg_01`–`zab_leg_06` (peripheral arterial disease)

#### Admission Physiology (~10)
`S_AD_KBRIG` (EMS systolic BP, 63.3% missing, retained with missingness indicator), `D_AD_KBRIG` (EMS diastolic BP, 63.3% missing), `S_AD_ORIT` (admission systolic BP, 15.7% missing), `D_AD_ORIT` (admission diastolic BP, 15.7% missing),

`O_L_POST` (pulmonary edema at admission), `K_SH_POST` (cardiogenic shock at admission), `MP_TP_POST` (pacemaker use at admission), `SVT_POST` / `GT_POST` / `FIB_G_POST` (arrhythmia types at admission)

#### ECG Features (~30)
MI Localization: `ant_im`, `lat_im`, `inf_im`, `post_im` (anterior/lateral/inferior/posterior MI), `IM_PG_P` (right ventricular MI)

Rhythm features: `ritm_ecg_p_01`–`ritm_ecg_p_08` (sinus, atrial fibrillation, SVT, VT, VF, AV block, etc.)

ECG morphology: `n_r_ecg_p_01`–`n_r_ecg_p_10` (heart rate variability, etc.), `n_p_ecg_p_01`–`n_p_ecg_p_12` (ST segment changes, T-wave abnormalities, pathological Q waves, etc.)

#### Fibrinolytic Therapy (~7)
`fibr_ter_01`–`fibr_ter_08` (streptokinase, tPA, etc., administration timing and dosage)

#### Blood Labs (9)
`GIPO_K`/`K_BLOOD` (hypokalemia/serum potassium, 21.8% missing), `GIPER_NA`/`NA_BLOOD` (hypernatremia/serum sodium, 22.1% missing), `ALT_BLOOD`/`AST_BLOOD` (liver enzymes, 16.7% missing), `L_BLOOD` (leukocytes, 7.4% missing), `ROE` (ESR, 11.9% missing)

#### Temporal Features (~20, across 3 ICU days)

**Admission Time**: `TIME_B_S` (onset-to-admission time, 7.4% missing)

**ICU Day 1**: `R_AB_1_n`–`R_AB_3_n` (pain recurrence count), `NA_KB`/`NOT_NA_KB` (narcotic/non-narcotic analgesics), `LID_KB` (lidocaine), `NITR_S` (IV nitrates)

**ICU Day 2**: `NA_R_1_n`–`NA_R_3_n` (day 2 narcotic analgesics), `NOT_NA_1_n`–`NOT_NA_3_n` (day 2 non-narcotic analgesics)

**ICU Day 3**: `LID_S_n`, `B_BLOK_S_n`, `ANT_CA_S_n`, `GEPAR_S_n`, `ASP_S_n`, `TIKL_S_n`, `TRENT_S_n` (oral medications: lidocaine, beta-blockers, calcium antagonists, heparin, aspirin, Ticlid, Trental)

### 1.3 Output Labels (12 Complications)

| Code | English Name | Prevalence | Pathological Category |
|------|-------------|------------|----------------------|
| FIBR_PREDS | Atrial Fibrillation | 10.0% | Arrhythmia |
| PREDS_TAH | Supraventricular Tachycardia | 1.2% | Arrhythmia |
| JELUD_TAH | Ventricular Tachycardia | 2.5% | Arrhythmia |
| FIBR_JELUD | Ventricular Fibrillation | 4.2% | Arrhythmia |
| A_V_BLOK | 3rd-degree AV Block | 3.4% | Arrhythmia |
| OTEK_LANC | Pulmonary Edema | 9.4% | Hemodynamic |
| RAZRIV | Myocardial Rupture | 3.2% | Structural |
| DRESSLER | Dressler Syndrome | 4.4% | Immunologic |
| ZSN | Chronic Heart Failure | 23.2% | Hemodynamic |
| REC_IM | Recurrent MI | 9.4% | Recurrent |
| P_IM_STEN | Post-infarction Angina | 8.7% | Recurrent |
| LET_IS | Lethal Outcome | 15.9% | Terminal |

> Note: LET_IS is originally an ordinal variable (0=survived, 1–7=types of death) and was binarized to 0/1 (survived/deceased) during preprocessing.

### 1.4 Label Co-occurrence Structure

**Top-10 Conditional Probabilities P(Col | Row):**

| Source Complication | Target Complication | P |
|--------------------|--------------------|-----|
| Myocardial Rupture | Lethal Outcome | 1.000 |
| Pulmonary Edema | Chronic Heart Failure | 0.410 |
| Recurrent MI | Chronic Heart Failure | 0.340 |
| Atrial Fibrillation | Chronic Heart Failure | 0.308 |
| Dressler Syndrome | Chronic Heart Failure | 0.298 |
| Ventricular Fibrillation | Lethal Outcome | 0.288 |
| Recurrent MI | Lethal Outcome | 0.283 |
| 3rd-degree AV Block | Lethal Outcome | 0.273 |
| Ventricular Tachycardia | Chronic Heart Failure | 0.259 |
| Pulmonary Edema | Recurrent MI | 0.238 |

**Key Statistics**:
- Median conditional probability < 0.15
- 18 co-occurrence pairs exceed P=0.2
- CHF and Lethal Outcome are the dominant "sink nodes"
- The co-occurrence structure is too sparse to support meaningful message passing in graph neural networks

---

## Part 2: System Architecture & Design

### 2.1 Overall Research Design

This study aims to establish a systematic benchmark for multi-label prediction of myocardial infarction complications. The research underwent four iterative phases:

```
v0 (Original)      Baseline establishment + GCN ablation
  ↓
v1 (Improvement)   MissForest + ECC + MLSMOTE
  ↓
v2 (Bug Fixes)     5 bug corrections + CatBoost + Robust thresholding
  ↓
v3 (Deep Refinement) Ensemble methods + Clinical features + Patient GAT (ineffective)
  ↓
Final Corrections   Validation set real weights + DeLong/Permutation tests + Bootstrap
```

### 2.2 Data Preprocessing Pipeline

```
Raw CSV (1700 × 124)
  │
  ├─ Exclude features: CPK (99.8% missing), IBS_NASL (95.8% missing)
  ├─ LET_IS binarization: ordinal 0-7 → binary 0/1
  ├─ EMS BP missingness indicators: S_AD_KBRIG_MISSING, D_AD_KBRIG_MISSING
  │
  ├─ Stratified split: 70% train / 10% validation / 20% test (stratified by LET_IS)
  │
  ├─ MICE imputation (13 features with 10-65% missing, RF estimator, 10 iterations)
  │    Fit on training set only; transform validation and test sets (no data leakage)
  │
  ├─ Median/mode imputation (remaining features with <10% missing)
  │
  └─ StandardScaler normalization (continuous numeric features only; binary 0/1 features unscaled)
      Critical: single Scaler instance, fit on training set only, transform val and test
```

### 2.3 Core Module Architecture

```
src/
├── config.py                    ← Global configuration hub
│   ├── All paths, label definitions, feature lists
│   └── Hyperparameter defaults, TabPFN settings
│
├── evaluation.py                ← Unified evaluation framework
│   ├── compute_all_metrics()    → Micro/Macro-AUC, AUPRC, F1, Hamming Loss
│   ├── bootstrap CI             → Percentile bootstrap 500 resamples
│   ├── robust_threshold_tuning()→ 5-fold CV median optimal threshold
│   └── print_metrics_table()   → Formatted output
│
├── preprocessing.py             ← Data preprocessing
│   └── MICE + missingness indicators + Scaler
│
└── [Experiment Scripts]
    ├── baselines.py             → 8 baseline models
    ├── comlc_mi_opt.py          → 18 GCN+CAL ablation configurations
    ├── temporal_experiments.py  → 4 temporal window experiments
    ├── shap_analysis.py         → SHAP interpretability
    ├── improvements_v2.py       → 6-model comparison after bug fixes
    ├── publication_fixes.py     → 2-model ensemble + Bootstrap + Stacking
    └── final_statistical_tests.py → DeLong + Permutation + FDR
```

### 2.4 Model Architecture (Phased Evolution)

#### Baseline Models (v0, 8 models)
- **Independent Classifiers**: XGBoost (per-label, scale_pos_weight for imbalance)
- **Classical Multi-label Methods**: Binary Relevance (LR), Classifier Chains (3-chain ensemble), Label Powerset (RF), RAkEL (k=4)
- **Deep Multi-label**: Shared-Encoder MLP (256→128→64, weighted BCE), Multitask DNN (shared encoder + per-label branches)

#### CoMLC-MI Architecture Exploration (v0, confirmed ineffective)
- **Feature Encoder**: MLP (111→256→128→64), BatchNorm + ReLU + Dropout
- **Label Co-occurrence GCN**: 2-layer GCNConv propagating on a 12-node label graph
- **Prediction Head**: dot product between patient embedding **h** and label embedding **z̃**: h·z̃
- **Clinical Asymmetric Loss (CAL)**: γ⁺=0, γ⁻=3, positive sample weight 1/prevalence(k)
- **Ablation Dimensions**: graph structure (none/sparse/symmetric) × architecture (dot/concat) × loss (W_BCE/CAL_g2/CAL_g3) = 18 configurations
- **Conclusion**: GCN degrades performance in all 18 configurations; CAL underperforms weighted BCE in all configurations

#### Temporal Analysis (v0)
- 4 prediction windows: admission (t0, 91 features), 24h (t1, 98 features), 48h (t2, 104 features), 72h (t3, 111 features)
- Finding: 24h is the optimal prediction window; ICU Day 1 data provides the largest marginal gain

#### Improved Methods (v2–v3)

**LP-RF (Label Powerset + Random Forest)**
```
Input features → RF (200 trees, max_depth=10)
               → Multi-label output (12-dim probability vector)
Enhancement: true MLSMOTE (multi-label interpolation, joint feature+label space)
```

**ECC-LGBM (Ensemble Classifier Chains)**
```
50 parallel chains, each:
  Random label order → Chained ClassifierChain
  → LightGBM (200 trees, max_depth=6, lr=0.03)
  → Previous label predictions as additional features for next label
Output: average across 50 chains
```

**TabPFN v2-BR (Tabular Foundation Model + Binary Relevance)**
```
Per-label independent TabPFN:
  top-80 features (aggregated RF importance across 12 labels)
  → TabPFNClassifier (zero hyperparameter tuning, full 1360 training samples)
  → 12 independent predictions → concatenated (N, 12) probability matrix
```

**CatBoost-BR (Declared Categorical Features + Ordered Boosting)**
```
Per-label independent CatBoost:
  80 binary (0/1) features declared as cat_features
  → class_weights=[1.0, n_neg/n_pos]
  → ordered boosting + Iter-type early stopping (od_wait=50)
```

**2-Model Ensemble (Final SOTA)**
```
LP-RF predicted probabilities (excels at extremely rare arrhythmias: SVT, VT, AVB)
  +
TabPFN predicted probabilities (excels at mid-prevalence complications: AF, PulEd, Leth)
  →
Equal-weight average (1/2, 1/2)
  → Macro-AUC = 0.7604, Micro-AUC = 0.8140
```

### 2.5 Statistical Inference Framework

```
Per-label DeLong Test (labels with n⁺ ≥ 15)
  ├── H₀: AUC(model_A) = AUC(model_B)
  ├── DeLong covariance matrix (patient-paired structure)
  └── Exact p-values (not truncated to <0.0001)

Per-label Permutation Test (labels with n⁺ < 15)
  ├── 10,000 random swaps of predictions (paired permutation)
  ├── No asymptotic assumptions, valid for any sample size
  └── p = (extreme permutations + 1) / (total permutations + 1)

Benjamini-Hochberg FDR Correction
  ├── 12-label p-values combined → p_fdr = p_raw × 12 / rank
  └── α = 0.05

Bootstrap Robustness
  ├── 500 resamples with replacement from test set
  └── Report mean + 95% percentile CI
```

### 2.6 Bug Fix History (5 Critical Fixes)

| Bug | Severity | Impact | Improvement After Fix |
|-----|----------|--------|----------------------|
| Scaler data leakage (3 independent StandardScalers) | 🔴 | Val/test sets normalized with own statistics | LP-RF +3.07 pts |
| TabPFN training samples truncated to 1000 | 🔴 | Lost 26% of training data | TabPFN +3.27 pts |
| Fake MLSMOTE (single-label imblearn.SMOTE) | 🔴 | Multi-label oversampling degraded to single-label | SVT AUC +31.32 pts |
| ECC serial training + weak hyperparameters | 🟡 | 50 chains serial, insufficient n_estimators | 20× speedup |
| Ensemble weights using hardcoded placeholders | 🔴 | Paper's core SOTA numbers based on fake weights | All re-measured |

### 2.7 Key Findings Summary

**Methods That Work:**
1. 2-model equal-weight ensemble (LP-RF + TabPFN): Macro-AUC=0.7604, surpassing best single model
2. True MLSMOTE targeting rare arrhythmia labels
3. CatBoost + categorical feature declaration: highest Macro-F1
4. 5-fold CV robust threshold tuning (rare label t* can go as low as 0.001)
5. Bootstrap robustness testing mitigates single-dataset concerns

**Methods Confirmed Ineffective:**
1. Label co-occurrence GCN (all 18 ablation configs underperform baseline)
2. Patient similarity GAT (9 graph parameter sweeps, best val=0.7223)
3. Clinical Asymmetric Loss CAL (γ⁻ focusing + 1/w inverse weighting double penalty)
4. AUC-weighted ensemble vs equal-weight (difference only 0.0001; Softmax insufficiently discriminative)

**Falsified Conclusions (Corrected by Permutation Tests):**
1. "TabPFN significantly outperforms ensemble on SVT" → DeLong false positive (n⁺=4), permutation p=0.774
2. "TabPFN significantly outperforms ensemble on AVB" → DeLong false positive (n⁺=9), permutation p=0.977

**Robust Clinical Findings:**
1. CHF: 4/4 expected SHAP features verified (AGE, ZSN_A, FK_STENOK, DLIT_AG)
2. ICU Day 1 data is the optimal temporal prediction window
3. Arrhythmias are predictable at admission; hemodynamic complications improve with ICU data accumulation

---

## Part 3: Reproduction Guide

### 3.1 Environment Requirements
- Python 3.12, NumPy 1.26, scikit-learn 1.7, LightGBM 4.6, CatBoost, TabPFN, PyTorch Geometric, SHAP

### 3.2 Minimal Reproduction Steps

```bash
# 1. Data preprocessing
python src/preprocessing.py
# Output: output/processed_data/{X,y}_{train,val,test}.csv

# 2. Train all models
python src/publication_fixes.py
# Output: output/improvements_v2/PUB_*_preds.npy

# 3. Statistical tests
python src/final_statistical_tests.py
# Output: output/improvements_v2/FINAL_STATISTICAL_RESULTS.json
```

### 3.3 Final Citable Numbers

| Model | Micro-AUC | Macro-AUC | Macro-AUPRC | Weight Source |
|-------|-----------|-----------|-------------|---------------|
| Ensemble 2-model (equal) | **0.8140** | **0.7604** | 0.2642 | Equal weights |
| TabPFN v2-BR | 0.8099 | 0.7587 | 0.2564 | Best single model |
| LP-RF | 0.8050 | 0.7570 | 0.2516 | 2nd best single model by Macro-AUC |
| ECC-LGBM | 0.7884 | 0.6960 | 0.2240 | — |
| Ensemble 3-model | 0.8123 | 0.7514 | 0.2559 | 3-model ensemble (deprecated) |

> All numbers based on 340-patient holdout test set. Ensemble weights derived from 170-patient validation set measured AUCs. Validated via 500-iteration bootstrap robustness testing. Per-label significance assessed via permutation test (n⁺<15) or DeLong test (n⁺≥15) + BH-FDR correction.

---

*Document completed 2026-06-21, covering all 126 experiments.*
