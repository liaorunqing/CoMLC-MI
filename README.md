# A Dependency-Aware Multi-Label Learning Framework for Myocardial Infarction Complication Prediction

> **Benchmarking, Statistical Validation, and Clinical Interpretation**
>
> Runqing Liao, Xiaopeng Yang, Zhining Wang — Hanshan Normal University

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![scikit-learn 1.9](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-liaorunqing%2FCoMLC--MI-lightgrey.svg?logo=github)](https://github.com/liaorunqing/CoMLC-MI.git)

---

> **Repository:** [https://github.com/liaorunqing/CoMLC-MI](https://github.com/liaorunqing/CoMLC-MI.git)

## Overview

When a patient survives the acute phase of myocardial infarction, the subsequent days often bring not one complication but several simultaneously — atrial fibrillation layered on top of pulmonary edema, or recurrent infarction progressing toward chronic heart failure. Predicting this outcome space **jointly**, rather than as a collection of independent binary decisions, is the natural formulation.

This repository provides the first systematic multi-label benchmark on the **Krasnoyarsk Myocardial Infarction Complications Database** (1,700 patients, 1992–1995). Eight methods spanning problem transformation, ensemble learning, deep networks, and tabular foundation models are evaluated under a unified protocol with **Macro-AUC** as the primary metric. The framework is **dependency-aware**: it computes label dependency metrics before model selection and uses them to narrow the space of viable architectures.

### Key Result

**TabPFN v2**, deployed without any task-specific hyperparameter tuning, achieves a Macro-AUC of **0.7587** — 97.9% of a retrospective oracle upper bound (0.7750). A simple 2-model ensemble (LP-RF + TabPFN v2) edges this to **0.7606**, but the gain is not statistically distinguishable from zero at the per-label level. Graph convolutional networks, despite their appeal for capturing label co-occurrence, provide **no measurable benefit** — the dataset's Label Dependency Score of ~0.13 sits well below the ~0.75 threshold at which GCNs become competitive.

---

## Framework Design

The framework has four interlocking stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Label Dependency Characterization (computed in seconds) │
│    LC (1.271) · LD (0.106) · LDS (~0.13) · Co-occurrence Matrix  │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Multi-Label Model Evaluation (8 methods + ensembles)    │
│    BR · CC · LP · RAkEL · MLP · Multitask DNN · GCN · TabPFN     │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Statistical Verification                                │
│    Per-label DeLong/Permutation · BH-FDR · Bootstrap 500×         │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: Clinical Interpretation                                  │
│    SHAP validation · Temporal horizons (4) · Framework ablation   │
└─────────────────────────────────────────────────────────────────┘
```

The dependency metrics serve a diagnostic function: when LDS is low (below ~0.15–0.20), graph-based methods are unlikely to outperform independent classifiers, and research effort is better directed toward feature engineering and stronger base learners.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Krasnoyarsk Interdistrict Clinical Hospital No. 20, Russia |
| Time window | 1992–1995 |
| Patients | 1,700 |
| Features | 119 (after exclusion, imputation, and clinical engineering) |
| Labels | 12 binary complications |
| Overall missing rate | 8.47% |
| Complete cases | 0 |
| Label Cardinality (LC) | 1.271 |
| Label Density (LD) | 0.106 |
| Label Dependency Score (LDS) | ~0.13 |

### Feature Groups (119 at 72h horizon)

| Group | Count | Examples |
|-------|-------|----------|
| Demographics | 2 | Age, Sex |
| Medical History & Comorbidities | 18 | Prior MI count, Angina grade, Hypertension, Diabetes, etc. |
| Admission Physiology | 10 | EMS/admission BP, Pulmonary edema, Cardiogenic shock |
| ECG Findings | 30 | MI localization, Rhythm, ST/T-wave, Q-wave abnormalities |
| Fibrinolytic Therapy | 7 | Streptokinase, tPA, timing and dosage |
| Blood Labs | 9 | K⁺, Na⁺, ALT, AST, Leukocytes, ESR |
| Onset-to-Admission Time | 1 | TIME_B_S |
| Clinical Interaction Features | 8 | Hemodynamic severity, ECG extent, Arrhythmia burden, etc. |
| ICU Day 1 Interventions | 7 | Pain recurrence, Analgesics, Lidocaine, IV nitrates |
| ICU Day 2 Medications | 6 | Narcotic/non-narcotic analgesics |
| ICU Day 3 Oral Medications | 7 | β-blockers, Ca-antagonists, Heparin, Aspirin, etc. |
| Missingness Indicators | 2 | EMS BP missing flags |

### Complication Labels

| Code | Complication | Prevalence | Category |
|------|-------------|------------|----------|
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
| LET_IS | Lethal Outcome | 15.9% | Endpoint |

---

## Project Structure

```
├── README.md
├── SYSTEM_ARCHITECTURE.md            # Detailed system documentation
├── requirements.txt
├── dataset/                          # Raw CSV + documentation PDFs
├── processed_data/                   # Preprocessed train/val/test splits
├── output/
│   ├── improvements_v2/              # Final predictions, ALL_RESULTS.json, timing
│   ├── synthetic/                    # Synthetic dependency experiment results
│   ├── temporal/                     # Temporal horizon analysis outputs
│   └── ablation/                     # Framework ablation outputs
├── paper/
│   ├── gai_rewrite.tex               # IEEE Access manuscript
│   └── figures/                      # Paper figures
└── src/
    ├── config.py                     # Centralized paths, labels, hyperparameters
    ├── evaluation.py                 # Unified metrics, bootstrap CI, threshold tuning
    ├── preprocessing.py              # MICE imputation, train/val/test split, scaler
    ├── run_all.py                    # Deterministic pipeline: trains all models
    ├── baselines.py                  # 8 baseline models (v0)
    ├── comlc_mi_opt.py               # 18-config GCN + CAL ablation
    ├── temporal_experiments.py       # 4-horizon temporal analysis
    ├── synthetic_experiments.py      # Controlled LDS/density/sample-size experiments
    ├── final_statistical_tests.py    # DeLong + Permutation + BH-FDR
    ├── framework_ablation.py         # Component ablation table generator
    └── timing_analysis.py            # Model runtime benchmarks
```

---

## Installation

```bash
git clone https://github.com/liaorunqing/CoMLC-MI.git
cd CoMLC-MI
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.26, <2.0 | Numerical computing |
| pandas | ≥2.0 | Data manipulation |
| scipy | ≥1.11 | Statistical tests |
| scikit-learn | 1.9.* | Classical ML, MICE (IterativeImputer) |
| lightgbm | 4.6.* | ECC-LGBM chains |
| catboost | ≥1.2 | Ordered boosting with categorical features |
| tabpfn | ≥0.2 | Tabular foundation model |
| torch | ≥2.4, <3.0 | Deep learning (MLP, GCN) |
| shap | ≥0.44 | SHAP interpretability |
| matplotlib, seaborn | ≥3.8 / ≥0.13 | Visualization |
| joblib | ≥1.3 | Parallel ECC chain training |

---

## Quick Start

Reproduce all paper results in 3 steps:

```bash
# Step 1: Data preprocessing
python src/preprocessing.py
# → processed_data/{X,y}_{train,val,test}.csv + scaler + imputer artifacts

# Step 2: Train all models (deterministic, seed=42)
python src/run_all.py
# → output/improvements_v2/PUB_*_preds.npy + ALL_RESULTS.json

# Step 3: Statistical tests
python src/final_statistical_tests.py
# → output/improvements_v2/FINAL_STATISTICAL_RESULTS.json
```

All results are 100% reproducible with `RANDOM_SEED = 42`.

---

## Evaluated Methods

### Problem Transformation Family

| Method | Base Learner | Label Dependency | Description |
|--------|-------------|-----------------|-------------|
| **Binary Relevance (BR)** | XGBoost, LightGBM, CatBoost | None | L independent classifiers |
| **Classifier Chains (ECC)** | LightGBM × 50 chains | Directed, ensemble-averaged | Chain order randomized; predictions propagate forward |
| **Label Powerset (LP-RF)** | Random Forest (200 trees, max_depth=10) | Full joint | Each label combination = one class |
| **RAkEL** | RF on k=4 random label subsets | Partial (subset-level) | Ensemble of LP over random label subsets |

### Deep Multi-Label Networks

| Method | Architecture | Training |
|--------|-------------|----------|
| **Shared-Encoder MLP** | 256→128→64, BatchNorm + ReLU + Dropout(0.3), sigmoid heads | Weighted BCE, AdamW, cosine annealing |
| **Multitask DNN** | Shared encoder 256→128 + per-task branches (32→1) | Same as above |
| **GCN (ablation only)** | MLP encoder + 2-layer GCNConv on label graph | 18 configs (graph × architecture × loss) |

### Tabular Foundation Model

| Method | Paradigm | Tuning |
|--------|----------|--------|
| **TabPFN v2-BR** | Per-label inference, top-80 RF-selected features | Zero hyperparameter tuning; single forward pass |

### Ensembles

| Ensemble | Constituents | Weighting |
|----------|-------------|-----------|
| **2-model (AUC-weighted)** | LP-RF + TabPFN v2 | Per-label softmax of validation AUC |
| **2-model (equal)** | LP-RF + TabPFN v2 | 1/2, 1/2 |
| **Stacking** | LP-RF + TabPFN v2 meta-features | Logistic regression meta-learner |

---

## Results

### Test-Set Performance (340 patients, seed=42)

| Method | Micro-AUC | Macro-AUC | Macro-AUPRC | Macro-F1 |
|--------|-----------|-----------|-------------|----------|
| **Ensemble 2-model (AUC-wt)** | **0.8103** | **0.7606** | **0.2568** | 0.0668 |
| Ensemble 2-model (equal) | 0.8102 | 0.7605 | 0.2566 | 0.0650 |
| **TabPFN v2-BR** | 0.8099 | **0.7587** | 0.2564 | 0.0809 |
| LP-RF | 0.7966 | 0.7423 | 0.2347 | 0.0257 |
| RAkEL-RF | 0.7984 | 0.7343 | 0.2531 | 0.0425 |
| Stacking (LR meta) | 0.6973 | 0.7216 | 0.2547 | 0.2473 |
| CatBoost-BR (cat. features) | 0.7266 | 0.7092 | 0.2470 | 0.2399 |
| ECC-LGBM (50 chains) | 0.7897 | 0.7007 | 0.2236 | 0.0877 |

> Bootstrap 95% CIs for the ensemble and TabPFN are nearly identical — the 0.0019 gap is not robust to resampling. TabPFN v2 reaches **97.9%** of the oracle upper bound (0.7750).

### Per-Label Statistical Significance (Ensemble vs. TabPFN v2, BH-FDR α=0.05)

| Complication | n⁺ | ΔAUC | p (FDR) | Significant? |
|-------------|-----|------|---------|:---:|
| Post-infarction Angina | 32 | +0.0182 | 0.0026 | ✓ Ensemble |
| Atrial Fibrillation | 35 | +0.0117 | 0.0140 | ✓ Ensemble |
| Pulmonary Edema | 40 | +0.0078 | 0.0330 | ✓ Ensemble |
| Chronic Heart Failure | 88 | −0.0119 | 0.0012 | ✓ TabPFN |
| Ventricular Fibrillation | 15 | −0.0029 | 0.7037 | — |
| SV Tachycardia | 4 | +0.0357 | 1.0000 | — |
| 3rd-degree AV Block | 9 | −0.0205 | 0.7744 | — |
| Ventricular Tachycardia | 12 | −0.0036 | 0.9279 | — |
| Myocardial Rupture | 9 | +0.0087 | 0.9787 | — |
| Dressler Syndrome | 20 | −0.0148 | 0.3961 | — |
| Recurrent MI | 38 | −0.0084 | 0.3317 | — |
| Lethal Outcome | 54 | +0.0019 | 0.7047 | — |

> **Key insight**: the aggregate DeLong test produces z=8.27 (highly "significant"), but this is a **false positive** — per-label + FDR correction reveals meaningful differences on only 4/12 labels.

---

## Key Findings

### What Works

1. **TabPFN v2 is near-oracle** — achieves 97.9% of the retrospective oracle upper bound (0.7587 vs. 0.7750) with zero hyperparameter tuning
2. **LP-RF complements on rare labels** — joint modeling of label combinations helps on extremely rare arrhythmias
3. **Equal-weight ensemble suffices** — AUC-weighted and equal-weight ensembles differ by only 0.0001; Softmax over validation AUCs is not discriminative enough
4. **CatBoost + categorical feature declaration** yields the highest Macro-F1 (0.2399), though lower Macro-AUC — a calibration-vs-discrimination tradeoff
5. **ICU Day 1 data** provides the largest marginal gain in temporal analysis
6. **4/4 CHF clinical predictors** (Age, Prior CHF, MI location, Functional class) confirmed in SHAP top-15

### What Doesn't Work

1. **GCNs** — all 18 ablation configurations underperform the no-graph baseline (best GCN: 0.7064 vs. no-graph: 0.7127)
2. **Clinical Asymmetric Loss** — underperforms weighted BCE in all 18 comparisons
3. **MLSMOTE augmentation** — degrades LP-RF by −0.28 Macro-AUC on average across 50 seeds; removed from final pipeline
4. **Bayesian network on residuals** — only 2 edges survive threshold >0.08, confirming near-conditional-independence after shared features

### The LDS Threshold Hypothesis

Synthetic experiments at four controlled dependency levels establish that GCNs only become competitive around **LDS ≈ 0.75**, well above the Krasnoyarsk dataset's LDS ≈ 0.13. When LDS < 0.20, independent classifiers with strong base learners are expected to match or exceed graph-based methods — a diagnostic that transfers to any new clinical multi-label dataset.

---

## Preprocessing Pipeline

```
Raw CSV (1700 × 124)
  │
  ├─ Drop: CPK (99.8% missing), IBS_NASL (95.8% missing)
  ├─ Binarize LET_IS: ordinal 0–7 → binary
  ├─ Missingness indicators: S_AD_KBRIG_MISSING, D_AD_KBRIG_MISSING
  │
  ├─ Stratified split: 70% train / 10% val / 20% test (by LET_IS)
  │
  ├─ MICE imputation: 13 features (10–65% missing), RF estimator, 10 iterations
  │    Fit on training set only → transform val & test (no leakage)
  ├─ Median/mode imputation: remaining <10% missing features
  │
  ├─ Clinical feature engineering: 8 interaction terms
  │    hemodynamic_severity, ecg_extent_score, arrhythmia_burden,
  │    metabolic_stress, age_fc_interaction, antithrombotic_adequacy,
  │    time_delay_severe, age_time_risk
  │
  └─ StandardScaler: continuous features only (binary 0/1 unscaled)
       Single scaler, fit on train only
```

---

## Statistical Verification Layer

```
Per-label significance
├── DeLong test (n⁺ ≥ 15): paired AUC, exact p-values
├── Permutation test (n⁺ < 15): 10,000 paired swaps, no asymptotics
└── BH-FDR correction: α = 0.05 across 12 labels

Confidence intervals
├── 500 bootstrap resamples (stratified, seed=42)
└── Report 95% percentile CI

Clinical validation
├── SHAP GradientExplainer on 200 test patients
└── Top-15 feature comparison against cardiology guidelines
```

---

## Citation

```bibtex
@article{liao2026dependency,
  title={A Dependency-Aware Multi-Label Learning Framework for Myocardial
         Infarction Complication Prediction: Benchmarking, Statistical
         Validation, and Clinical Interpretation},
  author={Liao, Runqing and Yang, Xiaopeng and Wang, Zhining},
  journal={IEEE Access},
  year={2026},
  note={Manuscript under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Last updated: 2026-07-18. All 126 experiments, 4 temporal horizons, 18 GCN ablation configs, fully reproducible with seed=42.*
