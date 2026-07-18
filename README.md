# CoMLC-MI: Multi-Label Myocardial Infarction Complication Prediction Benchmark

> **A comprehensive benchmark and ensemble framework for predicting 12 post-MI complications from 111 heterogeneous clinical features.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

CoMLC-MI is a systematic research framework for multi-label classification of myocardial infarction complications. It establishes a rigorous benchmark on the Krasnoyarsk MI dataset (1,700 patients, 1992–1995), evaluating 8+ model architectures across 126 experiments.

**Key Result:** A simple 2-model equal-weight ensemble (LP-RF + TabPFN) achieves state-of-the-art performance with **Macro-AUC = 0.7604** and **Micro-AUC = 0.8140**, validated via 500-iteration bootstrap and per-label statistical testing with BH-FDR correction.

### Why This Matters

- **12 diverse complications** spanning arrhythmias (AF, SVT, VT, VF, AVB), hemodynamic (Pulmonary Edema, CHF), structural (Myocardial Rupture), immunologic (Dressler Syndrome), recurrent (Re-MI, Post-infarction Angina), and terminal (Lethal Outcome) outcomes
- **Extreme class imbalance** — prevalence ranges from 1.2% (SVT) to 23.2% (CHF)
- **8.47% overall missing rate** requiring robust imputation (MICE for 13 features with 10–65% missingness)
- **Clinically grounded feature engineering** including hemodynamic severity, ECG extent, arrhythmia burden, and metabolic stress scores

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Krasnoyarsk Interdistrict Clinical Hospital No. 20, Russia |
| Time Window | 1992–1995 |
| Patients | 1,700 |
| Input Features | 111 (after excluding 2 features with >95% missing) |
| Output Labels | 12 binary complications |
| Overall Missing Rate | 8.47% |
| Complete Cases | 0 (all patients have at least one missing feature) |
| Avg. Label Cardinality | 1.27 complications per patient |

### Feature Groups

| Group | Count | Examples |
|-------|-------|----------|
| Demographics | 2 | Age, Sex |
| History & Comorbidities | ~18 | Prior MI, Angina grade, Hypertension, Diabetes, Obesity |
| Admission Physiology | ~10 | EMS/admission BP, Pulmonary Edema, Cardiogenic Shock |
| ECG Findings | ~30 | MI localization, Rhythm features, ST/T-wave changes, Pathological Q-waves |
| Fibrinolytic Therapy | ~7 | Streptokinase, tPA, timing and dosage |
| Blood Labs | 9 | Potassium, Sodium, ALT, AST, Leukocytes, ESR |
| Temporal (ICU Day 1–3) | ~20 | Pain recurrence, Analgesics, Lidocaine, Beta-blockers, Antiplatelets |

### Output Labels (12 Complications)

| Code | Name | Prevalence | Category |
|------|------|------------|----------|
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

---

## Project Structure

```
CoMLC-MI_Final_Submission_new1/
├── README.md                          # This file
├── SYSTEM_ARCHITECTURE.md             # Detailed system architecture (Chinese)
├── requirements.txt                   # Python dependencies
├── dataset/
│   ├── Myocardial infarction complications Database.csv
│   └── *.pdf                          # Dataset documentation
├── processed_data/                    # Preprocessed train/val/test splits
├── output/
│   ├── improvements_v2/               # Final model predictions & results
│   ├── synthetic/                     # Synthetic experiment results
│   ├── temporal/                      # Temporal horizon analysis
│   └── ablation/                      # Framework ablation results
├── paper/
│   ├── gai_rewrite.tex                # Paper source (LaTeX)
│   ├── gai_rewrite.pdf                # Compiled paper
│   └── figures/                       # Paper figures
└── src/
    ├── config.py                       # Centralized configuration
    ├── evaluation.py                   # Unified evaluation framework
    ├── preprocessing.py                # Data preprocessing pipeline
    ├── run_all.py                      # Unified deterministic pipeline
    ├── baselines.py                    # 8 baseline models
    ├── comlc_mi_opt.py                 # GCN + CAL ablation study (18 configs)
    ├── temporal_experiments.py         # 4 temporal prediction windows
    ├── synthetic_experiments.py        # Synthetic label dependency experiments
    ├── final_statistical_tests.py      # DeLong + Permutation + FDR
    ├── framework_ablation.py           # Framework component ablation
    └── timing_analysis.py              # Model runtime benchmarks
```

---

## Installation

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd CoMLC-MI_Final_Submission_new1

# Install dependencies
pip install -r requirements.txt
```

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.26, <2.0 | Numerical computing |
| pandas | ≥2.0 | Data manipulation |
| scipy | ≥1.11 | Statistical tests |
| scikit-learn | 1.9.* | Classical ML, MICE imputation |
| xgboost | ≥2.0 | Gradient boosting baseline |
| lightgbm | 4.6.* | ECC-LGBM chains |
| catboost | ≥1.2 | Ordered boosting with categorical features |
| tabpfn | ≥0.2 | Tabular foundation model (zero-shot) |
| torch | ≥2.4, <3.0 | Deep learning (MLP, GCN, GAT) |
| shap | ≥0.44 | Feature importance & interpretability |
| matplotlib | ≥3.8 | Visualization |
| seaborn | ≥0.13 | Statistical plots |

---

## Quick Start

Reproduce the paper's main results in 3 steps:

```bash
# Step 1: Preprocess the data
python src/preprocessing.py
# Output: processed_data/{X,y}_{train,val,test}.csv + scaler + imputer artifacts

# Step 2: Train all models and generate predictions
python src/run_all.py
# Output: output/improvements_v2/PUB_*_preds.npy + ALL_RESULTS.json

# Step 3: Run statistical tests
python src/final_statistical_tests.py
# Output: output/improvements_v2/FINAL_STATISTICAL_RESULTS.json
```

---

## Models

### Baselines (8 models)

| Model | Type | Description |
|-------|------|-------------|
| XGBoost-BR | Tree ensemble | Per-label XGBoost with scale_pos_weight |
| LightGBM-BR | Tree ensemble | Per-label LightGBM |
| Binary Relevance (LR) | Problem transformation | One LR classifier per label |
| Classifier Chains (Ensemble) | Problem transformation | 3-chain ensemble with LR base |
| Label Powerset (RF) | Problem transformation | RF on label combinations |
| RAkEL (k=4) | Ensemble of problem transformation | Random k-labelsets with LR |
| Shared-Encoder MLP | Deep MLC | 256→128→64 shared encoder + sigmoid heads |
| Multitask DNN | Deep MLC | Makhmudov-style per-task branches |

### Advanced Models (v2–v3)

| Model | Macro-AUC | Key Innovation |
|-------|-----------|----------------|
| **LP-RF** | 0.7570 | Label Powerset + RF (200 trees, max_depth=10) |
| **ECC-LGBM** | 0.6960 | 50 parallel Classifier Chains with LightGBM |
| **TabPFN v2-BR** | 0.7587 | Zero-shot tabular foundation model, top-80 RF-selected features |
| **CatBoost-BR** | — | Ordered boosting with declared categorical features |
| **Ensemble 2-model (equal)** | **0.7604** | LP-RF + TabPFN equal-weight average |

### Confirmed Ineffective Approaches

After 126 experiments, the following were **conclusively shown to not work**:

- **Label Co-occurrence GCN** — All 18 ablation configurations underperform the no-graph baseline
- **Patient Similarity GAT** — 9 graph parameter sweeps, best val = 0.7223 (worse than LP-RF)
- **Clinical Asymmetric Loss (CAL)** — Double-penalty (γ⁻ focusing + 1/w inverse weighting) degrades performance
- **AUC-weighted Ensemble** — Differs from equal-weight by only 0.0001 (Softmax fails to discriminate)

---

## Reproducible Results

### Final Performance (340-patient test set)

| Model | Micro-AUC | Macro-AUC | Macro-AUPRC |
|-------|-----------|-----------|-------------|
| **Ensemble 2-model (equal)** | **0.8140** | **0.7604** | 0.2642 |
| TabPFN v2-BR | 0.8099 | 0.7587 | 0.2564 |
| LP-RF | 0.8050 | 0.7570 | 0.2516 |
| ECC-LGBM | 0.7884 | 0.6960 | 0.2240 |
| CatBoost-BR | — | — | — |

> All numbers are 100% reproducible with `RANDOM_SEED = 42`. Validated via 500-iteration bootstrap with 95% percentile CIs. Per-label significance assessed via DeLong test (n⁺ ≥ 15) or permutation test (10,000 swaps, n⁺ < 15) with Benjamini-Hochberg FDR correction (α = 0.05).

### Statistical Validation Pipeline

```
Per-label Testing
├── DeLong Test (n⁺ ≥ 15): Paired AUC comparison with exact p-values
├── Permutation Test (n⁺ < 15): 10,000 paired swaps, no asymptotic assumptions
└── BH-FDR Correction: 12-label p-value adjustment (α = 0.05)

Bootstrap Robustness
├── 500 resamples with replacement
├── Report mean + 95% percentile CI
└── Local RandomState for full reproducibility
```

---

## Key Findings

### What Works

1. **Simple ensemble beats complex models** — Equal-weight averaging of LP-RF + TabPFN outperforms all single models and graph-based approaches
2. **TabPFN excels at mid-prevalence complications** — Strongest on AF (10.0%), Pulmonary Edema (9.4%), and Lethal Outcome (15.9%)
3. **LP-RF excels at rare arrhythmias** — Best on SVT (1.2%), VT (2.5%), and AVB (3.4%)
4. **CatBoost with categorical feature declaration** achieves the highest Macro-F1
5. **ICU Day 1 data** provides the largest marginal gain for temporal prediction
6. **CHF clinical validation** — All 4 expected features (AGE, ZSN_A, FK_STENOK, DLIT_AG) confirmed in SHAP top-15

### Bug Fixes That Mattered

| Bug | Severity | Impact | Fix |
|-----|----------|--------|-----|
| Scaler data leakage (3 independent StandardScalers) | 🔴 Critical | Val/test normalized with own statistics | Single fitted scaler |
| TabPFN truncated to 1000 samples | 🔴 Critical | Lost 26% of training data | Use all 1,360 samples |
| Fake MLSMOTE (single-label SMOTE) | 🔴 Critical | Multi-label oversampling degraded to single-label | True multi-label SMOTE |
| ECC serial training | 🟡 Moderate | 50 chains trained sequentially | joblib.Parallel (20× speedup) |
| Hardcoded ensemble weights | 🔴 Critical | Paper's SOTA numbers based on placeholder weights | Val-set measured weights |

---

## Citation

If you use this work, please cite:

```bibtex
@article{comlc-mi-2026,
  title={CoMLC-MI: A Comprehensive Multi-Label Benchmark for Myocardial Infarction Complication Prediction},
  author={...},
  journal={...},
  year={2026},
  note={Manuscript in preparation}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Documentation last updated: 2026-07-18. Covers all 126 experiments across 4 research phases.*
