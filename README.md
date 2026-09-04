# CoMLC-MI

Code, configurations, aggregate results, and reproducibility materials for the
CoMLC-MI benchmark of multi-label myocardial-infarction complication
prediction.

## Study scope

The internal benchmark evaluates ten model families for 12 source-coded
hospitalization outcomes in 1,700 patients from the Krasnoyarsk Myocardial
Infarction Complications Database. The primary `admission_safe_v1` contract has
89 variables identified by the source dictionary as admission-stage information
and two missingness indicators generated inside each training fold (91 inputs).
This is not claimed to be a strictly pretreatment feature set because exact
treatment timestamps are unavailable for seven retained fibrinolysis fields.

The Hungarian registry analysis is a separate cross-cohort mortality
transportability stress test. It uses only four harmonized admission variables
and therefore does not validate the other 11 outcomes, the 91-variable internal
benchmark, or label-dependency modeling. None of the released models is intended
for clinical deployment.

## Reproduce the benchmark

Install the recorded dependencies, then run:

```powershell
python -m src.run_benchmark --config configs/comlc_mi_benchmark.json
```

The configuration fixes the 5-repeat × 5-fold outer validation, four inner
folds, preprocessing, model hyperparameters, 2,000 paired bootstrap resamples,
label-wise tests, ten-seed graph ablation, temporal sensitivity analysis, and
controlled synthetic experiment. Checkpoints allow interrupted runs to resume;
optional stage and shard arguments change scheduling only, not the analysis
contract.

TabPFN results were produced with `tabpfn==8.0.1` and the TabPFN-3 classifier
checkpoint `tabpfn-v3-classifier-v3_default.ckpt` (SHA-256
`D0D865D54DFBC524F5703104BE90620182DCA7E5FB2C16DE72E9959EA18F3988`).
The licensed checkpoint is resolved through TabPFN's supported cache and is not
redistributed here.

## Methods implemented

- Fold-local imputation, scaling, feature selection, and training decisions.
- BR-LR, BR-XGBoost, BR-LightGBM, BR-CatBoost, ECC-LightGBM, LP-RF,
  RAkELd-RF, shared MLP, multitask DNN, and TabPFN-3.
- Equal and inner-OOF-weighted LP-RF–TabPFN-3 ensembles.
- AUROC, AUPRC, Brier score, equal-frequency 10-bin ECE, and fixed-threshold
  Macro-F1, with patient-level paired uncertainty analyses.
- A paired-seed 18-configuration label-graph experiment and a controlled
  synthetic dependency experiment using `BR-kNN (distance-weighted, k=10)`.

## Repository layout

- `configs/comlc_mi_benchmark.json`: canonical analysis specification.
- `src/run_benchmark.py`: canonical command-line entry point.
- `src/`: modeling, statistical analysis, figure generation, and audit code.
- `output/benchmark/`: machine-readable benchmark results and metadata.
- `figures/manuscript/`: publication figures generated from the released aggregate results.
- `tests/`: leakage, feature-contract, statistical, model, and packaging tests.

The machine-readable results include five outer-fold OOF predictions per
patient and their mean, fold indices, aggregate and per-label metrics,
calibration summaries, graph and synthetic experiment outputs, environment
metadata, and SHA-256 manifests. Reported bootstrap intervals condition on the
saved mean OOF predictions and do not propagate full model-retraining
uncertainty.

Run the automated checks with:

```powershell
python -m pytest -q
```

## Data and privacy

The internal source data are available from the UCI Machine Learning Repository.
The Hungarian registry extract is available from Mendeley Data at
<https://doi.org/10.17632/2v7n2r3xch.1>. Obtain it directly from the source and
set `HUNGARIAN_MI_XLSX` to the local workbook path when running the external
analysis.

This repository does not distribute the Hungarian workbook, patient-level
Hungarian predictions, external bootstrap arrays, or licensed TabPFN-3 weights.
Only aggregate external results and calibration-bin summaries are public.

## Ethics note

The study uses public, de-identified data and cites the governance statements of
the source cohorts. The manuscript's institution-specific secondary-analysis
ethics wording must be confirmed by the corresponding author before submission.
