# CoMLC-MI — IEEE Access R1 reproducibility package

This repository contains the technical reconstruction for manuscript
`Access-2026-35668`. The formal internal benchmark predicts 12 coded outcomes
from 1,700 patients in the historical Krasnoyarsk myocardial-infarction cohort.

## Scope and interpretation

- The primary internal feature contract is `admission_safe_v1`: 89 measured
  admission-time predictors plus two fold-generated missingness indicators
  (91 variables total).
- Nine interval-count variables, 11 variables with ambiguous inpatient timing,
  and eight invalid interactions formerly constructed after standardization are
  excluded from the primary analysis.
- The prospective sensitivity horizons contain 91, 94, 97, and 100 variables.
  The 111-variable source-dictionary set is retrospective sensitivity analysis
  only.
- The contemporary Hungarian registry is used only for a cross-cohort mortality
  transportability stress test with four shared admission variables. It does not
  validate the other 11 outcomes, label-dependency modeling, or the complete
  internal benchmark.
- The models are research benchmarks and are not suitable for clinical use.
- The label dependency score is a prevalence-dependent descriptive statistic,
  not a universal architecture-selection threshold.

## Formal reproduction command

```powershell
python -m src.run_revision --config configs/access_2026_35668_r1.json
```

The command executes the locked analysis:

1. five repeats of five outer multilabel-stratified folds;
2. four inner folds for ensemble weights and training-round decisions;
3. fold-local imputation, scaling, and TabPFN top-80 feature selection;
4. ten model families plus equal and inner-OOF-weighted LP-RF–TabPFN ensembles;
5. 2,000 paired patient bootstrap resamples, per-label DeLong/permutation tests,
   BH correction, calibration intervals, and reliability data;
6. the 18-configuration GCN ablation across seeds 42–51;
7. the four-horizon neural sensitivity analysis and controlled synthetic study;
8. the locked external mortality transportability stress test.

The external patient-level workbook is deliberately not distributed. On a
local machine, set `HUNGARIAN_MI_XLSX` to the downloaded workbook path; if the
variable is absent, the runner checks the standard Downloads folder. No
Hungarian patient-level data enter the repository or supplementary ZIP.

## Recovery and audit stages

The formal command is checkpointed by outer fold. For inspection or recovery,
the same entry point accepts `--stage` with `splits`, `internal`, `summarize`,
`gcn`, `temporal`, `synthetic`, or `external`. These switches do not change the
locked configuration; they only select which stage is executed.

Long internal, GCN, and synthetic stages also support disjoint recovery shards
(`--fold-shard-index/--fold-shards`, `--gcn-shard-index/--gcn-shards`,
`--temporal-shard-index/--temporal-shards`, and
`--synthetic-shard-index/--synthetic-shards`). Shards write only fold-, seed-,
horizon-, or repetition-level checkpoints. A final unsharded stage call is required to
assemble and statistically summarize the locked outputs.

The temporal stage may additionally be distributed over the 25 repeated outer
folds with `--temporal-fold-shard-index/--temporal-fold-shards`. Horizon and
fold sharding may be combined; they change scheduling only, not folds, seeds,
models, or estimates.

## Main files

- `configs/access_2026_35668_r1.json` — formal feature, validation, model,
  statistical, ablation, synthetic, and release specification.
- `src/run_revision.py` — only formal orchestration entry point.
- `src/revision_features.py` — feature contracts and fold-local preprocessing.
- `src/revision_models.py` — uniform model adapters and inner-OOF weighting.
- `src/revision_metrics.py` — paired bootstrap, calibration, DeLong,
  permutation, and BH analysis.
- `src/revision_gcn.py` — paired 18-configuration, 10-seed GCN ablation.
- `src/revision_temporal.py` — 91/94/97/100-variable horizon analysis.
- `src/generate_r1_figures.py` — vector-first manuscript figures.
- `tests/` — leakage, feature-count, loss, split, seed, statistics, GCN,
  synthetic, and external-count tests.

## Output contract

Formal outputs are written to `output/revision_r1/` and include:

- five OOF predictions per patient and their average;
- outer-fold assignments and per-fold decision metadata;
- model and per-label metrics;
- paired differences, bootstrap arrays, calibration bins, and intervals;
- full 10-seed GCN and synthetic results;
- external aggregate results and predictions required for audit;
- runtime package versions and a SHA-256 manifest.

Run the test gate with:

```powershell
python -m pytest -q
python -m src.revision_r1_qa --level analysis
```

After the clean manuscript, highlighted PDF, response letter, and supplement
have been compiled and visually inspected, build and verify the upload bundle:

```powershell
python -m src.package_r1_submission
python -m src.revision_r1_qa --level submission
```

The package builder uses an explicit whitelist. It excludes the Hungarian raw
workbook, every patient-level prediction archive, and all bootstrap draw arrays
from the supplementary ZIP.

The fixed public release tag is `access-2026-35668-r1`. Manuscript claims,
tables, figures, supplementary files, and the response letter must be generated
from the fixed-tag outputs; older working directories are not authoritative.

## Ethics and data boundary

This work is a secondary analysis of public, de-identified datasets. Under the
authors' institutional policy, no new ethics approval or informed consent is
required for this secondary analysis. The revised manuscript also cites the
ethics information reported for each original cohort. This wording requires
final confirmation by the corresponding author before resubmission.
