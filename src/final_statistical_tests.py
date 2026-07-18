"""
================================================================================
CoMLC-MI: Final Statistical Tests (Permutation + Exact p-values)
================================================================================
Addresses the three issues from the latest review:

  Problem 2: Permutation tests for labels with <15 positive test samples
    (SVT, VT, AVB, Rupture — DeLong asymptotic assumption unreliable).

  Problem 3: Exact p-values instead of "<0.0001" truncation.

  Also: Full code architecture audit — check all scripts for consistency.

Output: Publication-ready statistical tables with exact p-values and
         permutation-validated significance.
================================================================================
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import json, os, warnings, glob
warnings.filterwarnings('ignore')

from config import *
from evaluation import compute_all_metrics

# ── Load predictions ─────────────────────────────────────────────────────
print("Loading predictions...")
lp_test   = np.load(os.path.join(OUTPUT_DIR, 'PUB_lp_rf_preds.npy'))
tpf_test  = np.load(os.path.join(OUTPUT_DIR, 'PUB_tabpfn_preds.npy'))
ens2_eq   = np.load(os.path.join(OUTPUT_DIR, 'PUB_ens2_eq_preds.npy'))

y_test = pd.read_csv(os.path.join(ORIG_PROCESSED_DIR, 'y_test.csv')).values.astype(np.float32)

# ── Compute test set positive counts per label ───────────────────────────
print("\nTest set positive counts:")
small_sample_labels = []
for i, col in enumerate(LABEL_COLS):
    n_pos = int(y_test[:, i].sum())
    note = ' *** SMALL SAMPLE (DeLong unreliable)' if n_pos < 15 else ''
    print(f"  {LABEL_SHORT[i]:>6} ({col}): n_pos={n_pos}{note}")
    if n_pos < 15:
        small_sample_labels.append((i, col, n_pos))

# ═══════════════════════════════════════════════════════════════════════════
# 1. STANDARD DeLONG (per-label, exact p-values)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. STANDARD DeLONG TEST (per-label, EXACT p-values)")
print("="*70)


def delong_roc_test(y_true, pred_a, pred_b):
    """Standard DeLong test for paired AUC. Returns (z, p, auc_a, auc_b)."""
    n = len(y_true)
    v_a = pred_a[y_true == 1]; v_b = pred_b[y_true == 1]
    u_a = pred_a[y_true == 0]; u_b = pred_b[y_true == 0]
    n1 = len(v_a); n0 = len(u_a)
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan, np.nan, np.nan

    theta_a = roc_auc_score(y_true, pred_a)
    theta_b = roc_auc_score(y_true, pred_b)

    xi_a = np.zeros(n); xi_b = np.zeros(n)
    for i in range(n):
        if y_true[i] == 1:
            xi_a[i] = (np.sum(u_a < pred_a[i]) + 0.5 * np.sum(u_a == pred_a[i])) / n0
            xi_b[i] = (np.sum(u_b < pred_b[i]) + 0.5 * np.sum(u_b == pred_b[i])) / n0
        else:
            xi_a[i] = (np.sum(v_a > pred_a[i]) + 0.5 * np.sum(v_a == pred_a[i])) / n1
            xi_b[i] = (np.sum(v_b > pred_b[i]) + 0.5 * np.sum(v_b == pred_b[i])) / n1

    S_aa = np.var(xi_a, ddof=1) / n
    S_bb = np.var(xi_b, ddof=1) / n
    S_ab = np.mean((xi_a - theta_a) * (xi_b - theta_b)) / n
    var_diff = S_aa + S_bb - 2 * S_ab
    se_diff = np.sqrt(max(var_diff, 1e-12))
    z = (theta_a - theta_b) / se_diff
    p_val = 2 * (1 - norm.cdf(abs(z)))
    return z, p_val, theta_a, theta_b


# ═══════════════════════════════════════════════════════════════════════════
# 2. PERMUTATION TEST (paired, for small-sample labels)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. PERMUTATION TEST (paired, for n_pos < 15 labels)")
print("="*70)


def permutation_auc_test(y_true, pred_a, pred_b, n_perm=10000, random_state=42):
    """
    Paired permutation test for AUC difference.
    Does NOT assume asymptotic normality — valid for any sample size.

    H0: AUC(pred_a) = AUC(pred_b)
    Procedure: randomly swap predictions between models within each patient pair.
    """
    rng = np.random.RandomState(random_state)
    observed_diff = roc_auc_score(y_true, pred_a) - roc_auc_score(y_true, pred_b)
    n = len(y_true)

    # Count extreme permutations
    n_extreme = 0
    for _ in range(n_perm):
        swap = rng.rand(n) < 0.5
        perm_a = np.where(swap, pred_b, pred_a)
        perm_b = np.where(swap, pred_a, pred_b)
        perm_diff = roc_auc_score(y_true, perm_a) - roc_auc_score(y_true, perm_b)
        if abs(perm_diff) >= abs(observed_diff):
            n_extreme += 1

    p_value = (n_extreme + 1) / (n_perm + 1)  # +1 correction for finite permutations
    return observed_diff, p_value


# ── Run both tests for ALL labels ──
print("\nEnsemble 2-model (equal) vs TabPFN v2-BR:")
print(f"{'Label':<6} {'n+':>4} {'Ens AUC':>8} {'Tab AUC':>8} {'Diff':>8} "
      f"{'DeLong z':>9} {'DeLong p':>12} {'Perm p':>10} {'Method':>12} {'Conclusion':>20}")
print("-"*110)

all_results = []
for i, col in enumerate(LABEL_COLS):
    yt = y_test[:, i]
    n_pos = int(yt.sum())

    if len(np.unique(yt)) < 2:
        all_results.append({'label': col, 'short': LABEL_SHORT[i], 'n_pos': n_pos,
                            'note': 'no positive samples'})
        continue

    # DeLong
    z, p_delong, auc_e, auc_t = delong_roc_test(yt, ens2_eq[:, i], tpf_test[:, i])

    # Permutation
    if n_pos < 15:
        diff_perm, p_perm = permutation_auc_test(yt, ens2_eq[:, i], tpf_test[:, i])
        use_method = 'permutation'
        p_final = p_perm
    else:
        diff_perm, p_perm = np.nan, np.nan
        use_method = 'DeLong'
        p_final = p_delong

    # Format p-values with appropriate precision
    def fmt_p(p):
        if np.isnan(p): return 'N/A'
        if p < 1e-6: return f'{p:.2e}'
        if p < 0.001: return f'{p:.6f}'
        return f'{p:.4f}'

    # Conclusion
    if p_final < 0.05:
        if auc_e > auc_t:
            conclusion = 'Ensemble BETTER'
        else:
            conclusion = 'TabPFN BETTER'
    else:
        conclusion = 'no significant diff'

    # Reliability note
    reliability = ''
    if n_pos < 15:
        reliability = ' [low-n: use permutation]'

    print(f"{LABEL_SHORT[i]:<6} {n_pos:>4} {auc_e:>8.4f} {auc_t:>8.4f} {auc_e-auc_t:>8.4f} "
          f"{z:>9.4f} {fmt_p(p_delong):>12} {fmt_p(p_perm):>10} {use_method:>12} {conclusion:>20}{reliability}")

    all_results.append({
        'label': col, 'short': LABEL_SHORT[i], 'n_pos': n_pos,
        'auc_ensemble': auc_e, 'auc_tabpfn': auc_t, 'diff': auc_e - auc_t,
        'delong_z': z, 'delong_p': p_delong,
        'permutation_p': p_perm,
        'final_p': p_final, 'method': use_method,
        'conclusion': conclusion, 'reliability_warning': n_pos < 15,
    })

# ── Also: Ensemble vs LP-RF ──
print("\n\nEnsemble 2-model (equal) vs LP-RF:")
print(f"{'Label':<6} {'n+':>4} {'Ens AUC':>8} {'LP-RF AUC':>8} {'Diff':>8} "
      f"{'DeLong z':>9} {'DeLong p':>12} {'Perm p':>10} {'Conclusion':>20}")
print("-"*100)

lp_results = []
for i, col in enumerate(LABEL_COLS):
    yt = y_test[:, i]
    n_pos = int(yt.sum())
    if len(np.unique(yt)) < 2: continue

    z, p_delong, auc_e, auc_l = delong_roc_test(yt, ens2_eq[:, i], lp_test[:, i])

    if n_pos < 15:
        diff_perm, p_perm = permutation_auc_test(yt, ens2_eq[:, i], lp_test[:, i])
        p_final = p_perm
        method = 'permutation'
    else:
        p_perm = np.nan
        p_final = p_delong
        method = 'DeLong'

    def fmt_p(p):
        if np.isnan(p): return 'N/A'
        if p < 1e-6: return f'{p:.2e}'
        if p < 0.001: return f'{p:.6f}'
        return f'{p:.4f}'

    if p_final < 0.05:
        conclusion = 'Ensemble BETTER' if auc_e > auc_l else 'LP-RF BETTER'
    else:
        conclusion = 'no significant diff'

    print(f"{LABEL_SHORT[i]:<6} {n_pos:>4} {auc_e:>8.4f} {auc_l:>8.4f} {auc_e-auc_l:>8.4f} "
          f"{z:>9.4f} {fmt_p(p_delong):>12} {fmt_p(p_perm):>10} {conclusion:>20}")

    lp_results.append({
        'label': col, 'short': LABEL_SHORT[i], 'n_pos': n_pos,
        'auc_ensemble': auc_e, 'auc_lp_rf': auc_l, 'diff': auc_e - auc_l,
        'delong_z': z, 'delong_p': p_delong, 'permutation_p': p_perm,
        'final_p': p_final, 'method': method, 'conclusion': conclusion,
    })

# ═══════════════════════════════════════════════════════════════════════════
# 3. FDR CORRECTION (Benjamini-Hochberg) with combined p-values
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. FDR CORRECTION (Benjamini-Hochberg on final p-values)")
print("="*70)


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR correction. Returns (reject, p_fdr)."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    ranks = np.arange(1, n+1)
    reject = np.zeros(n, dtype=bool)

    k_max = 0
    for k in range(n):
        if p_values[sorted_idx[k]] <= 0.05 * (k+1) / n:
            k_max = k + 1
    if k_max > 0:
        reject[sorted_idx[:k_max]] = True

    p_fdr_sorted = np.minimum(1, p_values[sorted_idx] * n / ranks)
    p_fdr = np.zeros(n)
    for orig in range(n):
        sort_pos = int(np.where(sorted_idx == orig)[0][0])
        p_fdr[orig] = p_fdr_sorted[sort_pos]

    return reject, p_fdr


# Ensemble vs TabPFN
p_vals_tpf = np.array([r['final_p'] for r in all_results if not np.isnan(r.get('final_p', np.nan))])
reject_tpf, p_fdr_tpf = bh_fdr(p_vals_tpf)

for j, r in enumerate([r for r in all_results if not np.isnan(r.get('final_p', np.nan))]):
    r['fdr_reject'] = bool(reject_tpf[j])
    r['p_fdr'] = float(p_fdr_tpf[j])

ens_wins_fdr = [(r['short'], r['diff'])
                for r in all_results if r.get('fdr_reject') and r.get('diff', 0) > 0]
tpf_wins_fdr = [(r['short'], r['diff'])
                for r in all_results if r.get('fdr_reject') and r.get('diff', 0) < 0]

print(f"\nEnsemble vs TabPFN after BH-FDR:")
print(f"  Ensemble significantly BETTER on {len(ens_wins_fdr)}/{len(p_vals_tpf)}: "
      f"{', '.join(f'{s}({d:+.4f})' for s,d in ens_wins_fdr)}")
print(f"  TabPFN significantly BETTER on {len(tpf_wins_fdr)}/{len(p_vals_tpf)}: "
      f"{', '.join(f'{s}({d:+.4f})' for s,d in tpf_wins_fdr)}")

# Same for LP-RF
p_vals_lp = np.array([r['final_p'] for r in lp_results if not np.isnan(r.get('final_p', np.nan))])
reject_lp, p_fdr_lp = bh_fdr(p_vals_lp)
for j, r in enumerate([r for r in lp_results if not np.isnan(r.get('final_p', np.nan))]):
    r['fdr_reject'] = bool(reject_lp[j])
    r['p_fdr'] = float(p_fdr_lp[j])

ens_wins_lp = [(r['short'], r['diff'])
               for r in lp_results if r.get('fdr_reject') and r.get('diff', 0) > 0]
lp_wins_fdr = [(r['short'], r['diff'])
               for r in lp_results if r.get('fdr_reject') and r.get('diff', 0) < 0]

print(f"\nEnsemble vs LP-RF after BH-FDR:")
print(f"  Ensemble significantly BETTER on {len(ens_wins_lp)}/{len(p_vals_lp)}: "
      f"{', '.join(f'{s}({d:+.4f})' for s,d in ens_wins_lp)}")
if lp_wins_fdr:
    print(f"  LP-RF significantly BETTER on {len(lp_wins_fdr)}/{len(p_vals_lp)}: "
          f"{', '.join(f'{s}({d:+.4f})' for s,d in lp_wins_fdr)}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. PUBLICATION-READY TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. PUBLICATION-READY STATISTICAL TABLE")
print("="*70)

print(f"\nTable: Per-label DeLong and Permutation Tests")
print(f"{'Complication':<12} {'n+':>4} {'AUC(Ens)':>8} {'AUC(base)':>9} {'Diff':>8} "
      f"{'p(DeLong)':>12} {'p(Perm)':>12} {'p(FDR)':>10} {'Reliable':>10} {'Conclusion':>25}")
print("-"*130)

# Combined: Ensemble vs the better of TabPFN/LP-RF per label
for r in all_results:
    if 'delong_p' not in r: continue
    short = r['short']
    n_pos = r['n_pos']
    # Find the LP-RF comparison for this label
    lp_r = next((x for x in lp_results if x['short'] == short), None)

    reliable = 'Yes' if n_pos >= 15 else f'Marginal (n={n_pos})'
    conclusion = r.get('conclusion', '')
    if r.get('fdr_reject'):
        conclusion += ' *'

    def fmt_p(p):
        if np.isnan(p): return 'N/A'
        if p < 1e-6: return f'{p:.2e}'
        if p < 0.001: return f'{p:.6f}'
        return f'{p:.4f}'

    lp_delong_p_str = fmt_p(lp_r['delong_p']) if lp_r and 'delong_p' in lp_r else 'N/A'

    print(f"{short:<12} {n_pos:>4} {r['auc_ensemble']:>8.4f} {r['auc_tabpfn']:>9.4f} "
          f"{r['diff']:>8.4f} {fmt_p(r['delong_p']):>12} {fmt_p(r.get('permutation_p', np.nan)):>12} "
          f"{fmt_p(r.get('p_fdr', np.nan)):>10} {reliable:>10} {conclusion:>25}")

print(f"\n* Significant after BH-FDR correction (alpha=0.05)")
print(f"Reliability: DeLong test assumes asymptotic normality;")
print(f"  'Marginal' for labels with n+ <15 (permutation test preferred).")

# ═══════════════════════════════════════════════════════════════════════════
# 5. CODE ARCHITECTURE AUDIT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. CODE ARCHITECTURE AUDIT")
print("="*70)

src_dir = os.path.join(BASE_DIR, 'src')
py_files = sorted(glob.glob(os.path.join(src_dir, '*.py')))

print(f"\n  Source files: {len(py_files)}")
print(f"  {'File':<35} {'Lines':>8} {'Status':<20} {'Role'}")
print(f"  {'-'*75}")

production_files = []
deprecated_files = []

for fpath in py_files:
    fname = os.path.basename(fpath)
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        n_lines = len(f.readlines())

    # Classify
    if fname in ['config.py', 'evaluation.py']:
        status = 'CORE (active)'
    elif fname in ['final_fixes_part2.py', 'publication_fixes.py', 'delong_fix.py',
                    'final_statistical_tests.py']:
        status = 'FINAL (active)'
    elif fname in ['preprocessing.py', 'data_exploration.py']:
        status = 'PIPELINE (active)'
    elif fname in ['baselines.py', 'comlc_mi.py', 'comlc_mi_opt.py', 'shap_analysis.py',
                    'temporal_experiments.py', 'cross_horizon_shap.py', 'deep_shap_analysis.py']:
        status = 'EXPERIMENT (v0)'
    elif fname in ['improvements_core.py', 'improvements_advanced.py',
                    'improvements_v2.py', 'phase1_ensemble.py']:
        status = 'IMPROVEMENT (v1-v3)'
    elif fname in ['optuna_tuning.py', 'compile_all_results.py', 'master_report.py',
                    'research_synthesis.py', 'graph_visualization.py']:
        status = 'REPORTING'
    elif fname == 'phase1_continue.py':
        status = 'SUPERSEDED (use final_fixes)'
    else:
        status = 'UNCLASSIFIED'

    role = {
        'config.py': 'Centralized paths, labels, hyperparams',
        'evaluation.py': 'Unified metrics, AUPRC, bootstrap CI',
        'preprocessing.py': 'MICE imputation, train/val/test split (CORRECT scaler)',
        'final_fixes_part2.py': 'FINAL experiment: real val weights, CatBoost, DeLong',
        'publication_fixes.py': '2-model ensemble, bootstrap, Stacking',
        'delong_fix.py': 'Per-label DeLong + FDR correction',
        'final_statistical_tests.py': 'Permutation tests, exact p-values',
    }.get(fname, '')

    print(f"  {fname:<35} {n_lines:>8} {status:<20} {role}")

    if 'SUPERSEDED' in status or 'DEPRECATED' in status:
        deprecated_files.append(fname)
    else:
        production_files.append(fname)

# ── Check for remaining issues ──
print(f"\n  Architecture audit findings:")
print(f"  - Active production files: {len(production_files)}")
print(f"  - Deprecated/superseded files: {len(deprecated_files)}")

# Check for .values bug on numpy arrays
print(f"\n  Checking for remaining '.values' on numpy arrays...")
issues_found = 0
for fpath in py_files:
    fname = os.path.basename(fpath)
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # Look for patterns like X_val.values where X_val is loaded as numpy
    # This is a heuristic check — false positives possible
    import re
    for match in re.finditer(r'(X_\w+)\.values', content):
        line_num = content[:match.start()].count('\n') + 1
        if fname not in ['preprocessing.py', 'config.py']:  # These are OK
            print(f"    WARNING: {fname}:{line_num}: potential .values on numpy array: {match.group()}")
            issues_found += 1
if issues_found == 0:
    print(f"    No remaining .values issues found.")

# Check for hardcoded scaler
print(f"\n  Checking for independent StandardScaler instances...")
for fpath in py_files:
    fname = os.path.basename(fpath)
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # Count StandardScaler() calls
    scaler_calls = content.count('StandardScaler()')
    if scaler_calls > 1 and fname not in ['preprocessing.py']:
        print(f"    WARNING: {fname}: {scaler_calls} StandardScaler() instances")
    elif scaler_calls == 1 and fname not in ['preprocessing.py']:
        pass  # OK if it's from evaluation.py or just one instance
    elif scaler_calls == 0:
        pass  # No scaler — OK for non-data scripts

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("AUDIT SUMMARY")
print(f"{'='*70}")

# Check for unsaved predictions from key runs
key_pred_files = [
    'PUB_lp_rf_preds.npy', 'PUB_tabpfn_preds.npy', 'PUB_ens2_eq_preds.npy',
    'FINAL_lp_rf_preds.npy', 'FINAL_RESULTS.json', 'CORRECTED_DELONG_RESULTS.json',
]
missing = [f for f in key_pred_files if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
if missing:
    print(f"  MISSING key output files: {missing}")
else:
    print(f"  All key output files present.")

print(f"\n  Production pipeline (recommended):")
print(f"    1. preprocessing.py          → generates processed_data/")
print(f"    2. final_fixes_part2.py       → trains all models, saves predictions")
print(f"    3. publication_fixes.py       → 2-model ensemble, bootstrap, Stacking")
print(f"    4. final_statistical_tests.py → DeLong + Permutation + FDR + exact p")
print(f"    5. evaluation.py              → imported by all as metrics framework")
print(f"    6. config.py                  → imported by all as configuration")

# Save
results = {
    'metadata': {
        'version': 'FINAL with Permutation Tests',
        'timestamp': pd.Timestamp.now().isoformat(),
        'note': 'DeLong for large-sample labels, Permutation for n+<15 labels'
    },
    'ensemble_vs_tabpfn': all_results,
    'ensemble_vs_lp_rf': lp_results,
    'audit': {
        'total_files': len(py_files),
        'active_files': len(production_files),
        'deprecated_files': deprecated_files,
    }
}

out_path = os.path.join(OUTPUT_DIR, 'FINAL_STATISTICAL_RESULTS.json')
with open(out_path, 'w') as f:
    # Convert numpy types to Python native
    json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x) if isinstance(x, np.bool_) else None)

print(f"\nResults saved to {out_path}")
