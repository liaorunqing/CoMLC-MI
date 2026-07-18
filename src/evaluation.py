"""
================================================================================
CoMLC-MI v2: Unified Evaluation Framework
================================================================================
Standardised metric computation for all models. Key additions over v1:
  - AUPRC (Precision-Recall AUC) — essential for extreme class imbalance
  - Skill score (AUPRC - random baseline)
  - Per-label threshold optimization with 5-fold CV
  - Bootstrap confidence intervals for per-label AUC
================================================================================
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, hamming_loss, label_ranking_loss,
    coverage_error, average_precision_score, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from config import LABEL_COLS, N_LABELS, RANDOM_SEED


def compute_all_metrics(y_true, y_pred_proba, thresholds=None, compute_ci=False, n_bootstrap=500):
    """
    Compute comprehensive multi-label evaluation metrics.

    Parameters
    ----------
    y_true : ndarray (N, L)
    y_pred_proba : ndarray (N, L)
    thresholds : ndarray (L,) or None — per-label binarization thresholds
    compute_ci : bool — if True, compute bootstrap CIs for per-label AUC
    n_bootstrap : int — number of bootstrap samples

    Returns
    -------
    dict with all metrics
    """
    if thresholds is None:
        thresholds = np.full(N_LABELS, 0.5)

    y_pred = (y_pred_proba >= thresholds).astype(int)

    # ── Per-label metrics ────────────────────────────────────────────
    per_label = {}
    for i, col in enumerate(LABEL_COLS):
        yt = y_true[:, i]
        yp = y_pred_proba[:, i]
        n_pos = int(yt.sum())

        entry = {}
        if n_pos >= 2:
            entry['auc'] = roc_auc_score(yt, yp)
            entry['auprc'] = average_precision_score(yt, yp)
            entry['skill'] = entry['auprc'] - (n_pos / len(yt))  # above random
        else:
            entry['auc'] = np.nan
            entry['auprc'] = np.nan
            entry['skill'] = np.nan

        entry['f1'] = f1_score(yt, y_pred[:, i], zero_division=0)
        entry['n_pos'] = n_pos
        per_label[col] = entry

    # ── Aggregate metrics ────────────────────────────────────────────
    valid_aucs = [v['auc'] for v in per_label.values() if not np.isnan(v['auc'])]
    valid_auprcs = [v['auprc'] for v in per_label.values() if not np.isnan(v['auprc'])]
    valid_skills = [v['skill'] for v in per_label.values() if not np.isnan(v['skill'])]

    metrics = {
        'micro_auc': roc_auc_score(y_true.ravel(), y_pred_proba.ravel()),
        'macro_auc': np.mean(valid_aucs) if valid_aucs else np.nan,
        'macro_auprc': np.mean(valid_auprcs) if valid_auprcs else np.nan,
        'macro_skill': np.mean(valid_skills) if valid_skills else np.nan,
        'micro_f1': f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0),
        'macro_f1': np.mean([v['f1'] for v in per_label.values()]),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'ranking_loss': label_ranking_loss(y_true, y_pred_proba),
        'coverage': coverage_error(y_true, y_pred_proba),
        'per_label': per_label,
    }

    # ── Bootstrap CIs (percentile method) ────────────────────────────
    if compute_ci:
        n = len(y_true)
        # Collect bootstrap AUC samples for each label
        boot_aucs = {col: [] for col in LABEL_COLS}
        for seed in range(n_bootstrap):
            idx = np.random.RandomState(RANDOM_SEED * 1000 + seed).choice(n, n, replace=True)
            for i, col in enumerate(LABEL_COLS):
                yt = y_true[idx, i]
                if yt.sum() >= 2 and (1 - yt).sum() >= 2:
                    boot_aucs[col].append(roc_auc_score(yt, y_pred_proba[idx, i]))

        ci_per_label = {}
        for col in LABEL_COLS:
            if len(boot_aucs[col]) > 10:
                arr = np.sort(boot_aucs[col])
                ci_per_label[col] = {
                    'auc_low': arr[int(0.025 * len(arr))],
                    'auc_high': arr[int(0.975 * len(arr))],
                    'auc_std': np.std(arr),
                }
            else:
                ci_per_label[col] = {'auc_low': np.nan, 'auc_high': np.nan, 'auc_std': np.nan}

        metrics['bootstrap_ci'] = ci_per_label

    return metrics


def robust_threshold_tuning(X_train, y_train, model_fn, n_folds=5):
    """
    Find stable per-label thresholds using cross-validation.

    Unlike single-validation-set tuning, this uses 5-fold CV to get
    thresholds that generalise. For labels with < 3 positives in a fold,
    the default 0.5 is retained.

    Parameters
    ----------
    X_train : ndarray (N, F)
    y_train : ndarray (N, L)
    model_fn : callable -> returns fitted model with .predict_proba()
    n_folds : int

    Returns
    -------
    thresholds : ndarray (L,) — median optimal threshold per label
    """
    fold_thresholds = np.full((n_folds, N_LABELS), 0.5)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train[:, -1])):
        model = model_fn()
        model.fit(X_train[tr_idx], y_train[tr_idx])
        preds = model.predict_proba(X_train[va_idx])

        for i in range(N_LABELS):
            yt = y_train[va_idx, i]
            if yt.sum() >= 3:  # minimum 3 positives to tune
                prec, rec, thresh = precision_recall_curve(yt, preds[:, i])
                f1s = 2 * prec * rec / (prec + rec + 1e-8)
                best_idx = np.argmax(f1s)
                if best_idx < len(thresh):
                    fold_thresholds[fold, i] = thresh[best_idx]

    # Median across folds (robust to outliers)
    thresholds = np.median(fold_thresholds, axis=0)
    return thresholds


def print_metrics_table(name, metrics, show_auprc=True):
    """Pretty-print a single model's metrics."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Micro-AUC:     {metrics['micro_auc']:.4f}")
    print(f"  Macro-AUC:     {metrics['macro_auc']:.4f}")
    if show_auprc:
        print(f"  Macro-AUPRC:   {metrics['macro_auprc']:.4f}")
        print(f"  Macro-Skill:   {metrics['macro_skill']:.4f}")
    print(f"  Micro-F1:      {metrics['micro_f1']:.4f}")
    print(f"  Macro-F1:      {metrics['macro_f1']:.4f}")
    print(f"  Hamming Loss:  {metrics['hamming_loss']:.4f}")
    print(f"  Ranking Loss:  {metrics['ranking_loss']:.4f}")
    print(f"  Coverage:      {metrics['coverage']:.4f}")
    print(f"\n  Per-label details:")
    print(f"  {'Label':<8} {'AUC':>8} {'AUPRC':>8} {'Skill':>8} {'F1':>8} {'N+':>6}")
    print(f"  {'-'*50}")
    for col in LABEL_COLS:
        pl = metrics['per_label'][col]
        a_str = f"{pl['auc']:.4f}" if not np.isnan(pl['auc']) else 'N/A'
        p_str = f"{pl['auprc']:.4f}" if not np.isnan(pl['auprc']) else 'N/A'
        s_str = f"{pl['skill']:.4f}" if not np.isnan(pl['skill']) else 'N/A'
        f_str = f"{pl['f1']:.4f}"
        print(f"  {col:<8} {a_str:>8} {p_str:>8} {s_str:>8} {f_str:>8} {pl['n_pos']:>6}")
