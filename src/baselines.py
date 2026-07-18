"""
=============================================================================
CoMLC-MI Research: Phase 1 — Baseline Model Implementation & Evaluation
=============================================================================
Baselines (per Section 6.3):
  - XGBoost (per-label, independent)
  - LightGBM (per-label, independent)
  - Binary Relevance (BR) with LogisticRegression
  - Classifier Chains (CC)
  - Label Powerset (LP)
  - RAkEL (Random k-Labelsets)
  - Shared-encoder MLP (PyTorch)
  - Multitask DNN (Makhmudov et al., 2025 style)

Metrics (per Section 6.2):
  Micro-AUC, Macro-AUC, Micro-F1, Macro-F1, Hamming Loss,
  Ranking Loss, Coverage, Per-label AUC
=============================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, f1_score, hamming_loss, label_ranking_loss, coverage_error
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from skmultilearn.ensemble import RakelD
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────
DATA_DIR = r'C:\Users\liaoq\Desktop\MI_Research\output\processed_data'
OUTPUT_DIR = r'C:\Users\liaoq\Desktop\MI_Research\output\baselines'
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_COLS = [
    'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK',
    'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS'
]

LABEL_NAMES = {
    'FIBR_PREDS': 'Atrial Fibrillation',
    'PREDS_TAH': 'Supraventricular Tachycardia',
    'JELUD_TAH': 'Ventricular Tachycardia',
    'FIBR_JELUD': 'Ventricular Fibrillation',
    'A_V_BLOK': '3rd-degree AV Block',
    'OTEK_LANC': 'Pulmonary Edema',
    'RAZRIV': 'Myocardial Rupture',
    'DRESSLER': 'Dressler Syndrome',
    'ZSN': 'Chronic Heart Failure',
    'REC_IM': 'Recurrent MI',
    'P_IM_STEN': 'Post-infarction Angina',
    'LET_IS': 'Lethal Outcome',
}


# ── Metric Functions ─────────────────────────────────────────────────────
def compute_all_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute all multi-label evaluation metrics.

    Parameters
    ----------
    y_true : ndarray (N, L) — ground truth binary labels
    y_pred_proba : ndarray (N, L) — predicted probabilities
    threshold : float — binarization threshold

    Returns
    -------
    dict with all metrics
    """
    n_labels = y_true.shape[1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Per-label AUC
    per_label_auc = {}
    for i, col in enumerate(LABEL_COLS):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            per_label_auc[col] = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        else:
            per_label_auc[col] = np.nan

    # Micro metrics
    micro_auc = roc_auc_score(y_true.ravel(), y_pred_proba.ravel())
    micro_f1 = f1_score(y_true.ravel(), y_pred.ravel())

    # Macro metrics
    macro_auc = np.nanmean(list(per_label_auc.values()))
    per_label_f1 = [f1_score(y_true[:, i], y_pred[:, i]) for i in range(n_labels)]
    macro_f1 = np.mean(per_label_f1)

    # Other MLC metrics
    ham_loss = hamming_loss(y_true, y_pred)
    rank_loss = label_ranking_loss(y_true, y_pred_proba)
    coverage = coverage_error(y_true, y_pred_proba)

    return {
        'micro_auc': micro_auc,
        'macro_auc': macro_auc,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'hamming_loss': ham_loss,
        'ranking_loss': rank_loss,
        'coverage': coverage,
        'per_label_auc': per_label_auc,
    }


def print_metrics(name, metrics):
    """Pretty-print metrics for a model."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Micro-AUC:     {metrics['micro_auc']:.4f}")
    print(f"  Macro-AUC:     {metrics['macro_auc']:.4f}")
    print(f"  Micro-F1:      {metrics['micro_f1']:.4f}")
    print(f"  Macro-F1:      {metrics['macro_f1']:.4f}")
    print(f"  Hamming Loss:  {metrics['hamming_loss']:.4f}")
    print(f"  Ranking Loss:  {metrics['ranking_loss']:.4f}")
    print(f"  Coverage:      {metrics['coverage']:.4f}")
    print(f"  Per-label AUC:")
    for col in LABEL_COLS:
        auc = metrics['per_label_auc'][col]
        if np.isnan(auc):
            print(f"    {col}: N/A (no positive samples)")
        else:
            print(f"    {col}: {auc:.4f}")


# ── Data Loading ─────────────────────────────────────────────────────────
print("Loading processed data...")
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
X_val = pd.read_csv(os.path.join(DATA_DIR, 'X_val.csv'))
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv'))
y_val = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))

# Combine train+val for CV-based evaluation (more robust)
X_train_full = pd.concat([X_train, X_val], ignore_index=True)
y_train_full = pd.concat([y_train, y_val], ignore_index=True)

print(f"Train+Val: {len(X_train_full)} samples, {X_train_full.shape[1]} features")
print(f"Test: {len(X_test)} samples")

# Convert to numpy for sklearn
X_train_np = X_train_full.values.astype(np.float32)
y_train_np = y_train_full.values.astype(np.int32)
X_test_np = X_test.values.astype(np.float32)
y_test_np = y_test.values.astype(np.int32)

n_features = X_train_np.shape[1]
n_labels = len(LABEL_COLS)

print(f"\nDevice: {DEVICE}")
print(f"n_features={n_features}, n_labels={n_labels}")

# ── Baseline 1: Independent XGBoost ──────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE 1: Independent XGBoost (per-label)")
print("=" * 70)

xgb_preds_test = np.zeros((len(X_test_np), n_labels))
xgb_per_label_auc = {}

for i, col in enumerate(LABEL_COLS):
    y_train_col = y_train_full[col].values
    y_test_col = y_test[col].values

    # Scale_pos_weight for imbalance
    n_pos = y_train_col.sum()
    n_neg = len(y_train_col) - n_pos
    scale_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        verbosity=0,


    )
    model.fit(X_train_np, y_train_col)
    xgb_preds_test[:, i] = model.predict_proba(X_test_np)[:, 1]

    n_unique = len(np.unique(y_test_col))
    if n_unique > 1:
        xgb_per_label_auc[col] = roc_auc_score(y_test_col, xgb_preds_test[:, i])
    else:
        xgb_per_label_auc[col] = np.nan

xgb_metrics_test = compute_all_metrics(y_test_np, xgb_preds_test)
xgb_metrics_test['per_label_auc'] = xgb_per_label_auc
print_metrics("XGBoost (per-label)", xgb_metrics_test)

# ── Baseline 2: Independent LightGBM ─────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE 2: Independent LightGBM (per-label)")
print("=" * 70)

lgb_preds_test = np.zeros((len(X_test_np), n_labels))
lgb_per_label_auc = {}

for i, col in enumerate(LABEL_COLS):
    y_train_col = y_train_full[col].values
    y_test_col = y_test[col].values

    n_pos = y_train_col.sum()
    n_neg = len(y_train_col) - n_pos
    scale_weight = n_neg / max(n_pos, 1)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X_train_np, y_train_col)
    lgb_preds_test[:, i] = model.predict_proba(X_test_np)[:, 1]

    if len(np.unique(y_test_col)) > 1:
        lgb_per_label_auc[col] = roc_auc_score(y_test_col, lgb_preds_test[:, i])
    else:
        lgb_per_label_auc[col] = np.nan

lgb_metrics_test = compute_all_metrics(y_test_np, lgb_preds_test)
lgb_metrics_test['per_label_auc'] = lgb_per_label_auc
print_metrics("LightGBM (per-label)", lgb_metrics_test)

# ── Baseline 3: Binary Relevance (BR) ────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE 3: Binary Relevance (Logistic Regression)")
print("=" * 70)

br_clf = BinaryRelevance(
    classifier=LogisticRegression(max_iter=2000, random_state=RANDOM_SEED,
                                   class_weight='balanced')
)
br_clf.fit(X_train_np, y_train_np)
br_preds_raw = br_clf.predict_proba(X_test_np)
# Handle sparse output from scikit-multilearn
if hasattr(br_preds_raw, 'toarray'):
    br_preds = br_preds_raw.toarray()
elif isinstance(br_preds_raw, list):
    br_preds = np.column_stack([
        p.toarray()[:, 1] if hasattr(p, 'toarray') and p.shape[1] > 1
        else (p.toarray().ravel() if hasattr(p, 'toarray') else p.ravel())
        for p in br_preds_raw
    ])
else:
    br_preds = br_preds_raw
br_metrics = compute_all_metrics(y_test_np, br_preds)
print_metrics("Binary Relevance (LR)", br_metrics)

# ── Baseline 4: Classifier Chains (CC) ───────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE 4: Classifier Chains")
print("=" * 70)

cc_preds = np.zeros((len(X_test_np), n_labels))
cc_per_label_auc = {}

for order_seed in [0, 1, 2]:  # Ensemble of 3 chain orders
    chain_order = np.random.RandomState(order_seed).permutation(n_labels)
    cc_clf = ClassifierChain(
        LogisticRegression(max_iter=2000, random_state=RANDOM_SEED,
                           class_weight='balanced'),
        order=chain_order,
        random_state=RANDOM_SEED,
    )
    cc_clf.fit(X_train_np, y_train_np)
    cc_preds += cc_clf.predict_proba(X_test_np)[:, :, 1] if cc_clf.predict_proba(X_test_np).ndim > 2 else cc_clf.predict_proba(X_test_np)

cc_preds /= 3  # Average over ensemble

for i, col in enumerate(LABEL_COLS):
    y_test_col = y_test[col].values
    if len(np.unique(y_test_col)) > 1:
        cc_per_label_auc[col] = roc_auc_score(y_test_col, cc_preds[:, i])
    else:
        cc_per_label_auc[col] = np.nan

cc_metrics = compute_all_metrics(y_test_np, cc_preds)
cc_metrics['per_label_auc'] = cc_per_label_auc
print_metrics("Classifier Chains (Ensemble of 3)", cc_metrics)

# ── Baseline 5: Label Powerset (LP) ──────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE 5: Label Powerset (RF)")
print("=" * 70)

# Use RF for LP to handle the large label space
lp_clf = LabelPowerset(
    classifier=RandomForestClassifier(n_estimators=200, max_depth=10,
                                       random_state=RANDOM_SEED, n_jobs=-1)
)
lp_clf.fit(X_train_np, y_train_np)
lp_preds_raw = lp_clf.predict_proba(X_test_np)
if hasattr(lp_preds_raw, 'toarray'):
    lp_preds = lp_preds_raw.toarray()
elif isinstance(lp_preds_raw, list):
    lp_preds = np.column_stack([
        p.toarray()[:, 1] if hasattr(p, 'toarray') and p.shape[1] > 1
        else (p.toarray().ravel() if hasattr(p, 'toarray') else p.ravel())
        for p in lp_preds_raw
    ])
else:
    lp_preds = lp_preds_raw
lp_metrics = compute_all_metrics(y_test_np, lp_preds)
print_metrics("Label Powerset (RF)", lp_metrics)

# ── Baseline 6: RAkEL ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE 6: RAkEL (Random k-Labelsets)")
print("=" * 70)

rakel_clf = RakelD(
    base_classifier=LogisticRegression(max_iter=2000, random_state=RANDOM_SEED,
                                        class_weight='balanced'),
    labelset_size=4
)
rakel_clf.fit(X_train_np, y_train_np)
rakel_preds_raw = rakel_clf.predict_proba(X_test_np)
if hasattr(rakel_preds_raw, 'toarray'):
    rakel_preds = rakel_preds_raw.toarray()
elif isinstance(rakel_preds_raw, list):
    rakel_preds = np.column_stack([
        p.toarray()[:, 1] if hasattr(p, 'toarray') and p.shape[1] > 1
        else (p.toarray().ravel() if hasattr(p, 'toarray') else p.ravel())
        for p in rakel_preds_raw
    ])
else:
    rakel_preds = rakel_preds_raw
rakel_metrics = compute_all_metrics(y_test_np, rakel_preds)
print_metrics("RAkEL (k=4, LR)", rakel_metrics)

# ── Baseline 7: Shared-Encoder MLP ───────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE 7: Shared-Encoder MLP (Deep MLC, No Graph)")
print("=" * 70)


class SharedEncoderMLP(nn.Module):
    """MLP with shared encoder + independent sigmoid heads per label."""

    def __init__(self, input_dim, hidden_dims, n_labels, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden_dims[-1], 1) for _ in range(n_labels)])

    def forward(self, x):
        h = self.encoder(x)
        return torch.cat([torch.sigmoid(head(h)) for head in self.heads], dim=1)


# Prepare PyTorch data
X_train_t = torch.FloatTensor(X_train.values.astype(np.float32))
y_train_t = torch.FloatTensor(y_train.values.astype(np.float32))
X_val_t = torch.FloatTensor(X_val.values.astype(np.float32))
y_val_t = torch.FloatTensor(y_val.values.astype(np.float32))
X_test_t = torch.FloatTensor(X_test_np)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# Compute per-label pos_weight for BCE loss
pos_counts = y_train.values.sum(axis=0)
neg_counts = len(y_train) - pos_counts
pos_weights = torch.FloatTensor(neg_counts / (pos_counts + 1e-8)).to(DEVICE)

mlp_model = SharedEncoderMLP(
    input_dim=n_features,
    hidden_dims=[256, 128, 64],
    n_labels=n_labels,
    dropout=0.3,
).to(DEVICE)

optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCELoss()  # Weighted BCE applied manually
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

best_val_auc = 0.0
best_state = None
patience = 30
patience_counter = 0

print("Training Shared-Encoder MLP...")
for epoch in range(200):
    mlp_model.train()
    train_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        preds = mlp_model(bx)
        # Weighted BCE
        loss = 0
        for k in range(n_labels):
            weight_k = pos_weights[k]
            loss += -weight_k * (by[:, k] * torch.log(preds[:, k] + 1e-8) +
                                 (1 - by[:, k]) * torch.log(1 - preds[:, k] + 1e-8)).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    # Validation
    mlp_model.eval()
    with torch.no_grad():
        val_preds = mlp_model(X_val_t.to(DEVICE)).cpu().numpy()
        val_auc = roc_auc_score(y_val.values, val_preds, average='macro')

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch+1}, best val macro-AUC: {best_val_auc:.4f}")
        break

    if (epoch + 1) % 30 == 0:
        print(f"  Epoch {epoch+1:3d}: train_loss={train_loss/len(train_loader):.4f}, val_macro_auc={val_auc:.4f}")

# Restore best model
mlp_model.load_state_dict(best_state)
mlp_model.eval()
with torch.no_grad():
    mlp_preds_test = mlp_model(X_test_t.to(DEVICE)).cpu().numpy()
mlp_metrics = compute_all_metrics(y_test_np, mlp_preds_test)
print_metrics("Shared-Encoder MLP (3-layer, 256→128→64)", mlp_metrics)

# ── Baseline 8: Makhmudov-style Multitask DNN ────────────────────────────
print("\n" + "=" * 70)
print("BASELINE 8: Multitask DNN (Makhmudov et al., 2025 style)")
print("=" * 70)


class MultiTaskDNN(nn.Module):
    """Deeper multitask network with per-task branches."""

    def __init__(self, input_dim, shared_dims, task_dims, n_labels, dropout=0.3):
        super().__init__()
        # Shared encoder
        layers = []
        prev_dim = input_dim
        for hd in shared_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        self.shared = nn.Sequential(*layers)

        # Per-task branches
        self.branches = nn.ModuleList()
        for k in range(n_labels):
            branch = nn.Sequential(
                nn.Linear(shared_dims[-1], task_dims[0]),
                nn.BatchNorm1d(task_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(task_dims[0], 1),
            )
            self.branches.append(branch)

    def forward(self, x):
        h = self.shared(x)
        return torch.cat([torch.sigmoid(branch(h)) for branch in self.branches], dim=1)


mt_model = MultiTaskDNN(
    input_dim=n_features,
    shared_dims=[256, 128],
    task_dims=[32],
    n_labels=n_labels,
    dropout=0.3,
).to(DEVICE)

optimizer_mt = optim.AdamW(mt_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_mt = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_mt, T_0=30, T_mult=2)

best_val_auc = 0.0
best_state_mt = None
patience_counter = 0

print("Training Multitask DNN...")
for epoch in range(200):
    mt_model.train()
    train_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer_mt.zero_grad()
        preds = mt_model(bx)
        loss = 0
        for k in range(n_labels):
            weight_k = pos_weights[k]
            loss += -weight_k * (by[:, k] * torch.log(preds[:, k] + 1e-8) +
                                 (1 - by[:, k]) * torch.log(1 - preds[:, k] + 1e-8)).mean()
        loss.backward()
        optimizer_mt.step()
        train_loss += loss.item()
    scheduler_mt.step()

    mt_model.eval()
    with torch.no_grad():
        val_preds = mt_model(X_val_t.to(DEVICE)).cpu().numpy()
        val_auc = roc_auc_score(y_val.values, val_preds, average='macro')

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state_mt = {k: v.cpu().clone() for k, v in mt_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"  Early stopping at epoch {epoch+1}, best val macro-AUC: {best_val_auc:.4f}")
        break

    if (epoch + 1) % 30 == 0:
        print(f"  Epoch {epoch+1:3d}: train_loss={train_loss/len(train_loader):.4f}, val_macro_auc={val_auc:.4f}")

mt_model.load_state_dict(best_state_mt)
mt_model.eval()
with torch.no_grad():
    mt_preds_test = mt_model(X_test_t.to(DEVICE)).cpu().numpy()
mt_metrics = compute_all_metrics(y_test_np, mt_preds_test)
print_metrics("Multitask DNN (Makhmudov-style)", mt_metrics)

# ── Aggregate Results ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE COMPARISON — All Models on Test Set")
print("=" * 70)

all_results = {
    'XGBoost (per-label)': xgb_metrics_test,
    'LightGBM (per-label)': lgb_metrics_test,
    'Binary Relevance (LR)': br_metrics,
    'Classifier Chains (Ensemble)': cc_metrics,
    'Label Powerset (RF)': lp_metrics,
    'RAkEL (k=4, LR)': rakel_metrics,
    'Shared-Encoder MLP': mlp_metrics,
    'Multitask DNN (Makhmudov)': mt_metrics,
}

# Build comparison table
print(f"\n{'Model':<35} {'Micro-AUC':>10} {'Macro-AUC':>10} {'Micro-F1':>10} {'Macro-F1':>10} {'HamLoss':>10} {'RankLoss':>10}")
print("-" * 95)
for name, m in all_results.items():
    print(f"{name:<35} {m['micro_auc']:>10.4f} {m['macro_auc']:>10.4f} {m['micro_f1']:>10.4f} {m['macro_f1']:>10.4f} {m['hamming_loss']:>10.4f} {m['ranking_loss']:>10.4f}")

# Find best per metric
print(f"\nBest Micro-AUC: {max(all_results.items(), key=lambda x: x[1]['micro_auc'])[0]}")
print(f"Best Macro-AUC: {max(all_results.items(), key=lambda x: x[1]['macro_auc'])[0]}")
print(f"Best Macro-F1:  {max(all_results.items(), key=lambda x: x[1]['macro_f1'])[0]}")

# ── Per-label comparison (key models only) ───────────────────────────────
print(f"\n{'Label':<25} {'XGBoost':>10} {'LightGBM':>10} {'MLP':>10} {'MT-DNN':>10}")
print("-" * 65)
for i, col in enumerate(LABEL_COLS):
    xgb_a = xgb_metrics_test['per_label_auc'].get(col, np.nan)
    lgb_a = lgb_metrics_test['per_label_auc'].get(col, np.nan)
    mlp_a = mlp_metrics['per_label_auc'].get(col, np.nan)
    mt_a = mt_metrics['per_label_auc'].get(col, np.nan)
    print(f"{LABEL_NAMES[col]:<25} {xgb_a:>10.4f} {lgb_a:>10.4f} {mlp_a:>10.4f} {mt_a:>10.4f}")

# ── Statistical significance (DeLong test proxy: McNemar) ────────────────
print("\n" + "=" * 70)
print("STATISTICAL TEST: McNemar's test (XGBoost vs LightGBM)")
print("=" * 70)

# Binarize at 0.5
xgb_bin = (xgb_preds_test >= 0.5).astype(int)
lgb_bin = (lgb_preds_test >= 0.5).astype(int)

for i, col in enumerate(LABEL_COLS):
    y_true_col = y_test_np[:, i]
    # McNemar: discordant pairs
    b = ((xgb_bin[:, i] == 1) & (lgb_bin[:, i] == 0) & (y_true_col == y_true_col)).sum()
    c_count = ((xgb_bin[:, i] == 0) & (lgb_bin[:, i] == 1) & (y_true_col == y_true_col)).sum()
    if b + c_count > 0:
        chi2 = (abs(b - c_count) - 1)**2 / (b + c_count)
        p_val = 1 - stats.chi2.cdf(chi2, 1)
        sig = '*' if p_val < 0.05 else ''
        print(f"  {col}: chi2={chi2:.3f}, p={p_val:.4f} {sig}")

# ── Save Results ─────────────────────────────────────────────────────────
print(f"\nSaving results to {OUTPUT_DIR}...")

# Convert to serializable format
results_serializable = {}
for name, m in all_results.items():
    results_serializable[name] = {
        k: (v if not isinstance(v, dict) else {kk: float(vv) if not np.isnan(vv) else None for kk, vv in v.items()})
        for k, v in m.items()
    }

with open(os.path.join(OUTPUT_DIR, 'all_baseline_results.json'), 'w') as f:
    json.dump(results_serializable, f, indent=2)

# Save predictions for later comparison with CoMLC-MI
np.save(os.path.join(OUTPUT_DIR, 'xgb_preds.npy'), xgb_preds_test)
np.save(os.path.join(OUTPUT_DIR, 'lgb_preds.npy'), lgb_preds_test)
np.save(os.path.join(OUTPUT_DIR, 'mlp_preds.npy'), mlp_preds_test)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test_np)

# Save model weights
torch.save(best_state, os.path.join(OUTPUT_DIR, 'shared_mlp_best.pt'))
torch.save(best_state_mt, os.path.join(OUTPUT_DIR, 'multitask_dnn_best.pt'))

print("Baseline evaluation complete.")
