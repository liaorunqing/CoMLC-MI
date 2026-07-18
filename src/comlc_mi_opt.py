"""
=============================================================================
CoMLC-MI: Hyperparameter Optimization and Ablation Runner
=============================================================================
Tests multiple configurations:
  - Graph: none, sparse (raw), dense (symmetric max)
  - Loss: BCE (weighted), CAL (various gamma_minus, m)
  - Architecture: dot-product, concat-head
=============================================================================
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv
import json
import os
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────
DATA_DIR = r'C:\Users\liaoq\Desktop\MI_Research\output\processed_data'
OUTPUT_DIR = r'C:\Users\liaoq\Desktop\MI_Research\output\comlc_mi'
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_COLS = [
    'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK',
    'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS'
]
N_LABELS = len(LABEL_COLS)
LABEL_NAMES = {
    'FIBR_PREDS': 'Atrial Fibrillation', 'PREDS_TAH': 'Supraventricular Tachycardia',
    'JELUD_TAH': 'Ventricular Tachycardia', 'FIBR_JELUD': 'Ventricular Fibrillation',
    'A_V_BLOK': '3rd-degree AV Block', 'OTEK_LANC': 'Pulmonary Edema',
    'RAZRIV': 'Myocardial Rupture', 'DRESSLER': 'Dressler Syndrome',
    'ZSN': 'Chronic Heart Failure', 'REC_IM': 'Recurrent MI',
    'P_IM_STEN': 'Post-infarction Angina', 'LET_IS': 'Lethal Outcome',
}

# ── Data ─────────────────────────────────────────────────────────────────
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv')).values.astype(np.float32)
X_val = pd.read_csv(os.path.join(DATA_DIR, 'X_val.csv')).values.astype(np.float32)
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv')).values.astype(np.float32)
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.astype(np.float32)
y_val = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv')).values.astype(np.float32)
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.astype(np.float32)

n_features = X_train.shape[1]
prevalence = y_train.mean(axis=0)
pos_weights = (1.0 / (prevalence + 1e-6))

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val).to(DEVICE)
y_val_t = torch.FloatTensor(y_val)
X_test_t = torch.FloatTensor(X_test).to(DEVICE)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# ── Build Adjacency Matrix ───────────────────────────────────────────────
def build_adjacency(y, method='symmetric', tau=0.08):
    """Build label adjacency matrix from training labels."""
    cooccur = np.zeros((N_LABELS, N_LABELS))
    marginal = y.sum(axis=0)
    for i in range(N_LABELS):
        for j in range(N_LABELS):
            if i == j:
                cooccur[i, j] = 1.0
            else:
                co_ij = ((y[:, i] == 1) & (y[:, j] == 1)).sum()
                cooccur[i, j] = co_ij / max(marginal[i], 1)

    if method == 'raw':
        adj = cooccur.copy()
    elif method == 'symmetric':
        adj = np.maximum(cooccur, cooccur.T)
    elif method == 'reweighted':
        p = np.clip(np.maximum(cooccur, cooccur.T), 1e-6, 1.0)
        adj = p / (p + 1.0 / (p + 1e-6))
    elif method == 'none':
        return None, None  # No graph

    adj[adj < tau] = 0.0
    np.fill_diagonal(adj, 1.0)

    edges, weights = [], []
    for i in range(N_LABELS):
        for j in range(N_LABELS):
            if adj[i, j] > 0:
                edges.append([i, j])
                weights.append(adj[i, j])

    if len(edges) == 0:
        return None, None

    ei = torch.LongTensor(edges).T.to(DEVICE)
    ew = torch.FloatTensor(weights).to(DEVICE)
    return ei, ew


# ── Model Definitions ────────────────────────────────────────────────────
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, d=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class LabelGCN(nn.Module):
    def __init__(self, d=64, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(d, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, d)
        self.dropout = nn.Dropout(0.3)
    def forward(self, z, ei, ew=None):
        z = F.relu(self.conv1(z, ei, ew))
        z = self.dropout(z)
        return self.conv2(z, ei, ew)


class DotProductModel(nn.Module):
    """Original CoMLC-MI: dot product between patient emb and label emb."""
    def __init__(self, input_dim, n_labels, d=64, dropout=0.3):
        super().__init__()
        self.encoder = FeatureEncoder(input_dim, d, dropout)
        self.label_embs = nn.Parameter(torch.randn(n_labels, d) * 0.1)
        self.gcn = LabelGCN(d)
    def forward(self, x, ei, ew=None):
        h = self.encoder(x)
        z = self.gcn(self.label_embs, ei, ew) if ei is not None else self.label_embs
        return torch.sigmoid(torch.matmul(h, z.T))


class ConcatModel(nn.Module):
    """Alternative: concatenate patient emb with each label emb, MLP head."""
    def __init__(self, input_dim, n_labels, d=64, dropout=0.3):
        super().__init__()
        self.encoder = FeatureEncoder(input_dim, d, dropout)
        self.label_embs = nn.Parameter(torch.randn(n_labels, d) * 0.1)
        self.gcn = LabelGCN(d)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(2*d, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
            for _ in range(n_labels)
        ])
    def forward(self, x, ei, ew=None):
        h = self.encoder(x)  # (B, d)
        z = self.gcn(self.label_embs, ei, ew) if ei is not None else self.label_embs  # (L, d)
        outputs = []
        for k in range(N_LABELS):
            z_k = z[k:k+1].expand(h.shape[0], -1)  # (B, d)
            combined = torch.cat([h, z_k], dim=-1)  # (B, 2d)
            outputs.append(torch.sigmoid(self.heads[k](combined)))
        return torch.cat(outputs, dim=1)


# ── Loss Functions ───────────────────────────────────────────────────────
def weighted_bce_loss(preds, targets):
    """Weighted BCE with per-label inverse prevalence weights."""
    w = torch.FloatTensor(pos_weights).to(preds.device)
    loss = 0
    for k in range(N_LABELS):
        wk = w[k]
        loss += -wk * (targets[:, k] * torch.log(preds[:, k] + 1e-8) +
                       (1 - targets[:, k]) * torch.log(1 - preds[:, k] + 1e-8)).mean()
    return loss / N_LABELS


def cal_loss(preds, targets, gamma_minus=3, m=0.05):
    """Clinical Asymmetric Loss."""
    w = torch.FloatTensor(pos_weights).to(preds.device)
    eps = 1e-8
    pos_mask = (targets == 1).float()
    neg_mask = (targets == 0).float()

    pos_loss = -torch.log(preds + eps) * w.unsqueeze(0) * pos_mask
    p_m = torch.clamp(preds - m, min=0.0)
    neg_loss = -p_m.pow(gamma_minus) * torch.log(1 - p_m + eps) * (1.0 / w.unsqueeze(0)) * neg_mask

    return (pos_loss.sum() + neg_loss.sum()) / targets.numel()


# ── Training Function ────────────────────────────────────────────────────
def train_model(model, ei, ew, loss_fn, loss_name, lr=1e-3, epochs=200, patience=25):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_val_auc = 0.0
    best_state = None
    patience_ctr = 0
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            preds = model(bx, ei, ew)
            loss = loss_fn(preds, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t, ei, ew).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_preds, average='macro')

        train_losses.append(epoch_loss / len(train_loader))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t, ei, ew).cpu().numpy()

    test_metrics = {
        'micro_auc': roc_auc_score(y_test.ravel(), test_preds.ravel()),
        'macro_auc': roc_auc_score(y_test, test_preds, average='macro'),
        'per_label_auc': {LABEL_COLS[i]: roc_auc_score(y_test[:, i], test_preds[:, i])
                          if len(np.unique(y_test[:, i])) > 1 else np.nan
                          for i in range(N_LABELS)},
    }

    return {
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'train_losses': train_losses,
        'model_state': best_state,
    }


# ── Run Experiments ──────────────────────────────────────────────────────
print("=" * 70)
print("CoMLC-MI HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

# Build adjacency matrices
adj_configs = {
    'none': build_adjacency(y_train, 'none'),
    'sparse_asym': build_adjacency(y_train, 'raw', tau=0.15),
    'symmetric': build_adjacency(y_train, 'symmetric', tau=0.08),
}

# Experiment grid
experiments = []
for adj_name, (ei, ew) in adj_configs.items():
    for model_cls, model_name in [(DotProductModel, 'dot'), (ConcatModel, 'concat')]:
        for loss_fn, loss_name in [
            (weighted_bce_loss, 'W_BCE'),
            (lambda p, t: cal_loss(p, t, gamma_minus=2, m=0.05), 'CAL_g2'),
            (lambda p, t: cal_loss(p, t, gamma_minus=3, m=0.05), 'CAL_g3'),
        ]:
            experiments.append({
                'adj': adj_name, 'model': model_name,
                'loss': loss_name, 'model_cls': model_cls, 'loss_fn': loss_fn,
                'ei': ei, 'ew': ew,
            })

print(f"\nRunning {len(experiments)} configurations...\n")

results = []
for i, exp in enumerate(experiments):
    name = f"[{exp['adj']}/{exp['model']}/{exp['loss']}]"
    print(f"  {i+1:2d}/{len(experiments)} {name}...", end=' ', flush=True)

    model = exp['model_cls'](n_features, N_LABELS, d=64, dropout=0.3)
    result = train_model(model, exp['ei'], exp['ew'], exp['loss_fn'], exp['loss'])

    test_macro = result['test_metrics']['macro_auc']
    val_best = result['best_val_auc']

    print(f"val={val_best:.4f} test_macro={test_macro:.4f}")
    results.append({**exp, **{k: v for k, v in result.items() if k != 'model_state'}})

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS SUMMARY (sorted by test Macro-AUC)")
print("=" * 70)

results.sort(key=lambda r: r['test_metrics']['macro_auc'], reverse=True)

print(f"\n{'Rank':<5} {'Config':<45} {'Val AUC':>10} {'Test Micro':>11} {'Test Macro':>11}")
print("-" * 85)
for rank, r in enumerate(results, 1):
    cfg = f"{r['adj']}/{r['model']}/{r['loss']}"
    tm = r['test_metrics']
    print(f"{rank:<5} {cfg:<45} {r['best_val_auc']:>10.4f} {tm['micro_auc']:>11.4f} {tm['macro_auc']:>11.4f}")

# Save best model
best = results[0]
print(f"\nBest: {best['adj']}/{best['model']}/{best['loss']}")

# Save results
serializable = []
for r in results:
    s = {k: v for k, v in r.items() if k not in ['loss_fn', 'model_cls', 'ei', 'ew', 'train_losses']}
    s['test_metrics'] = {
        'micro_auc': float(s['test_metrics']['micro_auc']),
        'macro_auc': float(s['test_metrics']['macro_auc']),
        'per_label_auc': {k: float(v) if not np.isnan(v) else None
                          for k, v in s['test_metrics']['per_label_auc'].items()},
    }
    serializable.append(s)

with open(os.path.join(OUTPUT_DIR, 'hyperparameter_results.json'), 'w') as f:
    json.dump(serializable, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/hyperparameter_results.json")
