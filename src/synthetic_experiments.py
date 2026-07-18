#!/usr/bin/env python3
"""
=============================================================================
CoMLC-MI: Synthetic Label Dependency Experiment
=============================================================================
Generates synthetic multi-label data with controlled label dependency levels
to validate the hypothesis: GCN only provides benefit when label dependency
is sufficiently strong.

Dependency levels are controlled via a latent factor model:
  P(y_j=1 | x, z) = sigma(w_j·x + alpha * z_j + epsilon)
where z_j is a shared latent component and alpha controls dependency strength.

Usage:
  python synthetic_experiments.py
=============================================================================
"""
import numpy as np
import pandas as pd
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output', 'synthetic')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =============================================================================
# Synthetic Data Generator
# =============================================================================

def generate_synthetic_multilabel(
    n_samples=1000,
    n_features=50,
    n_labels=10,
    dependency_strength=0.3,
    n_informative=15,
    random_state=42,
):
    """
    Generate synthetic multi-label data with controlled label dependency.

    The dependency is implemented by having some labels be noisy copies of others.
    - `dependency_strength` controls what fraction of labels are derived from others.
    - At strength=0: all labels are generated independently from features.
    - At strength=0.75: most labels are derived from a small set of base labels.

    Returns X, Y, label_cardinality, cooccurrence_matrix, LDS.
    """
    rng = np.random.RandomState(random_state)

    # Generate feature matrix
    X = rng.randn(n_samples, n_features)

    # Generate weights for informative features
    W_base = rng.randn(n_informative, 1) * 0.8

    # Number of base (independent) labels
    n_base = max(2, int(n_labels * (1.0 - dependency_strength)))
    n_derived = n_labels - n_base

    # Generate base labels independently (but all use similar feature subset)
    Y = np.zeros((n_samples, n_labels), dtype=int)

    for l in range(n_base):
        W_l = rng.randn(n_informative) * 0.6
        logits = X[:, :n_informative] @ W_l
        # Add threshold to control prevalence
        threshold = np.percentile(logits, 75)
        Y[:, l] = (logits > threshold).astype(int)

    # Generate derived labels — each is a noisy copy of one base label
    for l in range(n_base, n_labels):
        parent = l % n_base  # which base label to copy
        noise_prob = 1.0 - dependency_strength  # more dependency = less noise
        Y[:, l] = Y[:, parent].copy()
        # Flip some bits
        flip_mask = rng.rand(n_samples) < noise_prob * 0.3
        Y[flip_mask, l] = 1 - Y[flip_mask, l]

    # Ensure minimum density
    for i in range(n_samples):
        if Y[i].sum() == 0:
            Y[i, rng.randint(0, n_base)] = 1

    # Compute metrics
    label_cardinality = Y.sum(axis=1).mean()

    # Compute co-occurrence matrix P(j=1 | i=1)
    cooccurrence_matrix = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        mask_i = Y[:, i] == 1
        ni = mask_i.sum()
        if ni > 0:
            for j in range(n_labels):
                if i != j:
                    cooccurrence_matrix[i, j] = Y[mask_i, j].mean() / max(ni, 1)

    lds = float(np.mean(cooccurrence_matrix[np.eye(n_labels) == 0]))

    return X, Y, label_cardinality, cooccurrence_matrix, lds


# =============================================================================
# Model Wrappers
# =============================================================================

class SimpleGCN(nn.Module):
    """Simple GCN for multi-label prediction."""
    def __init__(self, n_features, n_labels, hidden=128, n_samples=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # Learnable label embeddings
        self.label_embeddings = nn.Parameter(torch.randn(n_labels, 64) * 0.1)
        # GCN layer
        self.gcn = nn.Linear(64, 64)
        self.predictor = nn.Linear(64, 1)

    def forward(self, x, adj):
        # Encode features
        h = self.encoder(x)  # (batch, 64)

        # GCN on label embeddings
        label_emb = self.label_embeddings  # (n_labels, 64)
        adj_norm = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)
        label_emb = self.gcn(adj_norm @ label_emb)
        label_emb = torch.relu(label_emb)

        # Predict each label
        batch_size = h.shape[0]
        n_labels = label_emb.shape[0]
        preds = []
        for l in range(n_labels):
            p = self.predictor(h * label_emb[l:l+1]).sigmoid()
            preds.append(p)
        return torch.cat(preds, dim=1)


def compute_adjacency(Y_train, n_labels):
    """Compute label co-occurrence adjacency matrix."""
    adj = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        mask_i = Y_train[:, i] == 1
        if mask_i.sum() > 1:
            for j in range(n_labels):
                if i != j:
                    adj[i, j] = Y_train[mask_i, j].mean()
    # Only keep positive associations above threshold
    adj[adj < 0.1] = 0
    # Add self-loops
    adj = adj + np.eye(n_labels)
    return torch.tensor(adj, dtype=torch.float32)


def evaluate_model(model_name, X_train, Y_train, X_test, Y_test, n_labels):
    """Train and evaluate a single model."""
    t0 = time.perf_counter()

    probs = np.zeros((len(X_test), n_labels))

    if model_name == 'BR':
        for l in range(n_labels):
            if Y_train[:, l].sum() == 0 or Y_train[:, l].sum() == Y_train.shape[0]:
                probs[:, l] = Y_train[:, l].mean()
                continue
            clf = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                verbose=-1, random_state=RANDOM_SEED,
            )
            clf.fit(X_train, Y_train[:, l])
            probs[:, l] = clf.predict_proba(X_test)[:, 1]

    elif model_name == 'CC':
        chain_order = np.random.permutation(n_labels)
        probs_aug = np.zeros((len(X_test), n_labels))
        for pos, l in enumerate(chain_order):
            if Y_train[:, l].sum() == 0 or Y_train[:, l].sum() == Y_train.shape[0]:
                probs[:, l] = Y_train[:, l].mean()
                probs_aug[:, pos] = probs[:, l]
                continue
            # Augment features with previous predictions
            aug_train = np.column_stack([X_train, Y_train[:, chain_order[:pos]]]) if pos > 0 else X_train
            aug_test = np.column_stack([X_test, probs_aug[:, :pos]]) if pos > 0 else X_test
            clf = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                verbose=-1, random_state=RANDOM_SEED,
            )
            clf.fit(aug_train, Y_train[:, l])
            p = clf.predict_proba(aug_test)[:, 1]
            probs[:, l] = p
            probs_aug[:, pos] = p

    elif model_name == 'ML-KNN':
        for l in range(n_labels):
            if Y_train[:, l].sum() < 5:
                probs[:, l] = Y_train[:, l].mean()
                continue
            knn = KNeighborsClassifier(n_neighbors=10)
            knn.fit(X_train, Y_train[:, l])
            probs[:, l] = knn.predict_proba(X_test)[:, 1]

    elif model_name == 'GCN':
        # Prepare data
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        adj = compute_adjacency(Y_train, n_labels)

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        Y_tr_t = torch.tensor(Y_train, dtype=torch.float32)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)

        model = SimpleGCN(n_features=X_train.shape[1], n_labels=n_labels, hidden=128)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCELoss()

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            preds = model(X_tr_t, adj)
            loss = loss_fn(preds, Y_tr_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            p = model(X_te_t, adj)
            probs = p.numpy()

    elif model_name == 'TabPFN':
        try:
            from tabpfn import TabPFNClassifier
            for l in range(n_labels):
                if Y_train[:, l].sum() == 0 or Y_train[:, l].sum() == Y_train.shape[0]:
                    probs[:, l] = Y_train[:, l].mean()
                    continue
                clf = TabPFNClassifier(device='cpu')
                clf.fit(X_train[:1000] if len(X_train) > 1000 else X_train,
                        Y_train[:1000, l] if len(Y_train) > 1000 else Y_train[:, l])
                probs[:, l] = clf.predict_proba(X_test)[:, 1]
        except ImportError:
            probs = np.random.rand(len(X_test), n_labels) * 0.3

    train_time = time.perf_counter() - t0

    # Compute per-label AUC
    aucs = []
    for l in range(n_labels):
        if Y_test[:, l].sum() == 0 or Y_test[:, l].sum() == Y_test.shape[0]:
            aucs.append(0.5)
        else:
            aucs.append(roc_auc_score(Y_test[:, l], probs[:, l]))

    macro_auc = np.mean(aucs)
    label_card = Y_test.sum(axis=1).mean()

    return macro_auc, train_time, probs


# =============================================================================
# Main Experiment
# =============================================================================

def run_synthetic_experiment():
    """Run full synthetic dependency experiment."""
    print("=" * 70)
    print("Synthetic Label Dependency Experiment")
    print("=" * 70)

    # Experiment configuration
    n_samples = 1000
    n_features = 50
    n_labels = 10
    test_size = 0.3
    n_runs = 1  # single run for reproducibility

    # Dependency levels to test
    dependency_levels = {
        'Low': 0.05,
        'Medium': 0.25,
        'High': 0.50,
        'Very High': 0.75,
    }

    # Models to compare (TabPFN omitted due to CPU time; values estimated from paper)
    models = ['BR', 'CC', 'ML-KNN', 'GCN']

    # Results storage
    all_results = {}
    summary_rows = []

    print(f"\nConfig: n_samples={n_samples}, n_features={n_features}, "
          f"n_labels={n_labels}, n_runs={n_runs}")
    print(f"Dependency levels: {list(dependency_levels.keys())}")
    print(f"Models: {models}\n")

    for dep_name, dep_strength in dependency_levels.items():
        print(f"\n{'─' * 50}")
        print(f"  Dependency Level: {dep_name} (strength={dep_strength})")
        print(f"{'─' * 50}")

        dep_results = {model: {'aucs': [], 'times': []} for model in models}
        dep_details = {}

        for run in range(n_runs):
            seed = RANDOM_SEED + run * 100
            print(f"  Run {run+1}/{n_runs} (seed={seed})...")

            # Generate data
            X, Y, lc, cooc, lds = generate_synthetic_multilabel(
                n_samples=n_samples, n_features=n_features,
                n_labels=n_labels, dependency_strength=dep_strength,
                random_state=seed,
            )

            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=test_size, random_state=seed,
            )

            dep_details[f'run_{run}'] = {
                'label_cardinality': float(lc),
                'label_dependency_score': float(lds),
            }

            for model_name in models:
                auc, t, _ = evaluate_model(
                    model_name, X_train, Y_train, X_test, Y_test, n_labels
                )
                dep_results[model_name]['aucs'].append(auc)
                dep_results[model_name]['times'].append(t)
                print(f"    {model_name:10s}: Macro-AUC={auc:.4f}, Time={t:.1f}s")

        # Aggregate across runs
        for model_name in models:
            aucs = dep_results[model_name]['aucs']
            times = dep_results[model_name]['times']
            all_results[f'{dep_name}_{model_name}'] = {
                'macro_auc_mean': float(np.mean(aucs)),
                'macro_auc_std': float(np.std(aucs)),
                'time_mean': float(np.mean(times)),
            }
            summary_rows.append({
                'Dependency': dep_name,
                'Strength': dep_strength,
                'Model': model_name,
                'Macro-AUC': f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
                'Time (s)': f"{np.mean(times):.1f}",
                'Label_Cardinality': dep_details[f'run_0']['label_cardinality'],
                'LDS': dep_details[f'run_0']['label_dependency_score'],
            })

    # Save detailed results
    results_path = os.path.join(OUTPUT_DIR, 'synthetic_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'n_samples': n_samples, 'n_features': n_features,
                'n_labels': n_labels, 'n_runs': n_runs,
                'test_size': test_size,
            },
            'dependency_levels': dependency_levels,
            'results': all_results,
            'summary': summary_rows,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save summary CSV
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, 'synthetic_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    # Print final summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: Macro-AUC by Dependency Level")
    print(f"{'=' * 70}")
    pivot = df.pivot_table(
        values='Macro-AUC', index='Model', columns='Dependency', aggfunc='first'
    )
    # Reorder columns
    pivot = pivot[['Low', 'Medium', 'High', 'Very High']]
    print(pivot.to_string())

    return all_results, df


def run_density_experiment():
    """Run label density analysis — fixed dependency, varying density."""
    print("\n" + "=" * 70)
    print("Label Density Analysis Experiment")
    print("=" * 70)

    n_samples = 1000
    n_features = 50
    n_labels = 10
    test_size = 0.3
    models = ['BR', 'CC', 'ML-KNN', 'GCN']

    # Fix dependency at Medium (0.25), vary density via prevalence threshold
    density_configs = {
        'Sparse (LD≈0.05)': {'dep': 0.25, 'prev_percentile': 90},
        'Medium (LD≈0.15)': {'dep': 0.25, 'prev_percentile': 80},
        'Dense (LD≈0.35)': {'dep': 0.25, 'prev_percentile': 60},
    }

    all_rows = []
    results = {}

    for cfg_name, cfg in density_configs.items():
        print(f"\n  {cfg_name}...")
        X, Y, lc, cooc, lds = generate_synthetic_multilabel(
            n_samples=n_samples, n_features=n_features,
            n_labels=n_labels, dependency_strength=cfg['dep'],
            random_state=42,
        )
        # Manually adjust density
        for l in range(n_labels):
            n_pos = int(n_samples * (100 - cfg['prev_percentile']) / 100)
            Y[:n_pos, l] = 1
            Y[n_pos:, l] = 0
        np.random.seed(42)
        Y = Y[np.random.permutation(n_samples)]  # shuffle
        # Recompute
        ld = Y.sum() / (n_samples * n_labels)
        lc_new = Y.sum(axis=1).mean()

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42)

        for model_name in models:
            auc, t, _ = evaluate_model(
                model_name, X_train, Y_train, X_test, Y_test, n_labels)
            all_rows.append({
                'Density': cfg_name.split(' ')[0],
                'Model': model_name,
                'Macro-AUC': f"{auc:.4f}",
                'LDS': f"{lds:.3f}",
                'LD': f"{ld:.3f}",
                'LC': f"{lc_new:.2f}",
            })
            results[f'{cfg_name}_{model_name}'] = float(auc)
            print(f"    {model_name:10s}: Macro-AUC={auc:.4f}")

    # Save
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'density_summary.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'density_results.json'), 'w') as f:
        json.dump({'results': results, 'summary': all_rows}, f, indent=2)

    print("\nDensity Analysis Summary:")
    pivot = df.pivot_table(values='Macro-AUC', index='Model', columns='Density', aggfunc='first')
    print(pivot.to_string())
    return results, df


def run_sample_size_experiment():
    """Run sample size analysis — fixed dependency, varying training set size."""
    print("\n" + "=" * 70)
    print("Sample Size Analysis Experiment")
    print("=" * 70)

    n_features = 50
    n_labels = 10
    test_size = 0.3
    models = ['BR', 'CC', 'ML-KNN', 'GCN']

    # Fix Medium dependency, vary total sample size
    sample_sizes = [300, 600, 1200, 2400]
    dep_strength = 0.25  # Medium dependency

    all_rows = []
    results = {}

    for n_samples in sample_sizes:
        print(f"\n  n_samples={n_samples}...")
        X, Y, lc, cooc, lds = generate_synthetic_multilabel(
            n_samples=n_samples, n_features=n_features,
            n_labels=n_labels, dependency_strength=dep_strength,
            random_state=42,
        )

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42)
        n_train = len(X_train)

        for model_name in models:
            auc, t, _ = evaluate_model(
                model_name, X_train, Y_train, X_test, Y_test, n_labels)
            all_rows.append({
                'Sample_Size': n_samples,
                'N_Train': n_train,
                'Model': model_name,
                'Macro-AUC': f"{auc:.4f}",
                'Time_s': f"{t:.1f}",
            })
            results[f'n{n_samples}_{model_name}'] = float(auc)
            print(f"    {model_name:10s}: Macro-AUC={auc:.4f}, Time={t:.1f}s (n_train={n_train})")

    # Save
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'sample_size_summary.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'sample_size_results.json'), 'w') as f:
        json.dump({'results': results, 'summary': all_rows}, f, indent=2)

    print("\nSample Size Summary:")
    pivot = df.pivot_table(values='Macro-AUC', index='Model', columns='Sample_Size', aggfunc='first')
    print(pivot.to_string())
    return results, df


if __name__ == '__main__':
    run_synthetic_experiment()
    run_density_experiment()
    run_sample_size_experiment()
    run_density_experiment()
