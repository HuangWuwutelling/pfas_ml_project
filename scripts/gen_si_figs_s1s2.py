#!/usr/bin/env python3
"""Generate SI Figures S1 and S2.

Figure S1: Side-by-side LOO bar chart comparing RDKit-only vs Combined model
Figure S2: Predicted vs observed scatter plots for RDKit-only and Combined LOO results

Wrapped in `if __name__ == "__main__":` so that `import gen_si_figs_s1s2`
does not trigger the LOO training (the main flow runs 47-fold LOO and
takes ~2 minutes).
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Shared constants (paths + feature definitions) — see §13.1-13.3 of
# reproduction-report-2026-07-25.md.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared_config import NON_FEATURE, SOIL_FEATURES  # noqa: E402

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT, "data", "paper")
# Figures saved under data/paper/ to be consistent with paper_07_generate_figures.py
# and gen_graphical_abstract.py (13.7 — unify output directories). The old
# paper/figures/ output was inconsistent and not tracked by any other script.
FIG_DIR = os.path.join(DATA_DIR)  # DATA_DIR = data/paper/
os.makedirs(FIG_DIR, exist_ok=True)


def load_loo_results(filepath):
    """Read a paper_06-style LOO results CSV and return list of dicts."""
    rows = []
    with open(filepath) as f:
        for r in csv.DictReader(f):
            r2 = r.get('r2', '').strip()
            if r2 and r2.lower() != 'nan':
                rows.append({
                    'pfas': r['test_pfas'],
                    'n_test': int(r['n_test']),
                    'r2': float(r2),
                    'rmse': float(r['rmse']),
                })
    return rows


def extract_matrix(rows_list, desc_cols):
    """Build numeric feature matrix and target vector from a list of
    row-dicts, using the given list of descriptor columns. Missing values
    are imputed with the column median.
    """
    n = len(rows_list)
    p = len(desc_cols)
    X = np.zeros((n, p))
    y = np.full(n, np.nan)
    for j, col in enumerate(desc_cols):
        for i, row in enumerate(rows_list):
            v = row.get(col, '').strip()
            if v:
                try:
                    X[i, j] = float(v)
                except (ValueError, TypeError):
                    pass
    for i, row in enumerate(rows_list):
        v = row.get('log_Kd', '').strip()
        if v:
            try:
                y[i] = float(v)
            except (ValueError, TypeError):
                pass
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    # Impute NaN with column median
    for j in range(p):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)):
            continue
        X[:, j] = np.nan_to_num(col_vals, nan=np.nanmedian(col_vals))
    return X, y


if __name__ == "__main__":
    import xgboost as xgb
    from collections import defaultdict
    from sklearn.metrics import r2_score
    from matplotlib.patches import Patch

    # Figure S1: LOO bar chart (RDKit vs Combined)
    combined = load_loo_results(os.path.join(DATA_DIR, "kd_leave_one_out_results_combined.csv"))
    rdkit_only = load_loo_results(os.path.join(DATA_DIR, "kd_leave_one_out_results_rdkit.csv"))

    # Sort by combined R2
    combined_sorted = sorted(combined, key=lambda x: x['r2'])
    names_sorted = [c['pfas'] for c in combined_sorted]
    rdkit_lookup = {r['pfas']: r for r in rdkit_only}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("Figure S1. Leave-One-PFAS-Out Cross-Validation Comparison",
                 fontsize=13, fontweight='bold')

    # Re-build the RDKit panel in the same order as Combined
    rdkit_ordered = [rdkit_lookup[n] for n in names_sorted]

    for ax, results, title in [
        (ax1, combined_sorted, '(a) RDKit-only model'),
        (ax2, rdkit_ordered, '(b) Combined model'),
    ]:
        sorted_res = sorted(results, key=lambda x: x['r2'])
        names = [r['pfas'] for r in sorted_res]
        r2s = [r['r2'] for r in sorted_res]
        n_tests = [r['n_test'] for r in sorted_res]

        colors = []
        for v, n in zip(r2s, n_tests):
            if v > 0.5:
                colors.append('#4daf4a')
            elif v > 0:
                colors.append('#ffd700')
            elif n < 5:
                colors.append('#b0b0b0')
            else:
                colors.append('#e41a1c')

        bars = ax.barh(range(len(names)), r2s, color=colors, alpha=0.8,
                       edgecolor='gray', linewidth=0.3)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel('R$^2$', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-6.5, 1.1)

        for bar, n in zip(bars, n_tests):
            ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                    f'n={n}', va='center', fontsize=4.5, color='gray')

    legend_elements = [
        Patch(facecolor='#4daf4a', label='Good (R$^2$ > 0.5)'),
        Patch(facecolor='#ffd700', label='Moderate (0 < R$^2$ < 0.5)'),
        Patch(facecolor='#e41a1c', label='Poor (R$^2$ $\\leq$ 0)'),
        Patch(facecolor='#b0b0b0', label='Insufficient data (n < 5)'),
    ]
    ax2.legend(handles=legend_elements, fontsize=7, loc='lower left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_s1 = os.path.join(FIG_DIR, "figS1_loo_comparison.png")
    plt.savefig(path_s1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figure S1: {path_s1}")

    # Figure S2: Predicted vs observed scatter
    FEATURE_FILE = os.path.join(DATA_DIR, "feature_matrix_kd.csv")

    with open(FEATURE_FILE) as f:
        all_rows = list(csv.DictReader(f))
    fieldnames = list(all_rows[0].keys())
    desc_cols = [c for c in fieldnames if c not in NON_FEATURE]

    # Group by PFAS
    pfas_groups = defaultdict(list)
    for row in all_rows:
        pfas_groups[row['PFAS_name'].strip()].append(row)
    pfas_list = sorted(pfas_groups.keys())

    # Run LOO for Combined model
    all_true_combined = []
    all_pred_combined = []
    all_true_rdkit = []
    all_pred_rdkit = []

    print("Running LOO predictions for scatter plots...")

    for test_pfas in pfas_list:
        train_rows = []
        for name, group in pfas_groups.items():
            if name != test_pfas:
                train_rows.extend(group)
        test_rows = pfas_groups[test_pfas]

        if len(test_rows) == 0 or len(train_rows) < 10:
            continue

        X_train, y_train = extract_matrix(train_rows, desc_cols)
        X_test, y_test = extract_matrix(test_rows, desc_cols)

        # RDKit-only model: remove soil features
        soil_indices = [i for i, c in enumerate(desc_cols) if c in SOIL_FEATURES]
        rdkit_indices = [i for i in range(len(desc_cols)) if i not in soil_indices]

        # Combined model
        model_combined = xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1,
        )
        model_combined.fit(X_train, y_train)
        y_pred_c = model_combined.predict(X_test)
        all_true_combined.extend(y_test.tolist())
        all_pred_combined.extend(y_pred_c.tolist())

        # RDKit-only model
        if rdkit_indices:
            model_rdkit = xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1,
            )
            model_rdkit.fit(X_train[:, rdkit_indices], y_train)
            y_pred_r = model_rdkit.predict(X_test[:, rdkit_indices])
            all_true_rdkit.extend(y_test.tolist())
            all_pred_rdkit.extend(y_pred_r.tolist())

    all_true_combined = np.array(all_true_combined)
    all_pred_combined = np.array(all_pred_combined)
    all_true_rdkit = np.array(all_true_rdkit)
    all_pred_rdkit = np.array(all_pred_rdkit)

    r2_c = r2_score(all_true_combined, all_pred_combined)
    r2_r = r2_score(all_true_rdkit, all_pred_rdkit)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Figure S2. Leave-One-PFAS-Out Predicted vs. Observed log K$_{d}$",
                 fontsize=13, fontweight='bold')

    for ax, true, pred, r2v, title, color in [
        (ax1, all_true_rdkit, all_pred_rdkit, r2_r, '(a) RDKit-only model', '#377eb8'),
        (ax2, all_true_combined, all_pred_combined, r2_c, '(b) Combined model', '#e41a1c'),
    ]:
        ax.scatter(true, pred, s=8, alpha=0.3, c=color, edgecolors='none')
        min_v = min(true.min(), pred.min())
        max_v = max(true.max(), pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1, alpha=0.4)
        ax.set_xlabel('Observed log K$_{d}$', fontsize=10)
        ax.set_ylabel('Predicted log K$_{d}$', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.text(0.05, 0.95, f'Pooled R$^2$ = {r2v:.3f}', transform=ax.transAxes,
                fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path_s2 = os.path.join(FIG_DIR, "figS2_loo_scatter.png")
    plt.savefig(path_s2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figure S2: {path_s2}")

    print("\nDone! Both SI figures generated.")
