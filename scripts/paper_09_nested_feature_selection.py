#!/usr/bin/env python3
"""
paper_09_nested_feature_selection.py
=====================================
Nested feature selection via SHAP within each LOO fold.

Purpose: Address the potential circularity concern where SHAP-based
         feature selection is performed on the full dataset before
         model evaluation.

Approach: For each leave-one-PFAS-out fold:
  1. Train XGBoost on 46 PFAS (training set)
  2. Compute SHAP on the training set (test PFAS never seen)
  3. Select Top K features from SHAP
  4. Retrain XGBoost using only Top K features on the training set
  5. Predict on the held-out test PFAS
  6. Repeat for all 47 PFAS

This guarantees that feature selection is blind to the test compound.

Input: data/paper/feature_matrix_kd.csv
Output: data/paper/kd_nested_feature_selection.csv
"""
import csv, os, sys, warnings
import numpy as np
from collections import defaultdict
warnings.filterwarnings("ignore")

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SI_DIR = os.path.join(PROJECT, "data", "paper")
INPUT = os.path.join(SI_DIR, "feature_matrix_kd.csv")
OUTPUT = os.path.join(SI_DIR, "kd_nested_feature_selection.csv")

# Shared constants (centralised in scripts/_shared_config.py)
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared_config import NON_FEATURE, SOIL_FEATURES
TOP_K_VALUES = [2, 5]  # Test multiple feature budgets
RANDOM_SEED = 42

def load_data():
    with open(INPUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pfas_groups = defaultdict(list)
    for row in rows:
        pfas_groups[row["PFAS_name"].strip()].append(row)
    print(f"Data: {len(rows)} rows, {len(pfas_groups)} PFAS compounds")
    return rows, pfas_groups

def extract_matrix(rows_list, fieldnames, selected_feature_indices=None):
    """
    Extract X, y from a list of rows.
    If selected_feature_indices is not None, only use those columns.
    """
    all_desc = [c for c in fieldnames if c not in NON_FEATURE]

    if selected_feature_indices is not None:
        use_cols = [all_desc[i] for i in selected_feature_indices]
    else:
        use_cols = all_desc

    n = len(rows_list)
    p = len(use_cols)
    X = np.full((n, p), np.nan)
    y = np.full(n, np.nan)

    for j, col in enumerate(use_cols):
        for i, row in enumerate(rows_list):
            v = row.get(col, "").strip()
            if v:
                try: X[i, j] = float(v)
                except: pass

    for i, row in enumerate(rows_list):
        v = row.get("log_Kd", "").strip()
        if v:
            try: y[i] = float(v)
            except: pass

    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    if len(y) == 0:
        return X, y, use_cols

    for j in range(p):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)): continue
        X[:, j] = np.nan_to_num(col_vals, nan=np.nanmedian(col_vals))

    return X, y, use_cols

def compute_shap_importance(model, X_train, feature_names):
    """SHAP-based feature importance ranking."""
    import shap
    # Sample up to 500 points for SHAP speed
    n_shap = min(len(X_train), 500)
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X_train), n_shap, replace=False)
    X_sample = X_train[idx]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample, check_additivity=False)
    mean_abs = np.abs(shap_vals).mean(axis=0)

    # Return feature indices sorted by importance (descending)
    return np.argsort(mean_abs)[::-1], mean_abs

def run_nested_loo(rows, pfas_groups, top_k):
    """
    Leave-one-PFAS-out with nested SHAP feature selection.

    For each fold:
      - Train on 46 PFAS → SHAP → select Top K features
      - Retrain on 46 PFAS using only Top K features
      - Predict on held-out PFAS
    """
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error

    fieldnames = list(rows[0].keys())
    pfas_list = sorted(pfas_groups.keys())
    n_pfas = len(pfas_list)

    # Identify soil feature indices in the descriptor list
    all_desc = [c for c in fieldnames if c not in NON_FEATURE]
    soil_indices = [i for i, c in enumerate(all_desc) if c in SOIL_FEATURES]

    all_results = []
    all_y_true, all_y_pred = [], []
    feature_rankings = {}  # per-PFAS top features for analysis

    print(f"\n--- Top {top_k} nested SHAP ---")

    for i, test_pfas in enumerate(pfas_list):
        train_rows = []
        for name, group in pfas_groups.items():
            if name != test_pfas:
                train_rows.extend(group)
        test_rows = pfas_groups[test_pfas]

        if len(test_rows) == 0 or len(train_rows) < 10:
            continue

        # --- Step 1: Full model on training set ---
        X_train_full, y_train, _ = extract_matrix(train_rows, fieldnames)
        X_test_full, y_test, _ = extract_matrix(test_rows, fieldnames)

        model_full = xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        model_full.fit(X_train_full, y_train)

        # --- Step 2: SHAP on training set only ---
        # X_train_full columns follow the order of all_desc
        # SHAP returns importance indices in X_train_full column order
        # We want to select Top K molecular descriptors, always include soil

        shap_rank, shap_vals = compute_shap_importance(model_full, X_train_full, all_desc)

        # Select Top K molecular descriptors (skip soil features in ranking)
        selected_desc = []
        for idx in shap_rank:
            col_name = all_desc[idx]
            if col_name not in SOIL_FEATURES:
                selected_desc.append(idx)
                if len(selected_desc) >= top_k:
                    break

        # Always include soil features
        selected_indices = sorted(set(selected_desc + soil_indices))

        # --- Step 3: Retrain with Top K features ---
        X_train_topk = X_train_full[:, selected_indices]
        X_test_topk = X_test_full[:, selected_indices]

        model_topk = xgb.XGBRegressor(
            n_estimators=500, max_depth=4 if top_k <= 10 else 8,
            learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        model_topk.fit(X_train_topk, y_train)
        y_pred = model_topk.predict(X_test_topk)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        # Record which features were selected for this PFAS
        # Record which features were selected for this PFAS
        selected_names = [all_desc[j] for j in selected_indices]
        selected_desc_names = [all_desc[j] for j in selected_desc]
        feature_rankings[test_pfas] = {
            "selected_features": selected_desc_names[:top_k],
            "n_test": len(y_test),
        }

        tag = "✅" if r2 > 0 else "⚠️"
        print(f"  [{i+1}/{n_pfas}] {tag} {test_pfas:<20} "
              f"n_test={len(y_test):<3} R²={r2:.3f} "
              f"(top desc: {selected_desc_names[:3]})")

        all_results.append({
            "test_pfas": test_pfas,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
        })

    # Pooled R²
    pooled_r2 = r2_score(all_y_true, all_y_pred)
    return all_results, pooled_r2, feature_rankings


def main():
    rows, pfas_groups = load_data()

    print("=" * 60)
    print("  Nested Feature Selection (SHAP within each LOO fold)")
    print("=" * 60)
    print(f"\n  Feature budgets tested: {TOP_K_VALUES}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Soil features always included (9 total)\n")

    all_summaries = []

    # --- Baseline: Full combined model LOO (no feature selection) ---
    # Using existing paper_06b results
    print("  Baseline: Combined model (no feature selection)")
    print(f"    Pooled R² = 0.719 (from paper_06b_loo_combined_fix.py)\n")

    for top_k in TOP_K_VALUES:
        results, pooled_r2, rankings = run_nested_loo(rows, pfas_groups, top_k)

        n_positive = sum(1 for r in results if r["r2"] > 0)
        n_total = len(results)

        print(f"\n  Top {top_k} Nested SHAP Summary:")
        print(f"    Pooled R² = {pooled_r2:.4f}")
        print(f"    Positive R²: {n_positive}/{n_total}")

        all_summaries.append({
            "top_k": top_k,
            "pooled_r2": round(pooled_r2, 4),
            "n_positive": n_positive,
            "n_total": n_total,
        })

        # Save per-PFAS results
        out_file = os.path.join(SI_DIR, f"kd_nested_shap_top{top_k}.csv")
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["test_pfas", "n_train", "n_test", "r2", "rmse"])
            writer.writeheader()
            writer.writerows(results)
        print(f"    Saved: {out_file}")

    # ── Summary table ──
    print(f"\n{'=' * 60}")
    print(f"  Nested Feature Selection Results")
    print(f"{'=' * 60}")
    print(f"{'Top K':<10} {'Pooled R²':<12} {'Pos. R²':<12}")
    print("-" * 34)
    print(f"{'Baseline':<10} {'0.719':<12} {'24/47':<12}")
    for s in all_summaries:
        pos_ratio = f"{s['n_positive']}/{s['n_total']}"
        print(f"{s['top_k']:<10} {s['pooled_r2']:<12.4f} {pos_ratio:<12}")

    # Save summary
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["top_k", "pooled_r2", "n_positive", "n_total"])
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\n✅ Summary saved to: {OUTPUT}")

    # ── Feature stability analysis ──
    print(f"\n{'=' * 60}")
    print(f"  Feature Stability Across Folds (Top 2)")
    print(f"{'=' * 60}")
    # Collect top-2 features from each fold
    from collections import Counter
    feature_counter = Counter()
    for pfas, info in rankings.items():
        for feat in info["selected_features"][:2]:
            feature_counter[feat] += 1
    total_folds = len(rankings)
    print(f"  Across {total_folds} folds:")
    for feat, count in feature_counter.most_common(10):
        print(f"    {feat}: {count}/{total_folds} ({count/total_folds*100:.0f}%)")


if __name__ == "__main__":
    main()
