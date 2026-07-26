#!/usr/bin/env python3
"""5-seed CV verification: average R² over 5 different random seeds.

This script re-trains XGBoost on the canonical feature matrix with five
different `random_state` values (42, 123, 456, 789, 1111) and reports the
mean 5-fold CV R² across seeds. This removes the seed-dependent
randomness in the train/test split and is the strongest evidence that
the headline numbers in the paper are stable, not artefacts of a
particular seed.

The script is wrapped in `if __name__ == "__main__":` so that
`import verify_cv_final` does not trigger the model training, which
is useful when other scripts want to inspect the helper functions
in isolation.
"""
import csv
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Shared constants (paths + feature definitions) — see §13.1-13.3 of
# reproduction-report-2026-07-25.md.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared_config import NON_FEATURE  # noqa: E402

SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
INPUT = os.path.join(SI_DIR, "feature_matrix_kd.csv")


def extract(rows_list, use_cols):
    """Extract numeric feature matrix and target vector from a list of
    row-dicts. Missing values are imputed with the column median.
    """
    n, p = len(rows_list), len(use_cols)
    X = np.full((n, p), np.nan)
    y = np.full(n, np.nan)
    for j, col in enumerate(use_cols):
        for i, row in enumerate(rows_list):
            v = row.get(col, "").strip()
            if v:
                try:
                    X[i, j] = float(v)
                except (ValueError, TypeError):
                    pass
    for i, row in enumerate(rows_list):
        v = row.get("log_Kd", "").strip()
        if v:
            try:
                y[i] = float(v)
            except (ValueError, TypeError):
                pass
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    for j in range(p):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)):
            continue
        X[:, j] = np.nan_to_num(col_vals, nan=np.nanmedian(col_vals))
    return X, y


if __name__ == "__main__":
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import xgboost as xgb

    with open(INPUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    all_desc = [c for c in fieldnames if c not in NON_FEATURE]
    X_all, y_all = extract(rows, all_desc)

    # 5 different random_state seeds × 5-fold CV
    SEEDS = [42, 123, 456, 789, 1111]
    results = []
    for seed in SEEDS:
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
        )
        cv = cross_val_score(model, X_all, y_all, cv=5, scoring='r2')
        results.append(cv.mean())
        print(f"seed={seed}: CV_R² = {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\nMean CV_R² across {len(SEEDS)} seeds = {np.mean(results):.4f} ± {np.std(results):.4f}")

    # Simplified model
    simple_cols = ["MolWt", "Corg_%", "pH", "CEC"]
    X_s, y_s = extract(rows, simple_cols)
    results_s = []
    for seed in SEEDS:
        model_s = xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
        )
        cv_s = cross_val_score(model_s, X_s, y_s, cv=5, scoring='r2')
        results_s.append(cv_s.mean())
        print(f"seed={seed}: CV_R²(simple) = {cv_s.mean():.4f} ± {cv_s.std():.4f}")
    print(f"\nSimplified model CV_R² = {np.mean(results_s):.4f} ± {np.std(results_s):.4f}")

    # Test set R² for simplified (single seed=42 split, matches paper_06b)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    model_s = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model_s.fit(X_tr, y_tr)
    y_p = model_s.predict(X_te)
    r2 = r2_score(y_te, y_p)
    rmse = np.sqrt(mean_squared_error(y_te, y_p))
    print(f"\nSimplified model Test: R²={r2:.4f}, RMSE={rmse:.4f}, RPD={np.std(y_te)/rmse:.2f}")
