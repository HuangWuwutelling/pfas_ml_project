#!/usr/bin/env python3
"""
S3_model_kd.py
==============
XGBoost prediction of log Kd: three-model comparison experiment.

Model A: RDKit molecular descriptors only (225) — can molecular structure alone predict Kd?
Model B: Soil properties only (Corg, pH, Sand, Silt, Clay, CEC, Fe, Al)
Model C: Both combined — best performance

Evaluation: R², RMSE, MAE, RPD (Ratio of Performance to Deviation)

Reference: Fabregat-Palau et al. (2025) ES&T, RPD > 3.16

Input:  data/paper/feature_matrix_kd.csv
Output: data/paper/kd_model_results.csv (three-model performance comparison)
        data/paper/kd_shap_importance.csv (SHAP feature importance for Model C)
"""

import csv
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
INPUT_FILE = os.path.join(SI_DIR, "feature_matrix_kd.csv")
OUTPUT_RESULTS = os.path.join(SI_DIR, "kd_model_results.csv")
OUTPUT_SHAP = os.path.join(SI_DIR, "kd_shap_importance.csv")

# RDKit descriptor prefix (non-feature columns)
# Non-feature columns (unified naming, shared across all scripts)
NON_FEATURE = {"PFAS_name", "log_Kd", "Kd_L_kg", "log_Koc"}
SOIL_FEATURE_NAMES = ["Corg_%", "foc", "pH", "Sand", "Silt", "Clay", "CEC", "Fe_g_kg", "Al_g_kg"]


def load_data():
    """Load feature matrix."""
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  Loaded {len(rows)} rows x {len(reader.fieldnames)} columns")
    return rows, reader.fieldnames


def prepare_features(rows, all_fieldnames, feature_subset=None):
    """
    Extract feature matrix X and target y from data.

    Parameters:
      rows: list of csv rows
      all_fieldnames: all column names
      feature_subset: which features to use. None = all (excluding NON_FEATURE);
                      'soil' = soil only; 'desc_only' = molecular descriptors only
                      (excluding soil features)

    Returns: X, y, feature_names
    """
    # Determine feature columns
    desc_cols = [c for c in all_fieldnames if c not in NON_FEATURE]

    if feature_subset == "soil":
        # Keep only soil features
        feature_names = [c for c in desc_cols if c in SOIL_FEATURE_NAMES]
    elif feature_subset == "desc_only":
        # Keep only molecular descriptors (excluding soil features)
        feature_names = [c for c in desc_cols if c not in SOIL_FEATURE_NAMES]
    else:
        # All features (no exclusion); Model C includes Fe/Al.
        feature_names = desc_cols

    n = len(rows)
    p = len(feature_names)

    X = np.zeros((n, p))
    y = np.full(n, np.nan)
    nan_mask = np.zeros(n, dtype=bool)

    for j, col in enumerate(feature_names):
        for i, row in enumerate(rows):
            v = row.get(col, "").strip()
            if v:
                try:
                    X[i, j] = float(v)
                except (ValueError, TypeError):
                    X[i, j] = np.nan
            else:
                X[i, j] = np.nan
        # Check if entire column is NaN
        col_nan = np.isnan(X[:, j])
        if np.all(col_nan):
            print(f"  WARNING: column entirely missing: {col}")

    # Extract target variable
    for i, row in enumerate(rows):
        v = row.get("log_Kd", "").strip()
        if v:
            try:
                y[i] = float(v)
            except (ValueError, TypeError):
                pass

    # Drop rows with missing target
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]

    # Keep valid features by column (those not entirely NaN) — this is a
    # data-loading step, independent of any split, and only removes columns
    # that have no information. Real imputer/variance threshold are moved
    # into a sklearn.Pipeline in train_and_evaluate() to avoid train/test
    # information leakage.
    clean_features = []
    clean_cols = []
    for j, col in enumerate(feature_names):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)):
            print(f"  WARNING: column entirely missing: {col}")
            continue
        clean_features.append(col)
        clean_cols.append(j)

    X = X[:, clean_cols]

    print(f"  y: {len(y)} valid values, range=[{y.min():.2f}, {y.max():.2f}]")
    print(f"  X: {X.shape} ({len(clean_features)} features pre-imputation)")

    return X, y, clean_features


def train_and_evaluate(X, y, feature_names, model_label):
    """Train XGBoost, evaluate, and return result.

    Preprocessing (NaN median imputation, zero-variance column removal) is
    placed inside a sklearn.Pipeline so cross_val_score and train_test_split
    only fit on the training fold, preventing test-set information from leaking
    into the imputer/variance computation (this was the methodology hard
    issue flagged in the 2026-07-23 reproduction report).
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    import xgboost as xgb

    print(f"\n--- {model_label} ---")

    # Row-level random split (random_state=42 to match the original; this is
    # a random-split R² measuring prediction on held-out measurements of
    # *known* PFAS. For prediction on unseen PFAS, see LOO pooled R² ≈ 0.72).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: imputer + variance threshold + XGBoost (fit only on training fold)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=1e-10)),
        ("xgb", xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # RPD = SD / RMSE (paper's key metric)
    sd = np.std(y_test)
    rpd = sd / rmse if rmse > 0 else float('inf')

    # Cross-validation (Pipeline supports cv_score; imputer+variance fit only
    # on training portion per fold)
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)

    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE = {mae:.4f}")
    print(f"  RPD = {rpd:.2f} (paper: 3.16)")
    print(f"  R² CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test SD = {sd:.4f}")
    print(f"  n_train = {len(y_train)}, n_test = {len(y_test)}")

    # Feature importance (extracted from the Pipeline's XGBoost step)
    xgb_model = pipe.named_steps["xgb"]
    importance = xgb_model.feature_importances_
    # Feature names also need to be filtered by variance threshold (keep non-zero-variance columns)
    variance_mask = pipe.named_steps["variance"].get_support()
    filtered_feature_names = [f for f, keep in zip(feature_names, variance_mask) if keep]
    top_n = min(20, len(filtered_feature_names))
    top_idx = np.argsort(importance)[::-1][:top_n]
    print(f"  Top {top_n} features:")
    for idx in top_idx:
        print(f"    {filtered_feature_names[idx]}: {importance[idx]:.4f}")

    return {
        "model_label": model_label,
        "n_samples": len(y),
        "n_features": len(filtered_feature_names),
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "rpd": round(rpd, 2),
        "cv_r2": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "test_sd": round(sd, 4),
        "model": xgb_model,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": filtered_feature_names,
        "top_features": [(filtered_feature_names[i], float(importance[i])) for i in top_idx],
    }


def shap_analysis(model, X_test, feature_names, model_label):
    """SHAP analysis."""
    print(f"\n--- SHAP analysis: {model_label} ---")
    try:
        import shap
        # Ensure X_test is a 2D numpy float64 array
        X_test_np = np.array(X_test, dtype=np.float64)
        if X_test_np.ndim == 1:
            X_test_np = X_test_np.reshape(-1, 1)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_np, check_additivity=False)

        # Global feature importance (mean |SHAP|)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_n = min(30, len(feature_names))
        top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

        print(f"  Top {top_n} SHAP features:")
        shap_results = []
        for idx in top_idx:
            shap_results.append({
                "feature": feature_names[idx],
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 6),
            })
            print(f"    {feature_names[idx]}: {mean_abs_shap[idx]:.6f}")

        return shap_values, shap_results

    except ImportError:
        print("  WARNING: shap not installed, skipping SHAP analysis")
        return None, None


def main():
    print("=" * 60)
    print("  XGBoost prediction of log Kd — three-model comparison")
    print("=" * 60)

    # Load data
    rows, all_fieldnames = load_data()

    # ========== Model A: RDKit descriptors only ==========
    print("\n" + "=" * 60)
    print("  Model A: RDKit molecular descriptors only")
    print("=" * 60)
    X_a, y_a, feat_a = prepare_features(rows, all_fieldnames, feature_subset="desc_only")
    print(f"  Model A: {X_a.shape}")
    result_a = train_and_evaluate(X_a, y_a, feat_a, "Model A: RDKit only")

    # ========== Model B: Soil properties only ==========
    print("\n" + "=" * 60)
    print("  Model B: Soil properties only")
    print("=" * 60)
    X_b, y_b, feat_b = prepare_features(rows, all_fieldnames, feature_subset="soil")
    result_b = train_and_evaluate(X_b, y_b, feat_b, "Model B: Soil only")

    # ========== Model C: All features ==========
    print("\n" + "=" * 60)
    print("  Model C: RDKit descriptors + soil properties")
    print("=" * 60)
    X_c, y_c, feat_c = prepare_features(rows, all_fieldnames, feature_subset=None)
    result_c = train_and_evaluate(X_c, y_c, feat_c, "Model C: Combined")

    # ========== SHAP analysis (Model C) ==========
    # SHAP must be on preprocessed X (imputer + variance fit).
    # We reuse the same train_test_split seed; refit a fresh Pipeline to
    # expose the imputer and variance steps for SHAP, ensuring imputer
    # medians and variance mask come from the same training fold.
    print("\n--- Preparing test data for SHAP analysis ---")
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    import xgboost as xgb

    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
        X_c, y_c, test_size=0.2, random_state=42
    )
    pipe_for_shap = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=1e-10)),
        ("xgb", xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipe_for_shap.fit(X_c_train, y_c_train)
    # Take preprocessed test data (imputer + variance fit on training fold)
    X_c_test_transformed = pipe_for_shap[:-1].transform(X_c_test)
    print(f"  SHAP X_test (preprocessed): {X_c_test_transformed.shape}")

    # Feature names also need variance threshold filtering
    variance_mask_shap = pipe_for_shap.named_steps["variance"].get_support()
    feat_c_filtered = [f for f, keep in zip(feat_c, variance_mask_shap) if keep]

    shap_values, shap_results = shap_analysis(
        pipe_for_shap.named_steps["xgb"],
        X_c_test_transformed,
        feat_c_filtered,
        "Model C",
    )

    # ========== Save results ==========
    # Performance comparison table
    with open(OUTPUT_RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "n_samples", "n_features", "r2", "rmse", "mae",
            "rpd", "cv_r2", "cv_std", "test_sd"
        ])
        writer.writeheader()
        for r in [result_a, result_b, result_c]:
            writer.writerow({
                "model": r["model_label"],
                "n_samples": r["n_samples"],
                "n_features": r["n_features"],
                "r2": r["r2"],
                "rmse": r["rmse"],
                "mae": r["mae"],
                "rpd": r["rpd"],
                "cv_r2": r["cv_r2"],
                "cv_std": r["cv_std"],
                "test_sd": r["test_sd"],
            })
    print(f"\nResults: {OUTPUT_RESULTS}")

    # SHAP importance table
    if shap_results:
        with open(OUTPUT_SHAP, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature", "mean_abs_shap"])
            writer.writeheader()
            writer.writerows(shap_results)
        print(f"SHAP importance: {OUTPUT_SHAP}")

    # Print summary
    print("\n" + "=" * 60)
    print("  Three-model performance comparison")
    print("=" * 60)
    print(f"{'Model':<25} {'R²':<8} {'RMSE':<8} {'RPD':<8} {'n_feat':<6}")
    print("-" * 55)
    for r in [result_a, result_b, result_c]:
        print(f"{r['model_label']:<25} {r['r2']:<8.4f} {r['rmse']:<8.4f} {r['rpd']:<8.2f} {r['n_features']:<6}")
    print(f"\n  Paper (ES&T 2025): RPD > 3.16")

    print(f"\nS3 complete!")


if __name__ == "__main__":
    main()
