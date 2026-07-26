#!/usr/bin/env python3
"""
S6b_loo_combined_fix.py
re-runCombinedmodel LOO validation, detailCSV
also confirm simplified model (MolWt + Corg + pH + CEC) R²
"""
import csv, os, sys, numpy as np, warnings
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared_config import NON_FEATURE, SOIL_FEATURES
warnings.filterwarnings("ignore")

SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
INPUT = os.path.join(SI_DIR, "feature_matrix_kd.csv")
OUT_COMBINED = os.path.join(SI_DIR, "kd_leave_one_out_results_combined.csv")

NON_FEATURE = {"PFAS_name","log_Kd","Kd_L_kg","log_Koc"}
SOIL_FEATURES = {"Corg_%","foc","pH","Sand","Silt","Clay","CEC","Fe_g_kg","Al_g_kg"}

# ── loaded ──
from collections import defaultdict
with open(INPUT, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

pfas_groups = defaultdict(list)
for row in rows:
    pfas_groups[row["PFAS_name"].strip()].append(row)

print(f"totaldata: {len(rows)} row, PFAStypes: {len(pfas_groups)}")

# ── extract data ──
def extract(rows_list, use_soil=True):
    fieldnames = list(rows_list[0].keys())
    all_desc = [c for c in fieldnames if c not in NON_FEATURE]
    use_cols = all_desc if use_soil else [c for c in all_desc if c not in SOIL_FEATURES]
    n, p = len(rows_list), len(use_cols)
    X = np.full((n, p), np.nan)
    y = np.full(n, np.nan)
    for j, col in enumerate(use_cols):
        for i, row in enumerate(rows_list):
            v = row.get(col, "").strip()
            if v:
                try: X[i,j] = float(v)
                except: pass
    for i, row in enumerate(rows_list):
        v = row.get("log_Kd", "").strip()
        if v:
            try: y[i] = float(v)
            except: pass
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    for j in range(p):
        col_vals = X[:,j]
        if np.all(np.isnan(col_vals)): continue
        X[:,j] = np.nan_to_num(col_vals, nan=np.nanmedian(col_vals))
    return X, y, use_cols

if __name__ == "__main__":
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    pfas_list = sorted(pfas_groups.keys())
    n_pfas = len(pfas_list)

    all_results = []
    combined_y_true = []
    combined_y_pred = []

    # Also compute simplified model (MolWt + Corg + pH + CEC)
    def extract_simplified(rows_list):
        """extract simplified features: MolWt, Corg_%, pH, CEC"""
        # first find indices of these columns
        # We need to work from the feature matrix directly
        from collections import OrderedDict
        # Just grab the columns we need
        n = len(rows_list)
        X = np.full((n, 4), np.nan)
        y = np.full(n, np.nan)
        for i, row in enumerate(rows_list):
            for j, col in enumerate(["MolWt","Corg_%","pH","CEC"]):
                v = row.get(col, "").strip()
                if v:
                    try: X[i,j] = float(v)
                    except: pass
            v = row.get("log_Kd", "").strip()
            if v:
                try: y[i] = float(v)
                except: pass
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]
        for j in range(4):
            col_vals = X[:,j]
            if np.all(np.isnan(col_vals)): continue
            X[:,j] = np.nan_to_num(col_vals, nan=np.nanmedian(col_vals))
        return X, y

    print("\n=== Combined LOO ===")
    for i, test_pfas in enumerate(pfas_list):
        train_rows = []
        for name, group in pfas_groups.items():
            if name != test_pfas:
                train_rows.extend(group)
        test_rows = pfas_groups[test_pfas]
    
        X_train, y_train, _ = extract(train_rows, use_soil=True)
        X_test, y_test, _ = extract(test_rows, use_soil=True)
    
        if len(y_test) == 0 or len(y_train) < 10:
            continue
    
        model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        train_sd = np.std(y_train)
        rpd = train_sd / rmse if rmse > 0 else float('inf')
    
        combined_y_true.extend(y_test.tolist())
        combined_y_pred.extend(y_pred.tolist())
    
        perf = "✅" if r2 > 0 else "⚠️"
        print(f"  [{i+1}/{n_pfas}] {perf} {test_pfas:<20} n_test={len(y_test):<3} R²={r2:.3f}")
    
        all_results.append({
            "test_pfas": test_pfas, "n_train": len(y_train), "n_test": len(y_test),
            "r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
            "rpd": round(rpd, 2),
            "y_true_mean": round(float(np.mean(y_test)), 3),
            "y_pred_mean": round(float(np.mean(y_pred)), 3),
        })

    # Save
    with open(OUT_COMBINED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n✅ save: {OUT_COMBINED}")

    # Combined LOO summary
    combined_r2 = np.array([r["r2"] for r in all_results])
    combined_n_test = np.array([r["n_test"] for r in all_results])
    overall_r2 = r2_score(combined_y_true, combined_y_pred)

    print(f"\n=== Combined LOO summary ===")
    print(f"  Overall R² (pooled) = {overall_r2:.4f}")
    print(f"  R² = {np.mean(combined_r2):.4f}")
    print(f"  median R² = {np.median(combined_r2):.4f}")
    print(f"  R²: {sum(combined_r2 > 0)}/{len(combined_r2)}")

    # n≥10 vs n<10
    large = combined_r2[combined_n_test >= 10]
    small = combined_r2[combined_n_test < 10]
    print(f"  n≥10 mean R² = {np.mean(large):.4f} (n={len(large)})")
    print(f"  n<10 mean R² = {np.mean(small):.4f} (n={len(small)})")

    # R² > 0.5 / 0~0.5 / <0
    good = sum(combined_r2 > 0.5)
    med = sum((combined_r2 > 0) & (combined_r2 <= 0.5))
    poor = sum(combined_r2 <= 0)
    print(f"  distribution: good(>0.5)={good} medium(0~0.5)={med} bad(<=0)={poor}")

    # worst 5
    idx = np.argsort(combined_r2)
    print(f"\n  bad 5:")
    for j in idx[:5]:
        r = all_results[j]
        print(f"    ⚠️ {r['test_pfas']:<18} R²={r['r2']:.3f} n={r['n_test']}")

    # saveLOOaggregate to kd_leave_one_out_summary.csv(S6oldresult)
    OUT_SUMMARY = os.path.join(SI_DIR, "kd_leave_one_out_summary.csv")
    n_positive = sum(combined_r2 > 0)
    n_total = len(combined_r2)
    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "n_pfas", "n_total_samples", "overall_r2",
            "overall_rmse", "overall_mae", "avg_r2", "median_r2",
            "std_r2", "positive_r2_ratio"
        ])
        writer.writeheader()
        writer.writerow({
            "model": "RDKit + soil properties",
            "n_pfas": n_pfas,
            "n_total_samples": len(combined_y_true),
            "overall_r2": round(overall_r2, 4),
            "overall_rmse": round(np.sqrt(mean_squared_error(combined_y_true, combined_y_pred)), 4),
            "overall_mae": round(np.mean(np.abs(np.array(combined_y_true) - np.array(combined_y_pred))), 4),
            "avg_r2": round(np.mean(combined_r2), 4),
            "median_r2": round(np.median(combined_r2), 4),
            "std_r2": round(np.std(combined_r2), 4),
            "positive_r2_ratio": f"{n_positive}/{n_total}",
        })
    print(f"  ✅ LOOsummary saved: {OUT_SUMMARY}")

    # ── Simplified model: all data MolWt + Corg + pH + CEC ──
    # (single seed=42 result, and SI Table S2 consistent)
    print("\n\n=== Simplified model: MolWt + Corg + pH + CEC ===\n")
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error

    all_rows_list = []
    for name in sorted(pfas_groups.keys()):
        all_rows_list.extend(pfas_groups[name])

    X_simple, y_simple = extract_simplified(all_rows_list)
    print(f"  data: {len(y_simple)} samples")

    X_tr, X_te, y_tr, y_te = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)
    model_s = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                random_state=42, n_jobs=-1)
    model_s.fit(X_tr, y_tr)
    y_pred_s = model_s.predict(X_te)
    r2_s = r2_score(y_te, y_pred_s)
    rmse_s = np.sqrt(mean_squared_error(y_te, y_pred_s))
    rpd_s = np.std(y_te) / rmse_s

    cv_s = cross_val_score(model_s, X_simple, y_simple, cv=5, scoring='r2')

    print(f"  Simplified model (MolWt+Corg+pH+CEC):")
    print(f"    Test R² = {r2_s:.4f}")
    print(f"    RMSE = {rmse_s:.4f}")
    print(f"    RPD = {rpd_s:.2f}")
    print(f"    5-fold CV: mean={cv_s.mean():.4f} ± {cv_s.std():.4f}")
    print(f"    fraction of full model (0.868): {r2_s/0.868*100:.1f}%")

    # full model (all features) CV confirm
    X_all, y_all, _ = extract(all_rows_list, use_soil=True)
    model_full = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, n_jobs=-1)
    cv_full = cross_val_score(model_full, X_all, y_all, cv=5, scoring='r2')
    print(f"\n  full model 5-fold CV: mean={cv_full.mean():.4f} ± {cv_full.std():.4f}")

    # append simplified model results to kd_simplified_results.csv(auxiliary record)
    SIMPLIFIED_OUT = os.path.join(SI_DIR, "kd_simplified_results.csv")
    if os.path.exists(SIMPLIFIED_OUT):
        with open(SIMPLIFIED_OUT, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"MolWt+Corg+pH+CEC",
                4,
                round(r2_s, 4),
                round(rmse_s, 4),
                round(rpd_s, 2),
                round(cv_s.mean(), 4),
                round(cv_s.std(), 4),
            ])
        print(f"  ✅ appended to {SIMPLIFIED_OUT}")