#!/usr/bin/env python3
"""
S6_leave_one_out_validation.py
===============================
leave-one-compound cross-validation (Leave-One-PFAS-Out Cross Validation)

core question:
  can model predict"never seen"PFASKd? 
  80/20 split may overestimate performance(PFASmay appear in both train and test sets). 

method:
  loop 47(1PFASdo one test set):
    training set: exceptPFASall data outside
    test set: PFASalldata
    eval: R², RMSE, RPD

for:
  - Model A: onlyRDKitdescriptors (MolWtonly2)
  - Model C: RDKit + soil properties

input: data/paper/feature_matrix_kd.csv
output: data/paper/kd_leave_one_out_results.csv
      data/paper/kd_leave_one_out_summary.csv
      data/paper/kd_leave_one_out.png
"""

import csv
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
INPUT_FILE = os.path.join(SI_DIR, "feature_matrix_kd.csv")
OUTPUT_RESULTS = os.path.join(SI_DIR, "kd_leave_one_out_results.csv")
OUTPUT_SUMMARY = os.path.join(SI_DIR, "kd_leave_one_out_summary.csv")

# Shared constants (centralised in scripts/_shared_config.py)
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared_config import NON_FEATURE, SOIL_FEATURES


def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    fieldnames = reader.fieldnames
    
    # byPFASname group
    from collections import defaultdict
    pfas_groups = defaultdict(list)
    for row in rows:
        name = row["PFAS_name"].strip()
        pfas_groups[name].append(row)
    
    print(f"  totaldata: {len(rows)} row × {len(fieldnames)} column")
    print(f"  PFAStypes: {len(pfas_groups)}")
    for name, group in sorted(pfas_groups.items(), key=lambda x: -len(x[1])):
        print(f"    {name}: {len(group)} rows")
    
    return rows, fieldnames, pfas_groups


def extract_data(rows, fieldnames, feature_subset=None):
    """extract from row listXandy"""
    all_desc = [c for c in fieldnames if c not in NON_FEATURE]
    
    if feature_subset == "desc_only":
        # molecular descriptors only (exclude soil)
        use_cols = [c for c in all_desc if c not in SOIL_FEATURES]
    elif feature_subset == "combined":
        use_cols = all_desc
    else:
        use_cols = all_desc
    
    n = len(rows)
    p = len(use_cols)
    X = np.zeros((n, p))
    y = np.full(n, np.nan)
    
    for j, col in enumerate(use_cols):
        for i, row in enumerate(rows):
            v = row.get(col, "").strip()
            if v:
                try:
                    X[i, j] = float(v)
                except:
                    X[i, j] = np.nan
    
    for i, row in enumerate(rows):
        v = row.get("log_Kd", "").strip()
        if v:
            try:
                y[i] = float(v)
            except:
                pass
    
    # removeymissingrow
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    
    # imputationNaN
    for j in range(p):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)):
            continue
        median_val = np.nanmedian(col_vals)
        col_vals = np.nan_to_num(col_vals, nan=median_val)
        X[:, j] = col_vals
    
    return X, y, use_cols


def run_loo(rows, fieldnames, pfas_groups, feature_subset, model_label):
    """run leave-one-compound validation"""
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.model_selection import cross_val_score
    
    print(f"\n{'='*60}")
    print(f"  leave-one-compound validation: {model_label}")
    print(f"{'='*60}")
    
    # pre-extract all data
    all_rows_list = []
    for name in sorted(pfas_groups.keys()):
        all_rows_list.extend(pfas_groups[name])
    
    results = []
    all_y_true = []
    all_y_pred = []
    
    # PFASloop
    pfas_list = sorted(pfas_groups.keys())
    n_pfas = len(pfas_list)
    
    for i, test_pfas in enumerate(pfas_list):
        # training set = all non-test_pfasrow
        train_rows = []
        for name, group in pfas_groups.items():
            if name != test_pfas:
                train_rows.extend(group)
        
        test_rows = pfas_groups[test_pfas]
        
        # extract matrix
        X_train, y_train, _ = extract_data(train_rows, fieldnames, feature_subset)
        X_test, y_test, feat_names = extract_data(test_rows, fieldnames, feature_subset)
        
        # feature alignment: ensuretestset features andtrainset completely consistent
        # (extract_dataorder already guaranteed consistent)
        
        n_train = len(y_train)
        n_test = len(y_test)
        
        if n_test == 0 or n_train < 10:
            continue
        
        # train
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # RPD (use training set std as reference)
        train_sd = np.std(y_train)
        rpd = train_sd / rmse if rmse > 0 else float('inf')
        
        results.append({
            "test_pfas": test_pfas,
            "n_train": n_train,
            "n_test": n_test,
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "rpd": round(rpd, 2),
            "y_true_mean": round(float(np.mean(y_test)), 3),
            "y_pred_mean": round(float(np.mean(y_pred)), 3),
        })
        
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        
        # eachPFASR²label good/bad
        perf_tag = "✅" if r2 > 0 else "⚠️"
        print(f"  [{i+1}/{n_pfas}] {perf_tag} {test_pfas:<20} "
              f"n_test={n_test:<3} R²={r2:.3f} RMSE={rmse:.4f} RPD={rpd:.2f}")
        
        if r2 < -0.5:
            print(f"           ⚠️  r²={r2:.2f} - PFAS structure differs from other PFAS in dataset")
    
    # summary statistics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    overall_r2 = r2_score(all_y_true, all_y_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    
    # nan-safe aggregates: a single-sample test fold (n_test == 1, e.g. 4:2
    # FTOH) yields R^2 = nan because the denominator is zero. Falling
    # back to nanmean/nanmedian/nanstd keeps those rows in the count but
    # out of the aggregate; alternatively they could be filtered upstream.
    r2_array = np.array([r["r2"] for r in results], dtype=float)
    avg_r2 = float(np.nanmean(r2_array))
    median_r2 = float(np.nanmedian(r2_array))
    std_r2 = float(np.nanstd(r2_array))
    n_positive = sum(1 for r in results if r["r2"] > 0)
    n_total = len(results)
    
    summary = {
        "model": model_label,
        "n_pfas": n_pfas,
        "n_total_samples": len(all_y_true),
        "overall_r2": round(overall_r2, 4),
        "overall_rmse": round(overall_rmse, 4),
        "overall_mae": round(overall_mae, 4),
        "avg_r2": round(avg_r2, 4),
        "median_r2": round(median_r2, 4),
        "std_r2": round(std_r2, 4),
        "positive_r2_ratio": f"{n_positive}/{n_total}",
    }
    
    print(f"\n  📊 summary ({model_label}):")
    print(f"     Overall R² = {overall_r2:.4f} (all points merged)")
    print(f"     mean per-PFAS R² = {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"    median R² = {median_r2:.4f}")
    print(f"    R²: {n_positive}/{n_total}")
    
    return results, summary, (all_y_true, all_y_pred)


def save_visualization(all_results_list, all_predictions_list, model_labels):
    """save visualization"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        n_models = len(all_results_list)
        
        # ===== fig1: PFASR²bar chart =====
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        if n_models == 1:
            axes = [axes]
        
        for idx, (results, label) in enumerate(zip(all_results_list, model_labels)):
            ax = axes[idx]
            results_sorted = sorted(results, key=lambda x: x["r2"])
            names = [r["test_pfas"] for r in results_sorted]
            r2_vals = [r["r2"] for r in results_sorted]
            colors = ["#e41a1c" if v < 0 else "#4daf4a" for v in r2_vals]
            
            bars = ax.barh(range(len(names)), r2_vals, color=colors, alpha=0.7)
            ax.axvline(0, color="gray", linestyle="-", lw=0.5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel("R²")
            ax.set_title(f"Leave-One-PFAS-Out R² - {label}")
            
            # label good/medium/bad
            good = sum(1 for v in r2_vals if v > 0.5)
            med = sum(1 for v in r2_vals if 0 < v <= 0.5)
            bad = sum(1 for v in r2_vals if v <= 0)
            ax.text(0.95, 0.95, f"Good(>0.5):{good} Medium:{med} Poor(≤0):{bad}",
                   transform=ax.transAxes, fontsize=9, ha="right", va="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            
            # annotate value on bar
            for bar, val in zip(bars, r2_vals):
                if val < -1:
                    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                           f"{val:.2f}", va="center", fontsize=5)
                else:
                    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                           f"{val:.3f}", va="center", fontsize=6)
        
        plt.tight_layout()
        figpath = os.path.join(SI_DIR, "kd_leave_one_out.png")
        plt.savefig(figpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\n✅ fig: {figpath}")
        
        # ===== fig2:  vs ptfig(all points merged) =====
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, ((y_true, y_pred), label) in enumerate(zip(all_predictions_list, model_labels)):
            ax = axes[idx]
            ax.scatter(y_true, y_pred, alpha=0.4, s=10)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1, alpha=0.5)
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            ax.set_xlabel("Observed log Kd")
            ax.set_ylabel("Predicted log Kd")
            ax.set_title(f"{label}\nOverall R²={r2:.3f}, RMSE={rmse:.3f}")
            ax.set_aspect("equal")
        
        plt.tight_layout()
        figpath = os.path.join(SI_DIR, "kd_loo_predicted_vs_actual.png")
        plt.savefig(figpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ fig: {figpath}")
        
    except ImportError:
        print("  ⚠️ matplotlibnot installed, skip figures")
    except Exception as e:
        print(f"  ⚠️ figure save failed: {e}")


def main():
    print("=" * 60)
    print("  Leave-One-PFAS-Out Cross Validation")
    print("=" * 60)
    
    rows, fieldnames, pfas_groups = load_data()
    
    # run LOO validation for two models
    all_results = []
    all_summaries = []
    all_predictions = []
    model_configs = [
        ("desc_only", "RDKit descriptors only"),
        ("combined", "RDKit + soil properties"),
    ]
    
    for feature_subset, model_label in model_configs:
        results, summary, predictions = run_loo(
            rows, fieldnames, pfas_groups, feature_subset, model_label
        )
        all_results.append(results)
        all_summaries.append(summary)
        all_predictions.append(predictions)
        
        # save eachPFASdetailed result
        out_res = OUTPUT_RESULTS.replace(".csv", f"_{model_label.split()[0].lower()}.csv")
        with open(out_res, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  ✅ detailed result: {out_res}")
    
    # save summary
    with open(OUTPUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\n  ✅ summary: {OUTPUT_SUMMARY}")
    
    # visualization
    save_visualization(all_results, all_predictions, [m[1] for m in model_configs])
    
    print(f"\n✅ S6 complete!")


if __name__ == "__main__":
    main()
