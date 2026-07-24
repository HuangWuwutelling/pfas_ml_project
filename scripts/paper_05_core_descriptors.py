#!/usr/bin/env python3
"""
S5_core_descriptors_analysis.py
================================
core descriptor filter + relationship analysis

Part 1: fromSHAP Top 20 select core descriptors, build simplified model
Part 2: MolWt/ vs Kdrelationship analysis, interpret cluster results

input: data/paper/feature_matrix_kd.csv
output: data/paper/kd_simplified_results.csv
      data/paper/kd_core_model_comparison.png
      data/paper/kd_molwt_vs_logkd.png
      data/paper/kd_descriptor_correlation.csv
"""

import csv
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
INPUT_FILE = os.path.join(SI_DIR, "feature_matrix_kd.csv")
SHAP_FILE = os.path.join(SI_DIR, "kd_shap_importance.csv")
OUTPUT_RESULTS = os.path.join(SI_DIR, "kd_simplified_results.csv")
OUTPUT_CORR = os.path.join(SI_DIR, "kd_descriptor_correlation.csv")

NON_FEATURE = {"PFAS_name", "log_Kd", "Kd_L_kg", "log_Koc"}
SOIL_FEATURES = {"Corg_%", "foc", "pH", "Sand", "Silt", "Clay", "CEC", "Fe_g_kg", "Al_g_kg"}


def load_data():
    """load_data + fromSHAPget feature ranking"""
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    fieldnames = reader.fieldnames
    
    # loadedSHAP sort result
    shap_ranking = []
    if os.path.exists(SHAP_FILE):
        with open(SHAP_FILE, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                shap_ranking.append((row["feature"], float(row["mean_abs_shap"])))
        # by contribution, descending
        shap_ranking.sort(key=lambda x: -x[1])
    
    print(f"  data: {len(all_rows)} row × {len(fieldnames)} column")
    print(f"  SHAP sort: {len(shap_ranking)} features")
    show_top = [f"{f}: {v:.4f}" for f, v in shap_ranking[:15]]
    print(f"  Top 15 SHAP feature: {', '.join(show_top)}")
    
    return all_rows, fieldnames, shap_ranking


def extract_xy(rows, fieldnames, feature_names):
    """extract feature matrix from dataXandtargety"""
    all_desc = [c for c in fieldnames if c not in NON_FEATURE]
    
    # iffeaturesubset, take only these
    if feature_names is not None:
        use_cols = [c for c in feature_names if c in all_desc]
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
    
    # removeyNaNrow
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    
    # fill by columnNaN
    for j in range(p):
        col_vals = X[:, j]
        median_val = np.nanmedian(col_vals)
        col_vals = np.nan_to_num(col_vals, nan=median_val)
        X[:, j] = col_vals
    
    return X, y, use_cols


def filtered_feature_set(rows, fieldnames, shap_ranking, filter_type="top5", 
                          exclude_soil=True, corr_threshold=0.95):
    """
    generate feature subsets by strategy. 
    
    filter_type: 
      'top5' - SHAP5
      'top10' - SHAP10
      'top20' - SHAP20
      'nosoil' - remove soil features
      'lowcorr' - remove high-correlation(r>0.95)then takeTop
      'alldesc' - all descriptors(excluding soil)
    """
    all_desc = [c for c in fieldnames if c not in NON_FEATURE]
    
    if filter_type == "all":
        return all_desc
    
    # fromSHAPtake from rankingTop N
    top_n = {"top5": 5, "top10": 10, "top20": 20}
    if filter_type in top_n:
        top_feats = [f for f, _ in shap_ranking[:top_n[filter_type]]]
        # ifall_desc
        result = [f for f in top_feats if f in all_desc]
        if exclude_soil:
            result = [f for f in result if f not in SOIL_FEATURES]
        return result
    
    if filter_type == "nosoil":
        return [f for f in all_desc if f not in SOIL_FEATURES]
    
    if filter_type == "lowcorr":
        # first fromSHAP Top 30, remove high-correlation
        candidates = [f for f, _ in shap_ranking[:30] if f in all_desc]
        if exclude_soil:
            candidates = [f for f in candidates if f not in SOIL_FEATURES]
        
        X, y, _ = extract_xy(rows, fieldnames, candidates)
        corr_mat = np.abs(np.corrcoef(X.T))
        
        # greedy selection: starting from most important, remove highly correlated with selected(r>0.95)
        selected = []
        for i, feat in enumerate(candidates[:min(30, len(candidates))]):
            if not selected:
                selected.append(feat)
                continue
            # check max correlation with selected features
            idx = candidates.index(feat) if feat in candidates else i
            if idx >= X.shape[1]:
                continue
            max_corr = max(abs(np.corrcoef(X[:, idx], X[:, j])[0, 1]) 
                          for j in [candidates.index(s) for s in selected 
                                   if s in candidates and candidates.index(s) < X.shape[1]
                                   and idx < X.shape[1]]
                          if idx < X.shape[1])
            if max_corr < corr_threshold:
                selected.append(feat)
        
        print(f"  low-correlation filter: {len(candidates)}candidate → {len(selected)}selected")
        return selected
    
    return all_desc  # default all


def train_xgb(X, y, feature_names, model_label):
    """trainXGBoostand evaluate"""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    import xgboost as xgb
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    sd = np.std(y_test)
    rpd = sd / rmse if rmse > 0 else float('inf')
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
    
    return {
        "model": model_label,
        "n_features": X.shape[1],
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "rpd": round(rpd, 2),
        "cv_r2": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
    }


def run_simplified_models(rows, fieldnames, shap_ranking):
    """run multiple simplified model variants"""
    print("\n" + "=" * 60)
    print("  Part 1: core descriptor filter + simplified model comparison")
    print("=" * 60)
    
    model_configs = [
        ("Top 5 SHAP (molecular descriptors)", "top5", True),
        ("Top 10 SHAP (molecular descriptors)", "top10", True),
        ("Top 20 SHAP (molecular descriptors)", "top20", True),
        ("Low-correlation Top 30 (>0.95 removed)", "lowcorr", True),
        ("RDKit descriptors only (225)", "nosoil", True),
        ("All features (RDKit + soil)", "all", False),
        ("Corg + pH + CEC only", None, False),
    ]

    all_results = []
    for label, ftype, exclude_soil in model_configs:
        if ftype is None:
            # manual: Corg + pH + CEC only
            features = ["Corg_%", "pH", "CEC"]
        else:
            features = filtered_feature_set(rows, fieldnames, shap_ranking, 
                                            filter_type=ftype, exclude_soil=exclude_soil)
        
        X, y, feat_names = extract_xy(rows, fieldnames, features)
        print(f"\n  [{label}] {X.shape[1]} features, {len(y)} samples")
        
        result = train_xgb(X, y, feat_names, label)
        all_results.append(result)
        
        if X.shape[1] <= 10:
            print(f"    feature: {feat_names}")
        print(f"    R²={result['r2']:.4f}, RMSE={result['rmse']:.4f}, RPD={result['rpd']:.2f}")
    
    # save results
    with open(OUTPUT_RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "n_features", "r2", "rmse", "rpd", "cv_r2", "cv_std"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n✅ Simplified modelresult: {OUTPUT_RESULTS}")
    
    return all_results


def analyze_structure_kd_relationship(rows, fieldnames):
    """Part 2: analyze MolWt andKdrelationship"""
    print("\n" + "=" * 60)
    print("  Part 2: molecular structure vs Kdrelationship analysis")
    print("=" * 60)
    
    # byPFASname aggregation: log Kd + PFASsubfamily + load from descriptors filemolwt, carbon_count, fluorine_count
    from collections import defaultdict
    
    # load meanlog Kd
    pfas_logkd = {}
    for row in rows:
        name = row["PFAS_name"].strip()
        v = row.get("log_Kd", "").strip()
        if v:
            pfas_logkd[name] = pfas_logkd.get(name, 0.0) + float(v)
    pfas_count = {}
    for row in rows:
        name = row["PFAS_name"].strip()
        pfas_count[name] = pfas_count.get(name, 0) + 1
    pfas_mean = {n: pfas_logkd[n] / pfas_count[n] for n in pfas_logkd}
    
    # from the 1row extractMolWt(all rows identical, because samePFASmolecular descriptorssame)
    pfas_molwt = {}
    pfas_carbon = {}
    pfas_fluorine = {}
    for row in rows:
        name = row["PFAS_name"].strip()
        if name not in pfas_molwt:
            v = row.get("MolWt", "").strip()
            pfas_molwt[name] = float(v) if v else np.nan
            v = row.get("carbon_count", "").strip()
            pfas_carbon[name] = float(v) if v else np.nan
            v = row.get("fluorine_count", "").strip()
            pfas_fluorine[name] = float(v) if v else np.nan
    
    # subfamily(fromdata)
    # use existingCSVhassubfamilyinfofile
    from collections import Counter
    
    # aggregate by subfamily
    with open(SI_DIR + "/PFAS_Properties.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        subfam_map = {}
        for row in reader:
            name = row["PFAS abbreviation"].strip()
            subfam = row["subfamily"].strip()
            subfam_map[name] = subfam
    
    # print per-subfamilyMolWt vs log Kd
    print(f"\n  aggregate by subfamily:")
    print(f"  {'subfamily':<20} {'n':<6} {'MolWtrange':<18} {'log Kdrange':<16} {'correlation(r)':<10}")
    print("  " + "-" * 70)
    
    from scipy.stats import pearsonr
    
    # aggregate by subfamily
    subfam_groups = defaultdict(list)
    for name in pfas_mean:
        subfam = subfam_map.get(name, "Other")
        if name in pfas_molwt and not np.isnan(pfas_molwt[name]):
            subfam_groups[subfam].append({
                "name": name,
                "molwt": pfas_molwt[name],
                "logkd": pfas_mean[name],
                "carbon": pfas_carbon.get(name, np.nan),
                "fluorine": pfas_fluorine.get(name, np.nan),
            })
    
    correlation_results = []
    for subfam, items in sorted(subfam_groups.items(), key=lambda x: -len(x[1])):
        n = len(items)
        if n < 3:
            continue
        molwts = np.array([it["molwt"] for it in items])
        logkds = np.array([it["logkd"] for it in items])
        carbons = np.array([it["carbon"] for it in items])
        fluorines = np.array([it["fluorine"] for it in items])
        
        r, p = pearsonr(molwts, logkds)
        wt_range = f"{molwts.min():.0f}-{molwts.max():.0f}"
        kd_range = f"{logkds.min():.2f}-{logkds.max():.2f}"
        print(f"  {subfam:<20} {n:<6} {wt_range:<18} {kd_range:<16} {r:.3f}")
        
        # also computes carbon chain vs log Kd
        valid_c = ~np.isnan(carbons)
        r_c = r_c_f = 0
        if valid_c.sum() >= 3:
            r_c, _ = pearsonr(carbons[valid_c], logkds[valid_c])
            r_c_f, _ = pearsonr(fluorines[valid_c], logkds[valid_c])
        else:
            r_c, r_c_f = 0, 0
        
        correlation_results.append({
            "subfamily": subfam,
            "n": n,
            "molwt_vs_logkd_r": round(r, 3),
            "molwt_vs_logkd_p": f"{p:.2e}",
            "carbon_vs_logkd_r": round(r_c, 3),
            "fluorine_vs_logkd_r": round(r_c_f, 3),
        })
    
    # save correlation table
    if correlation_results:
        corr_path = os.path.join(SI_DIR, "kd_structure_correlation.csv")
        with open(corr_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=correlation_results[0].keys())
            writer.writeheader()
            writer.writerows(correlation_results)
        print(f"\n✅ structure correlation: {corr_path}")
    
    # setstatistics(fromLevel 2 verify file)
    print(f"\n  key findings:")
    print(f"  PFCA 12 homologs: MolWt vs log Kd correlation")
    for items in [subfam_groups.get("PFCA", [])]:
        if len(items) >= 3:
            m = np.array([it["molwt"] for it in items])
            l = np.array([it["logkd"] for it in items])
            c = np.array([it["carbon"] for it in items])
            r_ml, p_ml = pearsonr(m, l)
            r_cl, _ = pearsonr(c, l)
            print(f"    MolWt vs log Kd: r={r_ml:.3f} (p={p_ml:.2e})")
            print(f"    C number vs log Kd: r={r_cl:.3f}")
            print(f"    rationale: per additional 1 CF2, log Kd increases by {np.polyfit(c, l, 1)[0]:.3f}")
    
    return subfam_groups


def save_visualizations(all_results, subfam_groups, fieldnames):
    """save visualization"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # ===== fig1: Simplified modelcomparison bar chart =====
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # R²for
        ax = axes[0]
        labels = [r["model"][:20] for r in all_results]
        r2_vals = [r["r2"] for r in all_results]
        colors_bar = ["#4daf4a" if "Top" in r["model"] or "" in r["model"] 
                      else ("#377eb8" if "soil" in r["model"] else "#e41a1c") 
                      for r in all_results]
        bars = ax.barh(range(len(labels)), r2_vals, color=colors_bar, alpha=0.8)
        ax.axvline(0.8679, color="red", linestyle="--", alpha=0.5, label="Full model (0.868)")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("R²")
        ax.set_title("Model Performance Comparison")
        ax.legend(fontsize=7)
        
        # annotate value on bar
        for bar, val in zip(bars, r2_vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", fontsize=7)
        
        # RPDfor
        ax = axes[1]
        rpd_vals = [r["rpd"] for r in all_results]
        bars = ax.barh(range(len(labels)), rpd_vals, color=colors_bar, alpha=0.8)
        ax.axvline(2.75, color="red", linestyle="--", alpha=0.5, label="Full model (2.75)")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("RPD")
        ax.set_title("RPD (Ratio of Performance to Deviation)")
        ax.legend(fontsize=7)
        for bar, val in zip(bars, rpd_vals):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   f"{val:.2f}", va="center", fontsize=7)
        
        plt.tight_layout()
        figpath = os.path.join(SI_DIR, "kd_core_model_comparison.png")
        plt.savefig(figpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\n✅ fig1 (model comparison): {figpath}")
        
        # ===== fig2: MolWt vs log Kd (color by subfamily) =====
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # define subfamily colors
        cmap = plt.cm.Set2(np.linspace(0, 1, 12))
        subfam_colors = {}
        for i, (sf, items) in enumerate(sorted(subfam_groups.items(), key=lambda x: -len(x[1]))):
            if len(items) >= 2:
                subfam_colors[sf] = cmap[i % len(cmap)]
                m = [it["molwt"] for it in items]
                l = [it["logkd"] for it in items]
                ax.scatter(m, l, c=[cmap[i % len(cmap)]], s=80, alpha=0.8, 
                          edgecolors="k", linewidth=0.5, label=f"{sf} (n={len(items)})")
                # PFASname
                for it in items:
                    ax.annotate(it["name"], (it["molwt"], it["logkd"]),
                               fontsize=6, alpha=0.7, ha="center", va="bottom")
        
        ax.set_xlabel("Molecular Weight (g/mol)")
        ax.set_ylabel("Mean log Kd")
        ax.set_title("Molecular Weight vs log Kd by PFAS subfamily")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        
        plt.tight_layout()
        figpath = os.path.join(SI_DIR, "kd_molwt_vs_logkd.png")
        plt.savefig(figpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ fig2 (MolWt vs log Kd): {figpath}")
        
    except ImportError:
        print("  ⚠️ matplotlibnot installed, skip figures")
    except Exception as e:
        print(f"  ⚠️ figure save failed: {e}")


def main():
    print("=" * 60)
    print("  Core Descriptors + Structure-Kd Analysis")
    print("=" * 60)
    
    rows, fieldnames, shap_ranking = load_data()
    
    # Part 1: Simplified model
    all_results = run_simplified_models(rows, fieldnames, shap_ranking)
    
    # Part 2: structure-Kdrelationship
    subfam_groups = analyze_structure_kd_relationship(rows, fieldnames)
    
    # visualization
    save_visualizations(all_results, subfam_groups, fieldnames)
    
    print(f"\n✅ S5 complete!!")
    print(f"  result: {OUTPUT_RESULTS}")
    print(f"  correlation: {OUTPUT_CORR}")


if __name__ == "__main__":
    main()
