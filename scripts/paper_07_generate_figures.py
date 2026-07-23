#!/usr/bin/env python3
"""
generate_figures.py
===================
为修改后的论文生成全部 Figure 和 Table。

Figure 1: Predicted vs. observed log Kd (Combined model, test set)
Figure 2: Model performance comparison bar chart (R² and RPD)
Figure 3: Molecular weight vs. log Kd, colored by subfamily
Figure 4: Leave-one-PFAS-out R² per compound
Figure 5: Joint t-SNE chemical space (11K background + 47 Kd-labeled)
Figure 6: HDBSCAN clusters + OECD class comparison (from existing 08 script output)

Table 1: Model performance summary
Table 2: Intra-subfamily MolWt-Kd correlations

输出: paper/figures/ 和 paper/tables/ 下
"""

import csv
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SI_DIR = os.path.join(PROJECT, "data", "paper")
FIG_DIR = os.path.join(PROJECT, "paper", "figures")
TBL_DIR = os.path.join(PROJECT, "paper", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

FEATURE_FILE = os.path.join(SI_DIR, "feature_matrix_kd.csv")
SIMPLIFIED_FILE = os.path.join(SI_DIR, "kd_simplified_results.csv")
LOO_RDKIT_FILE = os.path.join(SI_DIR, "kd_leave_one_out_results_rdkit.csv")
LOO_COMBINED_FILE = os.path.join(SI_DIR, "kd_leave_one_out_results_combined.csv")
LOO_SUMMARY_FILE = os.path.join(SI_DIR, "kd_leave_one_out_summary.csv")
STRUCTURE_CORR_FILE = os.path.join(SI_DIR, "kd_structure_correlation.csv")
PFAS_PROP_FILE = os.path.join(SI_DIR, "PFAS_Properties.csv")

# ══════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════

def load_feature_matrix():
    with open(FEATURE_FILE) as f:
        reader = csv.DictReader(f)
        return list(reader)

def load_simplified_results():
    with open(SIMPLIFIED_FILE) as f:
        return list(csv.DictReader(f))

def load_loo_results():
    # Try combined file first
    fn = LOO_COMBINED_FILE if os.path.exists(LOO_COMBINED_FILE) else LOO_RDKIT_FILE
    with open(fn) as f:
        return list(csv.DictReader(f))

def load_loo_summary():
    with open(LOO_SUMMARY_FILE) as f:
        return list(csv.DictReader(f))

def load_subfamily_map():
    map_ = {}
    with open(PFAS_PROP_FILE) as f:
        for row in csv.DictReader(f):
            name = row["PFAS abbreviation"].strip()
            subfam = row["subfamily"].strip()
            map_[name] = subfam
    return map_

# ══════════════════════════════════════════════════════════
#  FIGURE 1: Predicted vs observed log Kd
# ══════════════════════════════════════════════════════════

def fig1_predicted_vs_actual(rows):
    """Generate pred vs actual scatter plot for Combined model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error

    # Prepare data: we need to retrain Combined model
    from collections import defaultdict
    NON_FEATURE = {"PFAS_name", "log_Kd", "Kd_L_kg", "log_Koc", "_n_soil_missing"}
    
    fieldnames = list(rows[0].keys())
    desc_cols = [c for c in fieldnames if c not in NON_FEATURE]
    
    n = len(rows)
    p = len(desc_cols)
    X = np.zeros((n, p))
    y = np.full(n, np.nan)
    names = []
    
    for j, col in enumerate(desc_cols):
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
        names.append(row["PFAS_name"].strip())
    
    valid = ~np.isnan(y)
    X, y, names = X[valid], y[valid], [names[i] for i, v in enumerate(valid) if v]
    
    # Impute
    for j in range(p):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)):
            continue
        median_val = np.nanmedian(col_vals)
        col_vals = np.nan_to_num(col_vals, nan=median_val)
        X[:, j] = col_vals
    
    # Split
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, np.array(names), test_size=0.2, random_state=42
    )
    
    # Train
    model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rpd = np.std(y_test) / rmse
    
    # ===== Plot =====
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Color by subfamily
    subfam_map = load_subfamily_map()
    colors = plt.cm.tab10(np.linspace(0, 1, 12))
    subfam_list = sorted(set(subfam_map.get(n, "Other") for n in names_test if subfam_map.get(n, "")))
    subfam_color = {s: colors[i % len(colors)] for i, s in enumerate(subfam_list)}
    
    for i, name in enumerate(names_test):
        sf = subfam_map.get(name, "Other")
        c = subfam_color.get(sf, "gray") if sf in subfam_color else "gray"
        ax.scatter(y_test[i], y_pred[i], c=[c], s=30, alpha=0.6, edgecolors="none")
    
    # Diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1, alpha=0.5)
    
    # Text box
    ax.text(0.05, 0.95, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nRPD = {rpd:.2f}\nn = {len(y_test)}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    ax.set_xlabel("Observed log Kd", fontsize=11)
    ax.set_ylabel("Predicted log Kd (Combined Model)", fontsize=11)
    ax.set_title("Figure 1. Predicted vs. Observed log Kd", fontsize=12)
    ax.set_aspect("equal")
    
    # Legend
    legend_handles = []
    for sf in subfam_list[:8]:  # Limit to avoid crowding
        legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                         markerfacecolor=subfam_color[sf], markersize=6, label=sf))
    ax.legend(handles=legend_handles, fontsize=6, ncol=2, loc="lower right")
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_predicted_vs_actual.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Fig 1: {path}")


# ══════════════════════════════════════════════════════════
#  FIGURE 2: Model comparison bar chart
# ══════════════════════════════════════════════════════════

def fig2_model_comparison():
    """Bar chart comparing R² and RPD across models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    results = load_simplified_results()
    
    # Filter to the key models
    key_models = [
        "Model A: RDKit only",
        "Model B: Soil only", 
        "Model C: Combined",
        "Top 5 SHAP (分子描述符)",
        "仅Corg + pH + CEC",
    ]
    
    filtered = [r for r in results if r["model"] in key_models]
    # Map names
    name_map = {
        "Model A: RDKit only": "RDKit only (136 feat.)",
        "Model B: Soil only": "Soil only (9 feat.)",
        "Model C: Combined": "Combined (145 feat.)",
        "Top 5 SHAP (分子描述符)": "Top 2 MolWt feat.",
        "仅Corg + pH + CEC": "Corg+pH+CEC",
    }
    
    labels = [name_map.get(r["model"], r["model"]) for r in filtered]
    r2 = [float(r["r2"]) for r in filtered]
    rpd = [float(r["rpd"]) if r["rpd"] else 0 for r in filtered]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    colors = ["#4daf4a", "#377eb8", "#e41a1c", "#ff7f00", "#984ea3"]
    
    # R²
    bars1 = ax1.barh(range(len(labels)), r2, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("R²", fontsize=11)
    ax1.set_title("(a) R² on Test Set", fontsize=11)
    ax1.set_xlim(0, 1)
    for bar, val in zip(bars1, r2):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    
    # RPD
    bars2 = ax2.barh(range(len(labels)), rpd, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("RPD", fontsize=11)
    ax2.set_title("(b) RPD (Ratio of Performance to Deviation)", fontsize=11)
    ax2.axvline(3.16, color="gray", linestyle="--", alpha=0.5, label="PFASorptionML (2025)")
    ax2.legend(fontsize=7)
    for bar, val in zip(bars2, rpd):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_model_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Fig 2: {path}")


# ══════════════════════════════════════════════════════════
#  FIGURE 3: Molecular weight vs log Kd by subfamily
#  (Reuse S5's cached result)
# ══════════════════════════════════════════════════════════

def fig3_molwt_vs_logkd():
    """MolWt vs log Kd scatter, colored by subfamily."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    rows = load_feature_matrix()
    subfam_map = load_subfamily_map()
    
    # Aggregate per PFAS
    pfas_data = defaultdict(lambda: {"logkd_sum": 0, "n": 0, "molwt": None, "subfam": ""})
    for row in rows:
        name = row["PFAS_name"].strip()
        v = row.get("log_Kd", "").strip()
        if v:
            try:
                val = float(v)
                pfas_data[name]["logkd_sum"] += val
                pfas_data[name]["n"] += 1
            except:
                pass
        if pfas_data[name]["molwt"] is None:
            v2 = row.get("MolWt", "").strip()
            if v2:
                try:
                    pfas_data[name]["molwt"] = float(v2)
                except:
                    pass
        if not pfas_data[name]["subfam"]:
            pfas_data[name]["subfam"] = subfam_map.get(name, "Other")
    
    # Compute means
    plot_data = []
    for name, data in pfas_data.items():
        if data["molwt"] and data["n"] > 0:
            plot_data.append({
                "name": name,
                "molwt": data["molwt"],
                "logkd": data["logkd_sum"] / data["n"],
                "subfam": data["subfam"],
            })
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Group by subfamily
    subfam_groups = defaultdict(list)
    for d in plot_data:
        subfam_groups[d["subfam"]].append(d)
    
    colors = plt.cm.Set2(np.linspace(0, 1, 12))
    color_idx = 0
    
    for sf, items in sorted(subfam_groups.items(), key=lambda x: -len(x[1])):
        if len(items) < 2:
            continue
        m = [d["molwt"] for d in items]
        l = [d["logkd"] for d in items]
        c = colors[color_idx % len(colors)]
        color_idx += 1
        
        ax.scatter(m, l, c=[c], s=80, alpha=0.8, edgecolors="k", linewidth=0.5,
                   label=f"{sf} (n={len(items)}, r={np.corrcoef(m, l)[0,1]:.2f})" if len(items) >= 3 else f"{sf} (n={len(items)})")
        
        # Annotate
        for d in items:
            ax.annotate(d["name"], (d["molwt"], d["logkd"]),
                       fontsize=5.5, alpha=0.75, ha="center", va="bottom")
    
    ax.set_xlabel("Molecular Weight (g/mol)", fontsize=11)
    ax.set_ylabel("Mean log Kd", fontsize=11)
    ax.set_title("Figure 3. Molecular Weight vs. log Kd by PFAS Subfamily", fontsize=12)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_molwt_vs_logkd.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Fig 3: {path}")


# ══════════════════════════════════════════════════════════
#  FIGURE 4: Leave-one-PFAS-out R² bar chart
# ══════════════════════════════════════════════════════════

def fig4_loo_bar():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Use combined LOO results (matches paper's main text)
    results = load_loo_results()
    
    # Sort by R²
    sorted_res = sorted(results, key=lambda r: float(r["r2"]))
    names = [r["test_pfas"] for r in sorted_res]
    r2 = [float(r["r2"]) for r in sorted_res]
    n_test = [int(r["n_test"]) for r in sorted_res]
    
    colors = []
    for v, n in zip(r2, n_test):
        if v > 0.5:
            colors.append("#4daf4a")  # green
        elif v > 0:
            colors.append("#ffd700")  # yellow
        elif n < 5:
            colors.append("#b0b0b0")  # gray (insufficient data)
        else:
            colors.append("#e41a1c")  # red
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(range(len(names)), r2, color=colors, alpha=0.8, edgecolor="gray", linewidth=0.3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("R² (Leave-One-PFAS-Out)", fontsize=11)
    ax.set_title("Figure 4. Leave-One-PFAS-Out Cross-Validation Results", fontsize=12)
    ax.set_xlim(-4, 1.1)
    
    # Annotate n_test
    for i, (bar, n) in enumerate(zip(bars, n_test)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f"n={n}", va="center", fontsize=5, color="gray")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4daf4a", label="Good (R² > 0.5)"),
        Patch(facecolor="#ffd700", label="Moderate (0 < R² ≤ 0.5)"),
        Patch(facecolor="#e41a1c", label="Poor (R² ≤ 0)"),
        Patch(facecolor="#b0b0b0", label="Insufficient data (n < 5)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower left")
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_loo_validation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Fig 4: {path}")


# ══════════════════════════════════════════════════════════
#  FIGURE 5: Joint t-SNE chemical space
#  (Reuse S4_fix_chemical_space.py's output, just copy)
# ══════════════════════════════════════════════════════════

def fig5_chemical_space():
    import shutil
    src = os.path.join(SI_DIR, "kd_chemical_space_annotated.png")
    dst = os.path.join(FIG_DIR, "fig5_chemical_space.png")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✅ Fig 5: {dst} (copied from S4 output)")
    else:
        print(f"  ⚠️ Fig 5: source not found at {src}")


# ══════════════════════════════════════════════════════════
#  FIGURE 6: HDBSCAN clusters (reuse existing 08 result)
# ══════════════════════════════════════════════════════════

def fig6_clusters():
    import shutil
    # The existing t-SNE cluster figure
    src = os.path.join(FIG_DIR, "cluster_t-sne.png")
    dst = os.path.join(FIG_DIR, "fig6_cluster_tsne.png")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✅ Fig 6: {dst} (from existing 08 script output)")
    else:
        print(f"  ⚠️ Fig 6: source not found at {src}")


# ══════════════════════════════════════════════════════════
#  TABLE 1: Model performance summary
# ══════════════════════════════════════════════════════════

def table1_model_performance():
    # Combine simplified_results + loo_summary
    simplified = load_simplified_results()
    loo_summary = load_loo_summary()
    
    with open(os.path.join(TBL_DIR, "table1_model_performance.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "n_features", "R²_test", "RMSE", "RPD", "CV_R²", "LOO_R²"])
        for r in simplified:
            model = r["model"]
            nf = r["n_features"]
            r2 = r["r2"]
            rmse = r["rmse"]
            rpd = r["rpd"]
            cv_r2 = r.get("cv_r2", "")
            # Find LOO_R² — only meaningful for descriptor-only and full-combined models
            loo_r2 = ""
            # 新格式: 只有1行 "RDKit + soil properties" → Combined model LOO
            # old format: 2行 "RDKit descriptors only" + "RDKit + soil properties"
            if len(loo_summary) == 1:
                # 只有combined LOO的行
                loo_row = loo_summary[0]
                if "soil" in model.lower() or "所有特征" in model.lower():
                    loo_r2 = loo_row["overall_r2"]
            else:
                # 旧格式兼容: 2行模型类型
                for s in loo_summary:
                    sm = s["model"].lower()
                    ml = model.lower()
                    if "rdkit" in sm and "only" in sm and ("top" in ml or "描述符" in ml or "去掉土壤" in ml):
                        loo_r2 = s["overall_r2"]
                    elif "soil" in sm and "所有特征" in ml:
                        loo_r2 = s["overall_r2"]
            writer.writerow([model, nf, r2, rmse, rpd, cv_r2, loo_r2])
    
    print(f"  ✅ Table 1: {os.path.join(TBL_DIR, 'table1_model_performance.csv')}")


# ══════════════════════════════════════════════════════════
#  TABLE 2: Intra-subfamily MolWt-Kd correlations
# ══════════════════════════════════════════════════════════

def table2_subfamily_correlations():
    from scipy.stats import pearsonr
    from collections import defaultdict
    
    rows = load_feature_matrix()
    subfam_map = load_subfamily_map()
    
    # Aggregate per PFAS
    pfas_data = defaultdict(lambda: {"logkd_sum": 0, "n": 0, "molwt": None, "subfam": ""})
    for row in rows:
        name = row["PFAS_name"].strip()
        v = row.get("log_Kd", "").strip()
        if v:
            try:
                val = float(v)
                pfas_data[name]["logkd_sum"] += val
                pfas_data[name]["n"] += 1
            except:
                pass
        if pfas_data[name]["molwt"] is None:
            v2 = row.get("MolWt", "").strip()
            if v2:
                try:
                    pfas_data[name]["molwt"] = float(v2)
                except:
                    pass
        if not pfas_data[name]["subfam"]:
            pfas_data[name]["subfam"] = subfam_map.get(name, "Other")
    
    results = []
    for sf in sorted(set(d["subfam"] for d in pfas_data.values())):
        items = [d for d in pfas_data.values() if d["subfam"] == sf and d["molwt"] and d["n"] > 0]
        if len(items) < 3:
            continue
        m = np.array([d["molwt"] for d in items])
        l = np.array([d["logkd_sum"] / d["n"] for d in items])
        r, p = pearsonr(m, l)
        slope = np.polyfit(m, l, 1)[0]
        results.append({
            "subfamily": sf,
            "n_compounds": len(items),
            "pearson_r": round(r, 3),
            "p_value": f"{p:.2e}",
            "slope_logKd_per_100MW": round(slope * 100, 3),
        })
    
    with open(os.path.join(TBL_DIR, "table2_subfamily_correlations.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subfamily", "n_compounds", "pearson_r", "p_value", "slope_logKd_per_100MW"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"  ✅ Table 2: {os.path.join(TBL_DIR, 'table2_subfamily_correlations.csv')}")


# ══════════════════════════════════════════════════════════
#  SI FIGURES (S3-S6)
# ══════════════════════════════════════════════════════════

def figS3_shap_bar(rows):
    """Figure S3: SHAP bar plot for Combined XGBoost model (Top 15 features)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    
    # Prepare data (same as fig1)
    NON_FEATURE = {"PFAS_name", "log_Kd", "Kd_L_kg", "log_Koc", "_n_soil_missing"}
    fieldnames = list(rows[0].keys())
    desc_cols = [c for c in fieldnames if c not in NON_FEATURE]
    
    n = len(rows)
    p = len(desc_cols)
    X = np.zeros((n, p))
    y = np.full(n, np.nan)
    for j, col in enumerate(desc_cols):
        for i, row in enumerate(rows):
            v = row.get(col, "").strip()
            if v:
                try: X[i, j] = float(v)
                except: X[i, j] = np.nan
    for i, row in enumerate(rows):
        v = row.get("log_Kd", "").strip()
        if v:
            try: y[i] = float(v)
            except: pass
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    for j in range(p):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)): continue
        X[:, j] = np.nan_to_num(col_vals, nan=np.nanmedian(col_vals))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    import shap
    X_test_np = np.array(X_test, dtype=np.float64)
    if X_test_np.ndim == 1:
        X_test_np = X_test_np.reshape(-1, 1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_np, check_additivity=False)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_n = 15
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_names = [desc_cols[i] for i in top_idx]
    top_vals = [mean_abs_shap[i] for i in top_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdBu_r(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), top_vals, color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (impact on model output)", fontsize=10)
    ax.set_title("Figure S3. SHAP Feature Importance — Combined Model", fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figS3_shap_bar_kd.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def figS4_shap_beeswarm(rows):
    """Figure S4: SHAP beeswarm summary plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    import shap
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    NON_FEATURE = {"PFAS_name", "log_Kd", "Kd_L_kg", "log_Koc", "_n_soil_missing"}
    fieldnames = list(rows[0].keys())
    desc_cols = [c for c in fieldnames if c not in NON_FEATURE]
    
    n = len(rows)
    p = len(desc_cols)
    X = np.zeros((n, p))
    y = np.full(n, np.nan)
    for j, col in enumerate(desc_cols):
        for i, row in enumerate(rows):
            v = row.get(col, "").strip()
            if v:
                try: X[i, j] = float(v)
                except: X[i, j] = np.nan
    for i, row in enumerate(rows):
        v = row.get("log_Kd", "").strip()
        if v:
            try: y[i] = float(v)
            except: pass
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    for j in range(p):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)): continue
        X[:, j] = np.nan_to_num(col_vals, nan=np.nanmedian(col_vals))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    X_test_np = np.array(X_test, dtype=np.float64)
    if X_test_np.ndim == 1:
        X_test_np = X_test_np.reshape(-1, 1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_np, check_additivity=False)
    
    shap.summary_plot(shap_values, X_test_np, feature_names=desc_cols,
                      show=False, max_display=20, alpha=0.6,
                      cmap=plt.get_cmap("RdBu_r"))
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    path = os.path.join(FIG_DIR, "figS4_shap_beeswarm_kd.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def figS5_simplified_scatter(rows):
    """Figure S5: Simplified model (MolWt + Corg + pH + CEC) pred vs actual."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    
    simple_cols = ["MolWt", "Corg_%", "pH", "CEC"]
    n = len(rows)
    p = len(simple_cols)
    X = np.zeros((n, p))
    y = np.full(n, np.nan)
    for j, col in enumerate(simple_cols):
        for i, row in enumerate(rows):
            v = row.get(col, "").strip()
            if v:
                try: X[i, j] = float(v)
                except: X[i, j] = np.nan
    for i, row in enumerate(rows):
        v = row.get("log_Kd", "").strip()
        if v:
            try: y[i] = float(v)
            except: pass
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    for j in range(p):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)): continue
        X[:, j] = np.nan_to_num(col_vals, nan=np.nanmedian(col_vals))
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    rpd = np.std(y_te) / rmse
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_te, y_pred, alpha=0.5, s=15, c="#377eb8", edgecolors="none")
    min_v = min(y_te.min(), y_pred.min())
    max_v = max(y_te.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=1, alpha=0.5)
    ax.text(0.05, 0.95, f"R\u00b2 = {r2:.4f}\nRMSE = {rmse:.4f}\nRPD = {rpd:.2f}\nn = {len(y_te)}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.set_xlabel("Observed log Kd", fontsize=11)
    ax.set_ylabel("Predicted log Kd (simplified)", fontsize=11)
    ax.set_title("Figure S5. Simplified Model (MolWt + Corg + pH + CEC)", fontsize=11)
    ax.set_aspect("equal")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figS5_simplified_model.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


def figS6_subfamily_faceted(rows):
    """Figure S6: Faceted MolWt-Kd scatter by subfamily with regression lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from scipy.stats import linregress
    
    subfam_map = load_subfamily_map()
    
    pfas_data = defaultdict(lambda: {"logkd_vals": [], "molwt": None, "subfam": ""})
    for row in rows:
        name = row["PFAS_name"].strip()
        v = row.get("log_Kd", "").strip()
        if v:
            try: pfas_data[name]["logkd_vals"].append(float(v))
            except: pass
        if pfas_data[name]["molwt"] is None:
            v2 = row.get("MolWt", "").strip()
            if v2:
                try: pfas_data[name]["molwt"] = float(v2)
                except: pass
        if not pfas_data[name]["subfam"]:
            pfas_data[name]["subfam"] = subfam_map.get(name, "Other")
    
    subfam_groups = defaultdict(list)
    for name, data in pfas_data.items():
        if data["molwt"] and data["logkd_vals"]:
            subfam_groups[data["subfam"]].append({
                "name": name, "molwt": data["molwt"],
                "logkd": np.mean(data["logkd_vals"]),
            })
    
    subfam_order = sorted(subfam_groups.keys(), key=lambda x: -len(subfam_groups[x]))
    n_sub = len(subfam_order)
    ncols = 3
    nrows = (n_sub + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
    
    for ax, sf in zip(axes_flat, subfam_order):
        items = subfam_groups[sf]
        m = np.array([d["molwt"] for d in items])
        l = np.array([d["logkd"] for d in items])
        ax.scatter(m, l, s=40, alpha=0.7, edgecolors="k", linewidth=0.5)
        for d in items:
            ax.annotate(d["name"], (d["molwt"], d["logkd"]),
                       fontsize=6, alpha=0.7, ha="center", va="bottom")
        if len(items) >= 3:
            slope, intercept, r_val, p_val, _ = linregress(m, l)
            x_line = np.linspace(m.min(), m.max(), 50)
            ax.plot(x_line, slope * x_line + intercept, "r--", lw=1, alpha=0.6)
            ax.text(0.95, 0.05, f"r = {r_val:.3f}", transform=ax.transAxes,
                   fontsize=8, ha="right", va="bottom",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        ax.set_xlabel("MolWt (g/mol)", fontsize=8)
        ax.set_ylabel("log Kd", fontsize=8)
        ax.set_title(f"{sf} (n={len(items)})", fontsize=9)
        ax.tick_params(labelsize=7)
    
    for ax in axes_flat[n_sub:]:
        ax.set_visible(False)
    
    plt.suptitle("Figure S6. Molecular Weight vs. log Kd by PFAS Subfamily", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "figS6_subfamily_faceted.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Generating Figures and Tables")
    print("=" * 60)
    
    print("\n--- Loading data ---")
    rows = load_feature_matrix()
    print(f"  {len(rows)} rows loaded")
    
    print("\n--- Figure 1: Predicted vs Actual ---")
    fig1_predicted_vs_actual(rows)
    
    print("\n--- Figure 2: Model Comparison ---")
    fig2_model_comparison()
    
    print("\n--- Figure 3: MolWt vs log Kd ---")
    fig3_molwt_vs_logkd()
    
    print("\n--- Figure 4: LOO Validation ---")
    fig4_loo_bar()
    
    print("\n--- Figure 5: Chemical Space ---")
    fig5_chemical_space()
    
    print("\n--- Figure 6: Clusters ---")
    fig6_clusters()
    
    print("\n--- Table 1: Model Performance ---")
    table1_model_performance()
    
    print("\n--- Table 2: Subfamily Correlations ---")
    table2_subfamily_correlations()
    
    # ── SI Figures ──
    print("\n--- SI Figure S3: SHAP Bar Plot ---")
    figS3_shap_bar(rows)
    
    print("\n--- SI Figure S4: SHAP Beeswarm ---")
    figS4_shap_beeswarm(rows)
    
    print("\n--- SI Figure S5: Simplified Model Scatter ---")
    figS5_simplified_scatter(rows)
    
    print("\n--- SI Figure S6: Subfamily Faceted ---")
    figS6_subfamily_faceted(rows)
    
    print("\n" + "=" * 60)
    print("  All figures and tables generated!")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  Tables:  {TBL_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()