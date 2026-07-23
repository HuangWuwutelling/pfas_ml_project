#!/usr/bin/env python3
"""
S6_leave_one_out_validation.py
===============================
留一化合物交叉验证 (Leave-One-PFAS-Out Cross Validation)

核心问题:
  模型能否预测"从未见过"的PFAS的Kd？
  随机80/20分割可能高估性能（同一PFAS可能同时出现在训练和测试集）。

方法:
  循环47次（每1种PFAS做一次测试集）:
    训练集: 除该PFAS外所有数据
    测试集: 该PFAS的全部数据
    评估: R², RMSE, RPD

对比:
  - Model A: 仅RDKit描述符 (MolWt只2个)
  - Model C: RDKit + 土壤性质

输入: data/paper/feature_matrix_kd.csv
输出: data/paper/kd_leave_one_out_results.csv
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

NON_FEATURE = {"PFAS_name", "log_Kd", "Kd_L_kg", "log_Koc"}
SOIL_FEATURES = {"Corg_%", "foc", "pH", "Sand", "Silt", "Clay", "CEC", "Fe_g_kg", "Al_g_kg"}


def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    fieldnames = reader.fieldnames
    
    # 按PFAS名称分组
    from collections import defaultdict
    pfas_groups = defaultdict(list)
    for row in rows:
        name = row["PFAS_name"].strip()
        pfas_groups[name].append(row)
    
    print(f"  总数据: {len(rows)} 行 × {len(fieldnames)} 列")
    print(f"  PFAS种类: {len(pfas_groups)}")
    for name, group in sorted(pfas_groups.items(), key=lambda x: -len(x[1])):
        print(f"    {name}: {len(group)} 条")
    
    return rows, fieldnames, pfas_groups


def extract_data(rows, fieldnames, feature_subset=None):
    """从行列表提取X矩阵和y向量"""
    all_desc = [c for c in fieldnames if c not in NON_FEATURE]
    
    if feature_subset == "desc_only":
        # 仅分子描述符 (排除土壤)
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
    
    # 去掉y缺失的行
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    
    # 填充NaN
    for j in range(p):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)):
            continue
        median_val = np.nanmedian(col_vals)
        col_vals = np.nan_to_num(col_vals, nan=median_val)
        X[:, j] = col_vals
    
    return X, y, use_cols


def run_loo(rows, fieldnames, pfas_groups, feature_subset, model_label):
    """运行留一化合物验证"""
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.model_selection import cross_val_score
    
    print(f"\n{'='*60}")
    print(f"  留一化合物验证: {model_label}")
    print(f"{'='*60}")
    
    # 预提取所有数据
    all_rows_list = []
    for name in sorted(pfas_groups.keys()):
        all_rows_list.extend(pfas_groups[name])
    
    results = []
    all_y_true = []
    all_y_pred = []
    
    # 每种PFAS一个循环
    pfas_list = sorted(pfas_groups.keys())
    n_pfas = len(pfas_list)
    
    for i, test_pfas in enumerate(pfas_list):
        # 训练集 = 所有非test_pfas的行
        train_rows = []
        for name, group in pfas_groups.items():
            if name != test_pfas:
                train_rows.extend(group)
        
        test_rows = pfas_groups[test_pfas]
        
        # 提取矩阵
        X_train, y_train, _ = extract_data(train_rows, fieldnames, feature_subset)
        X_test, y_test, feat_names = extract_data(test_rows, fieldnames, feature_subset)
        
        # 特征对齐：确保test集的特征与train集完全一致
        # (extract_data已经保证了顺序一致)
        
        n_train = len(y_train)
        n_test = len(y_test)
        
        if n_test == 0 or n_train < 10:
            continue
        
        # 训练
        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 评价
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # RPD (用训练集的标准差作为参考)
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
        
        # 给每个PFAS的R²标注好坏
        perf_tag = "✅" if r2 > 0 else "⚠️"
        print(f"  [{i+1}/{n_pfas}] {perf_tag} {test_pfas:<20} "
              f"n_test={n_test:<3} R²={r2:.3f} RMSE={rmse:.4f} RPD={rpd:.2f}")
        
        if r2 < -0.5:
            print(f"           ⚠️  r²={r2:.2f} — 该PFAS可能与其他PFAS结构差异大")
    
    # 汇总统计
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    overall_r2 = r2_score(all_y_true, all_y_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    
    avg_r2 = np.mean([r["r2"] for r in results])
    median_r2 = np.median([r["r2"] for r in results])
    std_r2 = np.std([r["r2"] for r in results])
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
    
    print(f"\n  📊 汇总 ({model_label}):")
    print(f"     Overall R² = {overall_r2:.4f} (所有点合并)")
    print(f"     平均每PFAS R² = {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"    中位数R² = {median_r2:.4f}")
    print(f"    正R²占比: {n_positive}/{n_total}")
    
    return results, summary, (all_y_true, all_y_pred)


def save_visualization(all_results_list, all_predictions_list, model_labels):
    """保存可视化"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        n_models = len(all_results_list)
        
        # ===== 图1: 每PFAS的R²柱状图 =====
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
            ax.set_title(f"Leave-One-PFAS-Out R² — {label}")
            
            # 标注好中差
            good = sum(1 for v in r2_vals if v > 0.5)
            med = sum(1 for v in r2_vals if 0 < v <= 0.5)
            bad = sum(1 for v in r2_vals if v <= 0)
            ax.text(0.95, 0.95, f"Good(>0.5):{good} Medium:{med} Poor(≤0):{bad}",
                   transform=ax.transAxes, fontsize=9, ha="right", va="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            
            # 在柱上标数值
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
        print(f"\n✅ 图表: {figpath}")
        
        # ===== 图2: 预测 vs 真实散点图（全部点合并） =====
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
        print(f"✅ 图表: {figpath}")
        
    except ImportError:
        print("  ⚠️ matplotlib未安装，跳过图表")
    except Exception as e:
        print(f"  ⚠️ 图表保存失败: {e}")


def main():
    print("=" * 60)
    print("  Leave-One-PFAS-Out Cross Validation")
    print("=" * 60)
    
    rows, fieldnames, pfas_groups = load_data()
    
    # 运行两个模型的留一验证
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
        
        # 保存每PFAS的详细结果
        out_res = OUTPUT_RESULTS.replace(".csv", f"_{model_label.split()[0].lower()}.csv")
        with open(out_res, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  ✅ 详细结果: {out_res}")
    
    # 保存汇总
    with open(OUTPUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\n  ✅ 汇总: {OUTPUT_SUMMARY}")
    
    # 可视化
    save_visualization(all_results, all_predictions, [m[1] for m in model_configs])
    
    print(f"\n✅ S6 完成!")


if __name__ == "__main__":
    main()
