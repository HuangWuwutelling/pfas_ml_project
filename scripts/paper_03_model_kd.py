#!/usr/bin/env python3
"""
S3_model_kd.py
==============
XGBoost预测log Kd: 三组模型对比实验。

模型A: 只用RDKit分子描述符（225个）— 分子结构能否独立预测Kd？
模型B: 只用土壤性质（Corg, pH, Sand, Silt, Clay, CEC, Fe, Al）
模型C: 两者都用 — 最优效果

评估: R², RMSE, RPD (Ratio of Performance to Deviation)
    
参照: Fabregat-Palau et al. (2025) ES&T, RPD > 3.16

输入: data/paper/feature_matrix_kd.csv
输出: data/paper/kd_model_results.csv (三组模型性能对比)
      data/paper/kd_shap_importance.csv (Model C的SHAP特征重要性)
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

# RDKit描述符前缀（非特征列）
# 非特征列（统一命名，供所有脚本引用）
NON_FEATURE = {"PFAS_name", "log_Kd", "Kd_L_kg", "log_Koc"}
SOIL_FEATURE_NAMES = ["Corg_%", "foc", "pH", "Sand", "Silt", "Clay", "CEC", "Fe_g_kg", "Al_g_kg"]


def load_data():
    """加载特征矩阵"""
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  加载 {len(rows)} 行 × {len(reader.fieldnames)} 列")
    return rows, reader.fieldnames


def prepare_features(rows, all_fieldnames, feature_subset=None):
    """
    从数据中提取特征矩阵 X 和目标变量 y。
    
    参数:
      rows: csv行列表
      all_fieldnames: 所有列名
      feature_subset: 指定使用的特征子集。None=全部(除NON_FEATURE), 'soil'=只用土壤, 'desc_only'=只用分子描述符(不含土壤)
    
    返回: X, y, feature_names
    """
    # 确定特征列
    desc_cols = [c for c in all_fieldnames if c not in NON_FEATURE]
    
    if feature_subset == "soil":
        # 只保留土壤特征
        feature_names = [c for c in desc_cols if c in SOIL_FEATURE_NAMES]
    elif feature_subset == "desc_only":
        # 只保留分子描述符（不包含土壤特征）
        feature_names = [c for c in desc_cols if c not in SOIL_FEATURE_NAMES]
    else:
        # 全部特征，但排除以Fe_g_kg/Al_g_kg开头的(因为它们缺失较多)
        # 注：Model C也包含Fe/Al
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
        # 该列全部NaN的检查
        col_nan = np.isnan(X[:, j])
        if np.all(col_nan):
            print(f"  ⚠️ 列全部缺失: {col}")
    
    # 提取目标变量
    for i, row in enumerate(rows):
        v = row.get("log_Kd", "").strip()
        if v:
            try:
                y[i] = float(v)
            except (ValueError, TypeError):
                pass
    
    # 去掉目标值缺失的行
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    
    # 按列保留有效特征（非全部 NaN）— 这是数据加载步骤，不依赖任何 split，
    # 仅移除在整列上完全缺失的列（没有信息量）。真正的 imputer/variance
    # threshold 移到 train_and_evaluate() 里的 sklearn.Pipeline，避免
    # train/test 信息泄漏。
    clean_features = []
    clean_cols = []
    for j, col in enumerate(feature_names):
        col_vals = X[:, j]
        if np.all(np.isnan(col_vals)):
            print(f"  ⚠️ 列全部缺失: {col}")
            continue
        clean_features.append(col)
        clean_cols.append(j)

    X = X[:, clean_cols]

    print(f"  y: {len(y)} 有效值, range=[{y.min():.2f}, {y.max():.2f}]")
    print(f"  X: {X.shape} ({len(clean_features)} features pre-imputation)")

    return X, y, clean_features


def train_and_evaluate(X, y, feature_names, model_label):
    """训练XGBoost, 评估并返回结果

    预处理（NaN 中位数填补、零方差列剔除）放在 sklearn.Pipeline 里，
    这样 cross_val_score 和 train_test_split 都只在训练 fold 上 fit，
    避免 test 集信息泄漏到 imputer/variance 计算（这是 2026-07-23 复现
    报告指出的方法学硬伤）。
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    import xgboost as xgb

    print(f"\n--- {model_label} ---")

    # 行级随机划分（random_state=42 保持与原版一致；这是 random-split R²，
    # 衡量"对已知 PFAS 新测量值的预测能力"。对全新 PFAS 的预测请看
    # LOO pooled R² ≈ 0.72）。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: imputer + variance threshold + XGBoost（只在 train fold fit）
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
    
    # 指标
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # RPD = SD / RMSE (原文关键指标)
    sd = np.std(y_test)
    rpd = sd / rmse if rmse > 0 else float('inf')
    
    # 交叉验证（Pipeline 也支持 cv_score，每折内 imputer+variance 只 fit 训练部分）
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)

    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE = {mae:.4f}")
    print(f"  RPD = {rpd:.2f} (原文: 3.16)")
    print(f"  R² CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test SD = {sd:.4f}")
    print(f"  n_train = {len(y_train)}, n_test = {len(y_test)}")

    # 特征重要性（从 Pipeline 里提取 XGBoost）
    xgb_model = pipe.named_steps["xgb"]
    importance = xgb_model.feature_importances_
    # 特征名也要经过 variance threshold 过滤（保留非零方差列）
    variance_mask = pipe.named_steps["variance"].get_support()
    filtered_feature_names = [f for f, keep in zip(feature_names, variance_mask) if keep]
    top_n = min(20, len(filtered_feature_names))
    top_idx = np.argsort(importance)[::-1][:top_n]
    print(f"  Top {top_n} 特征:")
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
    """SHAP分析"""
    print(f"\n--- SHAP分析: {model_label} ---")
    try:
        import shap
        # 确保X_test是2D numpy float64数组
        X_test_np = np.array(X_test, dtype=np.float64)
        if X_test_np.ndim == 1:
            X_test_np = X_test_np.reshape(-1, 1)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_np, check_additivity=False)
        
        # 全局特征重要性(平均|SHAP|)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_n = min(30, len(feature_names))
        top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
        
        print(f"  Top {top_n} SHAP特征:")
        shap_results = []
        for idx in top_idx:
            shap_results.append({
                "feature": feature_names[idx],
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 6),
            })
            print(f"    {feature_names[idx]}: {mean_abs_shap[idx]:.6f}")
        
        return shap_values, shap_results
        
    except ImportError:
        print("  ⚠️ shap未安装, 跳过SHAP分析")
        return None, None


def main():
    print("=" * 60)
    print("  XGBoost预测log Kd — 三组模型对比")
    print("=" * 60)
    
    # 加载数据
    rows, all_fieldnames = load_data()
    
    # ========== 模型A: 只用RDKit描述符 ==========
    print("\n" + "=" * 60)
    print("  模型A: 只用RDKit分子描述符")
    print("=" * 60)
    X_a, y_a, feat_a = prepare_features(rows, all_fieldnames, feature_subset="desc_only")
    print(f"  Model A: {X_a.shape}")
    result_a = train_and_evaluate(X_a, y_a, feat_a, "Model A: RDKit only")
    
    # ========== 模型B: 只用土壤性质 ==========
    print("\n" + "=" * 60)
    print("  模型B: 只用土壤性质")
    print("=" * 60)
    X_b, y_b, feat_b = prepare_features(rows, all_fieldnames, feature_subset="soil")
    result_b = train_and_evaluate(X_b, y_b, feat_b, "Model B: Soil only")
    
    # ========== 模型C: 全部特征 ==========
    print("\n" + "=" * 60)
    print("  模型C: RDKit描述符 + 土壤性质")
    print("=" * 60)
    X_c, y_c, feat_c = prepare_features(rows, all_fieldnames, feature_subset=None)
    result_c = train_and_evaluate(X_c, y_c, feat_c, "Model C: Combined")
    
    # ========== SHAP分析(模型C) ==========
    # SHAP 必须在预处理过的 X 上做（imputer + variance 已 fit）。
    # 我们重用 train_and_evaluate() 里已经 fit 好的 pipe，
    # 但那里 pipe 是局部变量 — 重新跑一次 fit 以拿到 pipe 实例供 SHAP 用。
    # 这样保证 imputer 的中位数和 variance 的 mask 来自同一训练 fold。
    print("\n--- 准备SHAP分析的测试数据 ---")
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
    # 取经过预处理的测试数据（imputer + variance 都已 fit 在训练 fold 上）
    X_c_test_transformed = pipe_for_shap[:-1].transform(X_c_test)
    print(f"  SHAP X_test (preprocessed): {X_c_test_transformed.shape}")

    # 特征名也要经过 variance threshold 过滤
    variance_mask_shap = pipe_for_shap.named_steps["variance"].get_support()
    feat_c_filtered = [f for f, keep in zip(feat_c, variance_mask_shap) if keep]

    shap_values, shap_results = shap_analysis(
        pipe_for_shap.named_steps["xgb"],
        X_c_test_transformed,
        feat_c_filtered,
        "Model C",
    )
    
    # ========== 保存结果 ==========
    # 性能对比表
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
    print(f"\n✅ 结果: {OUTPUT_RESULTS}")
    
    # SHAP重要性表
    if shap_results:
        with open(OUTPUT_SHAP, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature", "mean_abs_shap"])
            writer.writeheader()
            writer.writerows(shap_results)
        print(f"✅ SHAP重要性: {OUTPUT_SHAP}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("  三组模型性能对比")
    print("=" * 60)
    print(f"{'模型':<25} {'R²':<8} {'RMSE':<8} {'RPD':<8} {'n特征':<6}")
    print("-" * 55)
    for r in [result_a, result_b, result_c]:
        print(f"{r['model_label']:<25} {r['r2']:<8.4f} {r['rmse']:<8.4f} {r['rpd']:<8.2f} {r['n_features']:<6}")
    print(f"\n  原文 (ES&T 2025): RPD > 3.16")
    
    print(f"\n✅ S3 完成!")


if __name__ == "__main__":
    main()
