import csv, numpy as np

# Combined LOO
import os
CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
with open(os.path.join(CSV_DIR, "kd_leave_one_out_results_combined.csv")) as f:
    rows = list(csv.DictReader(f))

r2 = np.array([float(r["r2"]) for r in rows])
n_test = np.array([int(r["n_test"]) for r in rows])

# 从 kd_leave_one_out_summary.csv 读 pooled R²（之前是硬编码 0.7185，
# 与 kd_leave_one_out_summary.csv 里的 0.7304 不一致；2026-07-23 修复）
summary_path = os.path.join(CSV_DIR, "kd_leave_one_out_summary.csv")
pooled_r2_str = "unknown"
try:
    with open(summary_path) as f:
        summary_rows = list(csv.DictReader(f))
    if summary_rows:
        pooled_r2_str = summary_rows[0].get("overall_r2", "unknown")
except FileNotFoundError:
    pass

print("=== Combined LOO 准确汇总 ===")
print(f"总体R² (pooled) = {pooled_r2_str}  (来自 kd_leave_one_out_summary.csv)")
print(f"平均R² = {np.nanmean(r2):.4f}")
print(f"中位数R² = {np.nanmedian(r2):.4f}")

# n≥10 vs n<10
large = r2[n_test >= 10]
small = r2[n_test < 10]
print(f"n≥10: mean={np.nanmean(large):.4f} (n={len(large)})")
print(f"n<10: mean={np.nanmean(small):.4f} (n={len(small)})")

# R²分布
good = sum(r2 > 0.5)
med = sum((r2 > 0) & (r2 <= 0.5))
poor = sum(r2 <= 0)
print(f"分布: >0.5={good}  0~0.5={med}  ≤0={poor}")

# 过滤nan
r2_clean = r2[~np.isnan(r2)]
print(f"\n有效R² (非nan) = {len(r2_clean)}")
print(f"平均R² = {np.mean(r2_clean):.4f}")
print(f"中位数R² = {np.median(r2_clean):.4f}")

# n≥5
nt = n_test[~np.isnan(r2)]
r2_nt = r2_clean
valid_nt5 = nt >= 5
r2_nt5 = r2_nt[valid_nt5]
print(f"n≥5: mean={np.mean(r2_nt5):.4f} (n={len(r2_nt5)})")

# 简化模型数据
# paper 声称值（作为对照基准）
# 简化模型 R²=0.837, RMSE=0.376, RPD=2.48
# 占全模型(0.868)的 {0.837/0.868*100:.1f}%
# 全模型 5-fold CV mean=0.626 (paper 0.561；实测见 verify_cv.py 输出)
print("\n=== 简化模型 ===")
print("简化模型 (MolWt+Corg+pH+CEC) — paper: R²=0.837, RMSE=0.376, RPD=2.48")
print("                          占全模型(0.868) 的 96.4%")
print("全模型 5-fold CV — paper: 0.561；实测见 verify_cv.py（Pipeline 修复后 ≈ 0.548）")
