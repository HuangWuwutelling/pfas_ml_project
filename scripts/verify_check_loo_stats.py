import csv, numpy as np

# Combined LOO
import os
CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
with open(os.path.join(CSV_DIR, "kd_leave_one_out_results_combined.csv")) as f:
    rows = list(csv.DictReader(f))

r2 = np.array([float(r["r2"]) for r in rows])
n_test = np.array([int(r["n_test"]) for r in rows])

print("=== Combined LOO 准确汇总 ===")
print(f"总体R² (pooled) = 0.7185")
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
print("\n=== 简化模型 ===")
print(f"R²=0.8372, RMSE=0.3763, RPD=2.48")
print(f"占全模型(0.868)的 {0.8372/0.868*100:.1f}%")
print(f"\n全模型5-fold CV mean=0.6261")
