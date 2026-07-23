"""Combined LOO statistics: read per-fold R² from CSV and aggregate."""
import csv, numpy as np
import os

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")

# Combined LOO per-fold results
with open(os.path.join(CSV_DIR, "kd_leave_one_out_results_combined.csv")) as f:
    rows = list(csv.DictReader(f))

r2 = np.array([float(r["r2"]) for r in rows])
n_test = np.array([int(r["n_test"]) for r in rows])

# Read pooled R² from kd_leave_one_out_summary.csv (previously hard-coded as
# 0.7185, which was inconsistent with the 0.7304 in the summary file;
# fixed 2026-07-23).
summary_path = os.path.join(CSV_DIR, "kd_leave_one_out_summary.csv")
pooled_r2_str = "unknown"
try:
    with open(summary_path) as f:
        summary_rows = list(csv.DictReader(f))
    if summary_rows:
        pooled_r2_str = summary_rows[0].get("overall_r2", "unknown")
except FileNotFoundError:
    pass

print("=== Combined LOO accurate summary ===")
print(f"Overall R² (pooled) = {pooled_r2_str}  (from kd_leave_one_out_summary.csv)")
print(f"Mean R² = {np.nanmean(r2):.4f}")
print(f"Median R² = {np.nanmedian(r2):.4f}")

# n>=10 vs n<10
large = r2[n_test >= 10]
small = r2[n_test < 10]
print(f"n>=10: mean={np.nanmean(large):.4f} (n={len(large)})")
print(f"n<10: mean={np.nanmean(small):.4f} (n={len(small)})")

# R² distribution
good = sum(r2 > 0.5)
med = sum((r2 > 0) & (r2 <= 0.5))
poor = sum(r2 <= 0)
print(f"Distribution: >0.5={good}  0~0.5={med}  <=0={poor}")

# Filter nan
r2_clean = r2[~np.isnan(r2)]
print(f"\nValid R² (non-nan) = {len(r2_clean)}")
print(f"Mean R² = {np.mean(r2_clean):.4f}")
print(f"Median R² = {np.median(r2_clean):.4f}")

# n>=5
nt = n_test[~np.isnan(r2)]
r2_nt = r2_clean
valid_nt5 = nt >= 5
r2_nt5 = r2_nt[valid_nt5]
print(f"n>=5: mean={np.mean(r2_nt5):.4f} (n={len(r2_nt5)})")

# Simplified model reference values (paper-claimed, for cross-check only;
# measured values come from paper_05_core_descriptors.py output)
# Paper: R²=0.837, RMSE=0.376, RPD=2.48
# Paper: 96.4% of full model (0.868)
# Paper: 5-fold CV mean=0.561 (verified measured: 0.548 after Pipeline fix)
print("\n=== Simplified model ===")
print("Simplified model (MolWt+Corg+pH+CEC) — paper: R²=0.837, RMSE=0.376, RPD=2.48")
print("                          96.4% of full model (0.868)")
print("Full model 5-fold CV — paper: 0.561; measured (after Pipeline fix): 0.548")
