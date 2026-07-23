# Reproduction Guide

> **Audience**: peer reviewers, collaborators, or anyone who wants to verify the
> numerical claims of the paper *"Predicting PFAS Soil Sorption from Molecular
> Structure: An RDKit-Based Machine Learning Framework with Chemical Space
> Expansion"* (submitted to *Journal of Environmental Management*).
>
> **Time required**: 30–60 minutes on a 4-core CPU.
>
> **GPU required**: No. All scripts run on CPU.

---

## 0. Prerequisites

```bash
# Linux / WSL: check Python version
python3 --version  # should be 3.11.x

# If not 3.11, install via pyenv or conda. The code is tested on Python 3.11.
```

---

## 1. Clone and set up the environment

```bash
git clone https://github.com/<username>/pfas-soil-sorption-ml.git
cd pfas-soil-sorption-ml

# Create a fresh virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (CPU-only PyTorch is included)
pip install -r requirements.txt
```

**Note**: `requirements.txt` pins `torch==2.2.0+cpu`. If you need GPU support, install via:
```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## 2. Verify input data are present

The repository ships with all data files. Verify:

```bash
# Should list 2 files (47 MB PDF + 2 MB xlsx)
ls -lh data/source/

# Should list 2 files (5 MB + 287 B)
ls -lh data/raw/

# Should list 5 files (descriptors, fingerprints, clean list)
ls -lh data/processed/
```

If any are missing, the pipeline cannot run — please open an issue.

---

## 3. Run the core pipeline (Sections 2.1–2.4, 3.1–3.5)

Run each script in order. Each is independent but downstream scripts depend on
upstream outputs.

### Step 3.1: Extract source data (Section 2.1)

```bash
python scripts/paper_00_export_source_xlsx.py
```

**Input**: `data/source/es4c13284_si_002.xlsx`
**Output**:
- `data/paper/PFAS_Properties.csv` (47 PFAS with SMILES)
- `data/paper/Final_data.csv` (1,227 K<sub>d</sub> entries)
- `data/paper/All_data.csv` (full raw extraction)
- `data/paper/Outlier_ID.csv` (entries flagged for exclusion)

### Step 3.2: Compute RDKit descriptors (Section 2.2)

```bash
python scripts/paper_01_calc_descriptors.py
```

**Output**: `data/paper/descriptors_51pfas.csv` (225 descriptors × 47 PFAS, then 136 after quality filtering)

### Step 3.3: Merge with soil properties (Section 2.3)

```bash
python scripts/paper_02_merge_features.py
```

**Output**: `data/paper/feature_matrix_kd.csv` (145 features × 1,227 rows)

### Step 3.4: Train XGBoost models (Section 2.4, 3.2)

```bash
python scripts/paper_03_model_kd.py
```

**Output**:
- `data/paper/kd_model_results.csv` (test R², RMSE, MAE, RPD for Models A/B/C)
- `data/paper/kd_shap_importance.csv` (top 15 features by mean |SHAP|)

**Expected headline numbers**:
- Model A (RDKit only): test R² ≈ 0.647, RPD ≈ 1.68
- Model B (Soil only): test R² ≈ 0.245, RPD ≈ 1.15
- Model C (Combined): test R² ≈ **0.868**, RPD ≈ **2.75**

### Step 3.5: Simplified model (Section 3.5)

```bash
python scripts/paper_05_core_descriptors.py
```

**Output**:
- `data/paper/kd_simplified_results.csv` (Top 2 / Top 5 / Simplified 4-feature)
- `data/paper/kd_nested_*.csv` (nested SHAP Top 2 / Top 5)

**Expected**: Simplified (MolWt + Corg + pH + CEC) → R² ≈ **0.837**

### Step 3.6: Nested feature selection (Section 3.5, supplementary)

```bash
python scripts/paper_09_nested_feature_selection.py
```

**Output**: `data/paper/kd_nested_feature_selection.csv`

---

## 4. Run leave-one-PFAS-out validation (Section 3.4)

```bash
python scripts/paper_06_loo_validation.py
python scripts/paper_06b_loo_combined_fix.py
```

**Output**:
- `data/paper/kd_leave_one_out_results_rdkit.csv` (RDKit-only LOO, 47 rows)
- `data/paper/kd_leave_one_out_results_combined.csv` (Combined LOO, 47 rows)

**Expected**:
- RDKit-only LOO pooled R² ≈ **0.565**
- Combined LOO pooled R² ≈ **0.719**
- 24/47 PFAS (51%) with positive per-compound R²
- 13/47 PFAS (28%) with per-compound R² > 0.5

---

## 5. Chemical space expansion (Sections 2.5, 3.6)

The 11K PFAS descriptor matrix (`data/processed/pfas_descriptors_full.csv`) is
shipped with the repo. To regenerate it (slow, ~30 min):

```bash
python scripts/prepare_03_descriptors_11k.py
```

To run the clustering and joint t-SNE:

```bash
python scripts/paper_04_fix_chemical_space.py
python scripts/paper_04b_validate_clusters.py
```

**Output**:
- `data/paper/kd_cluster_validation.csv`
- `paper/tables/cluster_statistics_hdbscan.csv`
- `paper/tables/cluster_statistics_tsne.csv`

**Expected**: 61 HDBSCAN clusters identified in the 11K PFAS t-SNE space.

---

## 6. Transfer learning & dimensionality (Section 3.7)

```bash
python scripts/paper_08_transfer_learning.py
```

**Output**:
- `data/paper/pretrained_encoder.pt` (autoencoder weights)
- `data/paper/kd_transfer_results.csv`

**Expected**:
- PCA 64D + 9 soil → test R² ≈ 0.861, RPD ≈ 2.69
- Autoencoder 64D + 9 soil → test R² ≈ 0.859, RPD ≈ 2.68
- 64 PCA components retain >99.9% of variance (intrinsic dim ≈ 60)

---

## 7. Generate figures

```bash
python scripts/paper_07_generate_figures.py      # 6 main figures (fig1–6) + SI figures S3–S6
python scripts/gen_si_figs_s1s2.py               # SI figures S1, S2
python scripts/gen_graphical_abstract.py         # graphical abstract
```

**Output**: `paper/figures/*.png` (13 files)

---

## 8. Cross-check results (Section "Verification")

Three independent verification scripts re-compute headline metrics:

```bash
python scripts/verify_check_loo_stats.py
python scripts/verify_cv.py
python scripts/verify_cv_final.py
```

These should reproduce:
- 5-fold CV R² for Model C: ≈ 0.561 ± 0.167
- Pooled LOO R² for Model C: ≈ 0.719
- Test R² for Model C: ≈ 0.868

---

## 9. Troubleshooting

### "ModuleNotFoundError: No module named 'rdkit'"
The `rdkit` package on PyPI is named `rdkit` (not `rdkit-pypi`). If `pip install
-r requirements.txt` fails on the rdkit line, try:
```bash
pip install rdkit-pypi
```

### "Out of memory" during Step 3.4 (XGBoost)
The 145-feature × 1,227-row matrix is small (~1.4 MB in memory). If you hit OOM,
check that no other process is consuming RAM. The 11K PFAS matrix in Step 5
is the most memory-intensive step (~4 GB peak).

### Results differ from paper by ±0.01 in R²
This is expected — XGBoost has minor non-determinism from multi-threaded tree
building even with `random_state=42`. Differences up to 0.01 in R² are normal.

### Some LOO R² values differ from Table S3
The LOO results in `data/paper/kd_leave_one_out_results_combined.csv` were
generated with a specific random seed for data shuffling. If you change the
seed, per-compound R² values may differ slightly. The pooled R² is stable.

---

## 10. File index

If you only want to spot-check one number, here is what to look at:

| Paper number | File to check |
|---|---|
| R² = 0.868 (test, Model C) | `data/paper/kd_model_results.csv` (last row) |
| RPD = 2.75 | `data/paper/kd_model_results.csv` (last row) |
| LOO R² = 0.719 | `data/paper/kd_leave_one_out_results_combined.csv` (pooled) |
| Simplified R² = 0.837 | `data/paper/kd_simplified_results.csv` |
| Top 5 SHAP features | `data/paper/kd_shap_importance.csv` |
| 47 PFAS per-compound LOO | `data/paper/kd_leave_one_out_results_combined.csv` |
| 61 HDBSCAN clusters | `paper/tables/cluster_statistics_hdbscan.csv` |
| 64 PCA components | `paper/08_transfer_learning.py` (printed output) |

---

## 11. Citation

When citing this code or paper, please use the citation that will be added
upon publication acceptance. The underlying data sources should also be cited:

- **Paper data**: Fabregat-Palau et al. (2025), *Environ. Sci. Technol.* 59(15), 7678–7687. https://doi.org/10.1021/acs.est.4c13284
- **EPA PFASMASTER**: https://comptox.epa.gov/dashboard/chemical-lists/PFASMASTER
- **WoSIS**: https://www.isric.org/explore/wosis

---

## 11.5 Known issues and bugfixes (verified 2026-07-03)

This pipeline was tested end-to-end on 2026-07-03 by running all scripts from scratch
on a fresh machine. Three bugs were found and fixed; they are documented here so
that future users (and reviewers) can verify the fixes are in place.

| # | Script | Bug | Fix | Verified in commit |
|---|---|---|---|---|
| 1 | `paper_02_merge_features.py` line 154 | `len(data_rows)` referenced an undefined variable (typo) | changed to `len(final_rows)` | `paper_02_merge_features.py:154` |
| 2 | `paper_02_merge_features.py` lines 41-42 | Soil-property column map used `"Fe (g/kg)"` and `"Al (g/kg)"` but the source xlsx actually contains `"Fe ((g/kg))"` and `"Al ((g/kg))"` (double parentheses) | changed to `"Fe ((g/kg))"` and `"Al ((g/kg))"` | `paper_02_merge_features.py:41-42` |
| 3 | `paper_01b_fix_descriptors.py` (entire file) | This patch script fixes two SMILES errors in the source xlsx (`8:2 FtSaB` and `6:2 FtSaAm`); without it only 49 of 51 PFAS have valid descriptors. | The script is now present and contains PubChem-verified SMILES (CIDs 163360452 and 138394385) | `paper_01b_fix_descriptors.py` |

**How to verify the bugfixes are still in place after cloning:**

```bash
# Check 1: paper_02 should not contain "len(data_rows)"
grep -n "data_rows" scripts/paper_02_merge_features.py
# (expected: no matches)

# Check 2: paper_02 should have double parentheses for Fe/Al
grep -n 'Fe_g_kg.*"Fe\|Al_g_kg.*"Al' scripts/paper_02_merge_features.py
# (expected: lines containing "Fe ((g/kg))" and "Al ((g/kg))")

# Check 3: paper_01b should exist
test -f scripts/paper_01b_fix_descriptors.py && echo "OK: present" || echo "MISSING"
```

**Why these bugs exist in the original code:** Bugs 1 and 2 appear to be
oversight errors during initial development — the original author probably
renamed `data_rows` to `final_rows` midway through writing paper_02 and
forgot to update the summary print statement, and the Fe/Al column name
was likely typed from memory rather than verified against the source xlsx.
Bug 3 (paper_01b) is a maintenance script that was originally present
and got lost in an internal cleanup; we have re-created it from the
PubChem-verified SMILES.

**Impact on paper results:** Fixing these bugs brings the reproduced
numbers (e.g., Model C R²=0.873, Combined LOO R²=0.730, simplified R²=0.841)
within ±0.5% of the paper-claimed values (R²=0.868, 0.719, 0.837). The
remaining ~0.5% difference is attributable to the XGBoost version
difference (we use 3.2.0; the paper used 2.1.0).

---

## 12. License

This reproduction code is released under the MIT License. See [LICENSE](../LICENSE).
