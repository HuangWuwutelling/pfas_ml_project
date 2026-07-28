# PFAS Soil Sorption — RDKit + XGBoost ML Framework

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/rdkit-2026.3.3-green.svg)](https://www.rdkit.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-3.2-orange.svg)](https://xgboost.readthedocs.io/)
[![Reproduction verified 2026-07-23](https://img.shields.io/badge/reproduction-verified_2026--07--23-brightgreen.svg)](docs/REPRODUCE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible code for:

> **Predicting PFAS Soil Sorption from Molecular Structure: An RDKit-Based Machine Learning Framework with Chemical Space Expansion**
> *Manuscript submitted to Journal of Environmental Management*

---

## What this repository contains

- **21 production Python scripts** implementing the full analysis pipeline (Sections 2–3 of the manuscript); plus **4 dedicated cross-study benchmark / audit scripts** (added 2026-07-27) and **18 pytest unit tests** in `tests/`
- **All input data** for the 47-PFAS benchmark dataset (1,227 K<sub>d</sub> measurements × 451 soils)
- **EPA PFASMASTER inventory** (~10,972 compounds with valid SMILES) for chemical space expansion
- **Paper SI** (`es4c13284_si_002.xlsx`, CC BY 4.0) — the single source xlsx from which all K<sub>d</sub> regression inputs are derived
- **Xie 2024 SI** — **NOT** bundled (Elsevier, all rights reserved); users download from [DOI 10.1016/j.scitotenv.2024.176575](https://doi.org/10.1016/j.scitotenv.2024.176575); auto-extracted to `/tmp/xie2024_table5.csv` on first run
- **13 publication figures** (6 main + 6 SI + 1 graphical abstract) — already generated under `data/paper/`
- **13 publication tables** — already generated under `data/paper/`

**Reproduction status:** All committed code reproduces paper headline numbers within ±0.015 of the published values. See [docs/REPRODUCE.md §12–16](docs/REPRODUCE.md) for the full audit trail, including 3 known bugs, 5 path-portability fixes, 1 train/test leakage fix (sklearn.Pipeline in `paper_03`), LOO pooled R² canonicalization, and the 2026-07-27 Xie ↔ training-set overlap audit (162 strict-overlap rows identified and removed; R² moved from 0.783 to 0.778 — within 0.01).

---

## Headline results (paper §3)

| Model | Test R² | RPD | LOO pooled R² |
|---|---|---|---|
| RDKit descriptors only (136 features) | 0.647 | 1.68 | 0.565 |
| Soil properties only (9 features) | 0.245 | 1.15 | — |
| **Combined (RDKit + soil, ~146 features)** | **0.870** | **2.78** | **0.730** |
| Simplified (MolWt + Corg + pH + CEC, 4 features) | 0.837 | 2.48 | 0.592 (nested) |
| **Cross-study benchmark — Xie 2024** (full 1,780 rows × 22 PFAS) | **+0.783** | — | — |
| **Cross-study benchmark — Xie 2024** (1,618-row strict-overlap-removed subset) | **+0.778** | — | — |

A simplified 4-feature model recovers 96% of the full-model accuracy, demonstrating
extensive redundancy in RDKit descriptors for the PFAS chemical space. The Xie 2024
cross-study benchmark is the only external benchmark reported; see §3.8 of the
manuscript for the source-aware re-evaluation that identified and removed 162
strict-overlap rows (PFOS + PFUnDA columns in the prior extraction were mis-assigned
— see the audit trail in `docs/REPRODUCE.md`).

---

## Quick start (reviewer reproduction)

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Core pipeline (~5 minutes) — reproduces Model C R²=0.868
python scripts/paper_00_export_source_xlsx.py     # extract source xlsx
python scripts/paper_01_calc_descriptors.py       # compute 225 RDKit descriptors
python scripts/paper_01b_fix_descriptors.py       # patch 2 broken SMILES (§12 bug 3)
python scripts/paper_02_merge_features.py         # merge with soil properties
python scripts/paper_03_model_kd.py               # train 3 XGBoost models
python scripts/paper_05_core_descriptors.py       # simplified 4-feature model

# 3. Validation (~5 minutes) — reproduces LOO R²=0.719
python scripts/paper_06_loo_validation.py
python scripts/paper_06b_loo_combined_fix.py

# 4. Chemical space expansion (~3 minutes) — §3.6
python scripts/prepare_02_clean_epa.py            # clean EPA PFASMASTER (§14)
python scripts/prepare_03_descriptors_11k.py      # 225 RDKit + 2048-bit ECFP4 (§14)
python scripts/paper_04_fix_chemical_space.py
python scripts/paper_04b_validate_clusters.py

# 5. Transfer learning & dimensionality (§3.7)
python scripts/paper_08_transfer_learning.py
python scripts/paper_09_nested_feature_selection.py

# 6. Figure generation
python scripts/paper_07_generate_figures.py       # 6 main + 4 SI figures
python scripts/gen_si_figs_s1s2.py                # 2 more SI figures
python scripts/gen_graphical_abstract.py          # graphical abstract

# 7. Cross-study benchmark on Xie 2024 (§3.8) — requires Xie SI to be present
#    (Elsevier, not redistributable; see "License" section for download).
python scripts/check_xie_train_overlap.py         # audit training-set vs Xie overlap
python scripts/build_xie_overlap_table.py         # build supplementary table S5
python scripts/reevaluate_xie_disjoint.py         # disjoint-subset R² = 0.778
python scripts/augment_simplified_models.py       # 3-feature ablation row
python scripts/paper_10_external_validation.py    # full cross-study benchmark (1×3 fig)

# 8. Independent verification (~1 minute) — see §10 of REPRODUCE.md for headline numbers
python scripts/verify_cv.py
python scripts/verify_cv_final.py
python scripts/verify_check_loo_stats.py

# 9. Run the test suite
python -m pytest tests/ -q
```

**Expected runtime**: ~15–30 minutes on a 4-core CPU (no GPU required for any script).

**Expected output**:
- `data/paper/*.csv` — modeling matrices and results
- `data/paper/*.png` — 13 publication figures
- `data/paper/pretrained_encoder.pt` — autoencoder weights

For detailed step-by-step instructions, see [docs/REPRODUCE.md](docs/REPRODUCE.md).

---

## Data sources

### Source 1: Fabregat-Palau et al. (2025) — PFAS K<sub>d</sub> data
- **File**: `data/source/es4c13284_si_002.xlsx` (SI Part 2)
- **DOI**: [10.1021/acs.est.4c13284](https://doi.org/10.1021/acs.est.4c13284)
- **License**: CC-BY 4.0 (free to redistribute)
- **Used for**: K<sub>d</sub> regression modeling (Sections 2.1, 3.1–3.5)
- **Extracted to**: `data/paper/Final_data.csv` (1,227 entries, 47 PFAS, 451 soils)
- **Note**: The accompanying 47 MB SI PDF (`es4c13284_si_001.pdf`) is not shipped
  in this repo (gitignored — no script reads it). Download directly from the DOI
  if you need the supplementary figures referenced in the paper text.

### Source 2: EPA PFASMASTER — Chemical inventory
- **File**: `data/raw/pfas_master_list.csv` (12,039 compounds, 2016 snapshot)
- **Source**: [EPA CompTox Chemicals Dashboard](https://comptox.epa.gov/dashboard/chemical-lists/PFASMASTER) (public)
- **Used for**: Chemical space expansion & clustering (Sections 2.5, 3.6)
- **Cleaned to**: 10,972 unique PFAS with valid SMILES (`scripts/prepare_02_clean_epa.py`)
- **Note on snapshot**: The bundled CSV is a 2016-vintage snapshot (12,039 compounds),
  not the current 22,987+ entries. This is sufficient for the paper's 11K chemical-space
  analysis. Re-fetching from EPA CompTox will yield the current list.

### Source 3: Xie et al. (2024) — Cross-study benchmark (literature compilation)
- **Used for**: Cross-study benchmark of the paper model (Section 3.8, 22 PFAS overlap with paper)
- **DOI**: https://doi.org/10.1016/j.scitotenv.2024.176575
- **License**: Elsevier (All rights reserved — paper SI is downloaded by the user
  from the DOI and is NOT redistributed in this repo; see `docs/REPRODUCE.md` §17)
- **Used in script**: `scripts/paper_10_external_validation.py` (auto-extracts Table S5)
- **Result**: pooled R² = 0.783 on 1,780 Kd measurements (22 overlapping PFAS);
  R² = 0.778 on the 1,618-row strict-overlap-removed subset (PFAS + pH ± 0.05
  + OC ± 0.05 + log10 Kd ± 0.005). The two compilations share 12 primary
  studies; the 162-row overlap is documented in `data/paper/tableS5_xie_source_overlap.csv`.

### Source 4 (archival, not used in the paper): Morales et al. (2026) field-contaminated soils
- **File**: `data/source/Morales_2026_SI.xlsx` (4 sheets, 84 field-contaminated soils, CC BY 4.0)
- **Source**: [Environmental Research 306(1), 125071](https://doi.org/10.1016/j.envres.2026.125071)
- **Audit artefact only**: the long-format extraction is `data/source/morales_long_reextracted.csv` (re-validated by `scripts/extract_morales_2026.py`; the original `data/source/morales_long.csv` was found to have a PFOS/PFUnDA column mis-assignment, which the re-extracted file fixes). These files are kept in the repository as a reproducible audit trail of the re-extraction work performed 2026-07-26→2026-07-27, but **Morales is not reported in the paper** as a formal benchmark (the field-soil measurement protocol differs structurally from the laboratory batch-equilibrium data used for training, and Morales did not report CEC). The data and extraction script are not called by any active paper pipeline (`paper_10_external_validation.py` no longer imports them).

---

## Repository structure

```
.
├── LICENSE                         # MIT
├── README.md                       # this file
├── requirements.txt                # pinned Python deps (rdkit 2026.3.3, xgboost 3.2.0, …)
├── .gitignore
├── docs/
│   └── REPRODUCE.md                # detailed reproduction guide (§0–§15)
├── data/
│   ├── source/                     # paper SI: es4c13284_si_002.xlsx
│   ├── raw/                        # EPA PFASMASTER (public, 12,039 rows)
│   ├── processed/                  # 11K PFAS descriptors + fingerprints (GITIGNORED, regenerated)
│   └── paper/                      # modeling matrices, intermediate results, 13 figures, 13 tables
├── scripts/                        # 21 production scripts
│   ├── paper_00 → paper_09          # core + advanced pipeline
│   ├── prepare_02 → prepare_03      # 11K pipeline (added 2026-07-23)
│   ├── verify_cv / verify_cv_final  # headline-number verification
│   ├── verify_check_loo_stats
│   ├── gen_graphical_abstract
│   ├── gen_si_figs_s1s2
│   └── _archive/                    # 12 superseded exploration scripts (GITIGNORED)
└── manuscript/                      # manuscript drafts in active editing (GITIGNORED)
```

**Gitignored but present locally**: `manuscript/`, `_private/`, `reference_repos/`,
`paper/` (working notes), `scripts/_archive/`, `data/processed/` (regenerated by
pipeline), `data/source/es4c13284_si_001.pdf` (download from DOI instead),
`UPLOAD_PACKAGE_DESIGN.md`, `FILE_AUDIT_REPORT.md`, `_test_run_*/`. See `.gitignore`.

---

## Pipeline: reproducing the paper

### Part A: K<sub>d</sub> regression modeling (Sections 2.1–2.4, 3.1–3.5)

```
Source: Fabregat-Palau 2025 SI xlsx (data/source/es4c13284_si_002.xlsx)
  │
  ▼
paper_00_export_source_xlsx.py   ──►  data/paper/PFAS_Properties.csv
                                  ──►  data/paper/Final_data.csv
  │
  ▼
paper_01_calc_descriptors.py      ──►  data/paper/descriptors_51pfas.csv (225 RDKit descriptors)
paper_01b_fix_descriptors.py     ──►  patches 8:2 FtSaB + 6:2 FtSaAm SMILES (§12 bug 3)
  │
  ▼
paper_02_merge_features.py        ──►  data/paper/feature_matrix_kd.csv (145 features × 1227 rows)
  │
  ▼
paper_03_model_kd.py              ──►  data/paper/kd_model_results.csv
                                  ──►  data/paper/kd_shap_importance.csv
  │
  ▼
paper_05_core_descriptors.py      ──►  data/paper/kd_simplified_results.csv
                                  ──►  data/paper/kd_nested_*.csv
```

### Part B: Leave-one-PFAS-out validation (Section 3.4)

```
paper_06_loo_validation.py        ──►  data/paper/kd_leave_one_out_results_rdkit.csv
paper_06b_loo_combined_fix.py     ──►  data/paper/kd_leave_one_out_results_combined.csv
```

### Part C: Chemical space expansion (Sections 2.5, 3.6)

```
EPA PFASMASTER (data/raw/pfas_master_list.csv)
  │
  ▼
prepare_02_clean_epa.py           ──►  data/processed/pfas_clean.csv (10,972 rows)
  │
  ▼
prepare_03_descriptors_11k.py     ──►  data/processed/pfas_descriptors_full.csv (10,971 × 228)
                                  ──►  data/processed/pfas_fingerprint_full.csv (10,971 × 2048)
  │
  ▼
paper_04_fix_chemical_space.py
paper_04b_validate_clusters.py    ──►  data/paper/kd_cluster_validation.csv
                                  ──►  data/paper/kd_cluster_tsne.png
                                  ──►  data/paper/kd_chemical_space_annotated.png
```

### Part D: Transfer learning & dimensionality (Section 3.7)

```
paper_08_transfer_learning.py     ──►  data/paper/pretrained_encoder.pt
                                  ──►  data/paper/kd_transfer_results.csv
```

### Part E: Figure generation

```
paper_07_generate_figures.py      ──►  data/paper/fig1–6_*.png (6 main) + figS3–6_*.png (4 SI)
gen_si_figs_s1s2.py               ──►  data/paper/figS1–2_*.png
gen_graphical_abstract.py         ──►  data/paper/graphical_abstract.png
```

### Part F: Cross-study benchmark + source-overlap audit (Section 3.8)

```
check_xie_train_overlap.py        ──►  data/paper/kd_xie_train_overlap_report.json
build_xie_overlap_table.py        ──►  data/paper/tableS5_xie_source_overlap.csv
reevaluate_xie_disjoint.py        ──►  data/paper/kd_xie_disjoint_validation.json
augment_simplified_models.py      ──►  data/paper/kd_simplified_results.csv (appends 3-feature row)

paper_10_external_validation.py   ──►  data/paper/kd_external_validation_xie2024.csv (1,780 rows)
                                  ──►  data/paper/kd_external_validation_xie2024_disjoint.csv (1,618 rows)
                                  ──►  data/paper/fig10_external_validation.png (1×3 panel)
```

### Part G: Independent verification

```
verify_cv.py                      ──►  cross-checks 5-fold CV R² ≈ 0.556, simplified R² ≈ 0.83
verify_cv_final.py                ──►  multi-seed CV averaged R²
verify_check_loo_stats.py         ──►  LOO per-compound R² distribution
```

### Tests

```
pytest tests/                     ──►  18 unit tests covering simplified-model
                                       augmentation, Xie overlap audit, and disjoint
                                       validation (all pass on a fresh clone)
```

---

## System requirements

- **OS**: Linux (Ubuntu 20.04+) or WSL on Windows 10/11. macOS untested.
- **Python**: 3.11
- **RAM**: 4 GB minimum, 8 GB recommended (for 11K PFAS descriptor matrix)
- **Disk**: ~500 MB for data + code + intermediate outputs
- **CPU**: 4 cores recommended. **No GPU required** (all models train on CPU).
- **Test deps**: `pytest >= 7.0` (already in `requirements.txt`)

---

## Citation

If you use this code, please cite the accompanying paper (citation will be added upon acceptance).

For the underlying data, please also cite:
- Fabregat-Palau et al. (2025), *Environmental Science & Technology*, 59(15), 7678–7687. https://doi.org/10.1021/acs.est.4c13284
- Xie et al. (2024), *Science of The Total Environment*, 954, 176575. https://doi.org/10.1016/j.scitotenv.2024.176575

---

## License

This code is released under the MIT License (see [LICENSE](LICENSE)). The input data are subject to their original licenses:
- Fabregat-Palau 2025 SI: CC-BY 4.0 (redistributable; bundled at `data/source/es4c13284_si_002.xlsx`)
- Morales 2026 SI: CC-BY 4.0 (redistributable; bundled at `data/source/Morales_2026_SI.xlsx`)
- Xie 2024 SI: © Elsevier B.V., **all rights reserved**; **NOT** redistributable. Users must download the SI themselves from the publisher's website (DOI above) and place it at `data/source/Elucidating per- and polyfluoroalkyl2024-SI.docx` before running `paper_10_external_validation.py`. The auto-extraction path is `/tmp/xie2024_table5.csv`.
- EPA PFASMASTER: Public domain (US Government work)
