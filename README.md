# PFAS Soil Sorption — RDKit + XGBoost ML Framework

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/rdkit-2024.9-green.svg)](https://www.rdkit.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.1-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible code for:

> **Predicting PFAS Soil Sorption from Molecular Structure: An RDKit-Based Machine Learning Framework with Chemical Space Expansion**
> *Manuscript submitted to Journal of Environmental Management*

---

## What this repository contains

- **31 Python scripts** implementing the full analysis pipeline (Sections 2–3 of the manuscript)
- **All input data** for the 47-PFAS benchmark dataset (1,227 K<sub>d</sub> measurements × 451 soils)
- **EPA PFASMASTER inventory** (~11,000 compounds) for chemical space expansion
- **13 publication figures** (6 main + 6 SI + 1 graphical abstract)
- **13 publication tables**

---

## Headline results (paper §3)

| Model | Test R² | RPD | LOO pooled R² |
|---|---|---|---|
| RDKit descriptors only (136 features) | 0.647 | 1.68 | 0.565 |
| Soil properties only (9 features) | 0.245 | 1.12 | — |
| **Combined (RDKit + soil, 145 features)** | **0.868** | **2.75** | **0.719** |
| Simplified (MolWt + Corg + pH + CEC, 4 features) | 0.837 | 2.48 | 0.592 (nested) |

A simplified 4-feature model recovers 96% of the full-model accuracy, demonstrating extensive redundancy in RDKit descriptors for the PFAS chemical space.

---

## Quick start (5-step reproduction)

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Note: torch is CPU-only; for GPU, install via pytorch.org/whl/cpu

# 2. Run the pipeline (each step is independent)
python scripts/paper_00_export_source_xlsx.py  # extract source xlsx → data/paper/
python scripts/paper_01_calc_descriptors.py    # compute 225 RDKit descriptors
python scripts/paper_02_merge_features.py      # merge with soil properties
python scripts/paper_03_model_kd.py            # train 3 XGBoost models
python scripts/paper_07_generate_figures.py    # generate all 6 main figures

# 3. (Optional) Advanced analysis
python scripts/paper_06_loo_validation.py      # leave-one-PFAS-out CV
python scripts/paper_08_transfer_learning.py    # PCA + autoencoder (§3.7)
python scripts/paper_09_nested_feature_selection.py  # nested SHAP (§3.5)
```

**Expected runtime**: ~30 minutes on a 4-core CPU (no GPU required for any script).

**Expected output**:
- `data/paper/*.csv` — modeling matrices and results
- `paper/figures/*.png` — 13 publication figures
- `paper/tables/*.csv` — 13 publication tables

For detailed step-by-step instructions, see [docs/REPRODUCE.md](docs/REPRODUCE.md).

---

## Data sources

### Source 1: Fabregat-Palau et al. (2025) — PFAS K<sub>d</sub> data
- **File**: `data/source/es4c13284_si_001.pdf` (SI Part 1) and `data/source/es4c13284_si_002.xlsx` (SI Part 2)
- **DOI**: [10.1021/acs.est.4c13284](https://doi.org/10.1021/acs.est.4c13284)
- **License**: CC-BY 4.0 (free to redistribute)
- **Used for**: K<sub>d</sub> regression modeling (Sections 2.1, 3.1–3.5)
- **Extracted to**: `data/paper/Final_data.csv` (1,227 entries, 47 PFAS, 451 soils)

### Source 2: EPA PFASMASTER — Chemical inventory
- **File**: `data/raw/pfas_master_list.csv`
- **Source**: [EPA CompTox Chemicals Dashboard](https://comptox.epa.gov/dashboard/chemical-lists/PFASMASTER) (public)
- **Used for**: Chemical space expansion & clustering (Sections 2.5, 3.6)
- **Cleaned to**: 10,971 unique PFAS with valid SMILES

### Source 3: World Soil Information Service (WoSIS)
- **Used for**: Soil pH context (referenced in Data Availability Statement)
- **DOI**: [10.17027/isric-wdcsoils-20231130](https://doi.org/10.17027/isric-wdcsoils-20231130)
- **Note**: Not redistributed; can be queried at [isric.org/explore/wosis](https://www.isric.org/explore/wosis)

---

## Repository structure

```
.
├── LICENSE                 # MIT
├── README.md               # this file
├── requirements.txt        # pinned Python deps
├── UPLOAD_PACKAGE_DESIGN.md  # internal design notes (not for review)
├── docs/
│   └── REPRODUCE.md        # detailed reproduction guide
├── data/
│   ├── source/             # original SI files from Fabregat-Palau 2025
│   ├── raw/                # EPA PFASMASTER (public)
│   ├── processed/          # 11K PFAS descriptors + fingerprints
│   └── paper/              # modeling matrices and intermediate results
├── scripts/                # 21 production scripts (paper_NN_*.py)
│   └── _archive/           # 11 one-off exploration scripts (gitignored? no, kept locally)
├── paper/
│   ├── article_summary.md  # 1-page summary of the manuscript
│   ├── figures/            # 13 publication figures
│   └── tables/             # 13 publication tables
├── manuscript/             # manuscript drafts in active editing (GITIGNORED)
└── _private/               # private downloads (GITIGNORED)
```

---

## Pipeline: reproducing the paper

### Part A: K<sub>d</sub> regression modeling (Sections 2.1–2.4, 3.1–3.5)

```
Source: Fabregat-Palau 2025 SI
  │
  ▼
paper_00_export_source_xlsx.py  ──►  data/paper/PFAS_Properties.csv
                                 ──►  data/paper/Final_data.csv
  │
  ▼
paper_01_calc_descriptors.py    ──►  data/paper/descriptors_51pfas.csv (225 RDKit descriptors for 47 PFAS)
  │
  ▼
paper_02_merge_features.py      ──►  data/paper/feature_matrix_kd.csv (145 features)
  │
  ▼
paper_03_model_kd.py            ──►  data/paper/kd_model_results.csv
                                 ──►  data/paper/kd_shap_importance.csv
  │
  ▼
paper_05_core_descriptors.py    ──►  data/paper/kd_simplified_results.csv
                                 ──►  data/paper/kd_nested_*.csv
```

### Part B: Leave-one-PFAS-out validation (Section 3.4)

```
paper_06_loo_validation.py      ──►  data/paper/kd_leave_one_out_results_rdkit.csv
paper_06b_loo_combined_fix.py   ──►  data/paper/kd_leave_one_out_results_combined.csv
```

### Part C: Chemical space expansion (Sections 2.5, 3.6)

```
EPA PFASMASTER (data/raw/pfas_master_list.csv)
  │
  ▼
prepare_03_descriptors_11k.py  ──►  data/processed/pfas_descriptors_full.csv
                                 ──►  data/processed/pfas_fingerprint_full.csv
  │
  ▼
paper_04_fix_chemical_space.py
paper_04b_validate_clusters.py ──►  data/paper/kd_cluster_validation.csv
                                 ──►  paper/tables/cluster_statistics_*.csv
```

### Part D: Transfer learning & dimensionality (Section 3.7)

```
paper_08_transfer_learning.py   ──►  data/paper/pretrained_encoder.pt
                                 ──►  data/paper/kd_transfer_results.csv
```

### Part E: Figure generation

```
paper_07_generate_figures.py    ──►  paper/figures/fig1–6_*.png (6 main) + figS3–6_*.png (4 SI)
gen_si_figs_s1s2.py             ──►  paper/figures/figS1–2_*.png
gen_graphical_abstract.py       ──►  paper/figures/graphical_abstract.png
```

---

## System requirements

- **OS**: Linux (Ubuntu 20.04+) or WSL on Windows 10/11. macOS untested.
- **Python**: 3.11
- **RAM**: 4 GB minimum, 8 GB recommended (for 11K PFAS descriptor matrix)
- **Disk**: ~500 MB for data + code
- **CPU**: 4 cores recommended. No GPU required (all models train on CPU).

---

## Citation

If you use this code, please cite the accompanying paper (citation will be added upon acceptance).

For the underlying data, please also cite:
- Fabregat-Palau et al. (2025), *Environmental Science & Technology*, 59(15), 7678–7687. https://doi.org/10.1021/acs.est.4c13284

---

## License

This code is released under the MIT License (see [LICENSE](LICENSE)). The input data are subject to their original licenses:
- Fabregat-Palau 2025 SI: CC-BY 4.0
- EPA PFASMASTER: Public domain (US Government work)
