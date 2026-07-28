#!/usr/bin/env python3
"""
paper_10_external_validation.py
================================
External validation of the XGBoost Kd model on independent datasets.

Validates the model trained on Fabregat-Palau et al. (2025) (47 PFAS, 1227 rows)
against two independent Kd datasets:

  1. Xie et al. (2024) STOTEN 954, 176575
     - 26 PFAS, 2148 rows, 304 soils, RF model R²=0.93
     - log₂ Kd (L/kg) — converted to log₁₀
     - Features: pH, OC, CEC, Sand, Silt, Clay, MW, LogP, LogS, ATSm8, SpDiam

Strategy: Train 3 XGBoost variants on the 47-PFAS paper data, then test
on Xie:

  Variant 1: Full model (145 features) — paper §3.2
  Variant 2: Simplified model (4 features: MolWt + Corg + pH + CEC) — paper §3.5
  Variant 3: Xie-aligned 6-feature model (pH + OC + CEC + Sand + Silt + Clay)
              — uses only common soil features

  For Xie 2024 (which has MW/LogP etc), apply Variant 2 (uses our MolWt +
  3 soil features). Variant 3 uses Xie's soil features + our MW.
"""
import csv
import json
import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_predict, KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _shared_config import NON_FEATURE

# Paths from _shared_config (13.3 — unify path patterns across 21 scripts)
import _shared_config as _cfg
BASE = _cfg.PROJECT_ROOT
DATA_PAPER = _cfg.DATA_PAPER

# ----------------------------------------------------------------------
# Step 1: Load paper data and train 3 model variants
# ----------------------------------------------------------------------

def load_paper_data():
    df = pd.read_csv(os.path.join(DATA_PAPER, 'feature_matrix_kd.csv'))
    feature_cols = [c for c in df.columns if c not in NON_FEATURE]
    return df, feature_cols


def get_smi_map():
    path = os.path.join(DATA_PAPER, 'PFAS_Properties.csv')
    smi_map = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            abbrev = row['PFAS abbreviation'].strip()
            smi = row['Smiles'].strip()
            if abbrev and smi and smi.upper() not in ('N.A.', 'NA', 'NONE', ''):
                smi_map[abbrev] = smi
    return smi_map


def get_mw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol is not None else None


def _xie_disjoint_subset(xie_df, tol_pH=0.05, tol_OC=0.05, tol_Kd=0.005):
    """Drop Xie rows that match a row in the training set (Final_data.csv).

    The training set comes from data/paper/Final_data.csv; we use the same
    4-tuple key (PFAS, pH, OC, log_Kd) as scripts/reevaluate_xie_disjoint.py
    so the two scripts agree.
    """
    from collections import defaultdict
    train_csv = _cfg.FINAL_DATA
    train_keys: set[tuple[str, float, float, float]] = set()
    with open(train_csv) as f:
        for raw in csv.DictReader(f):
            pfas = (raw.get("PFAS (abreviation)") or "").strip()
            pH = pd.to_numeric(raw.get("pH (measured)"), errors="coerce")
            OC = pd.to_numeric(raw.get("Corg (%)"), errors="coerce")
            Kd = pd.to_numeric(raw.get("log Kd ([-])"), errors="coerce")
            if pfas and pd.notna(pH) and pd.notna(OC) and pd.notna(Kd):
                train_keys.add((pfas, round(float(pH), 4),
                                round(float(OC), 4), round(float(Kd), 6)))
    by_pfas: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for pfas, pH, OC, Kd in train_keys:
        by_pfas[pfas].append((pH, OC, Kd))
    keep = []
    for _, row in xie_df.iterrows():
        pfas = row["PFAS"]; pH = row["pH"]; OC = row["Corg_%"]; Kd = row["log_Kd"]
        if not pfas or pd.isna(pH) or pd.isna(OC) or pd.isna(Kd):
            continue
        matched = False
        for tpH, tOC, tKd in by_pfas.get(pfas, []):
            if (abs(pH - tpH) <= tol_pH
                and abs(OC - tOC) <= tol_OC
                and abs(Kd - tKd) <= tol_Kd):
                matched = True
                break
        if not matched:
            keep.append(row)
    return pd.DataFrame(keep)


# ----------------------------------------------------------------------
# Step 2: Load Xie 2024 Table 5
# ----------------------------------------------------------------------

def load_xie2024(smi_map):
    """Return DataFrame with: PFAS, MW_paper (RDKit), pH, OC, CEC, Sand, Silt, Clay,
    log_Kd (paper base 10). Convert from Xie's log₂ to log₁₀.

    NOTE on data access: The Xie et al. (2024) SI is published by Elsevier
    (Sci. Total Environ.) and is NOT publicly redistributable. To re-run
    this validation, download the SI from
    https://doi.org/10.1016/j.scitotenv.2024.176575 and extract Table S5
    to CSV. The expected columns are:
        PFAS, LogKd (Base 2, L/kg), pH, OC (%), CEC (cmol+/kg),
        Sand (%), Silt (%), Clay (%), MW (g/mol), LogP, LogS, ATSm8, SpDiam

    This function auto-extracts from
    data/source/Elucidating per- and polyfluoroalkyl2024-SI.docx
    (if present locally), or reads from a cached CSV in the system
    temp directory (if pre-extracted). The cache path uses
    `tempfile.gettempdir()` so the script is cross-platform
    (works on Linux / macOS / Windows / WSL).
    """
    import os
    import tempfile

    rows = []
    csv_path = None

    # Use system temp dir for cross-platform cache (13.6 — avoid hard-coded /tmp)
    cache_path = os.path.join(tempfile.gettempdir(), 'xie2024_table5.csv')

    # Try pre-extracted CSV first
    if os.path.exists(cache_path):
        csv_path = cache_path
    else:
        # Try auto-extract from the SI docx
        si_docx = _cfg.XIE_SI_DOCX
        if os.path.exists(si_docx):
            try:
                import docx
                doc = docx.Document(si_docx)
                for tbl in doc.tables:
                    if len(tbl.rows) > 0:
                        first = [c.text.strip() for c in tbl.rows[0].cells]
                        if 'PFAS' in first and 'LogKd' in first:
                            with open(cache_path, 'w', newline='') as f:
                                w = csv.writer(f)
                                for row in tbl.rows:
                                    w.writerow([c.text.strip() for c in row.cells])
                            csv_path = cache_path
                            print(f"  Auto-extracted {len(tbl.rows)} rows from Xie SI to {cache_path}")
                            break
            except Exception as e:
                print(f"  Could not auto-extract from SI: {e}")

    if csv_path is None:
        print("\n*** Xie 2024 SI not found locally. ***")
        print("    To run Xie external validation, either:")
        print("    1) Download from https://doi.org/10.1016/j.scitotenv.2024.176575")
        print("       and save to data/source/Elucidating per- and polyfluoroalkyl2024-SI.docx")
        print(f"    2) Or pre-extract Table S5 to {cache_path} with columns:")
        print("       PFAS, LogKd, pH, OC, CEC, Sand, Silt, Clay, MW, LogP, LogS, ATSm8, SpDiam")
        print("    Then re-run this script. Skipping Xie external validation for now.\n")
        return pd.DataFrame()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pfas = row['PFAS'].strip()
            if pfas not in smi_map:
                continue
            try:
                log_kd_log2 = float(row['LogKd'])
                ph = float(row['pH'])
                oc = float(row['OC'])
                cec = float(row['CEC'])
                sand = float(row['Sand'])
                silt = float(row['Silt'])
                clay = float(row['Clay'])
            except (ValueError, KeyError):
                continue
            # Use RDKit MW from SMILES (consistent with our paper)
            mw = get_mw(smi_map[pfas])
            if mw is None:
                continue
            log_kd_log10 = log_kd_log2 / np.log2(10)
            rows.append({
                'PFAS': pfas, 'MolWt': mw, 'pH': ph, 'Corg_%': oc, 'CEC': cec,
                'Sand': sand, 'Silt': silt, 'Clay': clay,
                'log_Kd': log_kd_log10, 'log_Kd_xie_log2': log_kd_log2,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Step 4: Train paper model variants
# ----------------------------------------------------------------------

def train_paper_variants(df, feature_cols):
    """Train 3 XGBoost variants and return dict of {name: (model, feat_names)}.

    Variants:
      A. Full: 145 features (RDKit post-filter + soil)
      B. Simplified: 4 features (MolWt + Corg + pH + CEC)
      C. Xie-aligned: 6 features (pH + OC + CEC + Sand + Silt + Clay)
    """
    X = df[feature_cols].values.astype(float)
    y = df['log_Kd'].values.astype(float)

    # Variant A: Full (max_depth=4 to be conservative for external)
    model_A = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model_A.fit(X, y)
    feat_A = feature_cols

    # Variant B: Simplified (4 features)
    simplified_feats = ['MolWt', 'Corg_%', 'pH', 'CEC']
    X_simp = df[simplified_feats].values.astype(float)
    model_B = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model_B.fit(X_simp, y)

    # Variant C: Xie-aligned (6 features)
    xie_feats = ['pH', 'Corg_%', 'CEC', 'Sand', 'Silt', 'Clay']
    X_xie = df[xie_feats].values.astype(float)
    model_C = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model_C.fit(X_xie, y)

    return {
        'A_Full_145feat': (model_A, feat_A, X, y),
        'B_Simplified_4feat': (model_B, simplified_feats, X_simp, y),
        'C_XieAligned_6feat': (model_C, xie_feats, X_xie, y),
    }


def predict_xie(models, xie_df):
    """Predict log10 Kd for Xie data using each variant.

    For variants with features not present in xie_df (e.g., the full 145-feat
    model), fill missing columns with 0 (XGBoost handles missing values).
    """
    results = {}
    for name, (model, feats, _, _) in models.items():
        # Reindex to ensure all feats exist, fill missing with 0
        X = xie_df.reindex(columns=feats, fill_value=0).values.astype(float)
        pred = model.predict(X)
        y = xie_df['log_Kd'].values
        r2 = r2_score(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        results[name] = {'pred': pred, 'r2': r2, 'rmse': rmse, 'y': y}
    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("  Cross-study benchmark: Xie et al. (2024)")
    print("=" * 60)

    # Step 1: Train 3 paper model variants
    print("\n[Step 1] Training 3 paper XGBoost variants on 47-PFAS data...")
    df, feat_cols = load_paper_data()
    print(f"  Paper: {df.shape[0]} rows, {len(feat_cols)} features")
    models = train_paper_variants(df, feat_cols)
    for name, (model, feats, _, _) in models.items():
        y_in = model.predict(models[name][2])
        r2_in = r2_score(models[name][3], y_in)
        print(f"  {name:25s}: in-sample R² = {r2_in:.4f} (trained on full 1227)")

    # Step 2: External validation on Xie 2024
    print("\n[Step 2] External validation: Xie et al. (2024) [25]...")
    smi_map = get_smi_map()
    xie_df = load_xie2024(smi_map)
    if len(xie_df) == 0:
        print("  ⚠️  Skipping Xie external validation (no overlapping data).")
        xie_results = {}
    else:
        print(f"  Xie overlap with paper: {xie_df['PFAS'].nunique()} unique PFAS, {len(xie_df)} rows")
        print(f"  PFAS: {sorted(xie_df['PFAS'].unique())}")
        xie_results = predict_xie(models, xie_df)
        for name, res in xie_results.items():
            print(f"  {name:25s}: R²={res['r2']:+.4f}, RMSE={res['rmse']:.4f}")
        xie_df.to_csv(os.path.join(DATA_PAPER, 'kd_external_validation_xie2024.csv'), index=False)

        # Write a disjoint-subset CSV for the figure panel + downstream use.
        # The disjoint subset removes the strict-overlap rows
        # (PFAS + pH +/- 0.05 + OC +/- 0.05 + log10 Kd +/- 0.005) so the
        # scatter reflects cross-study generalisation rather than
        # memorised training rows.
        disjoint_df = _xie_disjoint_subset(xie_df)
        disjoint_df.to_csv(
            os.path.join(DATA_PAPER, 'kd_external_validation_xie2024_disjoint.csv'),
            index=False,
        )
        print(f"  Xie disjoint subset: {len(disjoint_df)} rows")

    # Step 3: Disjoint-subset validation on Xie 2024.
    # The Xie SI Table S5 is itself a literature compilation that draws on
    # several of the same primary studies (Fabregat-Palau 2021, Knight 2019,
    # Milinovic 2015, etc.) that contributed to our training set.  After
    # removing the strict-overlap rows (PFAS + pH +/- 0.05 + OC +/- 0.05 +
    # log10 Kd +/- 0.005) the disjoint subset is evaluated instead.
    print("\n[Step 3] Disjoint-subset validation: Xie et al. (2024) [25]...")
    disjoint_path = os.path.join(DATA_PAPER, 'kd_xie_disjoint_validation.json')
    if not os.path.exists(disjoint_path):
        raise FileNotFoundError(
            f"Missing {disjoint_path}; run scripts/reevaluate_xie_disjoint.py first"
        )
    disjoint_report = json.loads(open(disjoint_path).read())
    xie_disjoint_r2 = disjoint_report["xie_disjoint_r2"]
    xie_disjoint_rmse = disjoint_report["xie_disjoint_rmse"]
    xie_disjoint_n = disjoint_report["xie_disjoint_rows"]
    print(f"  Xie disjoint subset: n={xie_disjoint_n}, R²={xie_disjoint_r2:+.4f}, "
          f"RMSE={xie_disjoint_rmse:.4f}")
    print(f"  (full-set R² = {disjoint_report['xie_full_r2']:+.4f}; "
          f"overlap rows removed = {disjoint_report['xie_overlap_rows_removed']})")

    # Step 4: Per-PFAS breakdown for the simplified model
    print("\n[Step 4] Per-PFAS R² for simplified model (Variant B)...")
    simp_model, simp_feats, _, _ = models['B_Simplified_4feat']
    print("  -- Xie 2024 --")
    xie_pred_simp = simp_model.predict(xie_df[simp_feats].values.astype(float))
    for pfas, grp in xie_df.groupby('PFAS'):
        idx = grp.index - xie_df.index[0]
        r2 = r2_score(grp['log_Kd'], xie_pred_simp[idx])
        rmse = np.sqrt(mean_squared_error(grp['log_Kd'], xie_pred_simp[idx]))
        print(f"    {pfas:12s} n={len(grp):4d}  R²={r2:+.4f}  RMSE={rmse:.4f}")

    # Step 5: Save figure
    print("\n[Step 5] Generating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Xie full model
    ax = axes[0]
    res = xie_results['A_Full_145feat']
    ax.scatter(res['y'], res['pred'], alpha=0.4, s=12, c='#3b82f6')
    lo, hi = min(res['y'].min(), res['pred'].min()), max(res['y'].max(), res['pred'].max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax.set_xlabel('Measured log₁₀ Kd (Xie 2024)')
    ax.set_ylabel('Predicted log₁₀ Kd (paper full model)')
    ax.set_title(f'A. Xie 2024 — Paper FULL model (145 feat)\nR²={res["r2"]:.3f}, RMSE={res["rmse"]:.3f}')
    ax.grid(alpha=0.3)

    # Panel B: Xie simplified model (full set)
    ax = axes[1]
    xie_pred_simp_arr = simp_model.predict(xie_df[simp_feats].values.astype(float))
    lo, hi = min(xie_df['log_Kd'].min(), xie_pred_simp_arr.min()), max(xie_df['log_Kd'].max(), xie_pred_simp_arr.max())
    ax.scatter(xie_df['log_Kd'], xie_pred_simp_arr, alpha=0.4, s=12, c='#10b981')
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    r2_simp_xie = r2_score(xie_df['log_Kd'], xie_pred_simp_arr)
    rmse_simp_xie = np.sqrt(mean_squared_error(xie_df['log_Kd'], xie_pred_simp_arr))
    ax.set_xlabel('Measured log₁₀ Kd (Xie 2024)')
    ax.set_ylabel('Predicted log₁₀ Kd (paper simplified model)')
    ax.set_title(f'B. Xie 2024 — Paper SIMPLIFIED model (4 feat)\n'
                 f'full set: R²={r2_simp_xie:.3f}, RMSE={rmse_simp_xie:.3f}')
    ax.grid(alpha=0.3)

    # Panel C: Xie simplified model on the strict-overlap-removed subset.
    ax = axes[2]
    disjoint = pd.read_csv(
        os.path.join(DATA_PAPER, 'kd_external_validation_xie2024_disjoint.csv')
    )
    disjoint_features = disjoint[['MolWt', 'Corg_%', 'pH', 'CEC']].values.astype(float)
    y_dis = disjoint['log_Kd'].values
    pred_dis = simp_model.predict(disjoint_features)
    r2_dis = r2_score(y_dis, pred_dis)
    rmse_dis = np.sqrt(mean_squared_error(y_dis, pred_dis))
    lo, hi = min(y_dis.min(), pred_dis.min()), max(y_dis.max(), pred_dis.max())
    ax.scatter(y_dis, pred_dis, alpha=0.4, s=12, c='#a855f7')
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax.set_xlabel('Measured log₁₀ Kd (Xie 2024, disjoint subset)')
    ax.set_ylabel('Predicted log₁₀ Kd (paper simplified model)')
    ax.set_title(f'C. Xie 2024 — DISJOINT subset (no training overlap)\n'
                 f'n={len(y_dis)}, R²={r2_dis:.3f}, RMSE={rmse_dis:.3f}')
    ax.grid(alpha=0.3)

    plt.suptitle('External cross-study benchmark of paper XGBoost Kd model on Xie et al. (2024) [25]',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(DATA_PAPER, 'fig10_external_validation.png'), dpi=200, bbox_inches='tight')
    print(f"  Saved: fig10_external_validation.png")

    # Summary
    print("\n" + "=" * 60)
    print("  CROSS-STUDY BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Paper model in-sample R²:                0.87 (paper §3.2)")
    print(f"  Xie 2024 — paper full model R²:         {xie_results['A_Full_145feat']['r2']:+.4f}")
    print(f"  Xie 2024 — paper simplified R²:         {r2_simp_xie:+.4f}")
    print(f"  Xie 2024 — Xie-aligned 6-feat R²:        {xie_results['C_XieAligned_6feat']['r2']:+.4f}")
    print(f"  Xie 2024 — disjoint subset R²:          {r2_dis:+.4f}  (n={len(y_dis)})")
    print("=" * 60)
