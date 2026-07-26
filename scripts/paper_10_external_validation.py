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

  2. Morales et al. (2026) Environ. Res. 306(1), 125071
     - 14 PFAS, 362 rows, 84 field-contaminated soils, Lyon France
     - log₁₀ Kd (L/kg) — already in our base

Strategy: Train 3 XGBoost variants on the 47-PFAS paper data, then test
on Xie and Morales:

  Variant 1: Full model (145 features) — paper §3.2
  Variant 2: Simplified model (4 features: MolWt + Corg + pH + CEC) — paper §3.5
  Variant 3: Xie-aligned 6-feature model (pH + OC + CEC + Sand + Silt + Clay)
              — uses only common soil features

  For Xie 2024 (which has MW/LogP etc), apply Variant 2 (uses our MolWt +
  3 soil features). Variant 3 uses Xie's soil features + our MW.
"""
import csv
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
# Step 3: Load Morales 2026 SI
# ----------------------------------------------------------------------

def load_morales2026(smi_map):
    """Return DataFrame with: PFAS, MW (RDKit), pH, OC, log_Kd.
    Morales 2026 has pH + OC% + Kd (in L/kg) per soil-PFAS.

    IMPORTANT UNIT CONVERSION:
    Morales reports log10_Kd in mL/g (= log10(L/kg) + 3). To convert to
    log10 Kd in L/kg (our paper's unit), subtract 3.
    """
    rows = []
    with open(_cfg.MORALES_LONG) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pfas = row['PFAS'].strip()
            if pfas not in smi_map:
                continue
            try:
                # Unit conversion: log10(Kd in L/kg) = log10_Kd_L_per_kg - 3
                log_kd_L_per_kg = float(row['log10_Kd_L_per_kg']) - 3.0
                oc_pct = float(row['OC_pct'])
                ph = float(row['pH'])
            except (ValueError, KeyError):
                continue
            mw = get_mw(smi_map[pfas])
            if mw is None:
                continue
            rows.append({
                'PFAS': pfas, 'MW': mw, 'MolWt': mw, 'pH': ph, 'OC': oc_pct,
                'Corg_%': oc_pct,  # alias for paper
                'log_Kd': log_kd_L_per_kg,
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


def predict_morales(models, morales_df):
    results = {}
    for name, (model, feats, _, _) in models.items():
        if 'Corg_%' in feats:
            # Reindex
            X = morales_df.reindex(columns=feats, fill_value=0).values.astype(float)
            pred = model.predict(X)
            y = morales_df['log_Kd'].values
            r2 = r2_score(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            results[name] = {'pred': pred, 'r2': r2, 'rmse': rmse, 'y': y}
    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("  External validation: Xie 2024 + Morales 2026")
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

    # Step 3: External validation on Morales 2026
    print("\n[Step 3] External validation: Morales et al. (2026) [26]...")
    morales_df = load_morales2026(smi_map)
    print(f"  Morales overlap: {morales_df['PFAS'].nunique()} unique PFAS, {len(morales_df)} rows (all cols)")
    # Drop rows with NaN in any required feature (Morales has 288 NaN in pH)
    # Note: Morales does not provide CEC, so for Morales we use 3 features
    # (MolWt, Corg_%, pH) instead of the 4-feature paper simplified model.
    morales_feats = ['MolWt', 'Corg_%', 'pH']
    morales_clean = morales_df.dropna(subset=morales_feats + ['log_Kd'])
    print(f"  Morales clean (pH not NaN): {len(morales_clean)} rows")
    print(f"  PFAS: {sorted(morales_clean['PFAS'].unique())}")
    # Predict with a Morales-specific 3-feature model (trained on paper 47 PFAS)
    model_M = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    paper_data = pd.read_csv(os.path.join(DATA_PAPER, 'feature_matrix_kd.csv'))
    X_M = paper_data[morales_feats].values.astype(float)
    y_M = paper_data['log_Kd'].values.astype(float)
    model_M.fit(X_M, y_M)
    X_mor_clean = morales_clean[morales_feats].values.astype(float)
    mor_pred = model_M.predict(X_mor_clean)
    r2_mor = r2_score(morales_clean['log_Kd'], mor_pred)
    rmse_mor = np.sqrt(mean_squared_error(morales_clean['log_Kd'], mor_pred))
    print(f"  Morales 3-feat (MolWt+Corg+pH): R²={r2_mor:+.4f}, RMSE={rmse_mor:.4f}")
    morales_results = {'Morales_3feat': {'pred': mor_pred, 'r2': r2_mor, 'rmse': rmse_mor,
                                       'y': morales_clean['log_Kd'].values}}
    morales_clean.to_csv(os.path.join(DATA_PAPER, 'kd_external_validation_morales2026.csv'), index=False)

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

    print("\n  -- Morales 2026 --")
    for pfas, grp in morales_clean.groupby('PFAS'):
        if len(grp) < 3:
            continue
        idx = grp.index - morales_clean.index[0]
        r2 = r2_score(grp['log_Kd'], mor_pred[idx])
        rmse = np.sqrt(mean_squared_error(grp['log_Kd'], mor_pred[idx]))
        print(f"    {pfas:12s} n={len(grp):4d}  R²={r2:+.4f}  RMSE={rmse:.4f}")

    # Step 5: Save figure
    print("\n[Step 5] Generating figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Xie full model
    ax = axes[0, 0]
    res = xie_results['A_Full_145feat']
    ax.scatter(res['y'], res['pred'], alpha=0.4, s=12, c='#3b82f6')
    lo, hi = min(res['y'].min(), res['pred'].min()), max(res['y'].max(), res['pred'].max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax.set_xlabel('Measured log₁₀ Kd (Xie 2024)')
    ax.set_ylabel('Predicted log₁₀ Kd (paper full model)')
    ax.set_title(f'A. Xie 2024 — Paper FULL model (145 feat)\nR²={res["r2"]:.3f}, RMSE={res["rmse"]:.3f}')
    ax.grid(alpha=0.3)

    # Panel B: Xie simplified model
    ax = axes[0, 1]
    xie_pred_simp_arr = simp_model.predict(xie_df[simp_feats].values.astype(float))
    lo, hi = min(xie_df['log_Kd'].min(), xie_pred_simp_arr.min()), max(xie_df['log_Kd'].max(), xie_pred_simp_arr.max())
    ax.scatter(xie_df['log_Kd'], xie_pred_simp_arr, alpha=0.4, s=12, c='#10b981')
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    r2_simp_xie = r2_score(xie_df['log_Kd'], xie_pred_simp_arr)
    rmse_simp_xie = np.sqrt(mean_squared_error(xie_df['log_Kd'], xie_pred_simp_arr))
    ax.set_xlabel('Measured log₁₀ Kd (Xie 2024)')
    ax.set_ylabel('Predicted log₁₀ Kd (paper simplified model)')
    ax.set_title(f'B. Xie 2024 — Paper SIMPLIFIED model (4 feat)\nR²={r2_simp_xie:.3f}, RMSE={rmse_simp_xie:.3f}')
    ax.grid(alpha=0.3)

    # Panel C: Morales simplified model
    ax = axes[1, 0]
    lo, hi = min(morales_clean['log_Kd'].min(), mor_pred.min()), max(morales_clean['log_Kd'].max(), mor_pred.max())
    ax.scatter(morales_clean['log_Kd'], mor_pred, alpha=0.4, s=12, c='#f59e0b')
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    r2_mor = r2_score(morales_clean['log_Kd'], mor_pred)
    rmse_mor = np.sqrt(mean_squared_error(morales_clean['log_Kd'], mor_pred))
    ax.set_xlabel('Measured log₁₀ Kd (Morales 2026)')
    ax.set_ylabel('Predicted log₁₀ Kd (paper 3-feat model)')
    ax.set_title(f'C. Morales 2026 — Paper 3-feat (MolWt+Corg+pH)\nR²={r2_mor:.3f}, RMSE={rmse_mor:.3f}, n={len(morales_clean)}')
    ax.grid(alpha=0.3)

    # Panel D: summary bar chart
    ax = axes[1, 1]
    bar_labels = ['Xie (full)', 'Xie (simplified)', 'Morales (3-feat)']
    bar_r2 = [xie_results['A_Full_145feat']['r2'], r2_simp_xie, r2_mor]
    x_pos = np.arange(len(bar_labels))
    ax.bar(x_pos, bar_r2, color=['#3b82f6', '#10b981', '#f59e0b'], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('External R²')
    ax.set_title(f'D. Summary: External R² (paper in-sample = 0.87)')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(0.87, color='gray', linestyle=':', linewidth=1, label='In-sample R²=0.87')
    ax.legend(fontsize=8)
    for i, v in enumerate(bar_r2):
        ax.text(i, v + (0.02 if v >= 0 else -0.05), f'{v:.3f}', ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('External validation of paper XGBoost Kd model on independent datasets', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(DATA_PAPER, 'fig10_external_validation.png'), dpi=200, bbox_inches='tight')
    print(f"  Saved: fig10_external_validation.png")

    # Summary
    print("\n" + "=" * 60)
    print("  EXTERNAL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Paper model in-sample R²:                0.87 (paper §3.2)")
    print(f"  Xie 2024 — paper full model R²:         {xie_results['A_Full_145feat']['r2']:+.4f}")
    print(f"  Xie 2024 — paper simplified R²:         {r2_simp_xie:+.4f}")
    print(f"  Xie 2024 — Xie-aligned 6-feat R²:        {xie_results['C_XieAligned_6feat']['r2']:+.4f}")
    print(f"  Morales 2026 — paper simplified R²:      {r2_mor:+.4f}")
    print("=" * 60)
