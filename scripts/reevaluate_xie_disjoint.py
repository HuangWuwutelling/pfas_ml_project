"""Re-run Xie external validation after removing strict-overlap rows.

The strict tolerance band is PFAS + pH +/- 0.05 + OC +/- 0.05 +
log10(Kd) +/- 0.005.  Under that band, 162 of the 1,780 Xie rows we
actually use for the external validation match a row in the
training set verbatim -- mostly the PFOA entries that came from the
same source publication.  We drop those rows before re-scoring the
paper's simplified 4-feature XGBoost model on the disjoint remainder.

Outputs:
  data/paper/kd_xie_disjoint_validation.json  -- summary numbers
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/paper/Final_data.csv"
XIE = Path("/tmp/xie2024_table5.csv")
FEATURE_MATRIX = ROOT / "data/paper/feature_matrix_kd.csv"
PFAS_PROPS = ROOT / "data/paper/PFAS_Properties.csv"
REPORT = ROOT / "data/paper/kd_xie_disjoint_validation.json"

STRICT = dict(pH=0.05, OC=0.05, Kd=0.005)


def to_f(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def load_train_keys() -> set[tuple[str, float, float, float]]:
    keys: set[tuple[str, float, float, float]] = set()
    with open(TRAIN) as f:
        for raw in csv.DictReader(f):
            pfas = (raw.get("PFAS (abreviation)") or "").strip()
            pH = to_f(raw.get("pH (measured)"))
            OC = to_f(raw.get("Corg (%)"))
            Kd = to_f(raw.get("log Kd ([-])"))
            if not (pfas and None not in (pH, OC, Kd)):
                continue
            keys.add((pfas, round(pH, 4), round(OC, 4), round(Kd, 6)))
    return keys


def load_xie() -> pd.DataFrame:
    rows = []
    with open(XIE) as f:
        reader = csv.reader(f)
        next(reader); next(reader)
        for raw in reader:
            pfas = (raw[0] or "").strip()
            lk2 = to_f(raw[1])
            if lk2 is None:
                continue
            pH = to_f(raw[2]); OC = to_f(raw[3]); CEC = to_f(raw[4])
            if None in (pH, OC, CEC):
                continue
            rows.append({
                "pfas": pfas,
                "pH": pH, "OC": OC, "CEC": CEC,
                "log_Kd": lk2 / math.log2(10.0),
            })
    return pd.DataFrame(rows)


def molwt_map() -> dict[str, float]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    smi: dict[str, str] = {}
    with open(PFAS_PROPS) as f:
        for raw in csv.DictReader(f):
            abbrev = (raw.get("PFAS abbreviation") or "").strip()
            s = (raw.get("Smiles") or "").strip()
            if abbrev and s and s.upper() not in ("N.A.", "NA", "NONE", ""):
                smi[abbrev] = s
    out: dict[str, float] = {}
    for a, s in smi.items():
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out[a] = Descriptors.MolWt(mol)
    return out


def main() -> None:
    train_keys = load_train_keys()
    xie = load_xie()
    mw = molwt_map()
    xie["MolWt"] = xie["pfas"].map(mw)
    xie = xie.dropna(subset=["MolWt", "pH", "OC", "CEC", "log_Kd"])
    fm = pd.read_csv(FEATURE_MATRIX)

    # Build an index from the training set for fast lookup, grouped by PFAS.
    train_by_pfas: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for pfas, pH, OC, Kd in train_keys:
        train_by_pfas[pfas].append((pH, OC, Kd))

    def is_overlap(row) -> bool:
        for pH, OC, Kd in train_by_pfas.get(row["pfas"], []):
            if (abs(row["pH"] - pH) <= STRICT["pH"]
                and abs(row["OC"] - OC) <= STRICT["OC"]
                and abs(row["log_Kd"] - Kd) <= STRICT["Kd"]):
                return True
        return False

    xie["overlap"] = xie.apply(is_overlap, axis=1)
    disjoint = xie[~xie["overlap"]].copy()
    overlap_count = int(xie["overlap"].sum())

    feats = ["MolWt", "OC", "pH", "CEC"]
    X_tr = fm[["MolWt", "Corg_%", "pH", "CEC"]].values.astype(float)
    y_tr = fm["log_Kd"].values.astype(float)

    X_xi_full = xie[feats].values.astype(float)
    y_xi_full = xie["log_Kd"].values
    X_xi_dis = disjoint[feats].values.astype(float)
    y_xi_dis = disjoint["log_Kd"].values

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    y_pred_full = model.predict(X_xi_full)
    y_pred_dis = model.predict(X_xi_dis)

    report = {
        "strict_tolerance": STRICT,
        "xie_input_rows": int(len(xie)),
        "xie_overlap_rows_removed": overlap_count,
        "xie_disjoint_rows": int(len(disjoint)),
        "model": "4-feature XGBoost (MolWt+Corg+pH+CEC) trained on 47-PFAS paper data",
        "xie_full_r2": round(float(r2_score(y_xi_full, y_pred_full)), 4),
        "xie_full_rmse": round(float(np.sqrt(mean_squared_error(y_xi_full, y_pred_full))), 4),
        "xie_disjoint_r2": round(float(r2_score(y_xi_dis, y_pred_dis)), 4),
        "xie_disjoint_rmse": round(float(np.sqrt(mean_squared_error(y_xi_dis, y_pred_dis))), 4),
        "overlap_removed_per_pfas": (
            xie[xie["overlap"]]["pfas"].value_counts().to_dict()
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
