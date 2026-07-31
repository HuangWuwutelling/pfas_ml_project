#!/usr/bin/env python3
"""Augment kd_simplified_results.csv with a MolWt + Corg + pH model row.

The four-feature simplified model in the paper (MolWt + Corg + pH + CEC)
out-performs the soil-only model (Corg + pH + CEC) by ~0.64 R^2 — but
the relative contribution of MolWt vs CEC to that jump is not yet
isolated. This script trains a *third* simplified model (MolWt + Corg +
pH) on the same 80/20 split and XGBoost hyperparameters as
``paper_05_core_descriptors.train_xgb`` so the numbers in
``kd_simplified_results.csv`` stay directly comparable, and appends
the row to a caller-supplied results CSV (does NOT touch the
committed ``data/paper/kd_simplified_results.csv`` in place).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

FEATURE_MATRIX = Path("data/paper/feature_matrix_kd.csv")
DEFAULT_OUTPUT = Path("data/paper/kd_simplified_results.csv")
PREDICTIONS_OUTPUT = Path("data/paper/kd_simplified_three_feat_predictions.csv")

THREE_FEAT_FEATURES: tuple[str, ...] = ("MolWt", "Corg_%", "pH")
MODEL_LABEL = "MolWt+Corg+pH"

# Match paper_05_core_descriptors.train_xgb exactly so the new row is
# directly comparable to the existing rows in kd_simplified_results.csv.
RANDOM_STATE = 42
TEST_SIZE = 0.2
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def _load_xy(
    feature_matrix: Path, features: Sequence[str]
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(feature_matrix)
    missing = [c for c in (*features, "log_Kd") if c not in df.columns]
    if missing:
        raise ValueError(f"{feature_matrix} missing columns: {missing}")
    return df[list(features)].copy(), df["log_Kd"].copy()


def run(
    feature_matrix: Path = FEATURE_MATRIX,
    output_results: Path = DEFAULT_OUTPUT,
    predictions_output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train the three-feature model and append the row to ``output_results``.

    By default the new row is **appended** to an existing results CSV (or
    a new file with header is created if none exists). This avoids
    silently wiping the rows produced by ``paper_05`` and ``paper_06b``.
    Pass ``overwrite=True`` to fall back to the legacy behaviour (replace
    the whole file with the single new row).
    """
    X, y = _load_xy(feature_matrix, THREE_FEAT_FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    rpd = float(np.std(y_test) / rmse) if rmse > 0 else float("inf")

    cv_scores = cross_val_score(
        xgb.XGBRegressor(**XGB_PARAMS),
        X,
        y,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    new_row = {
        "model": MODEL_LABEL,
        "n_features": len(THREE_FEAT_FEATURES),
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "rpd": round(rpd, 2),
        "cv_r2": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
    }

    fieldnames = ["model", "n_features", "r2", "rmse", "rpd", "cv_r2", "cv_std"]
    output_results.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not output_results.exists():
        mode = "w"
        write_header = True
    else:
        mode = "a"
        write_header = False

    with open(output_results, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(new_row)

    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        pred_df = X_test.reset_index(drop=True)
        pred_df["log_Kd"] = y_test.reset_index(drop=True)
        pred_df["pred_log_Kd"] = y_pred
        pred_df.to_csv(predictions_output, index=False)

    return {
        "model": MODEL_LABEL,
        "features": list(THREE_FEAT_FEATURES),
        "n_features": len(THREE_FEAT_FEATURES),
        "n_samples": int(len(X)),
        "r2": r2,
        "rmse": rmse,
        "rpd": rpd,
        "cv_r2": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "row": new_row,
        "write_mode": mode,
        "appended_existing": not write_header,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        default=FEATURE_MATRIX,
        help="Path to feature_matrix_kd.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the augmented kd_simplified_results.csv (default: append to %(default)s)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTIONS_OUTPUT,
        help="Where to write test-set predictions for plotting (use 'none' to skip)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace the entire results CSV with only the new row (legacy "
            "behaviour). Default is to append to the existing file so that "
            "rows from paper_05 / paper_06b are preserved."
        ),
    )
    args = parser.parse_args()

    predictions = None if str(args.predictions).lower() == "none" else args.predictions
    result = run(
        feature_matrix=args.feature_matrix,
        output_results=args.output,
        predictions_output=predictions,
        overwrite=args.overwrite,
    )
    verb = "Overwrote" if result["write_mode"] == "w" else "Appended"
    print(f"{verb} {result['model']} to {args.output} (mode={result['write_mode']})")
    print(
        f"  R²={result['r2']:.4f}  RMSE={result['rmse']:.4f}  "
        f"RPD={result['rpd']:.2f}  5-fold CV R²={result['cv_r2']:.4f}"
    )


if __name__ == "__main__":
    main()
