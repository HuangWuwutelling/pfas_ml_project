"""Tests for the MolWt + Corg + pH three-feature simplified model.

These tests pin down the behaviour we need from the augmentation script:

* the trained model uses the same train/test split and XGBoost hyper-
  parameters as the existing four-feature simplified model, so numbers
  in the simplified results table stay comparable;
* the new row is appended to ``data/paper/kd_simplified_results.csv``;
* the resulting R^2 must lie between the all-features R^2 (~0.87) and
  the Corg+pH+CEC R^2 (~0.20) — it is, by design, an intermediate
  simplified model.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "augment_simplified_models.py"
RESULTS_CSV = ROOT / "data" / "paper" / "kd_simplified_results.csv"
FEATURE_MATRIX = ROOT / "data" / "paper" / "feature_matrix_kd.csv"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("augment_simplified_models", SCRIPT)
    assert spec and spec.loader, f"Cannot import {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_matrix_has_required_columns() -> None:
    """The augmentation must find MolWt, Corg_%, pH and log_Kd."""
    with open(FEATURE_MATRIX) as f:
        reader = csv.reader(f)
        header = next(reader)
    for col in ("MolWt", "Corg_%", "pH", "log_Kd"):
        assert col in header, f"feature_matrix_kd.csv missing column {col}"


def test_results_csv_exists_with_baseline_rows() -> None:
    """Anchor: the existing simplified-results table is the input to augment."""
    assert RESULTS_CSV.exists(), f"missing baseline {RESULTS_CSV}"
    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))
    assert any(r["model"] == "MolWt+Corg+pH+CEC" for r in rows)
    assert any(r["model"] == "Corg + pH + CEC only" for r in rows)


def test_three_feature_model_metrics_are_intermediate(tmp_path: Path) -> None:
    """Run the augmentation in a sandboxed output dir and assert metrics.

    The MolWt+Corg+pH model must out-perform the soil-only (Corg+pH+CEC)
    model but stay below the full combined model — that's the whole
    scientific point of the row we are adding.
    """
    module = _load_module()
    output_csv = tmp_path / "kd_simplified_results.csv"
    result = module.run(
        feature_matrix=FEATURE_MATRIX,
        output_results=output_csv,
    )

    assert result["n_features"] == 3
    assert result["features"] == ["MolWt", "Corg_%", "pH"]
    assert result["n_samples"] > 1000, "expected the full 1227-row matrix"

    # Performance: must sit between soil-only and the full model.
    full_r2 = next(
        r for r in csv.DictReader(open(RESULTS_CSV))
        if r["model"] == "All features (RDKit + soil)"
    )["r2"]
    soil_r2 = next(
        r for r in csv.DictReader(open(RESULTS_CSV))
        if r["model"] == "Corg + pH + CEC only"
    )["r2"]

    full_r2 = float(full_r2)
    soil_r2 = float(soil_r2)
    r2 = result["r2"]

    assert soil_r2 < r2 < full_r2, (
        f"expected soil_r2={soil_r2:.3f} < three-feat R^2={r2:.3f} "
        f"< full_r2={full_r2:.3f}"
    )
    assert 0.0 < r2 < 1.0
    assert 0.0 < result["rmse"] < 1.0
    assert result["rpd"] > 1.0


def test_row_is_appended_to_simplified_results(tmp_path: Path) -> None:
    """The new row must land in the results CSV with a stable label."""
    module = _load_module()
    output_csv = tmp_path / "kd_simplified_results.csv"
    module.run(
        feature_matrix=FEATURE_MATRIX,
        output_results=output_csv,
    )

    with open(output_csv) as f:
        rows = list(csv.DictReader(f))

    three_feat_rows = [r for r in rows if r["model"] == "MolWt+Corg+pH"]
    assert len(three_feat_rows) == 1, three_feat_rows
    row = three_feat_rows[0]
    assert int(row["n_features"]) == 3
    assert row["r2"]
    assert row["rmse"]
    assert row["rpd"]


def test_does_not_mutate_baseline_results_csv(tmp_path: Path) -> None:
    """The script must take an output path, not edit the committed CSV in place."""
    before = RESULTS_CSV.read_text()
    module = _load_module()
    module.run(
        feature_matrix=FEATURE_MATRIX,
        output_results=tmp_path / "kd_simplified_results.csv",
    )
    after = RESULTS_CSV.read_text()
    assert before == after, "committed kd_simplified_results.csv was mutated"
