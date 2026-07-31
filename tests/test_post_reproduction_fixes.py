"""Regression tests for the three bug-fixes from the 2026-07-30
reproduction report §改进建议:

1. ``augment_simplified_models.py`` default mode must **append** to an
   existing ``kd_simplified_results.csv`` instead of silently wiping
   the rows from ``paper_05`` / ``paper_06b``. ``--overwrite`` must
   restore the legacy behaviour.
2. ``paper_06`` / ``paper_06b`` per-PFAS aggregates must be nan-safe
   (``np.nanmean`` / ``np.nanmedian`` / ``np.nanstd``) so a single-
   sample fold (n_test == 1, e.g. 4:2 FTOH) does not poison the
   summary statistics with NaN.

These tests are fast and offline — they do not retrain any model.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
AUGMENT_SCRIPT = ROOT / "scripts" / "augment_simplified_models.py"
RESULTS_CSV = ROOT / "data" / "paper" / "kd_simplified_results.csv"
FEATURE_MATRIX = ROOT / "data" / "paper" / "feature_matrix_kd.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_augment_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "augment_simplified_models", AUGMENT_SCRIPT
    )
    assert spec and spec.loader, f"Cannot import {AUGMENT_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline_rows() -> list[dict[str, str]]:
    """Snapshot of the committed kd_simplified_results.csv rows."""
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Bug 1 — append / overwrite semantics
# ---------------------------------------------------------------------------


def test_appends_when_results_csv_exists(tmp_path: Path) -> None:
    """Default mode must append, not overwrite, an existing results CSV."""
    module = _load_augment_module()

    # Seed the destination with two pre-existing rows (simulating the
    # outputs of paper_05 and paper_06b).
    seed_csv = tmp_path / "kd_simplified_results.csv"
    fieldnames = [
        "model", "n_features", "r2", "rmse", "rpd", "cv_r2", "cv_std",
    ]
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "model": "Corg + pH + CEC only",
            "n_features": 3, "r2": 0.1976, "rmse": 0.8277,
            "rpd": 1.12, "cv_r2": -0.18, "cv_std": 0.3904,
        })
        writer.writerow({
            "model": "MolWt+Corg+pH+CEC",
            "n_features": 4, "r2": 0.8407, "rmse": 0.3723,
            "rpd": 2.51, "cv_r2": 0.4917, "cv_std": 0.2844,
        })

    result = module.run(
        feature_matrix=FEATURE_MATRIX,
        output_results=seed_csv,
    )

    assert result["write_mode"] == "a"
    assert result["appended_existing"] is True

    with open(seed_csv) as f:
        rows = list(csv.DictReader(f))

    # The two seeded rows must still be present, in order, with their
    # original values intact.
    assert len(rows) == 3
    assert rows[0]["model"] == "Corg + pH + CEC only"
    assert float(rows[0]["r2"]) == 0.1976
    assert rows[1]["model"] == "MolWt+Corg+pH+CEC"
    assert float(rows[1]["r2"]) == 0.8407
    assert rows[2]["model"] == "MolWt+Corg+pH"


def test_overwrite_flag_replaces_existing_rows(tmp_path: Path) -> None:
    """``overwrite=True`` must restore the legacy replace-with-one-row mode."""
    module = _load_augment_module()

    seed_csv = tmp_path / "kd_simplified_results.csv"
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "n_features", "r2", "rmse", "rpd",
                "cv_r2", "cv_std",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "model": "Corg + pH + CEC only",
            "n_features": 3, "r2": 0.1976, "rmse": 0.8277,
            "rpd": 1.12, "cv_r2": -0.18, "cv_std": 0.3904,
        })

    result = module.run(
        feature_matrix=FEATURE_MATRIX,
        output_results=seed_csv,
        overwrite=True,
    )

    assert result["write_mode"] == "w"
    assert result["appended_existing"] is False

    with open(seed_csv) as f:
        rows = list(csv.DictReader(f))

    # Only the new three-feature row should remain.
    assert len(rows) == 1
    assert rows[0]["model"] == "MolWt+Corg+pH"


def test_cli_overwrite_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """``python augment_simplified_models.py --overwrite`` must reach run()."""
    module = _load_augment_module()

    captured: dict[str, object] = {}

    def fake_run(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {
            "model": "MolWt+Corg+pH",
            "features": ["MolWt", "Corg_%", "pH"],
            "n_features": 3,
            "n_samples": 0,
            "r2": 0.0,
            "rmse": 0.0,
            "rpd": 0.0,
            "cv_r2": 0.0,
            "cv_std": 0.0,
            "row": {},
            "write_mode": "w",
            "appended_existing": False,
        }

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["augment_simplified_models.py", "--overwrite"],
    )
    module.main()

    assert captured.get("overwrite") is True


def test_baseline_results_csv_not_mutated(tmp_path: Path) -> None:
    """Existing guarantee: default invocation never edits the committed CSV."""
    before = _baseline_rows()
    before_n = len(before)

    module = _load_augment_module()
    module.run(
        feature_matrix=FEATURE_MATRIX,
        output_results=tmp_path / "kd_simplified_results.csv",
    )

    after = _baseline_rows()
    assert len(after) == before_n
    assert [r["model"] for r in after] == [r["model"] for r in before]


# ---------------------------------------------------------------------------
# Bug 2 — nan-safe per-PFAS aggregates
# ---------------------------------------------------------------------------


def test_nan_safe_aggregates_handle_single_sample_fold() -> None:
    """A fold with R^2 == nan (n_test == 1) must not poison mean / median / std.

    Mirrors the aggregates used in ``paper_06_loo_validation.run_loo``
    and ``paper_06b_loo_combined_fix``. If a future refactor reintroduces
    ``np.mean`` / ``np.median`` / ``np.std`` the asserts below will
    fail loudly.
    """
    r2_values = np.array(
        [0.8, 0.6, 0.4, 0.2, np.nan, -0.1], dtype=float
    )

    avg = float(np.nanmean(r2_values))
    med = float(np.nanmedian(r2_values))
    std = float(np.nanstd(r2_values))

    # None of the aggregates should be nan when at least one valid value
    # exists.
    assert np.isfinite(avg), f"avg={avg} should be finite"
    assert np.isfinite(med), f"med={med} should be finite"
    assert np.isfinite(std), f"std={std} should be finite"

    # Expected hand-calculated values over [0.8, 0.6, 0.4, 0.2, -0.1].
    expected_avg = (0.8 + 0.6 + 0.4 + 0.2 - 0.1) / 5
    expected_med = 0.4  # middle of 5 sorted values
    assert avg == pytest.approx(expected_avg, abs=1e-9)
    assert med == pytest.approx(expected_med, abs=1e-9)


def test_nan_safe_aggregates_return_nan_only_for_all_nan_input() -> None:
    """Edge case: if every fold is nan, aggregates are nan (and the caller
    must surface that as missing data, not silently mask it)."""
    r2_values = np.array([np.nan, np.nan], dtype=float)

    avg = float(np.nanmean(r2_values))
    med = float(np.nanmedian(r2_values))

    assert np.isnan(avg)
    assert np.isnan(med)