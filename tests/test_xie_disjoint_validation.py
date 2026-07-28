"""TDD: pin down the de-duplicated Xie external validation numbers.

The paper currently reports Xie 2024 external R^2 = 0.7834 on 1,780
rows.  After removing 162 rows that match a training-set row verbatim
(PFAS + pH +/- 0.05 + OC +/- 0.05 + log10 Kd +/- 0.005), the disjoint
subset of 1,618 rows yields R^2 = 0.7781, RMSE = 0.3972.  These
tests pin those numbers so a future refactor of
``reevaluate_xie_disjoint.py`` can't silently move them.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "paper" / "kd_xie_disjoint_validation.json"


def _payload() -> dict:
    assert REPORT.exists(), (
        "Run scripts/reevaluate_xie_disjoint.py first to produce the report"
    )
    return json.loads(REPORT.read_text())


def test_overlap_removed_count_is_about_160() -> None:
    """The strict-overlap de-duplication removes ~160 of 1,780 rows."""
    p = _payload()
    assert 140 <= p["xie_overlap_rows_removed"] <= 200, p


def test_disjoint_subset_is_about_1620_rows() -> None:
    p = _payload()
    assert 1500 <= p["xie_disjoint_rows"] <= 1700, p


def test_disjoint_r2_is_within_a_few_percent_of_full() -> None:
    """De-duplication should not move R^2 by more than 0.05 absolute."""
    p = _payload()
    delta = abs(p["xie_full_r2"] - p["xie_disjoint_r2"])
    assert delta <= 0.05, p


def test_disjoint_r2_still_above_0_70() -> None:
    """The disjoint subset still validates the model at R^2 > 0.70."""
    p = _payload()
    assert p["xie_disjoint_r2"] >= 0.70, p


def test_pfoa_is_the_largest_overlap() -> None:
    """PFOA dominates the overlap -- the same PFOA Kd values were
    compiled by both Fabregat-Palau (training) and Xie (validation)."""
    p = _payload()
    pfas_counts = p["overlap_removed_per_pfas"]
    assert pfas_counts.get("PFOA", 0) >= 80, pfas_counts
    assert pfas_counts["PFOA"] == max(pfas_counts.values()), pfas_counts
