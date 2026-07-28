"""TDD: build Table S5 (Xie 2024 reference vs training-set row overlap).

This is a unit test that pins down the per-source overlap counts we want
to publish in supplementary table S5.  The script
``scripts/check_xie_train_overlap.py`` already produces the underlying
JSON; this test just asserts the *table* values the paper will cite.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "paper" / "kd_xie_train_overlap_report.json"
SI_TABLE = ROOT / "data" / "paper" / "tableS5_xie_source_overlap.csv"


def test_report_has_required_keys() -> None:
    payload = json.loads(REPORT.read_text())
    for k in (
        "train_total_rows", "xie_total_rows", "train_pfas_unique",
        "xie_pfas_unique", "pfas_in_both", "train_rows_in_xie_strict",
        "feature_matrix_rows_in_xie_strict", "by_author_top10",
    ):
        assert k in payload, f"missing {k} in overlap report"


def test_table_s5_file_is_published() -> None:
    assert SI_TABLE.exists(), (
        "scripts/build_xie_overlap_table.py has not been run yet"
    )
    with open(SI_TABLE) as f:
        rows = list(csv.DictReader(f))
    assert rows, "Table S5 is empty"
    cols = set(rows[0].keys())
    expected = {"source", "xie_cited", "train_rows", "strict_in_xie", "pct_strict"}
    assert expected <= cols, f"missing columns {expected - cols}"


def test_knight_2019_is_dominant_strict_overlap() -> None:
    payload = json.loads(REPORT.read_text())
    by_author = {
        (entry["author"], entry["year"]): entry
        for entry in payload["by_author_top10"]
    }
    knight = by_author[("Knight", 2019.0)]
    assert knight["total"] >= 95
    assert knight["strict_in_xie"] >= 90  # > 90% of Knight 2019 rows match Xie


def test_pfoa_is_in_top_overlapping_pfas() -> None:
    """The overlap_per_pfas dict in the disjoint JSON has PFOA as the
    single largest contributor; pin that down so the SI table can quote it.
    """
    disjoint = json.loads((ROOT / "data/paper/kd_xie_disjoint_validation.json").read_text())
    per_pfas = disjoint["overlap_removed_per_pfas"]
    assert per_pfas["PFOA"] >= 100
    assert per_pfas["PFOA"] == max(per_pfas.values())
