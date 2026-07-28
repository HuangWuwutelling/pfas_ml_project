"""TDD: pin down the behaviour of check_xie_train_overlap.

These tests assert the numbers we just observed in
``kd_xie_train_overlap_report.json`` so future refactors of the
overlap script can't silently change them.  They are the contract
we use to argue that the paper is (or is not) using the Xie 2024
data as a true external benchmark.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "paper" / "kd_xie_train_overlap_report.json"
XIE = Path("/tmp/xie2024_table5.csv")
TRAIN = ROOT / "data" / "paper" / "Final_data.csv"


def test_report_exists_and_parses() -> None:
    """The overlap script writes a JSON report we can grep for numbers."""
    assert REPORT.exists(), "Run scripts/check_xie_train_overlap.py first"
    payload = json.loads(REPORT.read_text())
    assert payload["train_total_rows"] == 1849
    assert payload["xie_total_rows"] == 2148
    assert payload["train_pfas_unique"] == 47
    assert payload["xie_pfas_unique"] == 26
    assert payload["pfas_in_both"] == 22


def test_strict_overlap_is_substantial() -> None:
    """Strict match: 161-169 rows is not 'no overlap' territory.

    This is the figure we have to disclose in the paper.  Pin it down
    so a future script edit cannot silently drop the disclosure.
    """
    payload = json.loads(REPORT.read_text())
    assert 150 <= payload["train_rows_in_xie_strict"] <= 200, payload
    assert 100 <= payload["feature_matrix_rows_in_xie_strict"] <= 200, payload


def test_per_source_overlap_is_dominated_by_known_citations() -> None:
    """Knight 2019 + Fabregat-Palau 2021 dominate the strict overlap.

    Both are on the Xie 2024 reference list.  This test pins down
    the per-author numbers so we can quote them in the paper text.
    """
    payload = json.loads(REPORT.read_text())
    by_author = {
        (entry["author"], entry["year"]): entry
        for entry in payload["by_author_top10"]
    }

    knight = by_author[("Knight", 2019.0)]
    assert knight["total"] == 100
    assert knight["strict_in_xie"] >= 95, knight  # 99 observed

    fabregat = by_author[("Fabregat-Palau", 2021.0)]
    assert fabregat["total"] == 56
    assert fabregat["strict_in_xie"] >= 40, fabregat  # 43 observed


def test_xie_top_pfas_are_present_in_train() -> None:
    """The big Xie compounds (PFOS, PFOA, PFHxS, PFDA) must be in train."""
    pfas_counts: dict[str, int] = {}
    with open(XIE) as f:
        reader = csv.reader(f)
        next(reader); next(reader)
        for row in reader:
            pfas_counts[row[0]] = pfas_counts.get(row[0], 0) + 1
    top = sorted(pfas_counts.items(), key=lambda kv: -kv[1])[:4]
    top_names = {n for n, _ in top}
    train_pfas: set[str] = set()
    with open(TRAIN) as f:
        for row in csv.DictReader(f):
            v = (row.get("PFAS (abreviation)") or "").strip()
            if v:
                train_pfas.add(v)
    missing = top_names - train_pfas
    assert not missing, f"top Xie PFAS missing from training: {missing}"
