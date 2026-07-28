"""Compute training-set vs Xie 2024 record-level overlap.

Strategy:
  - For each row in the training set (data/paper/Final_data.csv), look
    for matching rows in the Xie 2024 SI Table S5.
  - A "strict" match requires the same PFAS abbreviation, the same
    pH +/- 0.05, the same OC% +/- 0.05, and the same log10(Kd) within
    +/- 0.005 log units. These are the tolerances we see when
    re-keying the 56 Fabregat-Palau 2021 rows above -- the original
    CSV is just one of two identical copies, so the numerical
    agreement is to 4-5 decimal places.
  - A "loose" match uses +/- 0.2 on pH, +/- 0.5 on OC, and +/- 0.05
    on log Kd to catch re-typed or rounded copies.
  - A "PFAS-only" match just counts rows that share at least one
    PFAS in common (sanity check on coverage).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "paper" / "Final_data.csv"
XIE = Path("/tmp/xie2024_table5.csv")
OUT = ROOT / "data" / "paper" / "kd_xie_train_overlap_report.json"


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def load_train() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(TRAIN) as f:
        for raw in csv.DictReader(f):
            rows.append(
                {
                    "pfas": (raw.get("PFAS (abreviation)") or "").strip(),
                    "soil": (raw.get("Sample name (tag)") or "").strip(),
                    "author": (raw.get("First author (name)") or "").strip(),
                    "year": _to_float(raw.get("publication (year)")),
                    "doi": (raw.get("DOI (number)") or "").strip(),
                    "pH": _to_float(raw.get("pH (measured)")),
                    "OC": _to_float(raw.get("Corg (%)")),
                    "CEC": _to_float(raw.get("CEC (cmol+/kg)")),
                    "log_Kd": _to_float(raw.get("log Kd ([-])")),
                }
            )
    return rows


def load_xie() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(XIE) as f:
        reader = csv.reader(f)
        next(reader)  # header
        next(reader)  # unit row
        for raw in reader:
            pfas = (raw[0] or "").strip()
            log_kd_log2 = _to_float(raw[1])
            if log_kd_log2 is None:
                continue
            import math
            log_kd = log_kd_log2 / math.log2(10.0)
            rows.append(
                {
                    "pfas": pfas,
                    "pH": _to_float(raw[2]),
                    "OC": _to_float(raw[3]),
                    "CEC": _to_float(raw[4]),
                    "Sand": _to_float(raw[5]),
                    "Silt": _to_float(raw[6]),
                    "Clay": _to_float(raw[7]),
                    "log_Kd": log_kd,
                }
            )
    return rows


def index_xie_by_pfas(xie: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    index: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in xie:
        if row["pfas"]:
            index[row["pfas"]].append(row)
    return index


def match(train_row: dict[str, object], xie_index: dict[str, list[dict[str, object]]],
          pH_tol: float, OC_tol: float, Kd_tol: float) -> bool:
    candidates = xie_index.get(train_row["pfas"], [])
    if not candidates:
        return False
    pH = train_row["pH"]
    OC = train_row["OC"]
    Kd = train_row["log_Kd"]
    if pH is None or OC is None or Kd is None:
        return False
    for c in candidates:
        if c["pH"] is None or c["OC"] is None or c["log_Kd"] is None:
            continue
        if (
            abs(c["pH"] - pH) <= pH_tol
            and abs(c["OC"] - OC) <= OC_tol
            and abs(c["log_Kd"] - Kd) <= Kd_tol
        ):
            return True
    return False


def main() -> None:
    train = load_train()
    xie = load_xie()
    xie_index = index_xie_by_pfas(xie)

    # Per-PFAS coverage: which PFAS in train appear at all in xie?
    pfas_overlap = sorted({r["pfas"] for r in train} & set(xie_index))

    # Strict and loose match counts.
    strict = [match(r, xie_index, 0.05, 0.05, 0.005) for r in train]
    loose = [match(r, xie_index, 0.2, 0.5, 0.05) for r in train]

    # Breakdown by (author, year).
    by_author: dict[tuple[str, float], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "strict": 0, "loose": 0}
    )
    for r, s, l in zip(train, strict, loose):
        key = (r["author"], r["year"] or 0.0)
        by_author[key]["total"] += 1
        by_author[key]["strict"] += int(s)
        by_author[key]["loose"] += int(l)

    # Final-feature-matrix rows vs xie.  This is what the model actually
    # trained on (1,227 rows after filtering), not the raw 1,849.
    fm = ROOT / "data" / "paper" / "feature_matrix_kd.csv"
    fm_keys: set[tuple[str, float, float, float]] = set()
    if fm.exists():
        with open(fm) as f:
            for raw in csv.DictReader(f):
                pH = _to_float(raw.get("pH"))
                OC = _to_float(raw.get("Corg_%"))
                Kd = _to_float(raw.get("log_Kd"))
                pfas = (raw.get("PFAS_name") or "").strip()
                if pH is not None and OC is not None and Kd is not None and pfas:
                    fm_keys.add((pfas, pH, OC, Kd))
    fm_keys_in_xie_strict = sum(
        1 for (pfas, pH, OC, Kd) in fm_keys if match(
            {"pfas": pfas, "pH": pH, "OC": OC, "log_Kd": Kd},
            xie_index, 0.05, 0.05, 0.005,
        )
    )

    report = {
        "train_total_rows": len(train),
        "xie_total_rows": len(xie),
        "train_pfas_unique": len({r["pfas"] for r in train if r["pfas"]}),
        "xie_pfas_unique": len({r["pfas"] for r in xie}),
        "pfas_in_both": len(pfas_overlap),
        "train_rows_in_xie_strict": sum(strict),
        "train_rows_in_xie_loose": sum(loose),
        "train_pct_in_xie_strict": round(100 * sum(strict) / len(train), 1),
        "feature_matrix_rows": len(fm_keys),
        "feature_matrix_rows_in_xie_strict": fm_keys_in_xie_strict,
        "feature_matrix_pct_in_xie_strict": round(
            100 * fm_keys_in_xie_strict / max(len(fm_keys), 1), 1
        ),
        "by_author_top10": [
            {
                "author": a,
                "year": y,
                "total": v["total"],
                "strict_in_xie": v["strict"],
                "loose_in_xie": v["loose"],
                "pct_strict": round(100 * v["strict"] / v["total"], 1) if v["total"] else 0.0,
            }
            for (a, y), v in sorted(
                by_author.items(), key=lambda kv: -kv[1]["strict"]
            )[:15]
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
