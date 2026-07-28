"""Build Table S5: Xie 2024 reference vs training-set row overlap.

Reads ``kd_xie_train_overlap_report.json`` and writes a clean CSV that
the paper can include as supplementary table S5.  Columns:

  source         the (author, year) label of the original *Kd* study
  xie_cited      "yes" if Xie 2024 cites the study in its Methods
  train_rows     how many training rows came from that source
  strict_in_xie  how many of those rows reappear in Xie 2024 verbatim
                 (PFAS + pH +/- 0.05 + OC +/- 0.05 + log10 Kd +/- 0.005)
  pct_strict     strict_in_xie / train_rows * 100
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "paper" / "kd_xie_train_overlap_report.json"
OUT = ROOT / "data" / "paper" / "tableS5_xie_source_overlap.csv"

# Studies cited by Xie 2024 in its Methods section.  See the discussion
# in the paper's Section 4.6 and the diagnostics in
# scripts/check_xie_train_overlap.py.
XIE_CITATIONS: set[tuple[str, float]] = {
    ("Cai", 2022.0),
    ("Campos-Pereira", 2023.0),
    ("Campos-Pereira", 2022.0),
    ("Fabregat-Palau", 2021.0),
    ("Higgins and Luthy", 2006.0),
    ("Knight", 2019.0),
    ("Knight", 2021.0),
    ("Mejia-Avendaño", 2020.0),
    ("Mejia-Avendaño", 2021.0),
    ("Milinovic", 2015.0),
    ("Nguyen", 2020.0),
    ("Oliver", 2020.0),
    ("Umeh", 2021.0),
    ("Wang", 2022.0),
}


def main() -> None:
    payload = json.loads(REPORT.read_text())
    by_author = payload["by_author_top10"]
    # by_author_top10 only carries the 10 largest; re-derive all
    # (author, year) pairs from a fresh pass.
    rows: list[dict[str, object]] = []
    for entry in by_author:
        author = entry["author"]
        year = entry["year"]
        rows.append({
            "source": f"{author} ({int(year)})",
            "xie_cited": "yes" if (author, year) in XIE_CITATIONS else "no",
            "train_rows": entry["total"],
            "strict_in_xie": entry["strict_in_xie"],
            "pct_strict": entry["pct_strict"],
        })
    # Add rows for sources that we know are in the training set but
    # whose strict-match count is too small to appear in by_author_top10.
    known = {entry["author"] for entry in by_author}
    from collections import defaultdict
    by_author_full = defaultdict(lambda: {"total": 0, "strict": 0})
    train_csv = ROOT / "data/paper/Final_data.csv"
    with open(train_csv) as f:
        for raw in csv.DictReader(f):
            au = (raw.get("First author (name)") or "").strip()
            yr = raw.get("publication (year)")
            try:
                yr = float(yr)
            except (TypeError, ValueError):
                continue
            by_author_full[(au, yr)]["total"] += 1
    disjoint = json.loads((ROOT / "data/paper/kd_xie_disjoint_validation.json").read_text())
    # Re-derive strict counts from the disjoint validation JSON
    # (which has overlap_removed_per_pfas but not per-author).  For
    # simplicity, only the top-10 authors have strict numbers; for
    # others we just list total and leave strict = 0.
    for (au, yr), counts in by_author_full.items():
        if au in known:
            continue
        rows.append({
            "source": f"{au} ({int(yr)})",
            "xie_cited": "yes" if (au, yr) in XIE_CITATIONS else "no",
            "train_rows": counts["total"],
            "strict_in_xie": "n/a (not in top 10)",
            "pct_strict": "n/a",
        })

    rows.sort(key=lambda r: -int(r["train_rows"]) if isinstance(r["train_rows"], int) else 0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["source", "xie_cited", "train_rows", "strict_in_xie", "pct_strict"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
