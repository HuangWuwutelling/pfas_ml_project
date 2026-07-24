#!/usr/bin/env python3
"""
S1.5_clean_epa_list.py
|==============================
|[core] EPA PFASMASTERlist, 11Kdescriptorspipelineinput(§2.2 / §3.4)
|
|input: data/raw/pfas_master_list.csv   (EPA PFASMASTER, 22,987 original records)
|output: data/processed/pfas_clean.csv   (after cleaning ~10,972 rows containingSMILESrecord)
|
|key filters:
|  1. must haveSMILES(non-empty and not N/A)
|  2. SMILESlength reasonable (5 < len < 500)
|  3. deduplicate(SMILESkeep only first)
|  4. keep only organic molecules( C  c atom)
|
|row:
|  cd <project_root>
|  python scripts/EPA cleaner.py
|
|description:
|  prepare_03_descriptors_11k.py depends on this script's output. 
|  ifonly core pipeline(paper_03 → paper_09), needthis. 
|
|original script: scripts/_archive/EPA cleaner.py(restored on 2026-07-23, 
|         revise docstring; script logic unchanged)
"""
import csv
import os
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "raw", "pfas_master_list.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "processed", "pfas_clean.csv")

print("=" * 60)
print("  EPA PFASMASTER list cleaning")
print("=" * 60)

smiles_seen = set()
clean_rows = []
total = 0
skipped_no_smiles = 0
skipped_short = 0
skipped_long = 0
skipped_duplicate = 0
skipped_not_organic = 0

with open(INPUT_FILE, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        total += 1
        smiles = (row.get("SMILES") or "").strip()
        name = (row.get("PREFERRED NAME") or "").strip()

        # must haveSMILES
        if not smiles or smiles == "N/A":
            skipped_no_smiles += 1
            continue

        # length filter
        if len(smiles) < 5:
            skipped_short += 1
            continue
        if len(smiles) > 500:
            skipped_long += 1
            continue

        # deduplicate
        if smiles in smiles_seen:
            skipped_duplicate += 1
            continue
        smiles_seen.add(smiles)

        # basic organic check(C)
        if "C" not in smiles and "c" not in smiles:
            skipped_not_organic += 1
            continue

        # extractDTXSID(columnisURL)
        dtxsid_raw = row.get("DTXSID", "")
        if "/" in dtxsid_raw:
            dtxsid = dtxsid_raw.split("/")[-1]
        else:
            dtxsid = dtxsid_raw

        clean_rows.append({
            "DTXSID": dtxsid,
            "PREFERRED_NAME": name,
            "CASRN": row.get("CASRN", ""),
            "SMILES": smiles,
            "INCHIKEY": row.get("INCHIKEY", ""),
            "MOLECULAR_FORMULA": row.get("MOLECULAR_FORMULA", ""),
            "AVERAGE_MASS": row.get("AVERAGE MASS", ""),
        })

# outputstatistics
print(f"\n=== cleaning statistics ===")
print(f"  original record count: {total:,}")
print(f"  noSMILESskip: {skipped_no_smiles:,}")
print(f"  SMILEStoo short, skipped: {skipped_short:,}")
print(f"  SMILEStoo long, skipped: {skipped_long:,}")
print(f"  duplicateSMILESskip: {skipped_duplicate:,}")
print(f"  non-organic skipped: {skipped_not_organic:,}")
print(f"  ✅ valid records: {len(clean_rows):,}")

# output
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "DTXSID", "PREFERRED_NAME", "CASRN", "SMILES",
        "INCHIKEY", "MOLECULAR_FORMULA", "AVERAGE_MASS"
    ])
    writer.writeheader()
    writer.writerows(clean_rows)

print(f"\n  ✅ cleaned list saved: {OUTPUT_FILE}")
print(f"     {len(clean_rows):,} compounds")

# simpledistribution
lengths = [len(r["SMILES"]) for r in clean_rows]
print(f"\n  SMILES: min={min(lengths)}, max={max(lengths)}, "
      f"mean={sum(lengths)/len(lengths):.0f}")
