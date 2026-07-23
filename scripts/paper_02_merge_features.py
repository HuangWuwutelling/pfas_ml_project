#!/usr/bin/env python3
"""
S2_merge_features.py
====================
SIdata: S1RDKitdescriptors + Final_datasoil properties + Kd → training feature matrix. 

input:
  data/paper/descriptors_51pfas.csv   (51PFASRDKitdescriptors)
  data/paper/Final_data.csv           (1849experimental data rows)
output:
  data/paper/feature_matrix_kd.csv    (1227row × ~250column: feature + target)
  data/paper/feature_matrix_kd_info.txt  (datastatistics)

description:
  matching method: PFAS (PFAS abbreviationcolumn) → PFAS_name
  keep only log Kd rows with non-empty values and complete soil features. 
  target: log Kd (SI Excelcolumn58)
"""

import csv
import os
import math
import statistics

SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
DESC_FILE = os.path.join(SI_DIR, "descriptors_51pfas.csv")
FINAL_FILE = os.path.join(SI_DIR, "Final_data.csv")
OUTPUT_FILE = os.path.join(SI_DIR, "feature_matrix_kd.csv")
INFO_FILE = os.path.join(SI_DIR, "feature_matrix_kd_info.txt")

# soil/experimental feature columns(Final_dataselected from)
# column name mapping(fromFinal_data.csvauto-fetch from header row)
SOIL_FEATURE_MAP = {
    "Corg_%": "Corg (%)",
    "foc": "foc",
    "pH": "pH (measured)",
    "Sand": "Sand (% mineral)",
    "Silt": "Silt (% mineral)",
    "Clay": "Clay (% mineral)",
    "CEC": "CEC (cmol+/kg)",
    "Fe_g_kg": "Fe ((g/kg))",
    "Al_g_kg": "Al ((g/kg))",
}

# targetcolumn
TARGET_NAME = "log Kd ([-])"
TARGET_ALT_NAME = "log Koc ([-])"
KD_NAME = "Kd (L/Kg)"
PFAS_NAME_MAP = "PFAS (abreviation)"


def safe_float(v):
    """safely convert tofloat, handle missing value flags"""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in ("-", "", "N/A", "not reported", "N.A."):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def main():
    # 1. loadedRDKitdescriptors, byPFASname index
    with open(DESC_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        desc_rows = list(reader)
    
    desc_by_name = {}
    for row in desc_rows:
        name = row["PFAS_name"].strip()
        desc_by_name[name] = row
    
    desc_fieldnames = [k for k in desc_rows[0].keys() 
                       if k not in ("PFAS_name", "Original_SMILES", "RDKIT_SMILES")]
    
    print("=" * 60)
    print("  feature merging: RDKitdescriptors + soil properties + Kd")
    print("=" * 60)
    print(f"\n  RDKitdescriptors: {len(desc_rows)} PFAS, {len(desc_fieldnames)} column")

    # 2. readFinal_data, useDictReaderbycolumn
    with open(FINAL_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        final_rows = list(reader)

    print(f"  Final_data: {len(final_rows)} row")
    
    # 3. match row-by-row and merge
    merged = []
    unmatched_names = set()
    skipped_no_target = 0
    skipped_no_desc = 0
    skipped_no_soil = 0
    matched = 0
    
    for row in final_rows:
        if not row:
            continue

        # PFASname
        pfas_name = row.get(PFAS_NAME_MAP, "").strip()
        if not pfas_name:
            continue

        # find descriptor
        desc = desc_by_name.get(pfas_name)
        if desc is None:
            unmatched_names.add(pfas_name)
            skipped_no_desc += 1
            continue

        # target: log Kd
        log_kd = safe_float(row.get(TARGET_NAME))
        if log_kd is None:
            skipped_no_target += 1
            continue

        # alternative: Kd and log Koc
        kd = safe_float(row.get(KD_NAME))
        log_koc = safe_float(row.get(TARGET_ALT_NAME))

        # soil features(access via column name)
        soil_vals = {}
        for feat_name, col_name in SOIL_FEATURE_MAP.items():
            v = safe_float(row.get(col_name))
            soil_vals[feat_name] = v
        
        # statisticssoil featuresmissing
        n_soil_missing = sum(1 for v in soil_vals.values() if v is None)
        
        # construct merged rows
        merged_row = {
            "PFAS_name": pfas_name,
            "log_Kd": log_kd,
            "Kd_L_kg": kd,
            "log_Koc": log_koc,
        }
        # add soil features
        for feat_name, val in soil_vals.items():
            merged_row[feat_name] = val
        # RDKitdescriptors
        for fn in desc_fieldnames:
            merged_row[fn] = desc[fn]
        
        merged_row["_n_soil_missing"] = n_soil_missing
        merged.append(merged_row)
        matched += 1
    
    # 4. statistics
    info_lines = []
    info_lines.append(f"Total rows in Final_data: {len(final_rows)}")
    info_lines.append(f"")
    info_lines.append(f"Matched (have RDKit desc + log Kd): {matched}")
    info_lines.append(f"  Skipped: no RDKit desc: {skipped_no_desc}")
    info_lines.append(f"  Skipped: no log Kd target: {skipped_no_target}")
    info_lines.append(f"")
    
    if unmatched_names:
        info_lines.append(f"Unmatched PFAS names (in Final_data but not in descriptors):")
        for n in sorted(unmatched_names):
            info_lines.append(f"  - {n}")
    info_lines.append(f"")
    
    # soil feature missing statistics
    missing_counts = {k: 0 for k in SOIL_FEATURE_MAP}
    n_soil_complete = 0
    for r in merged:
        for fn in SOIL_FEATURE_MAP:
            if r[fn] is None:
                missing_counts[fn] += 1
        if r["_n_soil_missing"] == 0:
            n_soil_complete += 1
    
    info_lines.append(f"Soil feature completeness (out of {len(merged)} matched rows):")
    total = len(merged)
    for fn, cnt in missing_counts.items():
        pct = cnt / max(total, 1) * 100
        info_lines.append(f"  {fn}: {total - cnt}/{total} present ({100-pct:.1f}%)")
    info_lines.append(f"")
    info_lines.append(f"Rows with ALL soil features: {n_soil_complete}/{total}")
    info_lines.append(f"")
    
    # log Kd distribution
    logkd_vals = [r["log_Kd"] for r in merged]
    info_lines.append(f"log Kd distribution:")
    info_lines.append(f"  n={len(logkd_vals)}")
    info_lines.append(f"  mean={statistics.mean(logkd_vals):.2f}")
    info_lines.append(f"  median={statistics.median(logkd_vals):.2f}")
    info_lines.append(f"  range=[{min(logkd_vals):.2f}, {max(logkd_vals):.2f}]")
    info_lines.append(f"  std={statistics.stdev(logkd_vals):.2f}")
    info_lines.append(f"")

    # statistics
    for line in info_lines:
        print(f"  {line}")
    
    # infofile
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines))
    print(f"\n  statisticsinfo: {INFO_FILE}")

    # 5. outputCSV(remove_n_soil_missinginternal columns)
    out_fieldnames = [
        "PFAS_name", "log_Kd", "Kd_L_kg", "log_Koc",
    ] + list(SOIL_FEATURE_MAP.keys()) + desc_fieldnames
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    
    print(f"  feature matrix: {OUTPUT_FILE}")
    print(f"  {len(merged)} row × {len(out_fieldnames)} column")
    print(f"\n✅ S2 ! ")


if __name__ == "__main__":
    main()
