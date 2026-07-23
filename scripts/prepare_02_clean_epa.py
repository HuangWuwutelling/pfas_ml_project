#!/usr/bin/env python3
"""
S1.5_clean_epa_list.py
|==============================
|[论文核心] 清洗EPA PFASMASTER清单，作为11K描述符pipeline的输入（§2.2 / §3.4）
|
|输入: data/raw/pfas_master_list.csv   (EPA PFASMASTER, 22,987条原始记录)
|输出: data/processed/pfas_clean.csv   (清洗后约10,972条含SMILES的记录)
|
|关键过滤:
|  1. 必须有SMILES（非空且非 N/A）
|  2. SMILES长度合理 (5 < len < 500)
|  3. 去重（同一SMILES只保留第一条）
|  4. 只保留有机分子（含 C 或 c 原子）
|
|运行:
|  cd <project_root>
|  python scripts/prepare_02_clean_epa.py
|
|说明:
|  prepare_03_descriptors_11k.py 依赖此脚本的输出。
|  如果你只想跑 core pipeline（paper_03 → paper_09），不需要这个脚本。
|
|原脚本: scripts/_archive/prepare_02_clean_epa.py（恢复于 2026-07-23，
|         修缮 docstring；脚本逻辑无改动）
"""
import csv
import os
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "raw", "pfas_master_list.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "processed", "pfas_clean.csv")

print("=" * 60)
print("  EPA PFASMASTER 清单清洗")
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

        # 必须有SMILES
        if not smiles or smiles == "N/A":
            skipped_no_smiles += 1
            continue

        # 长度过滤
        if len(smiles) < 5:
            skipped_short += 1
            continue
        if len(smiles) > 500:
            skipped_long += 1
            continue

        # 去重
        if smiles in smiles_seen:
            skipped_duplicate += 1
            continue
        smiles_seen.add(smiles)

        # 基本有机性检查（含C）
        if "C" not in smiles and "c" not in smiles:
            skipped_not_organic += 1
            continue

        # 提取DTXSID（第一列是URL）
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

# 输出统计
print(f"\n=== 清洗统计 ===")
print(f"  原始记录数: {total:,}")
print(f"  无SMILES跳过: {skipped_no_smiles:,}")
print(f"  SMILES太短跳过: {skipped_short:,}")
print(f"  SMILES太长跳过: {skipped_long:,}")
print(f"  重复SMILES跳过: {skipped_duplicate:,}")
print(f"  非有机跳过: {skipped_not_organic:,}")
print(f"  ✅ 有效净记录: {len(clean_rows):,}")

# 输出
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "DTXSID", "PREFERRED_NAME", "CASRN", "SMILES",
        "INCHIKEY", "MOLECULAR_FORMULA", "AVERAGE_MASS"
    ])
    writer.writeheader()
    writer.writerows(clean_rows)

print(f"\n  ✅ 清洗后清单已保存: {OUTPUT_FILE}")
print(f"     {len(clean_rows):,} 种化合物")

# 简单长度分布
lengths = [len(r["SMILES"]) for r in clean_rows]
print(f"\n  SMILES长度: min={min(lengths)}, max={max(lengths)}, "
      f"mean={sum(lengths)/len(lengths):.0f}")
