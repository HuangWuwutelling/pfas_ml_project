#!/usr/bin/env python3
"""
06_calc_descriptors_full.py
============================
用 RDKit 批量计算 EPA PFASMASTER 全清单（~11,000种）的分子描述符和指纹。

输入: data/processed/pfas_clean.csv
       （清洗后的 EPA PFASMASTER 清单，10,972条含SMILES）
输出: data/processed/pfas_descriptors_full.csv
       （217个RDKit描述符 + PFAS特有特征）
      data/processed/pfas_fingerprint_full.csv
       （2048位ECFP4摩尔指纹）

与02_calc_descriptors.py的区别：
  - 输入从PubChem的小数据集改为EPAClean的全量清单
  - SMILES直接从 pfas_clean.csv 的 SMILES 列读取
  - 包含SMILES格式清理（去掉 |lp:...| 扩展标记）
  - 输出文件名加 _full 后缀

运行:
  python scripts/06_calc_descriptors_full.py

预计耗时: 2-5分钟（11,000条）
"""

import csv
import os
import sys
import time
import numpy as np

# 路径配置
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "processed", "pfas_clean.csv")
OUTPUT_DESC = os.path.join(DATA_DIR, "processed", "pfas_descriptors_full.csv")
OUTPUT_FP = os.path.join(DATA_DIR, "processed", "pfas_fingerprint_full.csv")

# ==========================================
# RDKit 导入检测
# ==========================================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
    print("✅ RDKit 已安装")
except ImportError:
    RDKIT_AVAILABLE = False
    print("❌ RDKit 未安装，请运行: pip install rdkit-pypi")
    print("   安装后重新运行本脚本")


def clean_smiles(smiles):
    """清理SMILES：去掉 |lp:...| 扩展标记等非标准格式"""
    if not smiles or not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s:
        return None
    # 去掉 |...| 扩展标记（RDKit不能解析），如 |lp:4:2,6:3...|
    pipe_pos = s.find("|")
    if pipe_pos != -1:
        s = s[:pipe_pos].strip()
    if not s:
        return None
    return s


def calc_all_descriptors(mol):
    """计算200+个标准RDKit分子描述符"""
    if mol is None:
        return {}
    desc_names = [d[0] for d in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    values = calculator.CalcDescriptors(mol)
    return dict(zip(desc_names, values))


def calc_fingerprint(mol, radius=2, nbits=2048):
    """计算 ECFP4 (Morgan) 分子指纹"""
    if mol is None:
        return None
    from rdkit.Chem import rdMolDescriptors
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.uint8)


def calc_pfas_specific_features(mol):
    """计算PFAS特有特征（基于RDKit解析后的分子对象）"""
    features = {
        "carbon_count": 0,       # 碳原子数
        "fluorine_count": 0,     # 氟原子数
        "has_sulfonate": 0,      # 含磺酸基 -S(=O)(=O)O
        "has_carboxyl": 0,       # 含羧基 -C(=O)O
        "has_ether": 0,          # 含醚键 C-O-C
        "has_aromatic": 0,       # 含芳香环
        "has_double_bond": 0,    # 含碳碳双键
        "perfluoro_ratio": 0.0,  # 氟化程度 (F原子数 / 总原子数)
    }
    if mol is None:
        return features

    atoms = mol.GetAtoms()
    total_atoms = len(atoms)
    c_count = sum(1 for a in atoms if a.GetAtomicNum() == 6)
    f_count = sum(1 for a in atoms if a.GetAtomicNum() == 9)

    features["carbon_count"] = c_count
    features["fluorine_count"] = f_count
    features["perfluoro_ratio"] = round(f_count / max(total_atoms, 1), 4)

    # 子结构匹配
    mol_s = mol
    if Chem.MolFromSmarts("[S](=O)(=O)[O]") and mol_s.HasSubstructMatch(Chem.MolFromSmarts("[S](=O)(=O)[O]")):
        features["has_sulfonate"] = 1
    if Chem.MolFromSmarts("[CX3](=O)[OX2]") and mol_s.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2]")):
        features["has_carboxyl"] = 1
    if Chem.MolFromSmarts("[CX4][OX2][CX4]") and mol_s.HasSubstructMatch(Chem.MolFromSmarts("[CX4][OX2][CX4]")):
        features["has_ether"] = 1
    if any(a.GetIsAromatic() for a in atoms):
        features["has_aromatic"] = 1
    # 碳碳双键
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() == 2.0:
            begin = bond.GetBeginAtom()
            end = bond.GetEndAtom()
            if begin.GetAtomicNum() == 6 and end.GetAtomicNum() == 6:
                features["has_double_bond"] = 1
                break

    return features


def main():
    if not RDKIT_AVAILABLE:
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_DESC), exist_ok=True)

    start_time = time.time()

    print("=" * 60)
    print("  EPA PFASMASTER 全量描述符计算")
    print("  输入: pfas_clean.csv (10,972条)")
    print("=" * 60)

    # 读取清洗后的PFAS清单
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n  读取 {len(rows):,} 种PFAS")
    print(f"  描述符: {len(Descriptors._descList)} 个")
    print(f"  指纹: 2048 位 ECFP4\n")

    # 逐条处理
    descriptor_names = [d[0] for d in Descriptors._descList]
    pfas_feature_names = [
        "carbon_count", "fluorine_count", "has_sulfonate",
        "has_carboxyl", "has_ether", "has_aromatic",
        "has_double_bond", "perfluoro_ratio"
    ]

    results_desc = []
    results_fp = []
    failed = 0
    skipped_no_smiles = 0
    skipped_parse_fail = 0

    for i, row in enumerate(rows):
        raw_smiles = row.get("SMILES", "")
        dtxsid = row.get("DTXSID", "?")

        # 清理SMILES
        smiles = clean_smiles(raw_smiles)
        if smiles is None:
            print(f"  [{i+1}/{len(rows)}] {dtxsid} ❌ 空SMILES")
            results_desc.append(None)
            results_fp.append(None)
            skipped_no_smiles += 1
            continue

        # 解析SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [{i+1}/{len(rows)}] {dtxsid} ❌ 解析失败: {smiles[:60]}")
            results_desc.append(None)
            results_fp.append(None)
            skipped_parse_fail += 1
            continue

        # 计算标准描述符
        desc = calc_all_descriptors(mol)

        # 获取RDKit规范化后的SMILES（用于跨数据集的键匹配）
        rdkit_smiles_norm = Chem.MolToSmiles(mol)

        # 计算PFAS特有特征
        pfas_feat = calc_pfas_specific_features(mol)
        desc.update(pfas_feat)

        # 计算指纹
        fp = calc_fingerprint(mol)

        results_desc.append({"desc": desc, "rdkit_smiles": rdkit_smiles_norm})
        results_fp.append(fp)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(rows)}] ✅ {rate:.0f} 条/秒, 预计剩余 {eta:.1f} 分钟")

    # 统计
    valid_idx = [i for i, d in enumerate(results_desc) if d is not None]
    print(f"\n  成功: {len(valid_idx):,}")
    print(f"  空SMILES: {skipped_no_smiles}")
    print(f"  解析失败: {skipped_parse_fail}")

    if not valid_idx:
        print("  ❌ 全部失败，退出")
        sys.exit(1)

    # 确定输出字段
    sample_desc = results_desc[valid_idx[0]]["desc"]
    all_fieldnames = ["DTXSID", "SMILES", "RDKIT_SMILES"] + list(sample_desc.keys())

    # 写入描述符CSV
    with open(OUTPUT_DESC, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for idx in valid_idx:
            row_data = {
                "DTXSID": rows[idx].get("DTXSID", ""),
                "SMILES": rows[idx].get("SMILES", ""),
                "RDKIT_SMILES": results_desc[idx]["rdkit_smiles"],
                **results_desc[idx]["desc"],
            }
            writer.writerow(row_data)

    # 写入指纹CSV
    fp_header = ["DTXSID", "RDKIT_SMILES"] + [f"FP_{i}" for i in range(2048)]
    with open(OUTPUT_FP, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fp_header)
        for idx in valid_idx:
            fp = results_fp[idx]
            fp_row = [rows[idx].get("DTXSID", ""), results_desc[idx]["rdkit_smiles"]] + fp.tolist()
            writer.writerow(fp_row)

    elapsed = time.time() - start_time
    print(f"\n  ✅ 完成! 耗时 {elapsed:.1f} 秒")
    print(f"  描述符: {OUTPUT_DESC}")
    print(f"     {len(valid_idx):,} 行 × {len(all_fieldnames)} 列")
    print(f"  指纹: {OUTPUT_FP}")
    print(f"     {len(valid_idx):,} 行 × 2048 列")

    # PFAS特征统计
    print(f"\n  PFAS特征统计:")
    for feat_name in pfas_feature_names:
        vals = [results_desc[i]["desc"][feat_name] for i in valid_idx
                if feat_name in results_desc[i]["desc"] and results_desc[i]["desc"][feat_name] is not None]
        if vals:
            if isinstance(vals[0], (int, float)):
                if all(v in (0, 1) for v in vals):
                    # 二值特征
                    ones = sum(v == 1 for v in vals)
                    print(f"    {feat_name}: {ones}/{len(vals)} ({ones/len(vals)*100:.1f}%)")
                else:
                    # 数值特征
                    print(f"    {feat_name}: min={min(vals):.2f}, max={max(vals):.2f}, mean={np.mean(vals):.2f}")

    # SMILES解析率
    total_valid = len(rows) - skipped_no_smiles
    parse_rate = len(valid_idx) / max(total_valid, 1) * 100
    print(f"\n  SMILES解析率: {len(valid_idx)}/{total_valid} = {parse_rate:.1f}%")


if __name__ == "__main__":
    main()
