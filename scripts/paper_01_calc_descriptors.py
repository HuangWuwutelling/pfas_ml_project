#!/usr/bin/env python3
"""
S1_calc_descriptors_51pfas.py
|==============================
|[论文核心] 计算51种PFAS的RDKit分子描述符（对应§2.2）
|
|输入: data/paper/PFAS_Properties.csv (51种PFAS, 含Smiles列)
|输出: data/paper/descriptors_51pfas.csv
|      (51行: PFAS名 + Smiles + RDKIT_SMILES + 225个RDKit描述符 + PFAS特有特征)
|
|运行:
|  cd <project_root>
|  .venv_py311/bin/python scripts/paper_01_calc_descriptors.py
"""

import csv
import os
import sys
import time
import numpy as np

# 路径配置
SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
INPUT_FILE = os.path.join(SI_DIR, "PFAS_Properties.csv")
OUTPUT_FILE = os.path.join(SI_DIR, "descriptors_51pfas.csv")

# RDKit导入
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
    print("✅ RDKit 已安装")
except ImportError:
    RDKIT_AVAILABLE = False
    print("❌ RDKit 未安装")
    sys.exit(1)


def clean_smiles(smiles):
    """清理SMILES：去掉 |lp:...| 扩展标记等非标准格式"""
    if not smiles or not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s:
        return None
    pipe_pos = s.find("|")
    if pipe_pos != -1:
        s = s[:pipe_pos].strip()
    if not s:
        return None
    # 特殊处理已知问题SMILES
    if s.upper() in ("", "N.A.", "N/A", "NA", "NONE"):
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


def calc_pfas_specific_features(mol):
    """计算PFAS特有特征"""
    features = {
        "carbon_count": 0,
        "fluorine_count": 0,
        "has_sulfonate": 0,
        "has_carboxyl": 0,
        "has_ether": 0,
        "has_aromatic": 0,
        "has_double_bond": 0,
        "perfluoro_ratio": 0.0,
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
    sulfonate = Chem.MolFromSmarts("[S](=O)(=O)[O]")
    carboxyl = Chem.MolFromSmarts("[CX3](=O)[OX2]")
    ether = Chem.MolFromSmarts("[CX4][OX2][CX4]")  # 仅饱和醚键 C-O-C，不含酯键
    if sulfonate and mol.HasSubstructMatch(sulfonate):
        features["has_sulfonate"] = 1
    if carboxyl and mol.HasSubstructMatch(carboxyl):
        features["has_carboxyl"] = 1
    if ether and mol.HasSubstructMatch(ether):
        features["has_ether"] = 1
    if any(a.GetIsAromatic() for a in atoms):
        features["has_aromatic"] = 1
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() == 2.0:
            begin = bond.GetBeginAtom()
            end = bond.GetEndAtom()
            if begin.GetAtomicNum() == 6 and end.GetAtomicNum() == 6:
                features["has_double_bond"] = 1
                break

    return features


def main():
    print("=" * 60)
    print("  51种PFAS的RDKit描述符计算")
    print("=" * 60)

    # 读取输入
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"\n  读取 {len(rows)} 种PFAS")

    # 确认列
    header = list(rows[0].keys()) if rows else []
    print(f"  列名 (前10): {header[:10]}")
    
    # 找SMILES列
    smiles_col = None
    name_col = None
    for c in header:
        cl = c.lower().strip()
        if "smiles" in cl or "smile" in cl:
            smiles_col = c
        if "abbreviation" in cl or "pfas" in cl:
            name_col = c
    
    if not name_col:
        name_col = header[0]  # 第一列是PFAS abbreviation
    print(f"  PFAS名称列: '{name_col}'")
    print(f"  SMILES列:   '{smiles_col}'")

    if not smiles_col:
        print("❌ 找不到SMILES列！")
        sys.exit(1)

    # 逐条处理
    desc_names_full = [d[0] for d in Descriptors._descList]
    pfas_feature_names = [
        "carbon_count", "fluorine_count", "has_sulfonate",
        "has_carboxyl", "has_ether", "has_aromatic",
        "has_double_bond", "perfluoro_ratio"
    ]

    results = []
    failed_smiles = []
    failed_parse = []
    succeeded = 0

    start_time = time.time()

    for i, row in enumerate(rows):
        pfas_name = row.get(name_col, f"PFAS_{i}").strip()
        raw_smiles = row.get(smiles_col, "")

        smiles = clean_smiles(raw_smiles)
        if smiles is None:
            failed_smiles.append(pfas_name)
            print(f"  [{i+1}/{len(rows)}] {pfas_name} ❌ 空/无效SMILES: '{raw_smiles}'")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed_parse.append(pfas_name)
            print(f"  [{i+1}/{len(rows)}] {pfas_name} ❌ 解析失败: {smiles[:60]}")
            continue

        desc = calc_all_descriptors(mol)
        rdkit_smiles = Chem.MolToSmiles(mol)
        pfas_feat = calc_pfas_specific_features(mol)
        desc.update(pfas_feat)

        results.append({
            "PFAS_name": pfas_name,
            "Original_SMILES": raw_smiles.strip(),
            "RDKIT_SMILES": rdkit_smiles,
            **desc,
        })
        succeeded += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(rows)}] ✅ 成功 {succeeded}, 耗时 {elapsed:.1f}s")

    # 统计
    elapsed = time.time() - start_time
    print(f"\n  ✅ 成功: {succeeded}/{len(rows)}")
    print(f"  ❌ 空SMILES: {len(failed_smiles)} — {failed_smiles}")
    print(f"  ❌ 解析失败: {len(failed_parse)} — {failed_parse}")
    print(f"  总耗时: {elapsed:.1f} 秒")

    if not results:
        print("❌ 全部失败！")
        sys.exit(1)

    # 写入CSV
    sample = results[0]
    fieldnames = ["PFAS_name", "Original_SMILES", "RDKIT_SMILES"] + \
                 [k for k in sample.keys() if k not in ("PFAS_name", "Original_SMILES", "RDKIT_SMILES")]
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for res in results:
            writer.writerow(res)

    print(f"\n  输出: {OUTPUT_FILE}")
    print(f"  {len(results)} 行 × {len(fieldnames)} 列")
    print(f"\n✅ S1 完成！")


if __name__ == "__main__":
    main()
