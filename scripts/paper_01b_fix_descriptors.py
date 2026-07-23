#!/usr/bin/env python3
"""
paper_01b_fix_descriptors.py
============================
修复 SI xlsx 中失败的 2 种 PFAS 的 SMILES 字段：
  - 8:2 FtSaB  (SMILES = "N.A."  → 用 PubChem CID 163360452 的 SMILES)
  - 6:2 FtSaAm  (SMILES 括号不匹配 → 加 1 个右括号)

来源：
  - PubChem CID 163360452 (8:2 FtSaB)
  - PubChem CID 138394385 (6:2 FtSaAm)

输入: data/paper/PFAS_Properties.csv   (含 51 种 PFAS 原始 SMILES)
      data/paper/descriptors_51pfas.csv (paper_01 输出, 49 行成功, 缺 2 行)
输出: data/paper/descriptors_51pfas.csv (追加/替换 2 个失败 PFAS → 51 行)

如果 paper_01 已经修过 SMILES 但本脚本也修, 是 no-op (脚本检查行数即可)
"""

import csv
import os

SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
INPUT_FILE = os.path.join(SI_DIR, "descriptors_51pfas.csv")
PROPS_FILE = os.path.join(SI_DIR, "PFAS_Properties.csv")

# PubChem 验证过的正确 SMILES (2026-07-03)
SMILES_FIX = {
    "8:2 FtSaB":  "C(C[NH2+]CC(=O)[O-])CNS(=O)(=O)CCC(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F",
    "6:2 FtSaAm": "C[NH+](C)CCCNS(=O)(=O)CCC(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F",
}

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def calc_all_descriptors(mol):
    desc_names = [d[0] for d in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    values = calculator.CalcDescriptors(mol)
    return dict(zip(desc_names, values))


def calc_pfas_specific_features(mol):
    features = {
        "carbon_count": 0, "fluorine_count": 0, "has_sulfonate": 0,
        "has_carboxyl": 0, "has_ether": 0, "has_aromatic": 0,
        "has_double_bond": 0, "perfluoro_ratio": 0.0,
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
    sulfonate = Chem.MolFromSmarts("[S](=O)(=O)[O]")
    carboxyl = Chem.MolFromSmarts("[CX3](=O)[OX2]")
    ether = Chem.MolFromSmarts("[C][O][C]")
    if sulfonate and mol.HasSubstructMatch(sulfonate):
        features["has_sulfonate"] = 1
    if carboxyl and mol.HasSubstructMatch(carboxyl):
        features["has_carboxyl"] = 1
    if ether and mol.HasSubstructMatch(ether):
        features["has_ether"] = 1
    if any(a.GetIsAromatic() for a in atoms):
        features["has_aromatic"] = 1
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE and not bond.GetIsAromatic():
            features["has_double_bond"] = 1
    return features


def main():
    # 1. 读取已有描述符
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = list(reader)
        fieldnames = reader.fieldnames

    existing_names = {r["PFAS_name"].strip() for r in existing}

    # 2. 读取 PFAS properties (用于 subfamily 等元数据)
    with open(PROPS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        props = {r["PFAS abbreviation"].strip(): r for r in reader}

    # 3. 找需要修复的 2 个 PFAS
    new_rows = []
    for pfas_name, correct_smiles in SMILES_FIX.items():
        if pfas_name in existing_names:
            print(f"  {pfas_name}: 已在 descriptors_51pfas.csv 中, 跳过")
            continue
        if pfas_name not in props:
            print(f"  {pfas_name}: 在 PFAS_Properties.csv 中找不到, 跳过")
            continue

        print(f"  修复 {pfas_name}...")
        print(f"    旧 SMILES: {props[pfas_name].get('Smiles', '?')}")
        print(f"    新 SMILES: {correct_smiles}")

        mol = Chem.MolFromSmiles(correct_smiles)
        if mol is None:
            print(f"    ❌ RDKit 仍无法解析, 跳过")
            continue

        # 计算所有 RDKit 描述符
        all_desc = calc_all_descriptors(mol)
        # 计算 PFAS 特有 features
        pfas_feat = calc_pfas_specific_features(mol)

        # 合并: PFAS_name, Original_SMILES, RDKIT_SMILES + desc + pfas_feat
        row = {"PFAS_name": pfas_name}
        row["Original_SMILES"] = props[pfas_name].get("Smiles", "")
        row["RDKIT_SMILES"] = correct_smiles
        # 提取 subfamily / Empirical formula 等
        for key in ["subfamily", "Empirical formula", "%F", "CAS number",
                    "C number", "F number", "H number", "N number", "S number",
                    "O number", "Cl number", "P number", "Molecular weight"]:
            if key in props[pfas_name]:
                row[key] = props[pfas_name][key]
        # 写入所有 RDKit 描述符
        for k, v in all_desc.items():
            row[k] = v
        # 写入 PFAS 特有 features
        for k, v in pfas_feat.items():
            row[k] = v

        new_rows.append(row)
        print(f"    ✅ 修复完成, 1 行 added")

    if not new_rows:
        print("\n  没有需要修复的 PFAS (no-op)")
        return

    # 4. 追加到 existing
    updated = 0
    appended = 0
    for nr in new_rows:
        found = False
        for i, er in enumerate(existing):
            if er["PFAS_name"].strip() == nr["PFAS_name"]:
                existing[i] = nr
                found = True
                updated += 1
                break
        if not found:
            existing.append(nr)
            appended += 1

    print(f"\n  更新: {updated} 行")
    print(f"  新增: {appended} 行")

    # 5. 写回
    with open(INPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)

    print(f"\n  输出: {INPUT_FILE}")
    print(f"  总行数: {len(existing)} 行")
    print("✅ 修复完成！")


if __name__ == "__main__":
    main()
