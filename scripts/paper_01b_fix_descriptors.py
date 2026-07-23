#!/usr/bin/env python3
"""
paper_01b_fix_descriptors.py
============================
修复 SI xlsx 中failed的 2  PFAS 的 SMILES field：
  - 8:2 FtSaB  (SMILES = "N.A."  → use PubChem CID 163360452 的 SMILES)
  - 6:2 FtSaAm  (SMILES parenthesis mismatch → 加 1 right parens)

source：
  - PubChem CID 163360452 (8:2 FtSaB)
  - PubChem CID 138394385 (6:2 FtSaAm)

input: data/paper/PFAS_Properties.csv   (含 51  PFAS original SMILES)
      data/paper/descriptors_51pfas.csv (paper_01 output, 49 rows succeeded, 缺 2 row)
output: data/paper/descriptors_51pfas.csv (追加/replace 2 failures PFAS → 51 row)

if paper_01 already fixed SMILES but this script also fixes, is no-op (just check row count)
"""

import csv
import os

SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
INPUT_FILE = os.path.join(SI_DIR, "descriptors_51pfas.csv")
PROPS_FILE = os.path.join(SI_DIR, "PFAS_Properties.csv")

# PubChem verified correct SMILES (2026-07-03)
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
    # 1. read existing descriptors
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = list(reader)
        fieldnames = reader.fieldnames

    existing_names = {r["PFAS_name"].strip() for r in existing}

    # 2. read PFAS properties (for subfamily 等元data)
    with open(PROPS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        props = {r["PFAS abbreviation"].strip(): r for r in reader}

    # 3. find what needs fixing 2 个 PFAS
    new_rows = []
    for pfas_name, correct_smiles in SMILES_FIX.items():
        if pfas_name in existing_names:
            print(f"  {pfas_name}:  descriptors_51pfas.csv , skip")
            continue
        if pfas_name not in props:
            print(f"  {pfas_name}:  PFAS_Properties.csv not found in, skip")
            continue

        print(f"   {pfas_name}...")
        print(f"    old SMILES: {props[pfas_name].get('Smiles', '?')}")
        print(f"    new SMILES: {correct_smiles}")

        mol = Chem.MolFromSmiles(correct_smiles)
        if mol is None:
            print(f"    ❌ RDKit still cannot parse, skip")
            continue

        # compute all RDKit descriptors
        all_desc = calc_all_descriptors(mol)
        # compute PFAS 特has features
        pfas_feat = calc_pfas_specific_features(mol)

        # merge: PFAS_name, Original_SMILES, RDKIT_SMILES + desc + pfas_feat
        row = {"PFAS_name": pfas_name}
        row["Original_SMILES"] = props[pfas_name].get("Smiles", "")
        row["RDKIT_SMILES"] = correct_smiles
        # extract subfamily / Empirical formula 等
        for key in ["subfamily", "Empirical formula", "%F", "CAS number",
                    "C number", "F number", "H number", "N number", "S number",
                    "O number", "Cl number", "P number", "Molecular weight"]:
            if key in props[pfas_name]:
                row[key] = props[pfas_name][key]
        # write all RDKit descriptors
        for k, v in all_desc.items():
            row[k] = v
        # write to PFAS 特has features
        for k, v in pfas_feat.items():
            row[k] = v

        new_rows.append(row)
        print(f"    ✅ fix complete, 1 row added")

    if not new_rows:
        print("\n  no fixes needed PFAS (no-op)")
        return

    # 4. append to existing
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

    print(f"\n  update: {updated} row")
    print(f"  new: {appended} row")

    # 5. write back
    with open(INPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)

    print(f"\n  output: {INPUT_FILE}")
    print(f"  total row count: {len(existing)} row")
    print("✅ fix complete！")


if __name__ == "__main__":
    main()
