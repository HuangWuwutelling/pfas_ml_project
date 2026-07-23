#!/usr/bin/env python3
"""
S1_calc_descriptors_51pfas.py
|==============================
|[core] compute51PFASRDKitmolecular descriptors(for§2.2)
|
|input: data/paper/PFAS_Properties.csv (51PFAS, Smilescolumn)
|output: data/paper/descriptors_51pfas.csv
|      (51row: PFAS + Smiles + RDKIT_SMILES + 225RDKitdescriptors + PFASspecific features)
|
|row:
|  cd <project_root>
|  .venv_py311/bin/python scripts/paper_01_calc_descriptors.py
"""

import csv
import os
import sys
import time
import numpy as np

# path config
SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
INPUT_FILE = os.path.join(SI_DIR, "PFAS_Properties.csv")
OUTPUT_FILE = os.path.join(SI_DIR, "descriptors_51pfas.csv")

# RDKitimport
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
    print("✅ RDKit installed")
except ImportError:
    RDKIT_AVAILABLE = False
    print("❌ RDKit not installed")
    sys.exit(1)


def clean_smiles(smiles):
    """SMILES: remove |lp:...| extension markers and other non-standard formats"""
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
    # special-case known issuesSMILES
    if s.upper() in ("", "N.A.", "N/A", "NA", "NONE"):
        return None
    return s


def calc_all_descriptors(mol):
    """compute200+standardRDKitmolecular descriptors"""
    if mol is None:
        return {}
    desc_names = [d[0] for d in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    values = calculator.CalcDescriptors(mol)
    return dict(zip(desc_names, values))


def calc_pfas_specific_features(mol):
    """computePFASspecific features"""
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

    # substructure match
    sulfonate = Chem.MolFromSmarts("[S](=O)(=O)[O]")
    carboxyl = Chem.MolFromSmarts("[CX3](=O)[OX2]")
    ether = Chem.MolFromSmarts("[CX4][OX2][CX4]")  # saturated ether bonds only C-O-C, 
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
    print("  51PFASRDKitdescriptor computation")
    print("=" * 60)

    # read input
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"\n  read {len(rows)} PFAS")

    # confirmcolumn
    header = list(rows[0].keys()) if rows else []
    print(f"  column (10): {header[:10]}")
    
    # SMILEScolumn
    smiles_col = None
    name_col = None
    for c in header:
        cl = c.lower().strip()
        if "smiles" in cl or "smile" in cl:
            smiles_col = c
        if "abbreviation" in cl or "pfas" in cl:
            name_col = c
    
    if not name_col:
        name_col = header[0]  # columnisPFAS abbreviation
    print(f"  PFASname column: '{name_col}'")
    print(f"  SMILEScolumn:   '{smiles_col}'")

    if not smiles_col:
        print("❌ not foundSMILEScolumn! ")
        sys.exit(1)

    # process per row
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
            print(f"  [{i+1}/{len(rows)}] {pfas_name} ❌ empty/invalidSMILES: '{raw_smiles}'")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed_parse.append(pfas_name)
            print(f"  [{i+1}/{len(rows)}] {pfas_name} ❌ parse failed: {smiles[:60]}")
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
            print(f"  [{i+1}/{len(rows)}] ✅ success {succeeded}, elapsed {elapsed:.1f}s")

    # statistics
    elapsed = time.time() - start_time
    print(f"\n  ✅ success: {succeeded}/{len(rows)}")
    print(f"  ❌ emptySMILES: {len(failed_smiles)} - {failed_smiles}")
    print(f"  ❌ parse failed: {len(failed_parse)} - {failed_parse}")
    print(f"  total elapsed: {elapsed:.1f} s")

    if not results:
        print("❌ all failed! ")
        sys.exit(1)

    # write toCSV
    sample = results[0]
    fieldnames = ["PFAS_name", "Original_SMILES", "RDKIT_SMILES"] + \
                 [k for k in sample.keys() if k not in ("PFAS_name", "Original_SMILES", "RDKIT_SMILES")]
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for res in results:
            writer.writerow(res)

    print(f"\n  output: {OUTPUT_FILE}")
    print(f"  {len(results)} row × {len(fieldnames)} column")
    print(f"\n✅ S1 ! ")


if __name__ == "__main__":
    main()
