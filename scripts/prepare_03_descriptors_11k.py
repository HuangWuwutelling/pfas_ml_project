#!/usr/bin/env python3
"""
06_calc_descriptors_full.py
============================
use RDKit batch compute EPA PFASMASTER full list(~11,000)molecular descriptorsandfingerprint. 

input: data/processed/pfas_clean.csv
       ( EPA PFASMASTER list, 10,972rows containingSMILES)
output: data/processed/pfas_descriptors_full.csv
       (217RDKitdescriptors + PFASspecific features)
      data/processed/pfas_fingerprint_full.csv
       (2048ECFP4Morgan fingerprint)

and02_calc_descriptors.py: 
  - inputfromPubChemdatasetEPACleanlist
  - SMILESfrom pfas_clean.csv  SMILES column read
  - containsSMILESformat cleanup(remove |lp:...| extension marker)
  - output filename + _full suffix

row:
  python scripts/06_calc_descriptors_full.py

estimated elapsed: 2-5min(11,000rows)
"""

import csv
import os
import sys
import time
import numpy as np

# path config
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "processed", "pfas_clean.csv")
OUTPUT_DESC = os.path.join(DATA_DIR, "processed", "pfas_descriptors_full.csv")
OUTPUT_FP = os.path.join(DATA_DIR, "processed", "pfas_fingerprint_full.csv")

# ==========================================
# RDKit import check
# ==========================================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
    print("✅ RDKit installed")
except ImportError:
    RDKIT_AVAILABLE = False
    print("❌ RDKit not installed, please run: pip install rdkit-pypi")
    print("   re-run this script after install")


def clean_smiles(smiles):
    """SMILES: remove |lp:...| extension markers and other non-standard formats"""
    if not smiles or not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s:
        return None
    # remove |...| extension marker(RDKit),  |lp:4:2,6:3...|
    pipe_pos = s.find("|")
    if pipe_pos != -1:
        s = s[:pipe_pos].strip()
    if not s:
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


def calc_fingerprint(mol, radius=2, nbits=2048):
    """compute ECFP4 (Morgan) molecular fingerprint"""
    if mol is None:
        return None
    from rdkit.Chem import rdMolDescriptors
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.uint8)


def calc_pfas_specific_features(mol):
    """computePFASspecific features(atRDKitparsed mol object)"""
    features = {
        "carbon_count": 0,       # atom count
        "fluorine_count": 0,     # F atom count
        "has_sulfonate": 0,      # contains sulfonate -S(=O)(=O)O
        "has_carboxyl": 0,       # contains carboxyl -C(=O)O
        "has_ether": 0,          # contains ether C-O-C
        "has_aromatic": 0,       # contains aromatic ring
        "has_double_bond": 0,    # contains C=C
        "perfluoro_ratio": 0.0,  # fluorination ratio (Fatom count / total atom count)
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
    mol_s = mol
    if Chem.MolFromSmarts("[S](=O)(=O)[O]") and mol_s.HasSubstructMatch(Chem.MolFromSmarts("[S](=O)(=O)[O]")):
        features["has_sulfonate"] = 1
    if Chem.MolFromSmarts("[CX3](=O)[OX2]") and mol_s.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2]")):
        features["has_carboxyl"] = 1
    if Chem.MolFromSmarts("[CX4][OX2][CX4]") and mol_s.HasSubstructMatch(Chem.MolFromSmarts("[CX4][OX2][CX4]")):
        features["has_ether"] = 1
    if any(a.GetIsAromatic() for a in atoms):
        features["has_aromatic"] = 1
    # 
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
    print("  EPA PFASMASTER full descriptor computation")
    print("  input: pfas_clean.csv (10,972rows)")
    print("=" * 60)

    # read cleanedPFASlist
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n  read {len(rows):,} PFAS")
    print(f"  descriptors: {len(Descriptors._descList)}")
    print(f"  fingerprint: 2048  ECFP4\n")

    # process per row
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

        # SMILES
        smiles = clean_smiles(raw_smiles)
        if smiles is None:
            print(f"  [{i+1}/{len(rows)}] {dtxsid} ❌ emptySMILES")
            results_desc.append(None)
            results_fp.append(None)
            skipped_no_smiles += 1
            continue

        # SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [{i+1}/{len(rows)}] {dtxsid} ❌ parse failed: {smiles[:60]}")
            results_desc.append(None)
            results_fp.append(None)
            skipped_parse_fail += 1
            continue

        # compute standard descriptors
        desc = calc_all_descriptors(mol)

        # takeRDKitnormalizedSMILES(for cross-dataset key matching)
        rdkit_smiles_norm = Chem.MolToSmiles(mol)

        # computePFASspecific features
        pfas_feat = calc_pfas_specific_features(mol)
        desc.update(pfas_feat)

        # compute fingerprints
        fp = calc_fingerprint(mol)

        results_desc.append({"desc": desc, "rdkit_smiles": rdkit_smiles_norm})
        results_fp.append(fp)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(rows)}] ✅ {rate:.0f} rows/s, estimated remaining {eta:.1f} min")

    # statistics
    valid_idx = [i for i, d in enumerate(results_desc) if d is not None]
    print(f"\n  success: {len(valid_idx):,}")
    print(f"  emptySMILES: {skipped_no_smiles}")
    print(f"  parse failed: {skipped_parse_fail}")

    if not valid_idx:
        print("  ❌ all failed, exit")
        sys.exit(1)

    # determineoutputfield
    sample_desc = results_desc[valid_idx[0]]["desc"]
    all_fieldnames = ["DTXSID", "SMILES", "RDKIT_SMILES"] + list(sample_desc.keys())

    # write descriptorsCSV
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

    # write fingerprintsCSV
    fp_header = ["DTXSID", "RDKIT_SMILES"] + [f"FP_{i}" for i in range(2048)]
    with open(OUTPUT_FP, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fp_header)
        for idx in valid_idx:
            fp = results_fp[idx]
            fp_row = [rows[idx].get("DTXSID", ""), results_desc[idx]["rdkit_smiles"]] + fp.tolist()
            writer.writerow(fp_row)

    elapsed = time.time() - start_time
    print(f"\n  ✅ ! elapsed {elapsed:.1f} s")
    print(f"  descriptors: {OUTPUT_DESC}")
    print(f"     {len(valid_idx):,} row × {len(all_fieldnames)} column")
    print(f"  fingerprint: {OUTPUT_FP}")
    print(f"     {len(valid_idx):,} row × 2048 column")

    # PFASfeature statistics
    print(f"\n  PFASfeature statistics:")
    for feat_name in pfas_feature_names:
        vals = [results_desc[i]["desc"][feat_name] for i in valid_idx
                if feat_name in results_desc[i]["desc"] and results_desc[i]["desc"][feat_name] is not None]
        if vals:
            if isinstance(vals[0], (int, float)):
                if all(v in (0, 1) for v in vals):
                    # binary features
                    ones = sum(v == 1 for v in vals)
                    print(f"    {feat_name}: {ones}/{len(vals)} ({ones/len(vals)*100:.1f}%)")
                else:
                    # numeric features
                    print(f"    {feat_name}: min={min(vals):.2f}, max={max(vals):.2f}, mean={np.mean(vals):.2f}")

    # SMILESparse rate
    total_valid = len(rows) - skipped_no_smiles
    parse_rate = len(valid_idx) / max(total_valid, 1) * 100
    print(f"\n  SMILESparse rate: {len(valid_idx)}/{total_valid} = {parse_rate:.1f}%")


if __name__ == "__main__":
    main()
