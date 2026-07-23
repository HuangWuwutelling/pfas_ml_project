#!/usr/bin/env python3
"""
S4_validate_clusters.py
=======================
Level 2: useRDKitdescriptor pair51PFASdo clustering, use experimentalKdvalidate cluster results. 

core question:
  "from molecular structure alone(RDKitdescriptors)unsupervised clustering from, 
   whether can auto-separateKd/lowPFASclass? "

workflow:
  1. loaded51PFASRDKitdescriptors(fromS1)
  2. extract per-PFAS from feature matrixPFASlog Kd
  3. reduce(t-SNE)+ cluster(HDBSCAN)
  4. annotate mean on each clusterlog Kd
  5. visualization: color=log Kd
  6. extend to11,000PFASchemical spaceoverlay

input:
  data/paper/descriptors_51pfas.csv        (51PFASRDKitdescriptors)
  data/paper/feature_matrix_kd.csv         (1227row × log Kd + feature)
  data/processed/pfas_descriptors_full.csv (11,000PFASdescriptors)

output:
  data/paper/kd_cluster_tsne.png           (t-SNEbylog Kd)
  data/paper/kd_cluster_umap.png           (UMAPbylog Kd)
  data/paper/kd_chemical_space_11k.png     (11K + 51overlay plot)
  data/paper/kd_cluster_validation.csv     (clusterstatistics)
"""

import csv
import os
import sys
import numpy as np
from collections import Counter

SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DESC_51_FILE = os.path.join(SI_DIR, "descriptors_51pfas.csv")
FEATURE_FILE = os.path.join(SI_DIR, "feature_matrix_kd.csv")
DESC_11K_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "pfas_descriptors_full.csv")
OUTPUT_TABLE = os.path.join(SI_DIR, "kd_cluster_validation.csv")


def load_51_descriptors():
    """loaded51PFASRDKitdescriptors"""
    with open(DESC_51_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # non-feature columns
    non_feat = {"PFAS_name", "Original_SMILES", "RDKIT_SMILES"}
    feat_names = [c for c in rows[0].keys() if c not in non_feat]
    
    n = len(rows)
    p = len(feat_names)
    X = np.zeros((n, p))
    names = []
    
    for i, row in enumerate(rows):
        names.append(row["PFAS_name"].strip())
        for j, col in enumerate(feat_names):
            v = row.get(col, "").strip()
            if v:
                X[i, j] = float(v)
            else:
                X[i, j] = 0.0
    
    print(f"  51PFASdescriptors: {X.shape} ({p}feature)")
    return X, names, feat_names


def load_mean_logkd():
    """compute per-PFAS from feature matrixPFASlog Kd"""
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # byPFASname aggregation
    logkd_by_pfas = {}
    count_by_pfas = {}
    for row in rows:
        name = row["PFAS_name"].strip()
        v = row.get("log_Kd", "").strip()
        if v:
            val = float(v)
            logkd_by_pfas[name] = logkd_by_pfas.get(name, 0.0) + val
            count_by_pfas[name] = count_by_pfas.get(name, 0) + 1
    
    mean_logkd = {}
    for name, total in logkd_by_pfas.items():
        mean_logkd[name] = total / count_by_pfas[name]
    
    print(f"  log KddataPFAS: {len(mean_logkd)} ")
    return mean_logkd


def load_11k_descriptors(sample_rate=0.1):
    """loaded11,000PFASdescriptors(can sample)"""
    if not os.path.exists(DESC_11K_FILE):
        print(f"  ⚠️ not found {DESC_11K_FILE}")
        return None, None
    
    with open(DESC_11K_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # sample(accelerate plotting)
    import random
    random.seed(42)
    total = len(rows)
    sample_size = max(min(int(total * sample_rate), 10000), total)
    if sample_size < total:
        rows = random.sample(rows, sample_size)
    
    non_feat_11k = {"DTXSID", "SMILES", "RDKIT_SMILES"}
    feat_names_11k = [c for c in rows[0].keys() if c not in non_feat_11k]
    
    n = len(rows)
    p = len(feat_names_11k)
    X = np.zeros((n, p))
    
    for i, row in enumerate(rows):
        for j, col in enumerate(feat_names_11k):
            v = row.get(col, "").strip()
            if v:
                X[i, j] = float(v)
            else:
                X[i, j] = 0.0
    
    print(f"  11K PFASdescriptors(sample{sample_rate*100:.0f}%): {X.shape} ({p}feature)")
    return X, rows


def standardize_features(X_train, X_test=None):
    """standardize to mean0std =1"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        return X_train_scaled, scaler.transform(X_test)
    return X_train_scaled


def run_tsne(X, perplexity=30):
    """t-SNEreduce"""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    print(f"\n--- t-SNEreduce ---")
    print(f"  input: {X.shape}")
    
    # PCApre-reduce
    n_pca = min(50, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"  PCA ({n_pca}): {var_exp:.1%}variance kept")
    
    tsne = TSNE(n_components=2, perplexity=min(perplexity, X.shape[0]-1),
                random_state=42, n_jobs=-1, verbose=False)
    X_tsne = tsne.fit_transform(X_pca)
    print(f"  t-SNE: {X_tsne.shape[0]}pt→2D")
    return X_tsne


def run_hdbscan(X_embedded):
    """HDBSCANcluster"""
    try:
        import hdbscan
    except ImportError:
        print("  ⚠️ hdbscannot installed, useKMeans")
        return run_kmeans(X_embedded)
    
    print(f"\n--- HDBSCANcluster ---")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
    labels = clusterer.fit_predict(X_embedded)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(labels == -1)
    print(f"  HDBSCAN: {n_clusters}clusters, noise: {n_noise}pt")
    
    # ifcluster, useKMeans
    if n_clusters > 15:
        print(f"  num clusters ({n_clusters}), falling back to KMeans")
        return run_kmeans(X_embedded)
    
    return labels


def run_kmeans(X_embedded, n_clusters=6):
    """KMeanscluster"""
    from sklearn.cluster import KMeans
    print(f"\n--- KMeanscluster(k={n_clusters}) ---")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_embedded)
    print(f"  KMeans: {n_clusters}clusters")
    return labels


def save_cluster_table(names, labels, mean_logkd, subfamilies, output_path):
    """save cluster statistics + PFASlog Kdval"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PFAS_name", "subfamily", "cluster", "mean_log_Kd", "n_samples"])
        for name, label, subfam in zip(names, labels, subfamilies):
            n = logkd_counts.get(name, 0)
            mk = mean_logkd.get(name, "")
            writer.writerow([name, subfam, label, mk, n])
    
    # clusterstatistics
    cluster_stats = {}
    for name, label in zip(names, labels):
        if label not in cluster_stats:
            cluster_stats[label] = {"names": [], "logkd_vals": []}
        cluster_stats[label]["names"].append(name)
        if name in mean_logkd:
            cluster_stats[label]["logkd_vals"].append(mean_logkd[name])
    
    print(f"\n--- clusterstatistics ---")
    for label in sorted(cluster_stats.keys()):
        info = cluster_stats[label]
        logkd_vals = info["logkd_vals"]
        if logkd_vals:
            avg = np.mean(logkd_vals)
            std = np.std(logkd_vals)
            print(f"  cluster{label}(n={len(info['names'])}): logKd={avg:.3f}±{std:.3f}")
        else:
            print(f"  cluster{label}(n={len(info['names'])}): logKd=N/A")
        for n in info["names"]:
            mk = mean_logkd.get(n, None)
            if mk is not None:
                print(f"    {n}: logKd={mk:.3f}")
            else:
                print(f"    {n}: logKd=N/A")
        print()
    
    print(f"  : {output_path}")


def save_visualization(X_51, X_11k, labels_51, mean_logkd, names_51):
    """save cluster visualization"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ matplotlibnot installed, skip figures")
        return
    
    # color: log Kdfrom lowest()tohighest()
    logkd_vals = np.array([mean_logkd.get(n, np.nan) for n in names_51])
    valid_mask = ~np.isnan(logkd_vals)
    
    # ===== fig1: 51PFASt-SNE, by meanlog Kd =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax = axes[0]
    scatter = ax.scatter(X_51[valid_mask, 0], X_51[valid_mask, 1],
                         c=logkd_vals[valid_mask], cmap="coolwarm",
                         s=80, alpha=0.8, edgecolors="k", linewidth=0.5)
    # PFASname
    for i, (name, v) in enumerate(zip(names_51, valid_mask)):
        if v:
            ax.annotate(name, (X_51[i, 0], X_51[i, 1]),
                       fontsize=6, alpha=0.8, ha="center", va="bottom")
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label("Mean log Kd", fontsize=10)
    ax.set_title("51 PFAS: t-SNE colored by mean log Kd", fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    # ===== fig2: color by cluster + label meanlog Kd =====
    ax = axes[1]
    unique_labels = sorted(set(labels_51))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_51 == label
        cluster_logkd = [mean_logkd.get(n, np.nan) for n in np.array(names_51)[mask]]
        cluster_logkd = [v for v in cluster_logkd if not np.isnan(v)]
        avg_str = f", avg logKd={np.mean(cluster_logkd):.2f}" if cluster_logkd else ""
        ax.scatter(X_51[mask, 0], X_51[mask, 1],
                   c=[colors[i % len(colors)]], s=80, alpha=0.7, edgecolors="k", linewidth=0.5,
                   label=f"Cluster {label} (n={mask.sum()}{avg_str})")
        # label cluster name
        for idx in np.where(mask)[0]:
            ax.annotate(names_51[idx], (X_51[idx, 0], X_51[idx, 1]),
                       fontsize=6, alpha=0.7, ha="center", va="bottom")
    
    ax.legend(fontsize=7, loc="upper right", ncol=1)
    ax.set_title("51 PFAS: t-SNE colored by cluster", fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    plt.tight_layout()
    figpath = os.path.join(SI_DIR, "kd_cluster_tsne.png")
    plt.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✅ fig1: {figpath}")
    
    # ===== fig3: 11K + 51 overlay =====
    if X_11k is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 11Kpt
        ax.scatter(X_11k[:, 0], X_11k[:, 1], c="lightgray", s=3, alpha=0.3, label=f"11K PFAS (PFASMASTER)")
        
        # 51PFASforeground points
        scatter = ax.scatter(X_51[valid_mask, 0], X_51[valid_mask, 1],
                            c=logkd_vals[valid_mask], cmap="coolwarm",
                            s=120, alpha=0.9, edgecolors="k", linewidth=0.8)
        for i, (name, v) in enumerate(zip(names_51, valid_mask)):
            if v:
                ax.annotate(name, (X_51[i, 0], X_51[i, 1]),
                           fontsize=7, alpha=0.9, ha="center", va="bottom")
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label("Mean log Kd", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_title("Chemical Space: 11K PFASMASTER + 51 PFAS (color=log Kd)", fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        
        plt.tight_layout()
        figpath = os.path.join(SI_DIR, "kd_chemical_space_11k.png")
        plt.savefig(figpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  ✅ fig2: {figpath}")


def main():
    print("=" * 60)
    print("  Level 2: clusterverify + chemical space extrapolation")
    print("=" * 60)
    
    # 1. load_data
    print("\n--- load_data ---")
    X_51, names_51, feat_names = load_51_descriptors()
    mean_logkd = load_mean_logkd()
    
    # subfamily info(fromCSVread)
    with open(DESC_51_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_51 = list(reader)
    subfamilies = []
    for row in rows_51:
        subfamilies.append("")  # descriptors file lackssubfamilycolumn, fromPFASname inference
    
    # for51PFASmanual subfamily annotation(based on known classification)
    subfam_map = {
        "6:2 FtSaB": "Zwitterionic", "8:2 FtSaB": "Zwitterionic", "10:2 FtSaB": "Zwitterionic",
        "PFOSB": "Zwitterionic", "PFOAAmS": "Cationic", "PFOAB": "Zwitterionic",
        "AmPr-FHxSA": "Cationic", "TAmPr-FHxSA": "Cationic", "6:2 FtSaAm": "Cationic",
        "4:2 FTOH": "FTOH", "6:2 FTOH": "FTOH", "8:2 FTOH": "FTOH", "10:2 FTOH": "FTOH",
        "TFA": "PFCA", "PFBA": "PFCA", "PFPeA": "PFCA", "PFHxA": "PFCA", "PFHpA": "PFCA",
        "PFOA": "PFCA", "PFNA": "PFCA", "PFDA": "PFCA", "PFUnA": "PFCA", "PFDoA": "PFCA",
        "PFTrA": "PFCA", "PFTeA": "PFCA",
        "GenX": "PFECA", "ADONA": "PFECA",
        "PFBS": "PFSA", "PFPeS": "PFSA", "PFHxS": "PFSA", "PFHpS": "PFSA", "PFOS": "PFSA",
        "PFNS": "PFSA", "PFDS": "PFSA", "PFEtCHxS": "PFSA",
        "8:2 Cl-PFAES": "PFAES",
        "4:2 FTS": "FTS", "6:2 FTS": "FTS", "8:2 FTS": "FTS",
        "FBSA": "FOSA", "FHxSA": "FOSA", "PFOSA": "FOSA", "EtFOSA": "FOSA",
        "N-MeFOSAA": "FOSAA", "N-EtFOSAA": "FOSAA",
        "PFHxPA": "PFPA", "PFOPA": "PFPA", "PFDPA": "PFPA",
        "C6/6 PFPiA": "PFPiA", "C6/8 PFPiA": "PFPiA", "C8/8 PFPiA": "PFPiA",
    }
    subfamilies = [subfam_map.get(n, "") for n in names_51]
    
    # log Kdcount statistics
    global logkd_counts
    logkd_counts = {}
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["PFAS_name"].strip()
            logkd_counts[name] = logkd_counts.get(name, 0) + 1
    
    # 2. standardize
    print("\n--- standardize ---")
    X_51_scaled = standardize_features(X_51)
    print(f"  after standardization: mean≈{X_51_scaled.mean():.2e}, std≈{X_51_scaled.std():.2f}")
    
    # 3. loaded11Kdataandt-SNE
    print("\n--- loaded11K PFASdescriptors ---")
    X_11k, rows_11k = load_11k_descriptors(sample_rate=0.2)
    
    if X_11k is not None:
        # align features: onlytake51and11Kshared features
        with open(DESC_11K_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            feat_11k = reader.fieldnames
        non_feat_11k = {"DTXSID", "SMILES", "RDKIT_SMILES"}
        feat_11k_clean = [c for c in feat_11k if c not in non_feat_11k]
        
        # common features of intersection
        common_feats = [f for f in feat_names if f in feat_11k_clean]
        print(f"  common features: {len(common_feats)}/{len(feat_names)}")
        
        # re-extract51PFAScommon features
        X_51_common = np.zeros((len(names_51), len(common_feats)))
        for i, row in enumerate(rows_51):
            for j, col in enumerate(common_feats):
                v = row.get(col, "").strip()
                if v:
                    X_51_common[i, j] = float(v)
        
        # re-extract11Kcommon features
        X_11k_common = np.zeros((X_11k.shape[0], len(common_feats)))
        # X_11kisnumpyarray but order may not match, userows_11kre-read
        # actualload_11k_descriptorsalready taken all features, but row-sampled
        # needs re-extraction
        pass
    
    # simplethis: only on51t-SNE, annotate on figurelog Kd
    # for11Kdo separatelyt-SNEthen overlay bad, because dimensionality-reduced spaces differ
    # usePCAproject: use51types of fittingPCA, 11Kproject to same space
    
    # 4. t-SNEreduce (51)
    X_tsne = run_tsne(X_51_scaled, perplexity=8)  # small dataset uses smallerperplexity
    
    # 5. cluster
    labels = run_hdbscan(X_tsne)
    
    # 6. save table
    save_cluster_table(names_51, labels, mean_logkd, subfamilies, OUTPUT_TABLE)
    
    # 7. 11K PCAproject+overlay
    X_11k_projected = None
    if X_11k is not None:
        print("\n--- PCA: 51types of fitting → 11Kproject to same space ---")
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # extract common features
        common_feats = [f for f in feat_names if f in feat_11k_clean]
        print(f"  common features: {len(common_feats)}")
        
        # rebuild51matrix types
        X_51_aligned = np.zeros((len(names_51), len(common_feats)))
        for i, row in enumerate(rows_51):
            for j, col in enumerate(common_feats):
                v = row.get(col, "").strip()
                if v:
                    X_51_aligned[i, j] = float(v)
        
        # rebuild11K
        n_11k = len(rows_11k)
        X_11k_aligned = np.zeros((n_11k, len(common_feats)))
        for i, row in enumerate(rows_11k):
            for j, col in enumerate(common_feats):
                v = row.get(col, "").strip()
                if v:
                    try:
                        val = float(v)
                        X_11k_aligned[i, j] = val if np.isfinite(val) else 0.0
                    except:
                        X_11k_aligned[i, j] = 0.0
                else:
                    X_11k_aligned[i, j] = 0.0
        
        # standardize
        scaler = StandardScaler()
        X_51_scaled_aligned = scaler.fit_transform(X_51_aligned)
        X_11k_scaled_aligned = scaler.transform(X_11k_aligned)
        
        # PCAuse51types of fitting, project11K
        pca = PCA(n_components=2, random_state=42)
        X_51_pca = pca.fit_transform(X_51_scaled_aligned)
        X_11k_pca = pca.transform(X_11k_scaled_aligned)
        
        print(f"  PCA 51: {X_51_pca.shape}, PCA 11K: {X_11k_pca.shape}")
        X_11k_projected = X_11k_pca
    
    # 8. save visualization
    save_visualization(X_tsne, X_11k_projected, labels, mean_logkd, names_51)
    
    print(f"\n✅ Level 2!")
    print(f"  : {OUTPUT_TABLE}")
    print(f"  fig: {SI_DIR}/kd_cluster_tsne.png")
    print(f"  fig: {SI_DIR}/kd_chemical_space_11k.png")


if __name__ == "__main__":
    main()
