#!/usr/bin/env python3
"""
S4_validate_clusters.py
=======================
Level 2: 用RDKit描述符对51种PFAS做聚类，用实验Kd值验证聚类结果。

核心问题:
  "仅从分子结构（RDKit描述符）出发的无监督聚类，
   是否能自动分离出Kd高/低的PFAS类别？"

流程:
  1. 加载51种PFAS的RDKit描述符（来自S1）
  2. 从特征矩阵提取每种PFAS的平均log Kd
  3. 降维（t-SNE）+ 聚类（HDBSCAN）
  4. 在每个簇上标注平均log Kd
  5. 可视化: 颜色=平均log Kd
  6. 扩展到11,000种PFAS的化学空间叠加

输入:
  data/paper/descriptors_51pfas.csv        (51种PFAS的RDKit描述符)
  data/paper/feature_matrix_kd.csv         (1227行 × log Kd + 特征)
  data/processed/pfas_descriptors_full.csv (11,000种PFAS的描述符)

输出:
  data/paper/kd_cluster_tsne.png           (t-SNE按log Kd着色)
  data/paper/kd_cluster_umap.png           (UMAP按log Kd着色)
  data/paper/kd_chemical_space_11k.png     (11K + 51叠加图)
  data/paper/kd_cluster_validation.csv     (簇统计)
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
DESC_11K_FILE = "/home/zaoquan/pfas_ml_project/data/processed/pfas_descriptors_full.csv"
OUTPUT_TABLE = os.path.join(SI_DIR, "kd_cluster_validation.csv")


def load_51_descriptors():
    """加载51种PFAS的RDKit描述符"""
    with open(DESC_51_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 非特征列
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
    
    print(f"  51种PFAS描述符: {X.shape} ({p}特征)")
    return X, names, feat_names


def load_mean_logkd():
    """从特征矩阵计算每种PFAS的平均log Kd"""
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 按PFAS名聚合
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
    
    print(f"  含log Kd数据的PFAS: {len(mean_logkd)} 种")
    return mean_logkd


def load_11k_descriptors(sample_rate=0.1):
    """加载11,000种PFAS的描述符(可采样)"""
    if not os.path.exists(DESC_11K_FILE):
        print(f"  ⚠️ 找不到 {DESC_11K_FILE}")
        return None, None
    
    with open(DESC_11K_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 采样(加速绘图)
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
    
    print(f"  11K PFAS描述符(采样{sample_rate*100:.0f}%): {X.shape} ({p}特征)")
    return X, rows


def standardize_features(X_train, X_test=None):
    """标准化到均值为0标准差为1"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        return X_train_scaled, scaler.transform(X_test)
    return X_train_scaled


def run_tsne(X, perplexity=30):
    """t-SNE降维"""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    print(f"\n--- t-SNE降维 ---")
    print(f"  输入: {X.shape}")
    
    # PCA预降维
    n_pca = min(50, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"  PCA ({n_pca}维): {var_exp:.1%}方差保留")
    
    tsne = TSNE(n_components=2, perplexity=min(perplexity, X.shape[0]-1),
                random_state=42, n_jobs=-1, verbose=False)
    X_tsne = tsne.fit_transform(X_pca)
    print(f"  t-SNE完成: {X_tsne.shape[0]}点→2D")
    return X_tsne


def run_hdbscan(X_embedded):
    """HDBSCAN聚类"""
    try:
        import hdbscan
    except ImportError:
        print("  ⚠️ hdbscan未安装，改用KMeans")
        return run_kmeans(X_embedded)
    
    print(f"\n--- HDBSCAN聚类 ---")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
    labels = clusterer.fit_predict(X_embedded)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(labels == -1)
    print(f"  HDBSCAN: {n_clusters}个簇, 噪声: {n_noise}点")
    
    # 如果簇太多，改用KMeans
    if n_clusters > 15:
        print(f"  簇数过多({n_clusters})，改用KMeans")
        return run_kmeans(X_embedded)
    
    return labels


def run_kmeans(X_embedded, n_clusters=6):
    """KMeans聚类"""
    from sklearn.cluster import KMeans
    print(f"\n--- KMeans聚类(k={n_clusters}) ---")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_embedded)
    print(f"  KMeans完成: {n_clusters}个簇")
    return labels


def save_cluster_table(names, labels, mean_logkd, subfamilies, output_path):
    """保存簇统计 + 各PFAS的log Kd均值"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PFAS_name", "subfamily", "cluster", "mean_log_Kd", "n_samples"])
        for name, label, subfam in zip(names, labels, subfamilies):
            n = logkd_counts.get(name, 0)
            mk = mean_logkd.get(name, "")
            writer.writerow([name, subfam, label, mk, n])
    
    # 簇统计摘要
    cluster_stats = {}
    for name, label in zip(names, labels):
        if label not in cluster_stats:
            cluster_stats[label] = {"names": [], "logkd_vals": []}
        cluster_stats[label]["names"].append(name)
        if name in mean_logkd:
            cluster_stats[label]["logkd_vals"].append(mean_logkd[name])
    
    print(f"\n--- 簇统计 ---")
    for label in sorted(cluster_stats.keys()):
        info = cluster_stats[label]
        logkd_vals = info["logkd_vals"]
        if logkd_vals:
            avg = np.mean(logkd_vals)
            std = np.std(logkd_vals)
            print(f"  簇{label}(n={len(info['names'])}): logKd={avg:.3f}±{std:.3f}")
        else:
            print(f"  簇{label}(n={len(info['names'])}): logKd=N/A")
        for n in info["names"]:
            mk = mean_logkd.get(n, None)
            if mk is not None:
                print(f"    {n}: logKd={mk:.3f}")
            else:
                print(f"    {n}: logKd=N/A")
        print()
    
    print(f"  表格: {output_path}")


def save_visualization(X_51, X_11k, labels_51, mean_logkd, names_51):
    """保存聚类可视化图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ matplotlib未安装，跳过图表")
        return
    
    # 颜色映射: log Kd从最低(蓝)到最高(红)
    logkd_vals = np.array([mean_logkd.get(n, np.nan) for n in names_51])
    valid_mask = ~np.isnan(logkd_vals)
    
    # ===== 图1: 51种PFAS的t-SNE, 按平均log Kd着色 =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax = axes[0]
    scatter = ax.scatter(X_51[valid_mask, 0], X_51[valid_mask, 1],
                         c=logkd_vals[valid_mask], cmap="coolwarm",
                         s=80, alpha=0.8, edgecolors="k", linewidth=0.5)
    # 标注PFAS名称
    for i, (name, v) in enumerate(zip(names_51, valid_mask)):
        if v:
            ax.annotate(name, (X_51[i, 0], X_51[i, 1]),
                       fontsize=6, alpha=0.8, ha="center", va="bottom")
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label("Mean log Kd", fontsize=10)
    ax.set_title("51 PFAS: t-SNE colored by mean log Kd", fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    # ===== 图2: 按簇着色 + 标注平均log Kd =====
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
        # 标注簇名
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
    print(f"  ✅ 图1: {figpath}")
    
    # ===== 图3: 11K + 51 叠加 =====
    if X_11k is not None:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 11K背景点
        ax.scatter(X_11k[:, 0], X_11k[:, 1], c="lightgray", s=3, alpha=0.3, label=f"11K PFAS (PFASMASTER)")
        
        # 51种PFAS前景点
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
        print(f"  ✅ 图2: {figpath}")


def main():
    print("=" * 60)
    print("  Level 2: 聚类验证 + 化学空间外推")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n--- 加载数据 ---")
    X_51, names_51, feat_names = load_51_descriptors()
    mean_logkd = load_mean_logkd()
    
    # 子家族信息（从CSV读取）
    with open(DESC_51_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_51 = list(reader)
    subfamilies = []
    for row in rows_51:
        subfamilies.append("")  # 描述符文件没有subfamily列，从PFAS名称推断
    
    # 对51种PFAS手动标注子家族（基于已知分类）
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
    
    # log Kd计数统计
    global logkd_counts
    logkd_counts = {}
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["PFAS_name"].strip()
            logkd_counts[name] = logkd_counts.get(name, 0) + 1
    
    # 2. 标准化
    print("\n--- 标准化 ---")
    X_51_scaled = standardize_features(X_51)
    print(f"  标准化后: mean≈{X_51_scaled.mean():.2e}, std≈{X_51_scaled.std():.2f}")
    
    # 3. 加载11K数据并做联合t-SNE
    print("\n--- 加载11K PFAS描述符 ---")
    X_11k, rows_11k = load_11k_descriptors(sample_rate=0.2)
    
    if X_11k is not None:
        # 对齐特征：只取51种和11K共有的特征
        with open(DESC_11K_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            feat_11k = reader.fieldnames
        non_feat_11k = {"DTXSID", "SMILES", "RDKIT_SMILES"}
        feat_11k_clean = [c for c in feat_11k if c not in non_feat_11k]
        
        # 交集的共同特征
        common_feats = [f for f in feat_names if f in feat_11k_clean]
        print(f"  共同特征: {len(common_feats)}/{len(feat_names)}")
        
        # 重新提取51种PFAS的共同特征
        X_51_common = np.zeros((len(names_51), len(common_feats)))
        for i, row in enumerate(rows_51):
            for j, col in enumerate(common_feats):
                v = row.get(col, "").strip()
                if v:
                    X_51_common[i, j] = float(v)
        
        # 重新提取11K的共同特征
        X_11k_common = np.zeros((X_11k.shape[0], len(common_feats)))
        # X_11k已经是numpy数组但顺序可能不对，用rows_11k重新读
        # 实际上load_11k_descriptors已经拿了所有特征，但采样了行
        # 需要重新提取
        pass
    
    # 简单版本：只对51种做t-SNE，在图上标注log Kd
    # 对11K单独做t-SNE然后叠加不好，因为降维空间不同
    # 改用PCA投影：用51种拟合PCA，把11K投影到同一空间
    
    # 4. t-SNE降维 (51种)
    X_tsne = run_tsne(X_51_scaled, perplexity=8)  # 小数据集用更小的perplexity
    
    # 5. 聚类
    labels = run_hdbscan(X_tsne)
    
    # 6. 保存表格
    save_cluster_table(names_51, labels, mean_logkd, subfamilies, OUTPUT_TABLE)
    
    # 7. 11K PCA投影+叠加
    X_11k_projected = None
    if X_11k is not None:
        print("\n--- PCA: 51种拟合 → 11K投影到同一空间 ---")
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # 提取共同特征
        common_feats = [f for f in feat_names if f in feat_11k_clean]
        print(f"  共同特征: {len(common_feats)}")
        
        # 重建51种矩阵
        X_51_aligned = np.zeros((len(names_51), len(common_feats)))
        for i, row in enumerate(rows_51):
            for j, col in enumerate(common_feats):
                v = row.get(col, "").strip()
                if v:
                    X_51_aligned[i, j] = float(v)
        
        # 重建11K矩阵
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
        
        # 标准化
        scaler = StandardScaler()
        X_51_scaled_aligned = scaler.fit_transform(X_51_aligned)
        X_11k_scaled_aligned = scaler.transform(X_11k_aligned)
        
        # PCA用51种拟合，投影11K
        pca = PCA(n_components=2, random_state=42)
        X_51_pca = pca.fit_transform(X_51_scaled_aligned)
        X_11k_pca = pca.transform(X_11k_scaled_aligned)
        
        print(f"  PCA 51: {X_51_pca.shape}, PCA 11K: {X_11k_pca.shape}")
        X_11k_projected = X_11k_pca
    
    # 8. 保存可视化
    save_visualization(X_tsne, X_11k_projected, labels, mean_logkd, names_51)
    
    print(f"\n✅ Level 2完成!")
    print(f"  表格: {OUTPUT_TABLE}")
    print(f"  图表: {SI_DIR}/kd_cluster_tsne.png")
    print(f"  图表: {SI_DIR}/kd_chemical_space_11k.png")


if __name__ == "__main__":
    main()
