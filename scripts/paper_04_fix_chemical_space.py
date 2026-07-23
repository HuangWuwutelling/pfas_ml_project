"""
fix chemical space figure: use jointt-SNEreduce (51 + 11Kjoint dimensionality reduction)。
PCAis linear dimensionality reduction，。
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
DESC_51_FILE = os.path.join(SI_DIR, "descriptors_51pfas.csv")
FEATURE_FILE = os.path.join(SI_DIR, "feature_matrix_kd.csv")
DESC_11K_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "pfas_descriptors_full.csv")

# 1. loaded51PFAS
with open(DESC_51_FILE, "r") as f:
    reader = csv.DictReader(f)
    rows_51 = list(reader)
non_feat = {"PFAS_name", "Original_SMILES", "RDKIT_SMILES"}
feat_names = [c for c in rows_51[0].keys() if c not in non_feat]
n_feat = len(feat_names)

X_51 = np.zeros((len(rows_51), n_feat))
names_51 = []
for i, row in enumerate(rows_51):
    names_51.append(row["PFAS_name"])
    for j, col in enumerate(feat_names):
        v = row.get(col, "").strip()
        X_51[i, j] = float(v) if v else 0.0

# 2. load alllog Kd
logkd_by_name = {}
with open(FEATURE_FILE, "r") as f:
    for row in csv.DictReader(f):
        name = row["PFAS_name"].strip()
        v = row.get("log_Kd", "").strip()
        if v:
            val = float(v)
            logkd_by_name[name] = logkd_by_name.get(name, 0.0) + val
count_by_name = {}
with open(FEATURE_FILE, "r") as f:
    for row in csv.DictReader(f):
        name = row["PFAS_name"].strip()
        count_by_name[name] = count_by_name.get(name, 0) + 1
mean_logkd = {n: logkd_by_name[n] / count_by_name[n] for n in logkd_by_name}

# 3. loaded11K
print("Loading 11K...")
with open(DESC_11K_FILE, "r") as f:
    reader = csv.DictReader(f)
    rows_11k = list(reader)

# take11Ksamefeature
feat_11k = reader.fieldnames
feat_11k_clean = [c for c in feat_11k if c not in {"DTXSID", "SMILES", "RDKIT_SMILES"}]
common = [f for f in feat_names if f in feat_11k_clean]
print(f"Common features: {len(common)}/{n_feat}")

# sample11Kto2000points（t-SNEruns too slow）
import random
random.seed(42)
n_11k = len(rows_11k)
sample_size = min(2000, n_11k)
rows_11k_sample = random.sample(rows_11k, sample_size)

# build11K（by common features）
X_11k = np.zeros((sample_size, len(common)))
feat_idx_map = {f: i for i, f in enumerate(common)}
for i, row in enumerate(rows_11k_sample):
    for col in common:
        j = feat_idx_map[col]
        v = row.get(col, "").strip()
        if v:
            try:
                X_11k[i, j] = float(v) if np.isfinite(float(v)) else 0.0
            except:
                X_11k[i, j] = 0.0

# rebuild51PFASfeature matrix
X_51_aligned = np.zeros((len(rows_51), len(common)))
for i, row in enumerate(rows_51):
    for col in common:
        j = feat_idx_map[col]
        v = row.get(col, "").strip()
        X_51_aligned[i, j] = float(v) if v else 0.0

# 5. standardize
print("Standardizing...")
X_all = np.vstack([X_11k, X_51_aligned])
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# 6. first usePCAreduce to50
print("PCA pre-reduction...")
pca = PCA(n_components=50, random_state=42)
X_all_pca = pca.fit_transform(X_all_scaled)
print(f"PCA variance retained: {pca.explained_variance_ratio_.sum():.2%}")

# 7. t-SNE
print("t-SNE on combined (11K+51)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1, verbose=True)
X_all_tsne = tsne.fit_transform(X_all_pca)

# split
X_11k_tsne = X_all_tsne[:sample_size]
X_51_tsne = X_all_tsne[sample_size:]

# 8. Plot
print("Plotting...")
fig, ax = plt.subplots(figsize=(14, 10))

# 11K
ax.scatter(X_11k_tsne[:, 0], X_11k_tsne[:, 1], 
           c="lightgray", s=3, alpha=0.3, label=f"PFASMASTER (n={sample_size})")

# 51PFAS
logkd_vals = np.array([mean_logkd.get(n, np.nan) for n in names_51])
valid = ~np.isnan(logkd_vals)

scatter = ax.scatter(X_51_tsne[valid, 0], X_51_tsne[valid, 1],
                     c=logkd_vals[valid], cmap="coolwarm",
                     s=100, alpha=0.9, edgecolors="k", linewidth=0.8)

# PFASname
for i, (name, v) in enumerate(zip(names_51, valid)):
    if v:
        ax.annotate(name, (X_51_tsne[i, 0], X_51_tsne[i, 1]),
                   fontsize=6, alpha=0.85, ha="center", va="bottom")

cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
cbar.set_label("Mean log Kd", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Chemical space: 11K PFASMASTER + 51 PFAS with known Kd (joint t-SNE)", fontsize=12)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")

# also save an unlabeled version for comparison
plt.tight_layout()
out = os.path.join(SI_DIR, "kd_chemical_space_11k_fixed.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out}")

# also save high-res annotated version
fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(X_11k_tsne[:, 0], X_11k_tsne[:, 1], 
           c="lightgray", s=2, alpha=0.25)
scatter = ax.scatter(X_51_tsne[valid, 0], X_51_tsne[valid, 1],
                     c=logkd_vals[valid], cmap="coolwarm",
                     s=120, alpha=0.9, edgecolors="k", linewidth=0.8)
for i, (name, v) in enumerate(zip(names_51, valid)):
    if v:
        logkd = mean_logkd[name]
        color = "red" if logkd > 1.5 else ("orange" if logkd > 0.5 else "blue")
        ax.annotate(f"{name}\n(logKd={logkd:.2f})", 
                   (X_51_tsne[i, 0], X_51_tsne[i, 1]),
                   fontsize=7, alpha=0.9, ha="center", va="bottom",
                   color=color, fontweight="bold")
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
cbar.set_label("Mean log Kd", fontsize=10)
ax.set_title("Chemical Space: 11K PFASMASTER + 51 PFAS with experimental Kd (joint t-SNE)", fontsize=12)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.tight_layout()
out2 = os.path.join(SI_DIR, "kd_chemical_space_annotated.png")
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {out2}")

print("Done!")
