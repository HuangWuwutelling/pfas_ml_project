#!/usr/bin/env python3
"""
paper_08_transfer_learning.py
==============================
Transfer learning for PFAS Kd prediction.

Phase 1: Pretrain a molecular representation on 11K unlabeled PFAS compounds
         using an autoencoder (PyTorch, CPU-only).
Phase 2: Transfer the pretrained encoder to the 47 PFAS with Kd labels,
         train XGBoost on the compressed representation + soil features.
Phase 3: Compare with the baseline (225 raw descriptors + soil).

Inputs:
  data/processed/pfas_descriptors_full.csv  (11K PFAS, 225 RDKit descriptors)
  data/paper/feature_matrix_kd.csv           (1,227 samples, Kd + features)

Outputs:
  data/paper/kd_transfer_results.csv         (performance summary)
  data/paper/pretrained_encoder.pt           (saved encoder weights)
"""
import csv, os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────
PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PAPER = os.path.join(PROJECT, "data", "paper")
DATA_PROCESSED = os.path.join(PROJECT, "data", "processed")
DESC_11K_FILE = os.path.join(DATA_PROCESSED, "pfas_descriptors_full.csv")
FEAT_FILE = os.path.join(DATA_PAPER, "feature_matrix_kd.csv")
OUT_RESULTS = os.path.join(DATA_PAPER, "kd_transfer_results.csv")
ENCODER_PATH = os.path.join(DATA_PAPER, "pretrained_encoder.pt")

# ── Config ─────────────────────────────────────────
LATENT_DIM = 64           # Bottleneck dimension
HIDDEN_DIM = 128          # Hidden layer size
EPOCHS = 200              # Autoencoder training epochs
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NON_FEATURE = {"PFAS_name", "log_Kd", "Kd_L_kg", "log_Koc", "_n_soil_missing"}
SOIL_FEATURES = ["Corg_%", "foc", "pH", "Sand", "Silt", "Clay", "CEC", "Fe_g_kg", "Al_g_kg"]

# ═══════════════════════════════════════════════════
#  Autoencoder Definition
# ═══════════════════════════════════════════════════

class Autoencoder(nn.Module):
    """Fully-connected autoencoder for molecular descriptor compression."""
    def __init__(self, input_dim=225, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        """Return latent representation only (for downstream tasks)."""
        return self.encoder(x)


def train_autoencoder(X, device='cpu'):
    """Train autoencoder and return the trained model."""
    input_dim_orig = X.shape[1]

    # Remove columns with inf values or extreme variance
    valid_cols = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.any(np.isinf(col)):
            continue
        if np.std(col) > 1e10:
            continue
        valid_cols.append(j)
    X = X[:, valid_cols]
    print(f"  Cleaned columns: {X.shape[1]} (removed {input_dim_orig - X.shape[1]} with inf/extreme values)")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/val split
    X_train, X_val = train_test_split(X_scaled, test_size=0.1, random_state=RANDOM_SEED)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    input_dim = X.shape[1]  # Use cleaned dimension
    model = Autoencoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                val_loss += criterion(recon, batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'hidden_dim': HIDDEN_DIM,
                'latent_dim': LATENT_DIM,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
            }, ENCODER_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

    # Load best checkpoint
    checkpoint = torch.load(ENCODER_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Reconstruct scaler from checkpoint
    scaler.mean_ = np.array(checkpoint['scaler_mean'])
    scaler.scale_ = np.array(checkpoint['scaler_scale'])

    return model, scaler, valid_cols

# ═══════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════

def load_11k_descriptors():
    """Load 11K PFASMASTER descriptors, return numpy array + SMILES."""
    print("=" * 60)
    print("  Phase 1: Pretraining Autoencoder on 11K PFAS")
    print("=" * 60)
    if not os.path.exists(DESC_11K_FILE):
        print(f"  ❌ File not found: {DESC_11K_FILE}")
        print("  Run: python3 scripts/prepare_03_descriptors_11k.py first")
        sys.exit(1)

    with open(DESC_11K_FILE) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Extract descriptor columns (skip DTXSID, SMILES, RDKIT_SMILES)
    skip_cols = {"DTXSID", "SMILES", "RDKIT_SMILES"}
    desc_cols = [c for c in reader.fieldnames if c not in skip_cols]

    n = len(rows)
    p = len(desc_cols)
    X = np.zeros((n, p))
    for j, col in enumerate(desc_cols):
        for i, row in enumerate(rows):
            v = row.get(col, "").strip()
            if v:
                try:
                    X[i, j] = float(v)
                except (ValueError, TypeError):
                    X[i, j] = np.nan

    # Remove columns that are all NaN
    valid_cols = [j for j in range(p) if not np.all(np.isnan(X[:, j]))]
    X = X[:, valid_cols]
    desc_cols_clean = [desc_cols[j] for j in valid_cols]

    # Impute remaining NaNs with median
    for j in range(X.shape[1]):
        col_vals = X[:, j]
        mask = np.isnan(col_vals)
        if mask.any():
            X[mask, j] = np.nanmedian(col_vals)

    # Remove rows with any remaining NaN
    row_valid = ~np.any(np.isnan(X), axis=1)
    X = X[row_valid]

    print(f"  Loaded {len(rows):,} PFAS records")
    print(f"  Usable descriptors: {X.shape[1]} (from {p})")
    print(f"  Clean samples: {X.shape[0]:,}")
    return X, desc_cols_clean


def load_kd_data():
    """Load 47 PFAS Kd features."""
    print("\n" + "=" * 60)
    print("  Phase 2: Transfer to Kd Prediction (47 PFAS)")
    print("=" * 60)

    with open(FEAT_FILE) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = reader.fieldnames
    all_desc = [c for c in fieldnames if c not in NON_FEATURE]

    n = len(rows)
    p = len(all_desc)
    X = np.zeros((n, p))
    y = np.full(n, np.nan)

    for j, col in enumerate(all_desc):
        for i, row in enumerate(rows):
            v = row.get(col, "").strip()
            if v:
                try:
                    X[i, j] = float(v)
                except:
                    X[i, j] = np.nan

    for i, row in enumerate(rows):
        v = row.get("log_Kd", "").strip()
        if v:
            try:
                y[i] = float(v)
            except:
                pass

    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]

    for j in range(p):
        col_vals = X[:, j]
        mask = np.isnan(col_vals)
        if mask.any():
            X[mask, j] = np.nanmedian(col_vals)

    # Split descriptors vs soil features
    desc_indices = [j for j, col in enumerate(all_desc) if col not in SOIL_FEATURES]
    soil_indices = [j for j, col in enumerate(all_desc) if col in SOIL_FEATURES]

    X_desc = X[:, desc_indices]
    X_soil = X[:, soil_indices]
    desc_names = [all_desc[j] for j in desc_indices]

    print(f"  Loaded {len(y):,} Kd samples")
    print(f"  Molecular descriptors: {X_desc.shape[1]}")
    print(f"  Soil features: {X_soil.shape[1]}")
    return X_desc, X_soil, y, desc_names


# ═══════════════════════════════════════════════════
#  Transfer & Evaluation
# ═══════════════════════════════════════════════════

def evaluate_model(X, y, model_label):
    """Train XGBoost and evaluate on train/test split."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr)
    y_pred = xgb_model.predict(X_te)

    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    rpd = np.std(y_te) / rmse

    cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2', n_jobs=-1)

    print(f"  {model_label}:")
    print(f"    R² = {r2:.4f}, RMSE = {rmse:.4f}, RPD = {rpd:.2f}")
    print(f"    CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        "model": model_label,
        "n_features": X.shape[1],
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "rpd": round(rpd, 2),
        "cv_r2": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
    }


def main():
    os.makedirs(DATA_PAPER, exist_ok=True)
    start = time.time()

    # ── Phase 1: Pretrain autoencoder ──
    X_11k, desc_cols_11k = load_11k_descriptors()
    print(f"\n  Training autoencoder (CPU): {X_11k.shape[0]} samples × {X_11k.shape[1]} dims → {LATENT_DIM} latent")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print()

    device = 'cpu'
    encoder, scaler, valid_cols = train_autoencoder(X_11k, device=device)

    # Clean 11K data with same column filter (inf + extreme variance) used in training
    valid_mask = np.array(valid_cols)
    X_11k_clean = X_11k[:, valid_cols]

    # Compute reconstruction quality
    X_scaled = scaler.transform(X_11k_clean)
    with torch.no_grad():
        recon_tensor, _ = encoder(torch.tensor(X_scaled, dtype=torch.float32))
        recon = recon_tensor.numpy()
    recon_mse = np.mean((X_scaled - recon) ** 2)
    recon_rmse = np.sqrt(recon_mse)
    orig_var = np.var(X_scaled)
    recon_r2 = 1 - recon_mse / orig_var
    print(f"\n  ✅ Autoencoder trained ({time.time() - start:.0f}s)")
    print(f"  Reconstruction: MSE={recon_mse:.6f}, RMSE={recon_rmse:.6f}, R²={recon_r2:.4f}")
    print(f"  Encoder saved to: {ENCODER_PATH}")

    # ── Phase 2: Transfer to Kd ──
    X_desc, X_soil, y, desc_names = load_kd_data()

    # Align 47 PFAS descriptors with 11K autoencoder columns
    # The 11K desc columns must match the 47 PFAS desc columns
    # Build mapping: 11K col name -> index
    col_to_idx_11k = {c: i for i, c in enumerate(desc_cols_11k)}

    # For 47 PFAS, find matching columns
    aligned_indices = []
    aligned_names = []
    for j, name in enumerate(desc_names):
        if name in col_to_idx_11k:
            aligned_indices.append(col_to_idx_11k[name])
            aligned_names.append(name)

    print(f"\n  Aligned descriptors: {len(aligned_indices)}/{len(desc_names)}")

    # Subset 47 PFAS to aligned columns + reorder to match 11K order
    X_desc_aligned = np.zeros((X_desc.shape[0], len(aligned_indices)))
    for k, idx_47k in enumerate([desc_names.index(n) for n in aligned_names]):
        X_desc_aligned[:, k] = X_desc[:, idx_47k]

    # Apply the same column filter as the autoencoder
    aligned_filtered_indices = [i for i in aligned_indices if i in valid_cols]
    X_desc_aligned_filtered = np.zeros((X_desc.shape[0], len(aligned_filtered_indices)))
    for k, autoencoder_col_idx in enumerate(aligned_filtered_indices):
        desc_col_name = desc_cols_11k[autoencoder_col_idx]
        idx_47k = desc_names.index(desc_col_name)
        X_desc_aligned_filtered[:, k] = X_desc[:, idx_47k]

    print(f"  Aligned descriptors after filter: {len(aligned_filtered_indices)}/{len(aligned_indices)}")

    # Apply encoder
    with torch.no_grad():
        X_desc_aligned_filtered_scaled = scaler.transform(X_desc_aligned_filtered)
        latent_tensor = encoder.encode(torch.tensor(X_desc_aligned_filtered_scaled, dtype=torch.float32))
        X_latent = latent_tensor.numpy()

    print(f"  Latent representation: {X_latent.shape[1]} dimensions")

    # ── Phase 3: Model comparison ──
    print("\n" + "=" * 60)
    print("  Phase 3: Model Comparison")
    print("=" * 60)

    all_results = []

    # Note: Baseline R² values come from paper_03 output for consistency
    # (kd_model_results.csv gives: Combined=0.868, RDKit=0.647, Soil=0.245)

    # Baseline: Top 2 SHAP (MolWt, ExactMolWt) + soil
    molwt_cols = []
    for c in ["MolWt", "ExactMolWt"]:
        if c in desc_names:
            molwt_cols.append(desc_names.index(c))
    if len(molwt_cols) >= 2:
        X_top2_desc = X_desc[:, molwt_cols]
        X_top2 = np.hstack([X_top2_desc, X_soil])
        all_results.append(evaluate_model(X_top2, y, "Top 2 SHAP (MolWt+ExactMolWt) + soil"))

    # Transfer: Autoencoder latent + soil
    X_transfer = np.hstack([X_latent, X_soil])
    all_results.append(evaluate_model(X_transfer, y, f"Transfer (AE {LATENT_DIM}D latent + soil)"))

    # Transfer: PCA baseline for comparison
    from sklearn.decomposition import PCA
    pca = PCA(n_components=LATENT_DIM, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_desc_aligned_filtered)
    X_pca_full = np.hstack([X_pca, X_soil])
    all_results.append(evaluate_model(X_pca_full, y, f"PCA ({LATENT_DIM}D) + soil"))
    pca_var = pca.explained_variance_ratio_.sum()
    print(f"    PCA variance retained: {pca_var:.2%}")

    # ── Save results ──
    with open(OUT_RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "n_features", "r2", "rmse", "rpd", "cv_r2", "cv_std"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'=' * 60}")
    print(f"  Transfer Learning Results")
    print(f"{'=' * 60}")
    print(f"{'Model':<45} {'R²':<10} {'RPD':<8} {'n_feat':<8}")
    print("-" * 71)
    for r in all_results:
        print(f"{r['model']:<45} {r['r2']:<10.4f} {r['rpd']:<8.2f} {r['n_features']:<8}")

    print(f"\n  ✅ Saved to: {OUT_RESULTS}")
    print(f"  Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
