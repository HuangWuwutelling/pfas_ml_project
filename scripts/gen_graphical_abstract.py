#!/usr/bin/env python3
"""Generate Graphical Abstract for PFAS Kd prediction paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "paper")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
})

fig = plt.figure(figsize=(12, 4.5), facecolor='white')
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.3)

# ── Panel 1: Input Data ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('1) Data & Features', fontsize=12, fontweight='bold', pad=8)

box1 = FancyBboxPatch((0.5, 7), 9, 2.5, boxstyle="round,pad=0.3",
                       facecolor='#e3f2fd', edgecolor='#1565c0', linewidth=1.5)
ax1.add_patch(box1)
ax1.text(5, 8.25, '1,227 Kd measurements', ha='center', fontsize=10, fontweight='bold')
ax1.text(5, 7.5, '47 PFAS × 451 soils', ha='center', fontsize=9, color='#555')

box_mol = FancyBboxPatch((0.8, 3.5), 4, 3, boxstyle="round,pad=0.2",
                          facecolor='#fff3e0', edgecolor='#e65100', linewidth=1.2)
ax1.add_patch(box_mol)
ax1.text(2.8, 6.0, '225 RDKit', ha='center', fontsize=9, fontweight='bold')
ax1.text(2.8, 5.3, 'Molecular Descriptors', ha='center', fontsize=8, color='#555')
ax1.text(2.8, 4.6, 'MolWt, LogP, TPSA, ...', ha='center', fontsize=7, color='#777')

box_soil = FancyBboxPatch((5.2, 3.5), 4, 3, boxstyle="round,pad=0.2",
                           facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=1.2)
ax1.add_patch(box_soil)
ax1.text(7.2, 6.0, '9 Soil', ha='center', fontsize=9, fontweight='bold')
ax1.text(7.2, 5.3, 'Properties', ha='center', fontsize=8, color='#555')
ax1.text(7.2, 4.6, 'pH, Corg, CEC, ...', ha='center', fontsize=7, color='#777')

ax1.annotate('', xy=(5, 3.5), xytext=(5, 6.8),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

ax1.text(5, 2.0, 'log Kd: -1.40 to 3.95', ha='center', fontsize=8, style='italic', color='#555')

# ── Panel 2: Model ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('2) XGBoost Modeling', fontsize=12, fontweight='bold', pad=8)

box_model = FancyBboxPatch((1, 2.5), 8, 5, boxstyle="round,pad=0.3",
                           facecolor='#f3e5f5', edgecolor='#6a1b9a', linewidth=2)
ax2.add_patch(box_model)

models = [
    ('RDKit only', 'R2 = 0.65', '#fff3e0'),
    ('Soil only',  'R2 = 0.24', '#e8f5e9'),
    ('Combined',   'R2 = 0.87\nRPD = 2.75', '#fce4ec'),
]
for i, (name, val, color) in enumerate(models):
    y_pos = 6.5 - i * 1.6
    bx = FancyBboxPatch((2, y_pos - 0.5), 6, 1.2, boxstyle="round,pad=0.15",
                        facecolor=color, edgecolor='#999', linewidth=0.8)
    ax2.add_patch(bx)
    ax2.text(3.2, y_pos, name, ha='center', fontsize=8, fontweight='bold')
    ax2.text(6.8, y_pos, val, ha='center', fontsize=9,
             color='#c62828' if '0.87' in val else '#333', fontweight='bold')

ax2.text(5, 1.8, 'Key driver: MolWt (SHAP=0.38) > Corg (0.20)', ha='center',
         fontsize=8, style='italic', color='#555')

# ── Panel 3: Results & Application ──────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('3) Validation & Application', fontsize=12, fontweight='bold', pad=8)

box_loo = FancyBboxPatch((0.8, 5.5), 8.4, 2, boxstyle="round,pad=0.3",
                          facecolor='#e3f2fd', edgecolor='#1565c0', linewidth=1.5)
ax3.add_patch(box_loo)
ax3.text(5, 6.9, 'Leave-One-PFAS-Out CV', ha='center', fontsize=9, fontweight='bold')
ax3.text(5, 6.2, 'Pooled R2 = 0.719', ha='center', fontsize=10, color='#c62828', fontweight='bold')
ax3.text(5, 5.7, '24/47 compounds with R2 > 0', ha='center', fontsize=7.5, color='#555')

box_simple = FancyBboxPatch((0.8, 2.8), 8.4, 2, boxstyle="round,pad=0.3",
                             facecolor='#fff8e1', edgecolor='#f9a825', linewidth=1.5)
ax3.add_patch(box_simple)
ax3.text(5, 4.3, 'Simplified Model', ha='center', fontsize=9, fontweight='bold')
ax3.text(5, 3.7, 'MolWt + Corg + pH + CEC', ha='center', fontsize=8)
ax3.text(5, 3.0, 'R2 = 0.84 (96% of full model)', ha='center', fontsize=8, color='#c62828', fontweight='bold')

box_app = FancyBboxPatch((0.8, 0.2), 8.4, 2, boxstyle="round,pad=0.3",
                           facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=1.5)
ax3.add_patch(box_app)
ax3.text(5, 1.6, 'Chemical Space Expansion', ha='center', fontsize=9, fontweight='bold')
ax3.text(5, 1.0, '47 -> 11,000 PFAS: visual prioritization', ha='center', fontsize=8, color='#555')

# Arrows between panels
fig.text(0.313, 0.55, '->', ha='center', va='center', fontsize=24, color='#999', fontweight='bold')
fig.text(0.645, 0.55, '->', ha='center', va='center', fontsize=24, color='#999', fontweight='bold')

# Bottom title
fig.text(0.5, 0.02,
         'Predicting PFAS Soil-Water Partitioning from Molecular Structure:\n'
         '  RDKit Descriptors + XGBoost + SHAP + Chemical Space Expansion',
         ha='center', va='bottom', fontsize=9, style='italic', color='#333')

# Color bar for Kd
cbar_ax = fig.add_subplot(gs[0, 3])
cbar_ax.axis('off')
cbar_ax.text(0.5, 0.9, 'Kd', ha='center', fontsize=8, fontweight='bold')
grad = np.linspace(0, 1, 100).reshape(-1, 1)
cbar_ax.imshow(grad, aspect='auto', cmap='viridis_r', extent=[0, 0.3, 0, 0.8])
cbar_ax.text(0.4, 0.8, 'high', ha='right', fontsize=6, color='#333')
cbar_ax.text(0.4, 0.0, 'low', ha='right', fontsize=6, color='#333')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

path = os.path.join(FIG_DIR, "graphical_abstract.png")
plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"SAVED: {path}")
print(f"DIMENSIONS: {12*300} x {4.5*300} px at 300 dpi")
