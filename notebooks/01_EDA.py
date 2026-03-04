"""
01_EDA.py
=========
Exploratory Data Analysis for the Rhea Soil Nutrient Prediction Challenge.

Run: python notebooks/01_EDA.py
Outputs saved to: notebooks/eda_outputs/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "notebooks" / "eda_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
train    = pd.read_csv(DATA_DIR / "Train.csv")
test     = pd.read_csv(DATA_DIR / "TestSet.csv")
mask     = pd.read_csv(DATA_DIR / "TargetPred_To_Keep.csv")
dates    = pd.read_csv(DATA_DIR / "Sample_Collection_Dates.csv")

NUTRIENTS = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']

# ── 1. Basic shape & dtypes ───────────────────────────────────────────────────
print("\n── Dataset shapes ──────────────────────────────────────────")
print(f"  Train  : {train.shape}")
print(f"  Test   : {test.shape}")
print(f"  Mask   : {mask.shape}")

print("\n── Train dtypes ────────────────────────────────────────────")
print(train.dtypes)

print("\n── First 3 rows ────────────────────────────────────────────")
print(train.head(3).to_string())

# ── 2. Missing values ─────────────────────────────────────────────────────────
print("\n── Missing values in Train (%) ─────────────────────────────")
missing = (train.isnull().mean() * 100).sort_values(ascending=False)
print(missing[missing > 0].to_string())

fig, ax = plt.subplots(figsize=(10, 5))
missing[missing > 0].plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title("Missing Values in Training Set (%)", fontsize=13)
ax.set_ylabel("Missing (%)")
ax.set_xlabel("")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUT_DIR / "missing_values.png", dpi=150)
plt.close()
print(f"  → saved missing_values.png")

# ── 3. Nutrient distributions (raw + log) ────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

available_nutrients = [n for n in NUTRIENTS if n in train.columns]
for i, nut in enumerate(available_nutrients):
    vals = train[nut].dropna()
    if vals.empty:
        continue
    ax = axes[i]
    # raw
    ax.hist(vals, bins=60, color='steelblue', edgecolor='none', alpha=0.6, label='raw')
    # log1p overlay on twin axis
    ax2 = ax.twinx()
    log_vals = np.log1p(vals.clip(lower=0))
    ax2.hist(log_vals, bins=60, color='tomato', edgecolor='none', alpha=0.4, label='log1p')
    ax.set_title(f"{nut}  (n={len(vals):,})", fontsize=9)
    ax.set_xlabel("mg/kg", fontsize=7)
    ax.tick_params(axis='both', labelsize=7)
    ax2.tick_params(axis='y', labelsize=7, colors='tomato')
    # legend
    lines = [plt.Line2D([0],[0],color='steelblue',lw=2,label='raw'),
             plt.Line2D([0],[0],color='tomato',  lw=2,label='log1p')]
    ax.legend(handles=lines, fontsize=6, loc='upper right')

# hide unused subplots
for j in range(len(available_nutrients), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Nutrient Distributions — Raw (blue) vs log1p (red)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "nutrient_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → saved nutrient_distributions.png")

# ── 4. Summary statistics for nutrients ──────────────────────────────────────
stats = train[available_nutrients].describe(percentiles=[.1,.25,.5,.75,.9]).T
stats['skew']  = train[available_nutrients].skew()
stats['CV%']   = (train[available_nutrients].std() / train[available_nutrients].mean() * 100).round(1)
stats['missing%'] = (train[available_nutrients].isnull().mean() * 100).round(1)
print("\n── Nutrient summary statistics ─────────────────────────────")
print(stats[['count','mean','std','min','50%','max','skew','CV%','missing%']].to_string())
stats.to_csv(OUT_DIR / "nutrient_stats.csv")
print("  → saved nutrient_stats.csv")

# ── 5. Correlation heatmap ────────────────────────────────────────────────────
corr = train[available_nutrients].corr()
mask_upper = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask_upper, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
ax.set_title("Inter-nutrient Pearson Correlation (Training Set)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "nutrient_correlation.png", dpi=150)
plt.close()
print("  → saved nutrient_correlation.png")

# ── 6. Spatial coverage ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, df, title, color in zip(
    axes, [train, test],
    ["Training Set (44,298)", "Test Set (6,070)"],
    ["steelblue", "tomato"]
):
    sc = ax.scatter(df['Longitude'], df['Latitude'],
                    s=1.5, alpha=0.4, c=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.3, alpha=0.5)

plt.suptitle("Spatial Coverage of Train and Test Sets", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "spatial_coverage.png", dpi=150)
plt.close()
print("  → saved spatial_coverage.png")

# colour by a nutrient
if 'Fe' in train.columns:
    fig, ax = plt.subplots(figsize=(10, 7))
    sub = train[['Longitude', 'Latitude', 'Fe']].dropna()
    sc = ax.scatter(sub['Longitude'], sub['Latitude'],
                    c=np.log1p(sub['Fe']), s=2, cmap='YlOrRd', alpha=0.5)
    plt.colorbar(sc, ax=ax, label='log1p(Fe) mg/kg')
    ax.set_title("Spatial Distribution of log1p(Fe) — Training Set", fontsize=12)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "spatial_Fe.png", dpi=150)
    plt.close()
    print("  → saved spatial_Fe.png")

# ── 7. Depth distribution ─────────────────────────────────────────────────────
print("\n── Depth distribution in Train ─────────────────────────────")
print(train['Depth_cm'].value_counts())

fig, ax = plt.subplots(figsize=(7, 4))
train['Depth_cm'].value_counts().plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title("Depth Interval Distribution — Training Set")
ax.set_xlabel("Depth_cm"); ax.set_ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUT_DIR / "depth_distribution.png", dpi=150)
plt.close()
print("  → saved depth_distribution.png")

# ── 8. Nutrient box-plots by depth ───────────────────────────────────────────
nutrients_to_plot = ['Al', 'Ca', 'Fe', 'Mg', 'K', 'N']
nutrients_to_plot = [n for n in nutrients_to_plot if n in train.columns]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, nut in enumerate(nutrients_to_plot):
    ax = axes[i]
    data = train[['Depth_cm', nut]].dropna()
    data[nut] = np.log1p(data[nut].clip(lower=0))
    order = sorted(data['Depth_cm'].unique())
    sns.boxplot(data=data, x='Depth_cm', y=nut, order=order, ax=ax,
                palette='Blues', flierprops=dict(marker='.', markersize=2))
    ax.set_title(f"log1p({nut}) by Depth", fontsize=10)
    ax.set_xlabel("Depth_cm"); ax.set_ylabel(f"log1p({nut})")
    ax.tick_params(axis='x', rotation=30)

for j in range(len(nutrients_to_plot), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Nutrient Values by Soil Depth (log scale)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "nutrients_by_depth.png", dpi=150)
plt.close()
print("  → saved nutrients_by_depth.png")

# ── 9. Mask analysis ──────────────────────────────────────────────────────────
mask_cols = ['Al','B','Ca','Cu','Fe','K','Mg','Mn','N','Na','P','S','Zn']
mask_sum  = mask[mask_cols].sum()

print("\n── TargetPred_To_Keep: # of test rows to PREDICT (1) ───────")
print(mask_sum.to_string())
print(f"\n  Total test entries  : {len(mask) * 13:,}")
print(f"  Entries to predict  : {mask_sum.sum():,}  ({mask_sum.sum()/(len(mask)*13)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(9, 4))
mask_sum.plot(kind='bar', ax=ax, color='seagreen', edgecolor='white')
ax.set_title("Number of Test Rows to Predict per Nutrient\n(from TargetPred_To_Keep)", fontsize=11)
ax.set_ylabel("Count (predict=1)")
ax.set_xlabel("")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(OUT_DIR / "mask_analysis.png", dpi=150)
plt.close()
print("  → saved mask_analysis.png")

# ── 10. pH and electrical_conductivity distributions ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ['ph', 'electrical_conductivity']):
    if col in train.columns:
        vals = train[col].dropna()
        ax.hist(vals, bins=60, color='mediumpurple', edgecolor='none', alpha=0.8)
        ax.axvline(vals.median(), color='red', lw=1.5, linestyle='--', label=f'median={vals.median():.2f}')
        ax.set_title(f"Distribution of {col}", fontsize=11)
        ax.set_xlabel(col); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "ph_ec_distributions.png", dpi=150)
plt.close()
print("  → saved ph_ec_distributions.png")

# ── 11. C_organic vs key nutrients scatter ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
pairs = [('C_organic','N'),('C_organic','Al'),('ph','Ca'),
         ('ph','Al'),('ph','Fe'),('electrical_conductivity','Na')]

for i, (xc, yc) in enumerate(pairs):
    if xc in train.columns and yc in train.columns:
        sub = train[[xc, yc]].dropna().sample(min(3000, len(train)), random_state=42)
        x = np.log1p(sub[xc].clip(lower=0)) if sub[xc].min() >= 0 else sub[xc]
        y = np.log1p(sub[yc].clip(lower=0)) if sub[yc].min() >= 0 else sub[yc]
        axes[i].scatter(x, y, s=2, alpha=0.3, color='steelblue')
        axes[i].set_xlabel(f"log1p({xc})" if sub[xc].min() >= 0 else xc, fontsize=8)
        axes[i].set_ylabel(f"log1p({yc})" if sub[yc].min() >= 0 else yc, fontsize=8)
        corr_val = np.corrcoef(x, y)[0,1]
        axes[i].set_title(f"{xc} vs {yc}  (r={corr_val:.2f})", fontsize=9)

plt.suptitle("Feature–Nutrient Scatter (log scale, 3k sample)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_pairs.png", dpi=150)
plt.close()
print("  → saved scatter_pairs.png")

print(f"\n✓ EDA complete. All outputs saved to: {OUT_DIR}")
