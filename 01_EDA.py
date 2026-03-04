"""
01_EDA.py — Exploratory Data Analysis
Rhea Soil Nutrient Prediction Challenge
Run: python 01_EDA.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data/raw")
PLOTS_DIR = Path("outputs/eda_plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data …")
train   = pd.read_csv(DATA_DIR / "Train.csv")
test    = pd.read_csv(DATA_DIR / "TestSet.csv")
mask    = pd.read_csv(DATA_DIR / "TargetPred_To_Keep.csv")
dates   = pd.read_csv(DATA_DIR / "Sample_Collection_Dates.csv")

TARGETS = ['Al','B','Ca','Cu','Fe','K','Mg','Mn','N','Na','P','S','Zn']

# ── 1. Basic info ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAIN SHAPE:", train.shape)
print("TEST  SHAPE:", test.shape)
print("\nTrain columns:", list(train.columns))
print("\nTarget missing % in train:")
for t in TARGETS:
    if t in train.columns:
        miss = train[t].isna().mean() * 100
        print(f"  {t:4s}: {miss:5.1f}% missing  |  "
              f"mean={train[t].mean():.2f}  median={train[t].median():.2f}  "
              f"max={train[t].max():.2f}")

# ── 2. Depth distribution ────────────────────────────────────────────────────
print("\nDepth distribution (train):")
print(train['Depth_cm'].value_counts())
print("\nDepth distribution (test):")
print(test['Depth_cm'].value_counts())

# ── 3. Mask analysis ─────────────────────────────────────────────────────────
mask_vals = mask[TARGETS].values
total     = mask_vals.size
nonzero   = mask_vals.sum()
print(f"\nTargetPred_To_Keep: {nonzero}/{total} entries to predict "
      f"({100*nonzero/total:.1f}%)")
print("Per-nutrient keep %:")
for c in TARGETS:
    pct = mask[c].mean() * 100
    print(f"  {c:4s}: {pct:5.1f}%")

# ── 4. Spatial coverage ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(train['Longitude'], train['Latitude'], s=1, alpha=0.3, c='steelblue')
axes[0].set_title(f'Train locations (n={len(train):,})', fontsize=12)
axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude')

axes[1].scatter(test['Longitude'], test['Latitude'], s=5, alpha=0.6, c='tomato')
axes[1].set_title(f'Test locations (n={len(test):,})', fontsize=12)
axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_spatial_coverage.png", dpi=120)
plt.close()
print("\nSaved: 01_spatial_coverage.png")

# ── 5. Target distributions (log scale) ─────────────────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()
for i, t in enumerate(TARGETS):
    if t not in train.columns:
        axes[i].set_visible(False); continue
    vals = train[t].dropna()
    vals_log = np.log1p(vals[vals > 0])
    axes[i].hist(vals_log, bins=60, color='steelblue', edgecolor='none', alpha=0.8)
    axes[i].set_title(f'{t}  (n={len(vals):,})', fontsize=10)
    axes[i].set_xlabel('log1p(value)')
    axes[i].set_ylabel('Count')
    skew = vals.skew()
    axes[i].text(0.97, 0.95, f'skew={skew:.1f}', transform=axes[i].transAxes,
                 ha='right', va='top', fontsize=8, color='darkred')

# hide unused subplots
for j in range(len(TARGETS), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Target nutrient distributions (log1p scale)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_target_distributions.png", dpi=120, bbox_inches='tight')
plt.close()
print("Saved: 02_target_distributions.png")

# ── 6. Correlation heatmap ───────────────────────────────────────────────────
avail = [t for t in TARGETS if t in train.columns]
corr  = train[avail].corr(method='spearman')

fig, ax = plt.subplots(figsize=(10, 8))
mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask_upper, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={'size':8})
ax.set_title('Spearman correlation between target nutrients (lower triangle)', fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_target_correlation.png", dpi=120)
plt.close()
print("Saved: 03_target_correlation.png")

# ── 7. Depth vs nutrient medians ─────────────────────────────────────────────
depth_stats = []
for depth in train['Depth_cm'].unique():
    row = {'Depth_cm': depth}
    sub = train[train['Depth_cm'] == depth]
    for t in avail:
        row[t] = sub[t].median()
    depth_stats.append(row)
depth_df = pd.DataFrame(depth_stats).set_index('Depth_cm')
print("\nMedian nutrient by depth:")
print(depth_df.round(1))

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
axes = axes.flatten()
for i, t in enumerate(avail):
    depths = train.groupby('Depth_cm')[t].median().sort_index()
    axes[i].bar(depths.index, depths.values, color='teal', alpha=0.8)
    axes[i].set_title(t, fontsize=10)
    axes[i].set_xlabel('Depth'); axes[i].tick_params(axis='x', rotation=30)
for j in range(len(avail), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Median nutrient value by soil depth', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_depth_vs_nutrients.png", dpi=120, bbox_inches='tight')
plt.close()
print("Saved: 04_depth_vs_nutrients.png")

# ── 8. pH vs nutrients scatter ───────────────────────────────────────────────
ph_targets = ['Al','Ca','Fe','Mg','Mn','K']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, t in enumerate(ph_targets):
    if t not in train.columns: continue
    sub = train[['ph', t]].dropna().sample(min(5000, len(train)))
    axes[i].scatter(sub['ph'], np.log1p(sub[t]), s=2, alpha=0.3, c='purple')
    # add trend line
    z = np.polyfit(sub['ph'], np.log1p(sub[t]), 1)
    p = np.poly1d(z)
    xs = np.linspace(sub['ph'].min(), sub['ph'].max(), 100)
    axes[i].plot(xs, p(xs), 'r-', linewidth=1.5)
    corr_val = sub['ph'].corr(np.log1p(sub[t]))
    axes[i].set_title(f'pH vs log1p({t})  r={corr_val:.2f}', fontsize=10)
    axes[i].set_xlabel('pH'); axes[i].set_ylabel(f'log1p({t})')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_ph_vs_nutrients.png", dpi=120)
plt.close()
print("Saved: 05_ph_vs_nutrients.png")

# ── 9. Geographic nutrient maps ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(22, 12))
axes = axes.flatten()
for i, t in enumerate(avail):
    sub = train[['Longitude','Latitude', t]].dropna()
    sc = axes[i].scatter(sub['Longitude'], sub['Latitude'],
                         c=np.log1p(sub[t]), s=1, cmap='YlOrRd', alpha=0.6)
    plt.colorbar(sc, ax=axes[i], pad=0.01)
    axes[i].set_title(f'{t} (log1p)', fontsize=9)
    axes[i].tick_params(labelsize=7)
for j in range(len(avail), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Spatial distribution of soil nutrients (log1p)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_spatial_nutrient_maps.png", dpi=120, bbox_inches='tight')
plt.close()
print("Saved: 06_spatial_nutrient_maps.png")

# ── 10. C_organic / C_total vs N ─────────────────────────────────────────────
if 'C_organic' in train.columns and 'N' in train.columns:
    sub = train[['C_organic','C_total','N']].dropna(subset=['N'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(np.log1p(sub['C_organic']), np.log1p(sub['N']),
                    s=2, alpha=0.3, c='green')
    r = sub['C_organic'].corr(sub['N'])
    axes[0].set_title(f'C_organic vs N  (r={r:.2f})', fontsize=11)
    axes[0].set_xlabel('log1p(C_organic)'); axes[0].set_ylabel('log1p(N)')

    axes[1].scatter(np.log1p(sub['C_total']), np.log1p(sub['N']),
                    s=2, alpha=0.3, c='olive')
    r2 = sub['C_total'].corr(sub['N'])
    axes[1].set_title(f'C_total vs N  (r={r2:.2f})', fontsize=11)
    axes[1].set_xlabel('log1p(C_total)'); axes[1].set_ylabel('log1p(N)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_carbon_vs_N.png", dpi=120)
    plt.close()
    print("Saved: 07_carbon_vs_N.png")

print("\n✅ EDA complete. All plots saved to:", PLOTS_DIR)
