"""
02_feature_engineering.py — Build features for Train + Test
Rhea Soil Nutrient Prediction Challenge

This script:
  1. Loads raw Train.csv / TestSet.csv / Sample_Collection_Dates.csv
  2. Merges collection dates onto test set
  3. Engineers all tabular features (depth, temporal, spatial, interactions)
  4. Optionally merges pre-downloaded external features (SoilGrids, WorldClim)
  5. Saves  data/processed/train_features.csv
           data/processed/test_features.csv

Run: python 02_feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────
RAW  = Path("data/raw")
EXT  = Path("data/external")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading raw data …")
train = pd.read_csv(RAW / "Train.csv")
test  = pd.read_csv(RAW / "TestSet.csv")
dates = pd.read_csv(RAW / "Sample_Collection_Dates.csv")

TARGETS = ['Al','B','Ca','Cu','Fe','K','Mg','Mn','N','Na','P','S','Zn']

# ── Attach dates to test set ─────────────────────────────────────────────────
test_dates = dates[dates['set'] == 'Test'].copy()
test = test.merge(test_dates[['ID','start_date','end_date']], on='ID', how='left')

# ── Shared feature engineering function ──────────────────────────────────────
def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()

    # ── 1. Depth encoding ────────────────────────────────────────────────────
    # Numeric midpoint
    def depth_midpoint(d):
        try:
            parts = str(d).replace(' ', '').split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        except Exception:
            return np.nan

    df['depth_mid'] = df['Depth_cm'].apply(depth_midpoint)
    df['depth_upper'] = df['Depth_cm'].apply(lambda d: float(str(d).replace(' ','').split('-')[0]) if '-' in str(d) else np.nan)
    df['depth_lower'] = df['Depth_cm'].apply(lambda d: float(str(d).replace(' ','').split('-')[1]) if '-' in str(d) else np.nan)

    # If horizon_upper / horizon_lower present (train), use them; otherwise derive from Depth_cm
    if 'horizon_upper' in df.columns:
        df['horizon_upper'] = pd.to_numeric(df['horizon_upper'], errors='coerce')
        df['horizon_lower'] = pd.to_numeric(df['horizon_lower'], errors='coerce')
        df['horizon_mid']   = (df['horizon_upper'] + df['horizon_lower']) / 2
        df['horizon_thick'] = df['horizon_lower'] - df['horizon_upper']
    else:
        df['horizon_upper'] = df['depth_upper']
        df['horizon_lower'] = df['depth_lower']
        df['horizon_mid']   = df['depth_mid']
        df['horizon_thick'] = df['depth_lower'] - df['depth_upper']

    # Depth category (0-20, 20-50, other)
    df['is_shallow'] = (df['depth_mid'] <= 20).astype(int)
    df['is_deep']    = (df['depth_mid'] > 20).astype(int)

    # ── 2. Temporal features ─────────────────────────────────────────────────
    for col in ['start_date','end_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    if 'start_date' in df.columns and df['start_date'].notna().any():
        df['year_start']  = df['start_date'].dt.year.fillna(2013)
        df['month_start'] = df['start_date'].dt.month.fillna(6)
    else:
        df['year_start']  = 2013
        df['month_start'] = 6

    if 'end_date' in df.columns and df['end_date'].notna().any():
        df['year_end']  = df['end_date'].dt.year.fillna(2018)
        df['month_end'] = df['end_date'].dt.month.fillna(12)
    else:
        df['year_end']  = 2018
        df['month_end'] = 12

    df['date_span_years'] = df['year_end'] - df['year_start']
    df['year_mid']        = (df['year_start'] + df['year_end']) / 2

    # Wet / dry season (rough for East/South Africa: long rains ~Mar-May)
    df['is_wet_season'] = df['month_start'].apply(lambda m: 1 if m in [3,4,5,10,11] else 0)

    # ── 3. Spatial features ───────────────────────────────────────────────────
    # Hemisphere & zone dummies
    df['lat_abs']   = df['Latitude'].abs()
    df['is_south']  = (df['Latitude'] < 0).astype(int)
    df['lon_sin']   = np.sin(np.radians(df['Longitude']))
    df['lon_cos']   = np.cos(np.radians(df['Longitude']))
    df['lat_sin']   = np.sin(np.radians(df['Latitude']))
    df['lat_cos']   = np.cos(np.radians(df['Latitude']))

    # Spatial blocks (0.5° grid)
    df['lat_block'] = (df['Latitude']  // 0.5).astype(int)
    df['lon_block'] = (df['Longitude'] // 0.5).astype(int)
    df['spatial_block'] = df['lat_block'].astype(str) + '_' + df['lon_block'].astype(str)

    # ── 4. Chemistry features (train-only columns) ───────────────────────────
    for col in ['ph','electrical_conductivity','C_organic','C_total']:
        if col not in df.columns:
            df[col] = np.nan

    df['ph'] = pd.to_numeric(df['ph'], errors='coerce')
    df['electrical_conductivity'] = pd.to_numeric(df['electrical_conductivity'], errors='coerce')
    df['C_organic'] = pd.to_numeric(df['C_organic'], errors='coerce')
    df['C_total']   = pd.to_numeric(df['C_total'], errors='coerce')

    df['ph_sq']              = df['ph'] ** 2
    df['log_C_organic']      = np.log1p(df['C_organic'].clip(lower=0))
    df['log_C_total']        = np.log1p(df['C_total'].clip(lower=0))
    df['C_org_x_ph']         = df['C_organic'] * df['ph']
    df['C_total_C_org_ratio'] = df['C_total'] / (df['C_organic'] + 1e-6)

    # ── 5. Agronomic ratio features (train targets as cross-features) ─────────
    # These are populated only at training time (or via OOF predictions at test)
    for t in TARGETS:
        if t in df.columns:
            df[f'log_{t}'] = np.log1p(df[t].clip(lower=0))

    # Classic agronomic ratios
    if all(c in df.columns for c in ['Ca','Mg']):
        df['Ca_Mg_ratio'] = df['Ca'] / (df['Mg'] + 1e-6)
    if all(c in df.columns for c in ['Fe','Mn']):
        df['Fe_Mn_ratio'] = df['Fe'] / (df['Mn'] + 1e-6)
    if all(c in df.columns for c in ['K','Na']):
        df['K_Na_ratio']  = df['K'] / (df['Na'] + 1e-6)

    return df


print("Engineering train features …")
train_fe = engineer_features(train, is_train=True)

print("Engineering test features …")
test_fe  = engineer_features(test,  is_train=False)

# ── K-Means spatial clusters (fit on train, apply to both) ───────────────────
print("Fitting spatial KMeans clusters …")
N_CLUSTERS = 50
coords_train = train_fe[['Latitude','Longitude']].values
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
train_fe['spatial_cluster'] = km.fit_predict(coords_train)
test_fe['spatial_cluster']  = km.predict(test_fe[['Latitude','Longitude']].values)

# ── Merge external features if available ─────────────────────────────────────
def try_merge_external(df, path, on='ID'):
    if Path(path).exists():
        ext_df = pd.read_csv(path)
        print(f"  Merging external: {path} ({len(ext_df.columns)-1} features)")
        return df.merge(ext_df, on=on, how='left')
    else:
        print(f"  [SKIP] External file not found: {path}")
        return df

print("Checking for external feature files …")
train_fe = try_merge_external(train_fe, EXT / "soilgrids_train.csv")
test_fe  = try_merge_external(test_fe,  EXT / "soilgrids_test.csv")
train_fe = try_merge_external(train_fe, EXT / "worldclim_train.csv")
test_fe  = try_merge_external(test_fe,  EXT / "worldclim_test.csv")
train_fe = try_merge_external(train_fe, EXT / "sentinel2_train.csv")
test_fe  = try_merge_external(test_fe,  EXT / "sentinel2_test.csv")

# ── Save ─────────────────────────────────────────────────────────────────────
train_fe.to_csv(PROC / "train_features.csv", index=False)
test_fe.to_csv( PROC / "test_features.csv",  index=False)
print(f"\n✅ Saved:")
print(f"   train_features.csv  shape={train_fe.shape}")
print(f"   test_features.csv   shape={test_fe.shape}")

# ── Feature column summary ────────────────────────────────────────────────────
# Decide which columns are usable as features for modeling
chemistry = ['ph','electrical_conductivity','C_organic','C_total',
             'ph_sq','log_C_organic','log_C_total','C_org_x_ph','C_total_C_org_ratio']
depth_feats = ['depth_mid','depth_upper','depth_lower',
               'horizon_upper','horizon_lower','horizon_mid','horizon_thick',
               'is_shallow','is_deep']
temporal = ['year_start','year_end','month_start','month_end',
            'date_span_years','year_mid','is_wet_season']
spatial  = ['Latitude','Longitude','lat_abs','is_south',
            'lon_sin','lon_cos','lat_sin','lat_cos',
            'lat_block','lon_block','spatial_cluster']

FEATURE_COLS = chemistry + depth_feats + temporal + spatial

# Add any external feature columns that got merged in
extra_ext_cols = [c for c in train_fe.columns
                  if c not in FEATURE_COLS + TARGETS + ['ID','Depth_cm','start_date','end_date',
                                                         'C_organic','C_total','spatial_block',
                                                         'lat_block','lon_block']
                  and not c.startswith('Target_')]
FEATURE_COLS += [c for c in extra_ext_cols if train_fe[c].dtype in [np.float64, np.int64, float, int]]

# Save feature list for use in training script
import json
with open(PROC / "feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)
print(f"\n   feature_cols.json  ({len(FEATURE_COLS)} features)")
