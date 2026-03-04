# Rhea Soil Nutrient Prediction Challenge — Project Design Blueprint

## 1. Problem Summary

**Task:** Multi-target regression — predict 13 soil nutrients (Al, B, Ca, Cu, Fe, K, Mg, Mn, N, Na, P, S, Zn) for 6,070 test locations across Africa where no lab tests are available.

**Metric:** RMSE (lower = better). Each predicted column is scored independently; the mean RMSE across all targets counts.

**Key constraint:** `TargetPred_To_Keep.csv` marks which nutrient/location pairs are valid (1) vs. forced zero (0). Only ~22% of test rows have all 13 nutrients present; the rest have partial targets. **Before any submission, zero out all flagged entries.**

---

## 2. Dataset Overview

| File | Rows | Notes |
|---|---|---|
| `Train.csv` | 44,298 | Soil samples with lat/lon, date range, depth, 13 target nutrients + extra features |
| `TestSet.csv` | 6,070 | Locations to predict (no nutrient values) |
| `TargetPred_To_Keep.csv` | 6,070 | Binary mask: 1 = predict, 0 = force to zero |
| `Sample_Collection_Dates.csv` | — | Date ranges per sample (train + test) |

### Training Features Available
- **Location:** `Latitude`, `Longitude`
- **Temporal:** `start_date`, `end_date` (date range 2008–2018 typical)
- **Depth:** `Depth_cm` (e.g. "0-20", "20-50"), `horizon_upper`, `horizon_lower`
- **Soil chemistry:** `pH`, `electrical_conductivity`, `C_organic`, `C_total`
- **Targets (also usable as cross-features):** Al, B, Ca, Cu, Fe, K, Mg, Mn, N, Na, P, S, Zn

---

## 3. Recommended Earth Observation (EO) Data Sources

Acquire and merge these by (lat, lon) and approximate date range for both train and test sets.

### 3.1 Sentinel-2 (ESA) — Spectral Indices
- **What to get:** Cloud-free composites (median) over the sample date range
- **Key bands:** B2–B8A, B11, B12
- **Derived indices:** NDVI, EVI, SAVI, NDWI, BSI (Bare Soil Index), Clay Ratio (B11/B8A), Iron Oxide Ratio (B4/B2)
- **Resolution:** 10–20m
- **Access:** Google Earth Engine (GEE) `ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")`
- **Depth of history:** 2017–present (supplement with Landsat for older samples)

```python
# GEE snippet — export S2 median composite
import ee
ee.Initialize()

def get_s2_composite(lon, lat, start, end, buffer=500):
    pt = ee.Geometry.Point([lon, lat])
    roi = pt.buffer(buffer)
    img = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(start, end)
           .filterBounds(roi)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
           .median()
           .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']))
    return img.reduceRegion(ee.Reducer.mean(), roi, 10).getInfo()
```

### 3.2 Landsat 7/8/9 (NASA)
- **Use for:** Samples dated pre-2017 (before Sentinel-2 coverage)
- **Key products:** Surface reflectance composites
- **Access:** GEE `ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")`
- **Derived:** Same indices as above; TIRS bands for land surface temperature

### 3.3 CHIRPS Climate Data
- **What:** Daily precipitation → aggregate to monthly/annual mean, seasonal totals, dry season length
- **Access:** GEE `ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")` or direct download from https://www.chc.ucsb.edu/data/chirps
- **Features to derive:** Annual rainfall, coefficient of variation, number of dry months

### 3.4 WorldClim v2
- **What:** 1km climate normals (temperature, precipitation, solar radiation, wind)
- **Access:** https://worldclim.org/data/worldclim21.html — static rasters, sample by lat/lon
- **Key variables:** `bio1` (mean annual temp), `bio12` (annual precip), `bio15` (precip seasonality), `bio4` (temp seasonality)

### 3.5 SoilGrids 2.0 (ISRIC)
- **What:** Global gridded soil property predictions at 250m — pH, clay%, silt%, sand%, SOC, CEC, bulk density
- **Access:** REST API at `https://rest.isric.org/soilgrids/v2.0/properties/query?lon=X&lat=Y`
- **Why it helps:** Strong prior for many nutrients, especially Ca, Mg, K, Al (highly correlated with clay and pH)

```python
import requests
def get_soilgrids(lat, lon, depths=["0-5cm","5-15cm","15-30cm"]):
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    props = ["phh2o","clay","sand","silt","soc","cec","bdod","nitrogen"]
    r = requests.get(url, params={"lon": lon, "lat": lat, "property": props, "depth": depths})
    return r.json()
```

### 3.6 OpenLandMap / MODIS
- **MODIS land cover:** `MCD12Q1` — classify each location (cropland, forest, grassland, etc.)
- **MODIS NDVI 16-day:** Long time-series NDVI features
- **OpenLandMap:** Pre-extracted soil and climate features available at https://openlandmap.org

---

## 4. Feature Engineering Pipeline

```
Raw inputs
    ├── Core tabular (lat/lon/depth/pH/EC/C_org)
    ├── EO spectral indices (S2 / Landsat)
    ├── SoilGrids properties
    ├── WorldClim bio variables
    ├── CHIRPS rainfall aggregates
    └── MODIS land cover + NDVI stats
            │
            ▼
    Feature Engineering
    ├── Depth encoding: one-hot OR numeric midpoint (e.g. "0-20" → 10)
    ├── Temporal: year of collection, season (wet/dry), decade
    ├── Spatial clustering: k-means on lat/lon → cluster_id (captures regional soil types)
    ├── Target cross-features: in training, known nutrients as predictors for others
    │     (use out-of-fold predictions at test time)
    ├── Nutrient ratios: Ca/Mg, Fe/Mn, K/Na (known agronomic indices)
    ├── Distance to features: distance to nearest river, elevation from SRTM/DEM
    └── Interaction terms: pH×C_organic, clay%×CEC
```

---

## 5. Modelling Strategy

### 5.1 Baseline (fast to run)
Train one **LightGBM** or **XGBoost** model per target nutrient using the same feature set.

```python
import lightgbm as lgb
from sklearn.model_selection import KFold

targets = ['Al','B','Ca','Cu','Fe','K','Mg','Mn','N','Na','P','S','Zn']

for target in targets:
    train_data = df_train[df_train[target].notna()]
    X = train_data[feature_cols]
    y = train_data[target]
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05,
                               num_leaves=127, subsample=0.8, colsample_bytree=0.8)
    # Cross-validate, then fit on full data
    model.fit(X, y)
    preds[target] = model.predict(X_test)
```

### 5.2 Multi-output / stacking approach
- **Stage 1:** Predict each nutrient independently with LightGBM
- **Stage 2:** Use Stage 1 OOF predictions as additional features in a second-level model
- This captures inter-nutrient correlations (e.g., Ca and Mg are strongly correlated)

### 5.3 Advanced options
| Method | Notes |
|---|---|
| **CatBoost** | Handles categorical depth/landcover natively |
| **Random Forest** | More stable for nutrients with sparse training data |
| **Neural network (MLP)** | Can share a backbone across all 13 targets simultaneously |
| **Gaussian Process (per target)** | Excellent for spatial interpolation tasks but slow at scale |
| **Spatial kriging** | Pure geostatistical approach; use as a feature or blending target |

### 5.4 Spatial cross-validation
**Critical:** Standard random K-fold will overfit spatially correlated data. Use:
- `sklearn`-compatible `GroupKFold` with spatial blocks (e.g., 0.5° × 0.5° grid cells as groups)
- Or use `scikit-learn`'s `KFold` on sorted latitude to create geographic folds

```python
from sklearn.model_selection import GroupKFold
import numpy as np

# Create spatial blocks
df_train['lat_block'] = (df_train['Latitude'] // 0.5).astype(int)
df_train['lon_block'] = (df_train['Longitude'] // 0.5).astype(int)
df_train['spatial_group'] = df_train['lat_block'].astype(str) + '_' + df_train['lon_block'].astype(str)

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=df_train['spatial_group'])):
    ...
```

---

## 6. Handling the `TargetPred_To_Keep` Mask

This is critical for your final score. After generating all predictions:

```python
import pandas as pd

preds_df = pd.read_csv('my_predictions.csv')        # your raw predictions
mask_df  = pd.read_csv('TargetPred_To_Keep.csv')    # 1=keep, 0=set to zero

target_cols = ['Target_Al','Target_B','Target_Ca','Target_Cu','Target_Fe',
               'Target_K','Target_Mg','Target_Mn','Target_N','Target_Na',
               'Target_P','Target_S','Target_Zn']
mask_cols   = ['Al','B','Ca','Cu','Fe','K','Mg','Mn','N','Na','P','S','Zn']

mask_df = mask_df.set_index('ID')
preds_df = preds_df.set_index('ID')

for pred_col, mask_col in zip(target_cols, mask_cols):
    preds_df[pred_col] = preds_df[pred_col] * mask_df[mask_col]

preds_df.reset_index().to_csv('submission_masked.csv', index=False)
```

---

## 7. Project Directory Structure

```
rhea-soil-challenge/
├── data/
│   ├── raw/               # Original CSVs from Zindi
│   └── external/          # Downloaded EO data
│       ├── sentinel2/
│       ├── soilgrids/
│       ├── worldclim/
│       └── chirps/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_EO_Data_Download.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Baseline_Model.ipynb
│   └── 05_Advanced_Models.ipynb
├── src/
│   ├── download_eo_data.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── submissions/
└── README.md
```

---

## 8. Step-by-Step Execution Plan

| Step | Action | Tools |
|---|---|---|
| 1 | EDA: distributions of 13 nutrients, missing values, spatial coverage | pandas, seaborn, folium |
| 2 | Download SoilGrids for all 44K train + 6K test points | requests (batch with throttling) |
| 3 | Download WorldClim bio variables (static rasters, point sampling) | rasterio |
| 4 | Download CHIRPS annual/seasonal rainfall | GEE or direct download |
| 5 | Download Sentinel-2 / Landsat composites via GEE | earthengine-api |
| 6 | Merge all EO features with train/test CSVs | pandas |
| 7 | Feature engineering (depth encoding, ratios, spatial clusters) | sklearn, numpy |
| 8 | Baseline LightGBM per-nutrient with spatial CV | lightgbm, sklearn |
| 9 | Tune + ensemble (LightGBM + XGBoost + CatBoost) | optuna for HPO |
| 10 | Apply TargetPred mask and format submission | pandas |
| 11 | Submit, check public LB, iterate | — |

---

## 9. Tips for Competitive Performance

1. **SoilGrids is your most powerful free feature** — pH, clay%, CEC, SOC closely predict many of the target nutrients.
2. **Log-transform skewed targets** before training (Al, Fe, Mn especially). Remember to inverse-transform predictions.
3. **Depth is a major signal** — "0-20cm" vs "20-50cm" samples can differ by 2-5× for many nutrients.
4. **Inter-nutrient correlation** — Ca, Mg, K are correlated; Fe, Mn, Al form another cluster (linked to soil acidity). Use cross-target features.
5. **Spatial autocorrelation** — nearby samples are similar. k-NN on lat/lon distance as a feature often boosts scores.
6. **Use training nutrient values as mutual features** — in a second-pass model, use OOF-predicted co-nutrients as features.
7. **Blend with median imputation per spatial cluster** as a regularization backstop.

---

## 10. Key Python Packages

```bash
pip install lightgbm xgboost catboost scikit-learn pandas numpy rasterio geopandas
pip install earthengine-api  # for GEE access
pip install requests tqdm    # for SoilGrids batch download
pip install optuna           # hyperparameter optimization
pip install shap             # feature importance explainability
```

---

*Prepared for the Rhea Soil Nutrient Prediction Challenge (Zindi, closes 9 Mar 2026)*
