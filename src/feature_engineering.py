"""
feature_engineering.py
=======================
Merges raw tabular data with external features, then engineers the full
feature matrix for training and inference.

Run: python src/feature_engineering.py
Outputs:
  data/processed/train_features.parquet
  data/processed/test_features.parquet
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
EXT_DIR  = ROOT / "data" / "external"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

NUTRIENTS = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']

# ── 1. Load raw data ──────────────────────────────────────────────────────────

def load_raw():
    train = pd.read_csv(DATA_DIR / "Train.csv")
    test  = pd.read_csv(DATA_DIR / "TestSet.csv")
    dates = pd.read_csv(DATA_DIR / "Sample_Collection_Dates.csv")

    # Merge collection dates (some date info may be richer in this file)
    train = train.merge(dates[["ID","start_date","end_date"]], on="ID", how="left",
                        suffixes=("", "_dates"))
    test  = test.merge( dates[["ID","start_date","end_date"]], on="ID", how="left",
                        suffixes=("", "_dates"))

    # Use _dates columns when the base ones are missing
    for col in ["start_date", "end_date"]:
        alt = col + "_dates"
        if alt in train.columns:
            train[col] = train[col].fillna(train[alt])
            test[col]  = test[col].fillna(test[alt]) if alt in test.columns else test[col]
            train.drop(columns=[alt], errors="ignore", inplace=True)
            test.drop( columns=[alt], errors="ignore", inplace=True)

    return train, test


# ── 2. Depth encoding ─────────────────────────────────────────────────────────

def encode_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'Depth_cm' string (e.g. '0-20') into numeric midpoint and flags."""
    def _midpoint(s):
        try:
            m = re.findall(r"[\d.]+", str(s))
            if len(m) >= 2:
                return (float(m[0]) + float(m[1])) / 2
            elif len(m) == 1:
                return float(m[0])
        except Exception:
            pass
        return np.nan

    df = df.copy()
    df["depth_mid"]   = df["Depth_cm"].apply(_midpoint)
    df["depth_upper"] = df.get("horizon_upper", pd.Series(dtype=float)).astype(float)
    df["depth_lower"] = df.get("horizon_lower", pd.Series(dtype=float)).astype(float)
    df["depth_range"] = df["depth_lower"] - df["depth_upper"]
    # One-hot for common depth intervals
    df["is_shallow"]  = (df["depth_mid"] <= 15).astype(int)
    df["is_deep"]     = (df["depth_mid"] > 30).astype(int)
    return df


# ── 3. Temporal features ──────────────────────────────────────────────────────

def encode_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["start_date", "end_date"]:
        parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
        prefix = col.replace("_date", "")
        df[f"{prefix}_year"]  = parsed.dt.year
        df[f"{prefix}_month"] = parsed.dt.month
    # Duration of observation period in days
    s = pd.to_datetime(df["start_date"], dayfirst=True, errors="coerce")
    e = pd.to_datetime(df["end_date"],   dayfirst=True, errors="coerce")
    df["obs_duration_days"] = (e - s).dt.days.fillna(0)
    df["collection_decade"] = (df["start_year"] // 10 * 10).fillna(2010)
    return df


# ── 4. Spatial features ───────────────────────────────────────────────────────

def add_spatial_features(df: pd.DataFrame, n_clusters: int = 30,
                          kmeans_model=None, fit: bool = True):
    """
    - Spatial k-means cluster on lat/lon (captures regional soil zones)
    - sin/cos encoding of lat/lon to remove circularity
    Returns (augmented_df, fitted_kmeans_model)
    """
    df = df.copy()
    coords = df[["Latitude", "Longitude"]].fillna(0).values

    if fit:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km.fit(coords)
        kmeans_model = km
    else:
        km = kmeans_model

    df["spatial_cluster"] = km.predict(coords).astype(str)

    # Distance to cluster centre (local spatial anomaly)
    centers = km.cluster_centers_
    labels  = km.predict(coords)
    df["dist_to_cluster_center"] = np.linalg.norm(
        coords - centers[labels], axis=1
    )

    # Cyclic encoding
    df["lat_sin"] = np.sin(np.radians(df["Latitude"].fillna(0)))
    df["lat_cos"] = np.cos(np.radians(df["Latitude"].fillna(0)))
    df["lon_sin"] = np.sin(np.radians(df["Longitude"].fillna(0)))
    df["lon_cos"] = np.cos(np.radians(df["Longitude"].fillna(0)))

    return df, km


# ── 5. Chemistry ratios & interactions ───────────────────────────────────────

def add_chemistry_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    # pH × C_organic interaction
    if "ph" in df.columns and "C_organic" in df.columns:
        df["ph_x_Corg"] = df["ph"] * df["C_organic"].fillna(0)

    if "C_organic" in df.columns and "C_total" in df.columns:
        df["C_inorganic"] = df["C_total"].fillna(0) - df["C_organic"].fillna(0)
        df["Corg_ratio"]  = df["C_organic"].fillna(0) / (df["C_total"].fillna(0) + eps)

    # Agronomic ratios from training targets (will be NaN for test — handled by model)
    nut = {n: df[n] if n in df.columns else pd.Series(np.nan, index=df.index)
           for n in NUTRIENTS}

    df["ratio_Ca_Mg"] = nut["Ca"] / (nut["Mg"] + eps)
    df["ratio_Fe_Mn"] = nut["Fe"] / (nut["Mn"] + eps)
    df["ratio_K_Na"]  = nut["K"]  / (nut["Na"] + eps)
    df["ratio_Ca_K"]  = nut["Ca"] / (nut["K"]  + eps)

    return df


# ── 6. External feature merge ─────────────────────────────────────────────────

def merge_external(df: pd.DataFrame) -> pd.DataFrame:
    ext_path = EXT_DIR / "all_external_features.csv"
    if not ext_path.exists():
        print(f"  WARNING: {ext_path} not found. Run download_eo_data.py first.")
        return df
    ext = pd.read_csv(ext_path)
    # Drop lat/lon from ext to avoid duplicates
    ext = ext.drop(columns=["Latitude","Longitude"], errors="ignore")
    return df.merge(ext, on="ID", how="left")


# ── 7. Log-transform targets ──────────────────────────────────────────────────

def log_transform_targets(df: pd.DataFrame, nutrients=NUTRIENTS) -> pd.DataFrame:
    """Apply log1p to nutrient columns (handles 0 and near-zero values)."""
    df = df.copy()
    for n in nutrients:
        if n in df.columns:
            df[n] = np.log1p(df[n].clip(lower=0))
    return df


def inverse_log_transform(arr: np.ndarray) -> np.ndarray:
    return np.expm1(arr)


# ── 8. Full pipeline ──────────────────────────────────────────────────────────

def build_features(log_targets: bool = True):
    print("Building feature matrices …")

    train_raw, test_raw = load_raw()

    # Step 1 — depth
    train = encode_depth(train_raw)
    test  = encode_depth(test_raw)

    # Step 2 — dates
    train = encode_dates(train)
    test  = encode_dates(test)

    # Step 3 — chemistry
    train = add_chemistry_features(train)
    test  = add_chemistry_features(test)

    # Step 4 — spatial clusters (fit on train, apply to test)
    train, km_model = add_spatial_features(train, n_clusters=30, fit=True)
    test,  _        = add_spatial_features(test,  n_clusters=30,
                                            kmeans_model=km_model, fit=False)

    # Step 5 — external EO data
    train = merge_external(train)
    test  = merge_external(test)

    # Step 6 — encode categorical depth string
    le = LabelEncoder()
    train["depth_label_enc"] = le.fit_transform(train["Depth_cm"].fillna("unknown"))
    test_depths = test["Depth_cm"].fillna("unknown")
    test["depth_label_enc"] = test_depths.apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

    # Step 7 — label-encode spatial cluster
    le2 = LabelEncoder()
    all_clusters = pd.concat([train["spatial_cluster"], test["spatial_cluster"]])
    le2.fit(all_clusters)
    train["spatial_cluster_enc"] = le2.transform(train["spatial_cluster"])
    test["spatial_cluster_enc"]  = le2.transform(test["spatial_cluster"])

    # Step 8 — log-transform targets
    if log_targets:
        train = log_transform_targets(train, NUTRIENTS)

    # Step 9 — drop columns not useful for modelling
    drop_cols = ["start_date","end_date","Depth_cm","spatial_cluster"]
    train.drop(columns=drop_cols, errors="ignore", inplace=True)
    test.drop( columns=drop_cols, errors="ignore", inplace=True)

    # Save
    train.to_parquet(PROC_DIR / "train_features.parquet", index=False)
    test.to_parquet( PROC_DIR / "test_features.parquet",  index=False)
    print(f"  Train features shape : {train.shape}")
    print(f"  Test  features shape : {test.shape}")
    print(f"  Saved to : {PROC_DIR}")

    return train, test


if __name__ == "__main__":
    build_features(log_targets=True)
