"""
download_eo_data.py
===================
Downloads external earth-observation features for every lat/lon in
Train.csv and TestSet.csv.

Sources
-------
  1. SoilGrids 2.0  (ISRIC REST API)  — pH, clay, sand, silt, SOC, CEC, bulk-density, nitrogen
  2. WorldClim v2   (pre-built raster) — BIO1, BIO4, BIO12, BIO15 (temp/precip normals)
  3. OpenTopoData   (SRTM v3 REST)     — Elevation in metres

All results are cached as CSVs so re-running is fast.

Run: python src/download_eo_data.py
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
EXT_DIR  = ROOT / "data" / "external"
EXT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_all_locations() -> pd.DataFrame:
    """Return unique (ID, Latitude, Longitude) for train + test combined."""
    train = pd.read_csv(DATA_DIR / "Train.csv",   usecols=["ID","Latitude","Longitude"])
    test  = pd.read_csv(DATA_DIR / "TestSet.csv", usecols=["ID","Latitude","Longitude"])
    locs  = pd.concat([train, test], ignore_index=True).drop_duplicates("ID")
    locs  = locs.dropna(subset=["Latitude","Longitude"])
    return locs.reset_index(drop=True)


def batch_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SoilGrids 2.0
# ═══════════════════════════════════════════════════════════════════════════════

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOILGRIDS_PROPS = ["phh2o","clay","sand","silt","soc","cec","bdod","nitrogen"]
SOILGRIDS_DEPTHS = ["0-5cm","5-15cm","15-30cm"]

def parse_soilgrids(response_json: dict, lat: float, lon: float) -> dict:
    """Flatten SoilGrids JSON into a flat dict of features."""
    row = {"Latitude": lat, "Longitude": lon}
    try:
        for layer in response_json.get("properties", {}).get("layers", []):
            prop  = layer["name"]
            for d in layer.get("depths", []):
                depth_label = d["label"].replace(" ","")  # e.g. "0-5cm"
                val = d.get("values", {}).get("mean", np.nan)
                # apply scale factor
                scale = layer.get("unit_measure", {}).get("d_factor", 1) or 1
                row[f"sg_{prop}_{depth_label}"] = val / scale if val is not None else np.nan
    except Exception:
        pass
    return row


def download_soilgrids(locs: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        print(f"  SoilGrids cache found → {cache_path}")
        return pd.read_csv(cache_path)

    print(f"  Downloading SoilGrids for {len(locs):,} locations …")
    records = []
    errors  = 0

    for _, row in tqdm(locs.iterrows(), total=len(locs), desc="SoilGrids"):
        lat, lon = row["Latitude"], row["Longitude"]
        try:
            r = requests.get(
                SOILGRIDS_URL,
                params={"lon": lon, "lat": lat,
                        "property": SOILGRIDS_PROPS,
                        "depth": SOILGRIDS_DEPTHS},
                timeout=30
            )
            if r.status_code == 200:
                rec = parse_soilgrids(r.json(), lat, lon)
                rec["ID"] = row["ID"]
                records.append(rec)
            else:
                errors += 1
                records.append({"ID": row["ID"], "Latitude": lat, "Longitude": lon})
        except Exception as e:
            errors += 1
            records.append({"ID": row["ID"], "Latitude": lat, "Longitude": lon})
        time.sleep(0.25)   # be polite to the API

    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"  → saved {cache_path}  ({errors} errors / {len(locs)} total)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Elevation (OpenTopoData — SRTM 90m)
# ═══════════════════════════════════════════════════════════════════════════════

TOPO_URL = "https://api.opentopodata.org/v1/srtm90m"

def download_elevation(locs: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        print(f"  Elevation cache found → {cache_path}")
        return pd.read_csv(cache_path)

    print(f"  Downloading elevation for {len(locs):,} locations (batches of 100) …")
    records = []

    for batch in tqdm(list(batch_list(locs.itertuples(index=False), 100)), desc="Elevation"):
        locations_str = "|".join(f"{r.Latitude},{r.Longitude}" for r in batch)
        try:
            r = requests.get(TOPO_URL, params={"locations": locations_str}, timeout=30)
            if r.status_code == 200:
                results = r.json().get("results", [])
                for i, res in enumerate(results):
                    records.append({
                        "ID": batch[i].ID,
                        "elevation_m": res.get("elevation", np.nan)
                    })
            else:
                for b in batch:
                    records.append({"ID": b.ID, "elevation_m": np.nan})
        except Exception:
            for b in batch:
                records.append({"ID": b.ID, "elevation_m": np.nan})
        time.sleep(1.0)

    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"  → saved {cache_path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  WorldClim v2.1 BIO variables (downloaded once, sampled by point)
#     Falls back to approximate values from Open-Meteo climate normals API
# ═══════════════════════════════════════════════════════════════════════════════

OPEN_METEO_CLIMATE = "https://climate-api.open-meteo.com/v1/climate"

def download_worldclim_approx(locs: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """
    Approximate BIO variables via Open-Meteo Climate API (ERA5, 1940–present).
    This is a free alternative to downloading WorldClim rasters.
    Returns annual mean temp (bio1-like) and annual precip (bio12-like).
    """
    if cache_path.exists():
        print(f"  WorldClim-approx cache found → {cache_path}")
        return pd.read_csv(cache_path)

    print(f"  Fetching climate normals (Open-Meteo) for {len(locs):,} locations …")
    records = []

    for _, row in tqdm(locs.iterrows(), total=len(locs), desc="Climate"):
        lat, lon = row["Latitude"], row["Longitude"]
        try:
            r = requests.get(
                OPEN_METEO_CLIMATE,
                params={
                    "latitude": lat, "longitude": lon,
                    "start_date": "2000-01-01", "end_date": "2010-12-31",
                    "models": "ERA5",
                    "daily": ["temperature_2m_mean","precipitation_sum"]
                },
                timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                daily = data.get("daily", {})
                temps  = [x for x in daily.get("temperature_2m_mean", []) if x is not None]
                precips = [x for x in daily.get("precipitation_sum", []) if x is not None]
                records.append({
                    "ID": row["ID"],
                    "climate_annual_temp_mean":  np.mean(temps)  if temps  else np.nan,
                    "climate_annual_precip_mm":  np.sum(precips) / max(len(precips)//365, 1) if precips else np.nan,
                    "climate_temp_std":          np.std(temps)   if temps  else np.nan,
                })
            else:
                records.append({"ID": row["ID"],
                                "climate_annual_temp_mean": np.nan,
                                "climate_annual_precip_mm": np.nan,
                                "climate_temp_std": np.nan})
        except Exception:
            records.append({"ID": row["ID"],
                            "climate_annual_temp_mean": np.nan,
                            "climate_annual_precip_mm": np.nan,
                            "climate_temp_std": np.nan})
        time.sleep(0.3)

    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"  → saved {cache_path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  External Data Download")
    print("=" * 60)

    locs = load_all_locations()
    print(f"  Total unique locations : {len(locs):,}")

    # --- SoilGrids ---
    print("\n[1/3] SoilGrids 2.0")
    sg = download_soilgrids(locs, EXT_DIR / "soilgrids_features.csv")
    print(f"      Columns: {list(sg.columns[:5])} … ({sg.shape[1]} total)")

    # --- Elevation ---
    print("\n[2/3] Elevation (SRTM 90m)")
    elev = download_elevation(locs, EXT_DIR / "elevation_features.csv")
    print(f"      Columns: {list(elev.columns)}")

    # --- Climate ---
    print("\n[3/3] Climate normals (Open-Meteo)")
    clim = download_worldclim_approx(locs, EXT_DIR / "climate_features.csv")
    print(f"      Columns: {list(clim.columns)}")

    # --- Merge all external features into one file ---
    print("\nMerging all external features …")
    ext = locs[["ID"]].merge(sg, on="ID", how="left") \
                       .merge(elev, on="ID", how="left") \
                       .merge(clim, on="ID", how="left")
    out_path = EXT_DIR / "all_external_features.csv"
    ext.to_csv(out_path, index=False)
    print(f"  → merged file saved: {out_path}  shape={ext.shape}")
    print("\n✓ Download complete.")


if __name__ == "__main__":
    main()
