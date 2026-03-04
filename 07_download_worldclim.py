"""
07_download_worldclim.py — Sample WorldClim v2.1 bio variables at each location
Rhea Soil Nutrient Prediction Challenge

Uses 10-minute resolution (~18 MB) for fast, reliable download.
Falls back across multiple mirrors automatically if one fails.

Run: python 07_download_worldclim.py
"""

import requests
import zipfile
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
RAW     = Path("data/raw")
EXT     = Path("data/external")
WC_DIR  = EXT / "worldclim"
WC_DIR.mkdir(parents=True, exist_ok=True)

# Use 10m (~18 MB) — fast and accurate enough for soil modelling
RESOLUTION = "10m"
BIO_VARS   = [1, 4, 7, 12, 15, 16]

# Multiple mirrors — script tries each in order
MIRRORS = [
    f"https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_{RESOLUTION}_bio.zip",
    f"https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_{RESOLUTION}_bio.zip",
    f"https://data.worldclim.org/base/wc2.1_{RESOLUTION}_bio.zip",
]


def download_worldclim():
    zip_path = WC_DIR / f"wc2.1_{RESOLUTION}_bio.zip"
    if zip_path.exists():
        print("WorldClim zip already downloaded — skipping")
        return

    for i, url in enumerate(MIRRORS):
        print(f"Trying mirror {i+1}/{len(MIRRORS)}: {url}")
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True,
                                                   desc="Downloading WorldClim") as pbar:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print("Download complete.")
            return
        except Exception as e:
            print(f"  Mirror {i+1} failed: {e}")
            if zip_path.exists():
                zip_path.unlink()  # remove partial file

    raise RuntimeError(
        "All mirrors failed. Please manually download the file from:\n"
        "  https://worldclim.org/data/worldclim21.html\n"
        f"Save the zip to: {zip_path}"
    )


def extract_worldclim():
    zip_path = WC_DIR / f"wc2.1_{RESOLUTION}_bio.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Not found: {zip_path} — run download first")
    tif_files = list(WC_DIR.glob("*.tif"))
    if tif_files:
        print(f"WorldClim TIFs already extracted ({len(tif_files)} files)")
        return
    print("Extracting TIF files …")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(WC_DIR)
    print("Extraction complete.")


def sample_raster(tif_path: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Sample a raster at given lat/lon coordinates. Returns 1-D array."""
    with rasterio.open(tif_path) as src:
        # rasterio uses (row, col) from (lon, lat)
        coords = list(zip(lons, lats))
        vals   = list(src.sample(coords))
        arr    = np.array([v[0] for v in vals], dtype=float)
        # Replace nodata with NaN
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return arr


def sample_worldclim(df: pd.DataFrame, label: str) -> pd.DataFrame:
    lats = df['Latitude'].values
    lons = df['Longitude'].values
    result = df[['ID']].copy()

    for bio_num in tqdm(BIO_VARS, desc=f"Sampling WorldClim [{label}]"):
        pattern = list(WC_DIR.glob(f"wc2.1_{RESOLUTION}_bio_{bio_num}.tif"))
        if not pattern:
            print(f"  [WARN] TIF not found for bio_{bio_num}")
            result[f'wc_bio{bio_num}'] = np.nan
            continue
        tif_path = pattern[0]
        result[f'wc_bio{bio_num}'] = sample_raster(tif_path, lats, lons)
    return result


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    download_worldclim()
    extract_worldclim()

    train = pd.read_csv(RAW / "Train.csv")[['ID','Latitude','Longitude']]
    test  = pd.read_csv(RAW / "TestSet.csv")[['ID','Latitude','Longitude']]

    train_wc = sample_worldclim(train, "train")
    train_wc.to_csv(EXT / "worldclim_train.csv", index=False)
    print(f"✅ Saved: data/external/worldclim_train.csv  shape={train_wc.shape}")

    test_wc = sample_worldclim(test, "test")
    test_wc.to_csv(EXT / "worldclim_test.csv", index=False)
    print(f"✅ Saved: data/external/worldclim_test.csv   shape={test_wc.shape}")
