"""
06_download_soilgrids.py — Fast parallel download of SoilGrids 2.0 features
Rhea Soil Nutrient Prediction Challenge

Uses 20 parallel threads → ~20x faster than sequential.
Automatically resumes from cache — safe to stop and restart anytime.

Run: python 06_download_soilgrids.py
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ── Config ────────────────────────────────────────────────────────────────────
RAW       = Path("data/raw")
EXT       = Path("data/external")
EXT.mkdir(parents=True, exist_ok=True)

CACHE_DIR = EXT / "soilgrids_cache"
CACHE_DIR.mkdir(exist_ok=True)

API_URL   = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPS     = ["phh2o","clay","sand","silt","soc","cec","bdod","nitrogen"]
DEPTHS    = ["0-5cm","5-15cm","15-30cm","30-60cm"]

MAX_WORKERS = 20    # parallel threads — increase to 30 if no rate limit errors
MAX_RETRY   = 3
TIMEOUT     = 30

# Thread-safe print lock
_print_lock = threading.Lock()


def fetch_point(lat: float, lon: float):
    params = {
        "lon": round(lon, 5),
        "lat": round(lat, 5),
        "property": PROPS,
        "depth": DEPTHS,
    }
    for attempt in range(MAX_RETRY):
        try:
            r = requests.get(API_URL, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(5 * (attempt + 1))   # back off on rate limit
            else:
                time.sleep(2)
        except Exception:
            time.sleep(2)
    return None


def parse_response(data, lat, lon):
    row = {'Latitude': lat, 'Longitude': lon}
    if data is None or 'properties' not in data:
        return row
    for layer in data['properties']['layers']:
        prop = layer['name']
        for depth_info in layer['depths']:
            depth_label = depth_info['label'].replace(' ', '')
            mean_val    = depth_info['values'].get('mean')
            row[f"sg_{prop}_{depth_label}"] = mean_val
    return row


def cache_path(lat, lon):
    return CACHE_DIR / f"{round(lat,4)}_{round(lon,4)}.json"


def load_cache(lat, lon):
    p = cache_path(lat, lon)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_cache(lat, lon, data):
    with open(cache_path(lat, lon), 'w') as f:
        json.dump(data, f)


def fetch_one(row_tuple):
    """Worker function: fetch one point, use cache if available."""
    id_, lat, lon = row_tuple
    cached = load_cache(lat, lon)
    if cached is not None:
        data = cached
    else:
        data = fetch_point(lat, lon)
        save_cache(lat, lon, data)
    parsed = parse_response(data, lat, lon)
    parsed['ID'] = id_
    return parsed


def process_set(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = list(zip(df['ID'], df['Latitude'], df['Longitude']))
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, r): r for r in rows}
        with tqdm(total=len(rows), desc=label) as pbar:
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    with _print_lock:
                        print(f"Error: {e}")
                pbar.update(1)

    result_df = pd.DataFrame(results)
    cols = ['ID'] + [c for c in result_df.columns if c != 'ID']
    return result_df[cols]


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train = pd.read_csv(RAW / "Train.csv")[['ID','Latitude','Longitude']].drop_duplicates('ID')
    test  = pd.read_csv(RAW / "TestSet.csv")[['ID','Latitude','Longitude']].drop_duplicates('ID')

    # Count already cached
    cached_count = len(list(CACHE_DIR.glob("*.json")))
    print(f"Train: {len(train)} points | Test: {len(test)} points")
    print(f"Already cached: {cached_count} points — resuming from where we left off")
    print(f"Running with {MAX_WORKERS} parallel threads\n")

    train_sg = process_set(train, "Train SoilGrids")
    train_sg.to_csv(EXT / "soilgrids_train.csv", index=False)
    print(f"\n✅ Saved: soilgrids_train.csv  shape={train_sg.shape}")

    test_sg = process_set(test, "Test SoilGrids")
    test_sg.to_csv(EXT / "soilgrids_test.csv", index=False)
    print(f"✅ Saved: soilgrids_test.csv   shape={test_sg.shape}")
