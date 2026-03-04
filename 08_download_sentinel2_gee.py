"""
08_download_sentinel2_gee.py — Full dual-window satellite feature extraction
Rhea Soil Nutrient Prediction Challenge

For every point this script extracts TWO sets of spectral features:

  1. CONTEMPORANEOUS — matched to when the soil was actually sampled
       • 2017+      → Sentinel-2 SR Harmonized
       • 2013-2016  → Landsat 8 Surface Reflectance
       • 2008-2012  → Landsat 7 Surface Reflectance
     Prefix: s2_contemp_* / l8_contemp_* / l7_contemp_*

  2. RECENT FIXED WINDOW (2020-2023) — stable land cover baseline
       • Always Sentinel-2
     Prefix: s2_recent_*

Total: up to ~26 spectral features per point.
Safe to stop (Ctrl+C) and restart anytime — resumes from cache.

Run: python 08_download_sentinel2_gee.py
"""

import ee
import pandas as pd
import json
import time
from pathlib import Path
from tqdm import tqdm

# ── Auth ───────────────────────────────────────────────────────────────────────
GEE_PROJECT = "gen-lang-client-0367276032"   # leave None to auto-detect, or set "ee-yourproject"

def init_gee():
    import subprocess
    project = GEE_PROJECT
    if not project:
        try:
            r = subprocess.run(['gcloud','config','get-value','project'],
                               capture_output=True, text=True, timeout=10)
            project = r.stdout.strip() or None
        except Exception:
            project = None
    if project:
        print(f"Using GEE project: {project}")
        try:
            ee.Initialize(project=project)
            return
        except Exception as e:
            print(f"  Init failed: {e}")
    print("Opening browser for earthengine authenticate …")
    ee.Authenticate()
    ee.Initialize(project=project) if project else ee.Initialize()

init_gee()

# ── Config ─────────────────────────────────────────────────────────────────────
RAW    = Path("data/raw")
EXT    = Path("data/external")
EXT.mkdir(parents=True, exist_ok=True)

BUFFER_M  = 500
CLOUD_PCT = 20
SLEEP     = 0.5

S2_COL = "COPERNICUS/S2_SR_HARMONIZED"
L7_COL = "LANDSAT/LE07/C02/T1_L2"
L8_COL = "LANDSAT/LC08/C02/T1_L2"


# ── Index computation helpers ──────────────────────────────────────────────────
def add_s2_indices(img):
    b2=img.select('B2'); b3=img.select('B3'); b4=img.select('B4')
    b8=img.select('B8'); b8a=img.select('B8A')
    b11=img.select('B11')
    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI')
    evi  = b8.subtract(b4).divide(b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1)).multiply(2.5).rename('EVI')
    ndwi = b3.subtract(b8).divide(b3.add(b8)).rename('NDWI')
    bsi  = b11.add(b4).subtract(b8a.add(b2)).divide(b11.add(b4).add(b8a).add(b2)).rename('BSI')
    clay = b11.divide(b8a).rename('ClayRatio')
    iron = b4.divide(b2).rename('IronOxide')
    savi = b8.subtract(b4).divide(b8.add(b4).add(0.5)).multiply(1.5).rename('SAVI')
    return img.select(['B2','B3','B4','B8','B11','B12']).addBands([ndvi,evi,ndwi,bsi,clay,iron,savi])

def add_l8_indices(img):
    b2=img.select('SR_B2'); b3=img.select('SR_B3'); b4=img.select('SR_B4')
    b5=img.select('SR_B5'); b6=img.select('SR_B6')
    ndvi = b5.subtract(b4).divide(b5.add(b4)).rename('NDVI')
    ndwi = b3.subtract(b5).divide(b3.add(b5)).rename('NDWI')
    bsi  = b6.add(b4).subtract(b5.add(b2)).divide(b6.add(b4).add(b5).add(b2)).rename('BSI')
    iron = b4.divide(b2).rename('IronOxide')
    savi = b5.subtract(b4).divide(b5.add(b4).add(0.5)).multiply(1.5).rename('SAVI')
    return img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6']).addBands([ndvi,ndwi,bsi,iron,savi])

def add_l7_indices(img):
    b1=img.select('SR_B1'); b2=img.select('SR_B2'); b3=img.select('SR_B3')
    b4=img.select('SR_B4'); b5=img.select('SR_B5')
    ndvi = b4.subtract(b3).divide(b4.add(b3)).rename('NDVI')
    ndwi = b2.subtract(b4).divide(b2.add(b4)).rename('NDWI')
    bsi  = b5.add(b3).subtract(b4.add(b1)).divide(b5.add(b3).add(b4).add(b1)).rename('BSI')
    iron = b3.divide(b1).rename('IronOxide')
    savi = b4.subtract(b3).divide(b4.add(b3).add(0.5)).multiply(1.5).rename('SAVI')
    return img.select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5']).addBands([ndvi,ndwi,bsi,iron,savi])

def sample(img, roi, scale):
    return img.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi,
                            scale=scale, maxPixels=1e9).getInfo()


# ── Core fetch function ────────────────────────────────────────────────────────
def fetch_all_windows(lon, lat, start_year, end_year):
    pt  = ee.Geometry.Point([lon, lat])
    roi = pt.buffer(BUFFER_M)
    out = {}

    # 1. Recent S2 fixed window — always
    try:
        img = (ee.ImageCollection(S2_COL)
               .filterDate('2020-01-01', '2023-12-31')
               .filterBounds(roi)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_PCT))
               .median())
        vals = sample(add_s2_indices(img), roi, 20)
        out.update({f's2_recent_{k}': v for k, v in (vals or {}).items()})
    except Exception:
        pass

    # 2. Contemporaneous window
    s = f"{int(start_year)}-01-01"
    e = f"{int(end_year)}-12-31"

    if start_year >= 2017:
        try:
            img = (ee.ImageCollection(S2_COL)
                   .filterDate(s, e).filterBounds(roi)
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_PCT))
                   .median())
            vals = sample(add_s2_indices(img), roi, 20)
            out.update({f's2_contemp_{k}': v for k, v in (vals or {}).items()})
        except Exception:
            pass
    elif start_year >= 2013:
        try:
            img = (ee.ImageCollection(L8_COL)
                   .filterDate(s, e).filterBounds(roi).median())
            vals = sample(add_l8_indices(img), roi, 30)
            out.update({f'l8_contemp_{k}': v for k, v in (vals or {}).items()})
        except Exception:
            pass
    else:
        try:
            img = (ee.ImageCollection(L7_COL)
                   .filterDate(s, e).filterBounds(roi).median())
            vals = sample(add_l7_indices(img), roi, 30)
            out.update({f'l7_contemp_{k}': v for k, v in (vals or {}).items()})
        except Exception:
            pass

    return out


# ── Cache helpers ──────────────────────────────────────────────────────────────
def load_done_ids(cache_path):
    done = set()
    if not cache_path.exists():
        return done
    with open(cache_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if any(k != 'ID' for k in rec.keys()):
                    done.add(rec['ID'])
            except Exception:
                pass
    return done


# ── Main processing ────────────────────────────────────────────────────────────
def process_df(df, label):
    cache_path = EXT / f"s2_cache_full_{label}.jsonl"
    done_ids   = load_done_ids(cache_path)
    print(f"  {len(done_ids)} cached | {len(df)-len(done_ids)} remaining")

    with open(cache_path, 'a') as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{label}]"):
            if row['ID'] in done_ids:
                continue
            try:
                features = fetch_all_windows(
                    row['Longitude'], row['Latitude'],
                    int(row.get('start_year', 2013)),
                    int(row.get('end_year',   2018))
                )
            except Exception:
                features = {}
            rec = {'ID': row['ID']}
            rec.update(features)
            f.write(json.dumps(rec) + '\n')
            time.sleep(SLEEP)

    rows = []
    with open(cache_path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    result = pd.DataFrame(rows)
    cols = ['ID'] + [c for c in result.columns if c != 'ID']
    return result[cols]


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    dates = pd.read_csv(RAW / "Sample_Collection_Dates.csv")

    def parse_year(s, default):
        try:
            return int(str(s).strip()[-4:])
        except Exception:
            return default

    dates['start_year'] = dates['start_date'].apply(lambda s: parse_year(s, 2013))
    dates['end_year']   = dates['end_date'].apply(lambda s: parse_year(s, 2018))

    train = pd.read_csv(RAW / "Train.csv")[['ID','Latitude','Longitude']]
    test  = pd.read_csv(RAW / "TestSet.csv")[['ID','Latitude','Longitude']]

    train = train.merge(dates[['ID','start_year','end_year']], on='ID', how='left')
    test  = test.merge( dates[['ID','start_year','end_year']], on='ID', how='left')

    train['start_year'] = train['start_year'].fillna(2013).astype(int)
    train['end_year']   = train['end_year'].fillna(2018).astype(int)
    test['start_year']  = test['start_year'].fillna(2013).astype(int)
    test['end_year']    = test['end_year'].fillna(2018).astype(int)

    print(f"\nTrain: {len(train)} | Test: {len(test)}")
    print("Each point gets: s2_recent (2020-2023) + contemporaneous S2/L8/L7\n")

    print("── Train ──")
    train_s2 = process_df(train, 'train')
    train_s2.to_csv(EXT / "sentinel2_train.csv", index=False)
    print(f"✅ sentinel2_train.csv  {train_s2.shape}  ({len(train_s2.columns)-1} features)")

    print("\n── Test ──")
    test_s2 = process_df(test, 'test')
    test_s2.to_csv(EXT / "sentinel2_test.csv", index=False)
    print(f"✅ sentinel2_test.csv   {test_s2.shape}  ({len(test_s2.columns)-1} features)")
