"""
04_make_submission.py — Apply TargetPred mask and format final submission
Rhea Soil Nutrient Prediction Challenge

Run: python 04_make_submission.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
RAW  = Path("data/raw")
PROC = Path("data/processed")
SUB  = Path("submissions")
SUB.mkdir(parents=True, exist_ok=True)

# ── Choose which stage to use (stage2 if it exists, else stage1) ──────────────
stage2_path = PROC / "test_preds_stage2.csv"
stage1_path = PROC / "test_preds_stage1.csv"

if stage2_path.exists():
    preds_df = pd.read_csv(stage2_path)
    print("Using Stage-2 predictions")
else:
    preds_df = pd.read_csv(stage1_path)
    print("Using Stage-1 predictions")

# ── Load test IDs ─────────────────────────────────────────────────────────────
test  = pd.read_csv(RAW / "TestSet.csv")
mask  = pd.read_csv(RAW / "TargetPred_To_Keep.csv")
sample_sub = pd.read_csv(RAW / "SampleSubmission.csv")

print(f"\nTest IDs:   {len(test)}")
print(f"Mask rows:  {len(mask)}")
print(f"Pred rows:  {len(preds_df)}")

# Target column names in submission vs mask
TARGETS      = ['Al','B','Ca','Cu','Fe','K','Mg','Mn','N','Na','P','S','Zn']
TARGET_COLS  = [f'Target_{t}' for t in TARGETS]

# ── Build submission DataFrame ────────────────────────────────────────────────
sub = pd.DataFrame()
sub['ID'] = test['ID'].values

for pred_col, t in zip(TARGET_COLS, TARGETS):
    if t in preds_df.columns:
        sub[pred_col] = preds_df[t].values
    else:
        sub[pred_col] = 0.0

# ── Apply TargetPred_To_Keep mask ─────────────────────────────────────────────
mask = mask.set_index('ID')
sub  = sub.set_index('ID')

for pred_col, t in zip(TARGET_COLS, TARGETS):
    if t in mask.columns:
        sub[pred_col] = sub[pred_col] * mask.loc[sub.index, t].values
    else:
        sub[pred_col] = 0.0

sub = sub.reset_index()

# Clip negatives to 0 (nutrients can't be negative)
for col in TARGET_COLS:
    sub[col] = sub[col].clip(lower=0)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("\nSubmission shape:", sub.shape)
print("Non-zero entries per target:")
for col in TARGET_COLS:
    n_nonzero = (sub[col] > 0).sum()
    print(f"  {col:15s}: {n_nonzero:,} non-zero   mean={sub[col][sub[col]>0].mean():.2f}")

# Check against sample submission columns
sample_cols = list(sample_sub.columns)
sub_cols    = list(sub.columns)
assert sub_cols == sample_cols, (
    f"Column mismatch!\n  Expected: {sample_cols}\n  Got: {sub_cols}"
)
print("\n✅ Column order matches SampleSubmission.csv")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = SUB / "submission.csv"
sub.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
print(sub.head())
