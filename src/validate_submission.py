"""
validate_submission.py
======================
Checks that a submission CSV is valid before uploading to Zindi:
  - Correct columns
  - No NaN values
  - Mask entries are 0
  - Non-negative predictions
  - Correct number of rows

Run: python src/validate_submission.py submissions/submission_final.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
NUTRIENTS = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']


def validate(submission_path: str):
    sub_path = Path(submission_path)
    if not sub_path.exists():
        print(f"ERROR: File not found: {sub_path}")
        sys.exit(1)

    sub   = pd.read_csv(sub_path)
    mask  = pd.read_csv(DATA_DIR / "TargetPred_To_Keep.csv")
    sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")

    print(f"\n{'='*55}")
    print(f"  Validating: {sub_path.name}")
    print(f"{'='*55}")
    ok = True

    # ── 1. Shape ──────────────────────────────────────────────────────────
    expected_rows = len(sample)
    if len(sub) != expected_rows:
        print(f"  ✗ Row count: {len(sub)} (expected {expected_rows})")
        ok = False
    else:
        print(f"  ✓ Row count: {len(sub)}")

    # ── 2. Columns ────────────────────────────────────────────────────────
    missing_cols = set(sample.columns) - set(sub.columns)
    extra_cols   = set(sub.columns) - set(sample.columns)
    if missing_cols:
        print(f"  ✗ Missing columns: {missing_cols}")
        ok = False
    elif extra_cols:
        print(f"  ⚠ Extra columns (harmless): {extra_cols}")
    else:
        print(f"  ✓ Columns match sample submission")

    target_cols    = [f"Target_{n}" for n in NUTRIENTS]
    present_target = [c for c in target_cols if c in sub.columns]

    # ── 3. NaN check ──────────────────────────────────────────────────────
    nan_counts = sub[present_target].isnull().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"  ✗ NaN values found:\n{nan_counts[nan_counts>0]}")
        ok = False
    else:
        print(f"  ✓ No NaN values")

    # ── 4. Negative values ────────────────────────────────────────────────
    neg_counts = (sub[present_target] < 0).sum()
    total_neg  = neg_counts.sum()
    if total_neg > 0:
        print(f"  ✗ Negative predictions:\n{neg_counts[neg_counts>0]}")
        ok = False
    else:
        print(f"  ✓ No negative values")

    # ── 5. Mask compliance ────────────────────────────────────────────────
    sub_m  = sub.set_index("ID")
    mask_m = mask.set_index("ID").reindex(sub_m.index)
    mask_violations = 0

    for nut in NUTRIENTS:
        pred_col = f"Target_{nut}"
        if pred_col not in sub_m.columns or nut not in mask_m.columns:
            continue
        should_be_zero = mask_m[nut] == 0
        nonzero_when_should_be = (sub_m.loc[should_be_zero, pred_col] != 0).sum()
        mask_violations += nonzero_when_should_be

    if mask_violations > 0:
        print(f"  ✗ Mask violations: {mask_violations} entries should be 0")
        ok = False
    else:
        print(f"  ✓ Mask compliant (all zeroed entries are 0)")

    # ── 6. Value range summary ────────────────────────────────────────────
    print(f"\n  Predicted value ranges (non-zero):")
    for col in present_target:
        vals = sub_m[col]
        nonzero = vals[vals > 0]
        if len(nonzero):
            print(f"    {col:<15}  min={nonzero.min():8.2f}  "
                  f"median={nonzero.median():8.2f}  max={nonzero.max():10.2f}  "
                  f"n={len(nonzero):,}")

    # ── 7. Final verdict ──────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    if ok:
        print("  ✓ Submission is VALID — ready to upload to Zindi!")
    else:
        print("  ✗ Submission has ISSUES — fix before uploading.")
    print(f"{'─'*55}\n")
    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "submissions/submission_final.csv"
    validate(path)
