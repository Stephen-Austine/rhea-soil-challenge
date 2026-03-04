"""
setup_data.py
=============
Copy the challenge CSV files into data/raw/.
Run ONCE before anything else.

Usage:
  python setup_data.py --src /path/to/downloaded/files
  python setup_data.py  # looks in current directory
"""

import shutil
import argparse
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FILES = [
    "Train.csv",
    "TestSet.csv",
    "TargetPred_To_Keep.csv",
    "Sample_Collection_Dates.csv",
    "SampleSubmission.csv",
    "data_dictionary.csv",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=".", help="Source directory with challenge CSVs")
    args = parser.parse_args()

    src_dir = Path(args.src)
    print(f"Source directory : {src_dir.resolve()}")
    print(f"Target directory : {DATA_DIR.resolve()}")
    print()

    all_found = True
    for fname in REQUIRED_FILES:
        src = src_dir / fname
        dst = DATA_DIR / fname
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"  ✓ Copied  {fname}")
            else:
                print(f"  ✓ Already exists  {fname}")
        else:
            print(f"  ✗ NOT FOUND: {src}")
            all_found = False

    if all_found:
        print("\n✓ All files ready in data/raw/")
    else:
        print("\n✗ Some files are missing. Place them in the source directory and re-run.")


if __name__ == "__main__":
    main()
