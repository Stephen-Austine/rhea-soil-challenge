# Rhea Soil Nutrient Prediction Challenge

Multi-target regression to predict 13 soil nutrients for 6,070 test locations across Africa.

## Setup

```bash
pip install -r requirements.txt
```

Place challenge CSV files in `data/raw/`:
- Train.csv
- TestSet.csv
- TargetPred_To_Keep.csv
- Sample_Collection_Dates.csv
- SampleSubmission.csv

## Workflow

Run scripts in order:

```bash
# 1. EDA — understand the data
python notebooks/01_EDA.py

# 2. Download external data (SoilGrids, WorldClim)
python src/download_eo_data.py

# 3. Feature engineering
python src/feature_engineering.py

# 4. Train models & generate submission
python src/train.py
```

Final submission will be saved to `submissions/submission_final.csv`.
