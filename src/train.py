"""
train.py
========
Trains one LightGBM model per soil nutrient using spatial cross-validation,
then generates a properly masked submission file.

Run: python src/train.py

Options (edit CONFIG block below):
  QUICK_RUN   — fewer trees, no tuning, fast iteration
  USE_OPTUNA  — run Optuna HPO (slow but often +0.05 RMSE improvement)
  N_FOLDS     — number of spatial CV folds
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
QUICK_RUN   = False    # set True for fast debugging
USE_OPTUNA  = False   # set True for HPO (requires optuna)
N_FOLDS     = 5
RANDOM_SEED = 42

ROOT        = Path(__file__).resolve().parent.parent
PROC_DIR    = ROOT / "data" / "processed"
DATA_DIR    = ROOT / "data" / "raw"
SUBMIT_DIR  = ROOT / "submissions"
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

NUTRIENTS   = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']

# ── LightGBM base params ──────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    objective         = "regression_l1",    # MAE loss — robust to outliers
    metric            = "rmse",
    n_estimators      = 200 if QUICK_RUN else 2000,
    learning_rate     = 0.05,
    num_leaves        = 63,
    max_depth         = -1,
    min_child_samples = 20,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    n_jobs            = -1,
    random_state      = RANDOM_SEED,
    verbose           = -1,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_feature_cols(df: pd.DataFrame, nutrients=NUTRIENTS) -> list:
    """Return columns to use as model inputs (drop ID and all target nutrients)."""
    exclude = set(["ID"] + nutrients)
    # also exclude raw nutrient columns that might be present under alternate names
    exclude.update(["Target_" + n for n in nutrients])
    return [c for c in df.columns if c not in exclude]


def make_spatial_groups(df: pd.DataFrame, grid_size: float = 0.5) -> pd.Series:
    """Assign each sample to a 0.5° × 0.5° grid cell (spatial CV groups)."""
    lat_block = (df["Latitude"] // grid_size).astype(int)
    lon_block = (df["Longitude"] // grid_size).astype(int)
    return (lat_block.astype(str) + "_" + lon_block.astype(str))


# ── Optuna HPO (optional) ─────────────────────────────────────────────────────

def tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials=30):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "rmse",
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "n_jobs": -1, "random_state": RANDOM_SEED, "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
        pred = model.predict(X_val)
        return rmse(y_val, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best.update({"objective": "regression_l1", "metric": "rmse",
                 "n_estimators": 2000, "n_jobs": -1,
                 "random_state": RANDOM_SEED, "verbose": -1})
    return best


# ── Per-nutrient training ─────────────────────────────────────────────────────

def train_one_nutrient(train_df: pd.DataFrame,
                       test_df:  pd.DataFrame,
                       nutrient: str,
                       feature_cols: list,
                       spatial_groups: pd.Series):
    """
    Train LightGBM with spatial GroupKFold CV.
    Returns (oof_predictions, test_predictions, cv_rmse).
    """
    mask_notna = train_df[nutrient].notna()
    df  = train_df[mask_notna].copy()
    grp = spatial_groups[mask_notna].values

    X   = df[feature_cols].values
    y   = df[nutrient].values

    X_test = test_df[feature_cols].values

    gkf       = GroupKFold(n_splits=N_FOLDS)
    oof_preds = np.zeros(len(df))
    test_preds_folds = []
    fold_rmses = []
    params = LGBM_PARAMS.copy()

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=grp)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if USE_OPTUNA and fold == 0:
            # Tune only on first fold, then reuse params
            params = tune_lgbm(X_tr, y_tr, X_val, y_val)

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100 if not QUICK_RUN else 20, verbose=False),
                lgb.log_evaluation(-1)
            ]
        )
        oof_preds[val_idx] = model.predict(X_val)
        test_preds_folds.append(model.predict(X_test))

        fold_rmse = rmse(y_val, oof_preds[val_idx])
        fold_rmses.append(fold_rmse)

    cv_rmse     = np.mean(fold_rmses)
    test_preds  = np.mean(test_preds_folds, axis=0)

    # Map OOF predictions back to full training index
    oof_full = np.full(len(train_df), np.nan)
    oof_full[mask_notna.values] = oof_preds

    return oof_full, test_preds, cv_rmse


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Rhea Soil Nutrient Prediction — Training")
    print("=" * 60)

    # ── Load processed features ───────────────────────────────────────────
    train_path = PROC_DIR / "train_features.parquet"
    test_path  = PROC_DIR / "test_features.parquet"

    if not train_path.exists() or not test_path.exists():
        print("  Processed features not found — running feature_engineering.py …")
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from feature_engineering import build_features
        train_df, test_df = build_features(log_targets=True)
    else:
        train_df = pd.read_parquet(train_path)
        test_df  = pd.read_parquet(test_path)
        print(f"  Loaded train {train_df.shape}, test {test_df.shape}")

    # ── Load mask ─────────────────────────────────────────────────────────
    mask_df = pd.read_csv(DATA_DIR / "TargetPred_To_Keep.csv").set_index("ID")

    # ── Feature columns ───────────────────────────────────────────────────
    feature_cols = get_feature_cols(train_df)
    print(f"  Feature columns : {len(feature_cols)}")
    print(f"  Sample features : {feature_cols[:8]} …")

    # Fill remaining NaNs in features with median
    medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    test_df[feature_cols]  = test_df[feature_cols].fillna(medians)

    # ── Spatial groups for CV ─────────────────────────────────────────────
    spatial_groups = make_spatial_groups(train_df)

    # ── Train per nutrient ────────────────────────────────────────────────
    test_ids        = test_df["ID"].values
    all_test_preds  = pd.DataFrame({"ID": test_ids})
    all_oof_preds   = pd.DataFrame({"ID": train_df["ID"].values})
    cv_scores       = {}

    for nutrient in NUTRIENTS:
        if nutrient not in train_df.columns:
            print(f"  [{nutrient}] not in train — skipping")
            all_test_preds[f"Target_{nutrient}"] = 0.0
            continue

        n_valid = train_df[nutrient].notna().sum()
        print(f"\n  [{nutrient}]  n_train={n_valid:,}", end="  ")

        oof, test_preds, cv_rmse = train_one_nutrient(
            train_df, test_df, nutrient, feature_cols, spatial_groups
        )

        cv_scores[nutrient] = cv_rmse
        print(f"CV RMSE (log1p space) = {cv_rmse:.4f}")

        # Inverse transform (log1p was applied during feature engineering)
        all_test_preds[f"Target_{nutrient}"] = np.expm1(test_preds)
        all_oof_preds[nutrient]              = np.expm1(oof)

    # ── CV summary ────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  CV RMSE summary (log1p-inverse space):")
    for nut, score in cv_scores.items():
        print(f"    {nut:<4} : {score:.4f}")
    mean_score = np.mean(list(cv_scores.values()))
    print(f"  Mean RMSE : {mean_score:.4f}")
    print("─" * 50)

    # ── Apply TargetPred_To_Keep mask ─────────────────────────────────────
    print("\n  Applying TargetPred_To_Keep mask …")
    preds_masked = all_test_preds.set_index("ID")
    mask_aligned = mask_df.reindex(preds_masked.index)

    nutrient_cols   = [f"Target_{n}" for n in NUTRIENTS]
    mask_nutrient   = NUTRIENTS  # column names in mask file

    for pred_col, mask_col in zip(nutrient_cols, mask_nutrient):
        if pred_col in preds_masked.columns and mask_col in mask_aligned.columns:
            preds_masked[pred_col] = preds_masked[pred_col] * mask_aligned[mask_col].fillna(0)

    submission = preds_masked.reset_index()

    # Ensure all required columns are present
    sample_sub = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    for col in sample_sub.columns:
        if col not in submission.columns:
            submission[col] = 0.0

    submission = submission[sample_sub.columns]  # reorder to match sample

    # Clip to non-negative (soil nutrients can't be negative)
    for col in nutrient_cols:
        if col in submission.columns:
            submission[col] = submission[col].clip(lower=0)

    # Save
    out_path = SUBMIT_DIR / "submission_final.csv"
    submission.to_csv(out_path, index=False)
    print(f"\n✓ Submission saved → {out_path}")
    print(f"  Shape : {submission.shape}")

    # Also save OOF for stacking or analysis
    oof_path = SUBMIT_DIR / "oof_predictions.csv"
    all_oof_preds.to_csv(oof_path, index=False)
    print(f"  OOF predictions → {oof_path}")


if __name__ == "__main__":
    main()
