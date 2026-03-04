"""
ensemble.py
===========
Stage-2 stacking: uses OOF predictions from train.py as additional features
to train a second-level model, capturing inter-nutrient correlations.

Also provides a simple weighted average blender.

Run: python src/ensemble.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT / "data" / "processed"
DATA_DIR   = ROOT / "data" / "raw"
SUBMIT_DIR = ROOT / "submissions"
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

NUTRIENTS  = ['Al', 'B', 'Ca', 'Cu', 'Fe', 'K', 'Mg', 'Mn', 'N', 'Na', 'P', 'S', 'Zn']
RANDOM_SEED = 42

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_spatial_groups(df, grid_size=0.5):
    lat_block = (df["Latitude"] // grid_size).astype(int)
    lon_block = (df["Longitude"] // grid_size).astype(int)
    return (lat_block.astype(str) + "_" + lon_block.astype(str))


def stage2_train(train_df, test_df, oof_df, stage1_test_preds_df):
    """
    Build a 2nd-level model for each nutrient:
      features = [original features] + [OOF predictions of all OTHER nutrients]
    """
    from feature_engineering import get_feature_cols

    base_feature_cols = [c for c in train_df.columns
                         if c not in ["ID"] + NUTRIENTS]
    spatial_groups = make_spatial_groups(train_df)

    all_test_preds = pd.DataFrame({"ID": test_df["ID"].values})
    cv_scores = {}

    for target in NUTRIENTS:
        if target not in train_df.columns:
            all_test_preds[f"Target_{target}"] = 0.0
            continue

        mask_notna = train_df[target].notna()
        df_t = train_df[mask_notna].copy()
        grp  = spatial_groups[mask_notna].values

        # Add OOF predictions of other nutrients as features
        other_nutrients = [n for n in NUTRIENTS if n != target]
        oof_others = oof_df[other_nutrients].reindex(df_t.index)
        oof_cols   = [f"oof_{n}" for n in other_nutrients]
        df_t[oof_cols] = oof_others.values

        feature_cols = base_feature_cols + oof_cols
        X    = df_t[feature_cols].fillna(df_t[feature_cols].median()).values
        y    = df_t[target].values

        # For test: use stage-1 test predictions as "OOF" features
        test_stage2 = test_df.copy()
        for n in other_nutrients:
            col = f"Target_{n}"
            if col in stage1_test_preds_df.columns:
                oof_col = f"oof_{n}"
                test_stage2[oof_col] = np.log1p(
                    stage1_test_preds_df.set_index("ID").reindex(test_df["ID"])[col].values.clip(0)
                )
        X_test = test_stage2[feature_cols].fillna(0).values

        gkf       = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(df_t))
        test_folds = []
        fold_rmses = []

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=grp)):
            model = lgb.LGBMRegressor(
                objective="regression_l1", metric="rmse",
                n_estimators=1500, learning_rate=0.04,
                num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=RANDOM_SEED, verbose=-1
            )
            model.fit(X[tr_idx], y[tr_idx],
                      eval_set=[(X[val_idx], y[val_idx])],
                      callbacks=[lgb.early_stopping(80, verbose=False),
                                 lgb.log_evaluation(-1)])
            oof_preds[val_idx] = model.predict(X[val_idx])
            test_folds.append(model.predict(X_test))

            fold_rmses.append(rmse(y[val_idx], oof_preds[val_idx]))

        cv_scores[target] = np.mean(fold_rmses)
        print(f"  [{target}]  Stage-2 CV RMSE = {cv_scores[target]:.4f}")
        all_test_preds[f"Target_{target}"] = np.expm1(np.mean(test_folds, axis=0))

    return all_test_preds, cv_scores


def blend_submissions(paths_weights: list) -> pd.DataFrame:
    """
    Simple weighted average of multiple submission CSVs.

    paths_weights: list of (path_str, weight) tuples
    """
    target_cols = [f"Target_{n}" for n in NUTRIENTS]
    base = None
    total_w = 0

    for path, w in paths_weights:
        df = pd.read_csv(path)
        if base is None:
            base = df.copy()
            base[target_cols] = 0.0

        for col in target_cols:
            if col in df.columns:
                base[col] += df[col].fillna(0) * w
        total_w += w

    base[target_cols] /= total_w
    return base


def main():
    print("=" * 60)
    print("  Stage-2 Ensemble / Stacking")
    print("=" * 60)

    # Load processed data
    train_df = pd.read_parquet(PROC_DIR / "train_features.parquet")
    test_df  = pd.read_parquet(PROC_DIR / "test_features.parquet")
    mask_df  = pd.read_csv(DATA_DIR / "TargetPred_To_Keep.csv").set_index("ID")

    # Load stage-1 OOF and test predictions
    oof_path   = SUBMIT_DIR / "oof_predictions.csv"
    stage1_path = SUBMIT_DIR / "submission_final.csv"

    if not oof_path.exists() or not stage1_path.exists():
        print("  Stage-1 outputs not found — run train.py first.")
        return

    oof_df      = pd.read_csv(oof_path).set_index("ID")
    stage1_preds = pd.read_csv(stage1_path)

    # Log-transform OOF (they were saved in original scale)
    for n in NUTRIENTS:
        if n in oof_df.columns:
            oof_df[n] = np.log1p(oof_df[n].clip(lower=0))

    # Stage-2 training
    stage2_preds, cv_scores = stage2_train(train_df, test_df, oof_df, stage1_preds)

    # Apply mask
    nutrient_cols = [f"Target_{n}" for n in NUTRIENTS]
    preds_masked  = stage2_preds.set_index("ID")
    mask_aligned  = mask_df.reindex(preds_masked.index)

    for pred_col, mask_col in zip(nutrient_cols, NUTRIENTS):
        if pred_col in preds_masked.columns and mask_col in mask_aligned.columns:
            preds_masked[pred_col] = preds_masked[pred_col] * mask_aligned[mask_col].fillna(0)

    stage2_sub = preds_masked.reset_index()
    for col in nutrient_cols:
        if col in stage2_sub.columns:
            stage2_sub[col] = stage2_sub[col].clip(lower=0)

    stage2_path = SUBMIT_DIR / "submission_stage2.csv"
    sample_sub  = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
    stage2_sub  = stage2_sub[sample_sub.columns]
    stage2_sub.to_csv(stage2_path, index=False)
    print(f"\n  Stage-2 submission saved → {stage2_path}")

    # Blend stage1 + stage2 (0.4 / 0.6 weights — tune as needed)
    print("\n  Blending stage-1 and stage-2 …")
    blended = blend_submissions([
        (str(stage1_path), 0.35),
        (str(stage2_path), 0.65),
    ])
    # Apply mask one more time to blended
    blended_m = blended.set_index("ID")
    mask_aligned2 = mask_df.reindex(blended_m.index)
    for pred_col, mask_col in zip(nutrient_cols, NUTRIENTS):
        if pred_col in blended_m.columns and mask_col in mask_aligned2.columns:
            blended_m[pred_col] = blended_m[pred_col] * mask_aligned2[mask_col].fillna(0)
            blended_m[pred_col] = blended_m[pred_col].clip(lower=0)

    blended_sub = blended_m.reset_index()[sample_sub.columns]
    blended_path = SUBMIT_DIR / "submission_blended.csv"
    blended_sub.to_csv(blended_path, index=False)
    print(f"  Blended submission saved → {blended_path}")

    print("\n── Stage-2 CV Summary ────────────────────────────────────")
    for n, s in cv_scores.items():
        print(f"    {n:<4} : {s:.4f}")
    print(f"  Mean : {np.mean(list(cv_scores.values())):.4f}")


if __name__ == "__main__":
    main()
