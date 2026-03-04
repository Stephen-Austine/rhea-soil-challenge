"""
03_train.py — Train one LightGBM model per nutrient target
Rhea Soil Nutrient Prediction Challenge

Strategy
--------
  • Spatial GroupKFold (5 folds) to avoid data leakage
  • Log1p-transform right-skewed targets → inverse-transform predictions
  • Stage-1: independent per-target LightGBM models
  • Stage-2 (optional): append OOF predictions as extra cross-target features
    and re-train (improves correlated nutrients like Ca/Mg)
  • Save OOF RMSE per target and per-fold models

Run: python 03_train.py
"""

import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
PROC      = Path("data/processed")
MODELS    = Path("models")
MODELS.mkdir(parents=True, exist_ok=True)

N_FOLDS   = 5
SEED      = 42

TARGETS   = ['Al','B','Ca','Cu','Fe','K','Mg','Mn','N','Na','P','S','Zn']

# LightGBM hyperparameters — reasonable defaults; tune with Optuna (04_tune.py)
LGB_PARAMS = dict(
    objective        = 'regression',
    metric           = 'rmse',
    n_estimators     = 2000,
    learning_rate    = 0.03,
    num_leaves       = 127,
    max_depth        = -1,
    min_child_samples= 20,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = SEED,
    n_jobs           = -1,
    verbose          = -1,
)

# Targets that are heavily right-skewed → log1p transform
LOG_TARGETS = {'Al','B','Cu','Fe','Mn','Na','P','Zn','S','K','Mg','Ca','N'}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading processed features …")
train  = pd.read_csv(PROC / "train_features.csv")
test   = pd.read_csv(PROC / "test_features.csv")

with open(PROC / "feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

# Drop feature cols that are not in train (safety check)
FEATURE_COLS = [c for c in FEATURE_COLS if c in train.columns and c in test.columns]
print(f"Using {len(FEATURE_COLS)} feature columns")

X_train_full = train[FEATURE_COLS].copy()
X_test       = test[FEATURE_COLS].copy()

# Fill any remaining NaNs with median (per-column, fitted on train)
medians = X_train_full.median()
X_train_full = X_train_full.fillna(medians)
X_test       = X_test.fillna(medians)
medians.to_csv(PROC / "feature_medians.csv")

# Spatial groups for cross-validation
train['spatial_block'] = (train['Latitude'] // 0.5).astype(str) + '_' + (train['Longitude'] // 0.5).astype(str)
groups = train['spatial_block']

gkf = GroupKFold(n_splits=N_FOLDS)

# ── Stage-1: Train per-target models ─────────────────────────────────────────
oof_preds   = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
test_preds  = pd.DataFrame(np.zeros((len(test), len(TARGETS))), columns=TARGETS)
rmse_log    = {}

print("\n" + "="*60)
print("STAGE 1 — Per-target LightGBM with spatial GroupKFold")
print("="*60)

for target in TARGETS:
    if target not in train.columns:
        print(f"  {target}: not in train — skipping")
        continue

    # Rows with non-null target
    valid_mask = train[target].notna()
    X_t = X_train_full[valid_mask]
    y_t = train.loc[valid_mask, target].values
    g_t = groups[valid_mask]

    # Log transform
    if target in LOG_TARGETS:
        y_t_fit = np.log1p(np.clip(y_t, 0, None))
    else:
        y_t_fit = y_t.copy()

    oof_target   = np.full(valid_mask.sum(), np.nan)
    fold_models  = []
    test_fold_preds = np.zeros((len(test), N_FOLDS))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_t, y_t_fit, groups=g_t)):
        X_tr, X_val = X_t.iloc[tr_idx], X_t.iloc[val_idx]
        y_tr, y_val = y_t_fit[tr_idx], y_t_fit[val_idx]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(-1)]
        )

        val_pred = model.predict(X_val)
        oof_target[val_idx] = val_pred
        test_fold_preds[:, fold] = model.predict(X_test)
        fold_models.append(model)

    # Inverse-transform OOF
    if target in LOG_TARGETS:
        oof_raw = np.expm1(oof_target)
    else:
        oof_raw = oof_target.copy()

    oof_preds.loc[valid_mask, target] = oof_raw

    # RMSE on original scale
    rmse = np.sqrt(mean_squared_error(y_t, np.clip(oof_raw, 0, None)))
    rmse_log[target] = rmse
    print(f"  {target:4s}: OOF RMSE = {rmse:10.4f}  |  "
          f"n_train = {len(y_t):,}  |  "
          f"best_iter = {fold_models[-1].best_iteration_}")

    # Test predictions: mean over folds, inverse-transform
    test_mean = test_fold_preds.mean(axis=1)
    if target in LOG_TARGETS:
        test_mean = np.expm1(test_mean)
    test_preds[target] = np.clip(test_mean, 0, None)

    # Save models
    with open(MODELS / f"lgb_{target}_folds.pkl", 'wb') as fh:
        pickle.dump(fold_models, fh)

# Summary
print("\n" + "-"*40)
mean_rmse = np.mean(list(rmse_log.values()))
print(f"Mean OOF RMSE across targets: {mean_rmse:.4f}")
print("-"*40)
for t, r in sorted(rmse_log.items(), key=lambda x: -x[1]):
    print(f"  {t:4s}: {r:.4f}")

# ── Save Stage-1 OOF and test predictions ─────────────────────────────────────
oof_preds.to_csv(PROC / "oof_stage1.csv", index=False)
test_preds.to_csv(PROC / "test_preds_stage1.csv", index=False)
pd.DataFrame({'target': list(rmse_log.keys()), 'rmse': list(rmse_log.values())}) \
    .to_csv(PROC / "oof_rmse_stage1.csv", index=False)
print("\n✅ Stage-1 complete.  Saved oof_stage1.csv, test_preds_stage1.csv")

# ── Stage-2: Cross-target features ───────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 2 — Re-train with OOF cross-target features")
print("="*60)

# Append OOF nutrient predictions as new features
cross_feats = [f'oof_{t}' for t in TARGETS if t in oof_preds.columns]
oof_renamed = oof_preds.copy()
oof_renamed.columns = [f'oof_{c}' for c in oof_renamed.columns]

X_train_s2 = pd.concat([X_train_full.reset_index(drop=True),
                         oof_renamed.reset_index(drop=True)], axis=1)

# Test gets stage-1 test predictions as cross-features
test_oof_like = test_preds.copy()
test_oof_like.columns = [f'oof_{c}' for c in test_oof_like.columns]
X_test_s2 = pd.concat([X_test.reset_index(drop=True),
                        test_oof_like.reset_index(drop=True)], axis=1)

# Fill NaN in the new columns too
new_medians = X_train_s2[cross_feats].median()
X_train_s2[cross_feats] = X_train_s2[cross_feats].fillna(new_medians)
X_test_s2[cross_feats]  = X_test_s2[cross_feats].fillna(new_medians)

FEATURE_COLS_S2 = FEATURE_COLS + cross_feats
test_preds_s2 = pd.DataFrame(np.zeros((len(test), len(TARGETS))), columns=TARGETS)
rmse_log_s2   = {}

for target in TARGETS:
    if target not in train.columns:
        # Remove own OOF to avoid leakage
        feat_s2 = [c for c in FEATURE_COLS_S2 if c != f'oof_{target}']

        valid_mask = train[target].notna()
        X_t = X_train_s2.loc[valid_mask, feat_s2]
        y_t = train.loc[valid_mask, target].values
        g_t = groups[valid_mask]

        if target in LOG_TARGETS:
            y_t_fit = np.log1p(np.clip(y_t, 0, None))
        else:
            y_t_fit = y_t.copy()

        oof_target       = np.full(valid_mask.sum(), np.nan)
        fold_models_s2   = []
        test_fold_preds  = np.zeros((len(test), N_FOLDS))

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_t, y_t_fit, groups=g_t)):
            X_tr, X_val = X_t.iloc[tr_idx], X_t.iloc[val_idx]
            y_tr, y_val = y_t_fit[tr_idx], y_t_fit[val_idx]

            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(-1)])

            val_pred = model.predict(X_val)
            oof_target[val_idx] = val_pred
            test_fold_preds[:, fold] = model.predict(X_test_s2[feat_s2])
            fold_models_s2.append(model)

        if target in LOG_TARGETS:
            oof_raw = np.expm1(oof_target)
        else:
            oof_raw = oof_target.copy()

        rmse = np.sqrt(mean_squared_error(y_t, np.clip(oof_raw, 0, None)))
        rmse_log_s2[target] = rmse

        test_mean = test_fold_preds.mean(axis=1)
        if target in LOG_TARGETS:
            test_mean = np.expm1(test_mean)
        test_preds_s2[target] = np.clip(test_mean, 0, None)

        with open(MODELS / f"lgb_s2_{target}_folds.pkl", 'wb') as fh:
            pickle.dump(fold_models_s2, fh)

        print(f"  {target:4s}: OOF RMSE = {rmse:10.4f}")
    else:
        # Nutrient not in train at all — keep stage-1 predictions
        if target in test_preds.columns:
            test_preds_s2[target] = test_preds[target]
        print(f"  {target:4s}: not in train, using Stage-1 preds")

mean_rmse_s2 = np.mean([v for v in rmse_log_s2.values()])
print(f"\nMean OOF RMSE Stage-2: {mean_rmse_s2:.4f}  (Stage-1: {mean_rmse:.4f})")

test_preds_s2.to_csv(PROC / "test_preds_stage2.csv", index=False)
pd.DataFrame({'target': list(rmse_log_s2.keys()), 'rmse': list(rmse_log_s2.values())}) \
    .to_csv(PROC / "oof_rmse_stage2.csv", index=False)

print("\n✅ Stage-2 complete.  Saved test_preds_stage2.csv")
