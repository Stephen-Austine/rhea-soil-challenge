"""
03_train.py — Train one XGBoost model per nutrient target
Rhea Soil Nutrient Prediction Challenge
Uses XGBoost instead of LightGBM for Mac stability.
"""

import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROC   = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
SEED    = 42

TARGETS     = ['Al','B','Ca','Cu','Fe','K','Mg','Mn','N','Na','P','S','Zn']
LOG_TARGETS = {'Al','B','Cu','Fe','Mn','Na','P','Zn','S','K','Mg','Ca','N'}

XGB_PARAMS = dict(
    objective        = 'reg:squarederror',
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    min_child_weight = 20,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = SEED,
    n_jobs           = 2,
    tree_method      = 'hist',   # fast histogram method, low memory
    verbosity        = 0,
    early_stopping_rounds = 50,
)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading processed features …")
train = pd.read_csv(PROC / "train_features.csv")
test  = pd.read_csv(PROC / "test_features.csv")

with open(PROC / "feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

FEATURE_COLS = [c for c in FEATURE_COLS if c in train.columns and c in test.columns]
print(f"Using {len(FEATURE_COLS)} feature columns")

X_train_full = train[FEATURE_COLS].copy()
X_test       = test[FEATURE_COLS].copy()

medians = X_train_full.median()
X_train_full = X_train_full.fillna(medians)
X_test       = X_test.fillna(medians)
medians.to_csv(PROC / "feature_medians.csv")

# Spatial groups
if 'lat_block' in train.columns:
    train['spatial_block'] = train['lat_block'].astype(str) + '_' + train['lon_block'].astype(str)
else:
    train['spatial_block'] = (train.index // 100).astype(str)
groups = train['spatial_block']

gkf = GroupKFold(n_splits=N_FOLDS)

# ── Stage 1 ───────────────────────────────────────────────────────────────────
oof_preds  = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
test_preds = pd.DataFrame(np.zeros((len(test), len(TARGETS))), columns=TARGETS)
rmse_log   = {}

print("\n" + "="*60)
print("STAGE 1 — Per-target XGBoost with spatial GroupKFold")
print("="*60)

for target in TARGETS:
    if target not in train.columns:
        print(f"  {target}: not in train — skipping")
        continue

    valid_mask = train[target].notna()
    X_t = X_train_full[valid_mask]
    y_t = train.loc[valid_mask, target].values
    g_t = groups[valid_mask]

    if target in LOG_TARGETS:
        y_fit = np.log1p(np.clip(y_t, 0, None))
    else:
        y_fit = y_t.copy()

    oof_target      = np.full(valid_mask.sum(), np.nan)
    fold_models     = []
    test_fold_preds = np.zeros((len(test), N_FOLDS))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_t, y_fit, groups=g_t)):
        X_tr, X_val = X_t.iloc[tr_idx], X_t.iloc[val_idx]
        y_tr, y_val = y_fit[tr_idx], y_fit[val_idx]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        val_pred = model.predict(X_val)
        oof_target[val_idx] = val_pred
        test_fold_preds[:, fold] = model.predict(X_test)
        fold_models.append(model)

    if target in LOG_TARGETS:
        oof_raw = np.expm1(oof_target)
    else:
        oof_raw = oof_target.copy()

    oof_preds.loc[valid_mask, target] = oof_raw
    rmse = np.sqrt(mean_squared_error(y_t, np.clip(oof_raw, 0, None)))
    rmse_log[target] = rmse

    best_iter = max(m.best_iteration for m in fold_models)
    print(f"  {target:4s}: OOF RMSE = {rmse:10.4f}  |  n={len(y_t):,}  |  best_iter={best_iter}")

    test_mean = test_fold_preds.mean(axis=1)
    if target in LOG_TARGETS:
        test_mean = np.expm1(test_mean)
    test_preds[target] = np.clip(test_mean, 0, None)

    with open(MODELS / f"xgb_{target}_folds.pkl", 'wb') as fh:
        pickle.dump(fold_models, fh)

print("\n" + "-"*40)
mean_rmse = np.mean(list(rmse_log.values()))
print(f"Mean OOF RMSE: {mean_rmse:.4f}")
for t, r in sorted(rmse_log.items(), key=lambda x: -x[1]):
    print(f"  {t:4s}: {r:.4f}")

oof_preds.to_csv(PROC / "oof_stage1.csv", index=False)
test_preds.to_csv(PROC / "test_preds_stage1.csv", index=False)
pd.DataFrame({'target': list(rmse_log.keys()), 'rmse': list(rmse_log.values())}) \
    .to_csv(PROC / "oof_rmse_stage1.csv", index=False)

print("\n✅ Stage-1 complete.")

# ── Stage 2 — cross-target features ──────────────────────────────────────────
print("\n" + "="*60)
print("STAGE 2 — Re-train with cross-target OOF features")
print("="*60)

oof_renamed = oof_preds.copy()
oof_renamed.columns = [f'oof_{c}' for c in oof_renamed.columns]
cross_feats = list(oof_renamed.columns)

X_train_s2 = pd.concat([X_train_full.reset_index(drop=True),
                         oof_renamed.reset_index(drop=True)], axis=1)
test_oof    = test_preds.copy()
test_oof.columns = [f'oof_{c}' for c in test_oof.columns]
X_test_s2   = pd.concat([X_test.reset_index(drop=True),
                          test_oof.reset_index(drop=True)], axis=1)

X_train_s2[cross_feats] = X_train_s2[cross_feats].fillna(X_train_s2[cross_feats].median())
X_test_s2[cross_feats]  = X_test_s2[cross_feats].fillna(X_test_s2[cross_feats].median())

FEATURE_COLS_S2  = FEATURE_COLS + cross_feats
test_preds_s2    = pd.DataFrame(np.zeros((len(test), len(TARGETS))), columns=TARGETS)
rmse_log_s2      = {}

for target in TARGETS:
    if target not in train.columns:
        if target in test_preds.columns:
            test_preds_s2[target] = test_preds[target]
        continue

    feat_s2    = [c for c in FEATURE_COLS_S2 if c != f'oof_{target}']
    valid_mask = train[target].notna()
    X_t  = X_train_s2.loc[valid_mask, feat_s2]
    y_t  = train.loc[valid_mask, target].values
    g_t  = groups[valid_mask]

    if target in LOG_TARGETS:
        y_fit = np.log1p(np.clip(y_t, 0, None))
    else:
        y_fit = y_t.copy()

    oof_target      = np.full(valid_mask.sum(), np.nan)
    fold_models_s2  = []
    test_fold_preds = np.zeros((len(test), N_FOLDS))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_t, y_fit, groups=g_t)):
        X_tr, X_val = X_t.iloc[tr_idx], X_t.iloc[val_idx]
        y_tr, y_val = y_fit[tr_idx], y_fit[val_idx]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        oof_target[val_idx] = model.predict(X_val)
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

    with open(MODELS / f"xgb_s2_{target}_folds.pkl", 'wb') as fh:
        pickle.dump(fold_models_s2, fh)

    print(f"  {target:4s}: OOF RMSE = {rmse:10.4f}")

mean_s2 = np.mean(list(rmse_log_s2.values()))
mean_s1 = np.mean(list(rmse_log.values()))
print(f"\nMean RMSE — Stage-1: {mean_s1:.4f}  Stage-2: {mean_s2:.4f}")

test_preds_s2.to_csv(PROC / "test_preds_stage2.csv", index=False)
pd.DataFrame({'target': list(rmse_log_s2.keys()), 'rmse': list(rmse_log_s2.values())}) \
    .to_csv(PROC / "oof_rmse_stage2.csv", index=False)

print("\n✅ Stage-2 complete. Run 04_make_submission.py next.")
