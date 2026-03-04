"""
05_tune_optuna.py — Hyperparameter optimisation with Optuna
Rhea Soil Nutrient Prediction Challenge

Tunes LightGBM for each target independently using spatial GroupKFold.
Run after 02_feature_engineering.py.

Usage:
    python 05_tune_optuna.py --target Al --n_trials 100
    python 05_tune_optuna.py --target all --n_trials 50
"""

import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────
PROC    = Path("data/processed")
TUNING  = Path("tuning")
TUNING.mkdir(parents=True, exist_ok=True)

TARGETS = ['Al','B','Ca','Cu','Fe','K','Mg','Mn','N','Na','P','S','Zn']
LOG_TARGETS = {'Al','B','Cu','Fe','Mn','Na','P','Zn','S','K','Mg','Ca','N'}
N_FOLDS = 5
SEED    = 42


def load_data():
    train = pd.read_csv(PROC / "train_features.csv")
    with open(PROC / "feature_cols.json") as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in train.columns]
    X = train[feature_cols].fillna(train[feature_cols].median())
    groups = (train['Latitude'] // 0.5).astype(str) + '_' + (train['Longitude'] // 0.5).astype(str)
    return train, X, groups, feature_cols


def make_objective(X, y, groups):
    gkf = GroupKFold(n_splits=N_FOLDS)

    def objective(trial):
        params = dict(
            objective         = 'regression',
            metric            = 'rmse',
            n_estimators      = trial.suggest_int('n_estimators', 500, 3000),
            learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            num_leaves        = trial.suggest_int('num_leaves', 31, 255),
            max_depth         = trial.suggest_int('max_depth', 4, 12),
            min_child_samples = trial.suggest_int('min_child_samples', 10, 100),
            subsample         = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            random_state      = SEED,
            n_jobs            = -1,
            verbose           = -1,
        )

        rmses = []
        for tr_idx, val_idx in gkf.split(X, y, groups=groups):
            model = lgb.LGBMRegressor(**params)
            model.fit(X.iloc[tr_idx], y[tr_idx],
                      eval_set=[(X.iloc[val_idx], y[val_idx])],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
            pred = model.predict(X.iloc[val_idx])
            rmses.append(mean_squared_error(y[val_idx], pred, squared=False))
        return np.mean(rmses)

    return objective


def tune_target(target, n_trials, train, X, groups):
    if target not in train.columns:
        print(f"  {target}: not in train — skip"); return

    valid  = train[target].notna()
    X_t    = X[valid]
    y_t    = train.loc[valid, target].values
    g_t    = groups[valid]

    if target in LOG_TARGETS:
        y_fit = np.log1p(np.clip(y_t, 0, None))
    else:
        y_fit = y_t.copy()

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_objective(X_t, y_fit, g_t), n_trials=n_trials,
                   show_progress_bar=True)

    best = study.best_params
    best_rmse_log = study.best_value

    # Convert back to original scale for reporting
    print(f"\n  {target}:  best log-scale RMSE = {best_rmse_log:.4f}")
    print(f"  Best params: {json.dumps(best, indent=4)}")

    out_path = TUNING / f"best_params_{target}.json"
    with open(out_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',   default='Al',
                        help='Nutrient to tune, or "all"')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials per target')
    args = parser.parse_args()

    train, X, groups, _ = load_data()

    if args.target == 'all':
        for t in TARGETS:
            print(f"\nTuning {t} …")
            tune_target(t, args.n_trials, train, X, groups)
    else:
        tune_target(args.target, args.n_trials, train, X, groups)

    print("\n✅ Tuning complete. Best params saved to tuning/")
    print("   Re-run 03_train.py after updating LGB_PARAMS with best params.")
