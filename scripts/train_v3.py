"""
FairSquare — Model Training v3
===============================
Uses 67k transactions (2023-2025) + street-level + grid encodings.
Target: MAE < 1800, R² > 0.50

New features vs v2:
  - voie_target_enc  : street-level Bayesian target encoding (+~10% R²)
  - grid_target_enc  : 500m grid cell encoding
  - lat_sq, lon_sq, lat_lon_cross : quadratic spatial

Run:
    python scripts/train_v3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.features_v2 import (
    FEATURE_COLS_V2,
    compute_target_encodings,
    prepare_features_v2,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED     = ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"
METRICS_PATH  = ROOT / "data" / "outputs" / "ml" / "metrics.json"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE    = 42
N_OPTUNA_TRIALS = 80   # more trials on larger dataset
CV_FOLDS        = 5


# ── Evaluation ────────────────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    m = {
        "model":  name,
        "MAE":    round(float(mean_absolute_error(y_true, y_pred)), 1),
        "RMSE":   round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "R2":     round(float(r2_score(y_true, y_pred)), 4),
        "MAPE_%": round(_mape(y_true, y_pred), 2),
    }
    print(f"  {name}: MAE={m['MAE']:,.1f}  RMSE={m['RMSE']:,.1f}  "
          f"R²={m['R2']:.4f}  MAPE={m['MAPE_%']:.2f}%")
    return m


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_split():
    print("Loading data…")
    df = pd.read_parquet(PROCESSED)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"  Train: {len(df_train):,}  Test: {len(df_test):,}")

    # Compute ALL target encodings from train split only (no leakage)
    print("  Computing target encodings (arr + street + grid)…")
    encs = compute_target_encodings(df_train)
    print(f"  Streets encoded: {len(encs['voie_enc']):,}  Grid cells: {len(encs['grid_enc']):,}")

    X_train, y_train = prepare_features_v2(
        df_train,
        arr_target_enc=encs["arr_enc"],
        global_mean=encs["global_mean"],
        voie_enc=encs["voie_enc"],
        grid_enc=encs["grid_enc"],
    )
    X_test, y_test = prepare_features_v2(
        df_test,
        arr_target_enc=encs["arr_enc"],
        global_mean=encs["global_mean"],
        voie_enc=encs["voie_enc"],
        grid_enc=encs["grid_enc"],
    )

    print(f"  Features: {len(FEATURE_COLS_V2)}")
    return X_train, X_test, y_train, y_test, encs


# ── Optuna objective (LightGBM) ───────────────────────────────────────────────

def lgb_objective(trial, X_cv: pd.DataFrame, y_cv: pd.Series) -> float:
    import lightgbm as lgb

    params = {
        "objective":         "regression_l1",
        "metric":            "mae",
        "n_estimators":      trial.suggest_int("n_estimators", 400, 3000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 512),
        "max_depth":         trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      1,
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "verbose":           -1,
        "random_state":      RANDOM_STATE,
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    maes = []
    for tr_idx, val_idx in kf.split(X_cv):
        X_tr, X_val = X_cv.iloc[tr_idx], X_cv.iloc[val_idx]
        y_tr, y_val = y_cv.iloc[tr_idx], y_cv.iloc[val_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(60, verbose=False),
                         lgb.log_evaluation(-1)])
        maes.append(mean_absolute_error(y_val, m.predict(X_val)))

    return float(np.mean(maes))


def train_lightgbm(X_train, y_train, X_test, y_test) -> tuple[object, dict]:
    import lightgbm as lgb

    print(f"\n[LightGBM] Optuna search ({N_OPTUNA_TRIALS} trials, {CV_FOLDS}-fold CV)…")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: lgb_objective(t, X_train, y_train),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    best = study.best_params
    print(f"  Best CV MAE: {study.best_value:,.1f}  params: {best}")

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        verbose=-1,
        random_state=RANDOM_STATE,
        **best,
    )
    model.fit(X_train, y_train)
    metrics = evaluate(y_test.values, model.predict(X_test), "LightGBM_v3")
    return model, metrics


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test) -> tuple[object, dict]:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  XGBoost not installed — skipping")
        return None, {}

    print("\n[XGBoost] Training…")
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:absoluteerror",
        eval_metric="mae",
        random_state=RANDOM_STATE,
        verbosity=0,
        early_stopping_rounds=60,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    metrics = evaluate(y_test.values, model.predict(X_test), "XGBoost_v3")
    return model, metrics


# ── CatBoost ──────────────────────────────────────────────────────────────────

def train_catboost(X_train, y_train, X_test, y_test) -> tuple[object, dict]:
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("  CatBoost not installed — skipping")
        return None, {}

    print("\n[CatBoost] Training…")
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=60,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    metrics = evaluate(y_test.values, model.predict(X_test), "CatBoost_v3")
    return model, metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FairSquare — Model Training v3")
    print("  Dataset:  2023-2025 (67k rows)")
    print("  New feat: street enc + grid enc + quad spatial")
    print("=" * 60)

    X_train, X_test, y_train, y_test, encs = load_and_split()

    print("\n[1/3] LightGBM with Optuna")
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)

    print("\n[2/3] XGBoost")
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)

    print("\n[3/3] CatBoost")
    cat_model, cat_metrics = train_catboost(X_train, y_train, X_test, y_test)

    # ── Ensemble ──────────────────────────────────────────────────────
    active_models = [(lgb_model, lgb_metrics), (xgb_model, xgb_metrics), (cat_model, cat_metrics)]
    active_models = [(m, met) for m, met in active_models if m is not None]

    ens_metrics = {}
    if len(active_models) > 1:
        print("\n[Ensemble] Averaging all models…")
        ens_pred = np.mean([m.predict(X_test) for m, _ in active_models], axis=0)
        ens_metrics = evaluate(y_test.values, ens_pred, "Ensemble_v3")

    # ── Pick best ─────────────────────────────────────────────────────
    all_results = {
        "LightGBM_v3": (lgb_model, lgb_metrics),
        "XGBoost_v3":  (xgb_model,  xgb_metrics),
        "CatBoost_v3": (cat_model,  cat_metrics),
    }
    if ens_metrics:
        all_results["Ensemble_v3"] = ([m for m, _ in active_models], ens_metrics)

    best_name = min(
        {k: v for k, v in all_results.items() if v[0] is not None},
        key=lambda k: all_results[k][1].get("MAE", 9999),
    )
    best_obj, best_met = all_results[best_name]
    print(f"\nBest model: {best_name}  MAE={best_met['MAE']:,.1f}  R²={best_met['R2']:.4f}")

    # ── Save artifact ──────────────────────────────────────────────────
    artifact = {
        "model":        best_obj,
        "arr_enc":      encs["arr_enc"],
        "voie_enc":     encs["voie_enc"],
        "grid_enc":     encs["grid_enc"],
        "global_mean":  encs["global_mean"],
        "feature_cols": FEATURE_COLS_V2,
    }
    path = ARTIFACTS_DIR / "best_model.pkl"
    joblib.dump(artifact, path)
    print(f"Saved → {path}")

    # ── Update metrics.json ────────────────────────────────────────────
    old = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else []
    old_names = {m["model"] for m in old}

    new_metrics = [m for _, m in active_models if m]
    if ens_metrics:
        new_metrics.append(ens_metrics)

    for nm in new_metrics:
        if nm.get("model") not in old_names:
            old.append(nm)
        else:
            old = [nm if m["model"] == nm["model"] else m for m in old]

    METRICS_PATH.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated → {METRICS_PATH}")

    print("\n" + "=" * 60)
    target_ok = "✓" if best_met["MAE"] < 1800 else "✗"
    r2_ok     = "✓" if best_met["R2"]  > 0.50 else "✗"
    print(f"MAE < 1800 : {target_ok}  ({best_met['MAE']:.0f})")
    print(f"R²  > 0.50 : {r2_ok}   ({best_met['R2']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
