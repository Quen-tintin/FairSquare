"""
FairSquare — Improved ML Training Pipeline
==========================================
Uses Optuna to tune LightGBM, then trains XGBoost for an ensemble.
Target: MAE < 1800, R² > 0.50

Run:
    python scripts/train_improved_model.py
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
    compute_arr_target_enc,
    prepare_features_v2,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED     = ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"
METRICS_PATH  = ROOT / "data" / "outputs" / "ml" / "metrics.json"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_OPTUNA_TRIALS = 60
CV_FOLDS = 5


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    m = {
        "model":   name,
        "MAE":     round(float(mean_absolute_error(y_true, y_pred)), 1),
        "RMSE":    round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "R2":      round(float(r2_score(y_true, y_pred)), 4),
        "MAPE_%":  round(_mape(y_true, y_pred), 2),
    }
    print(f"  {name}: MAE={m['MAE']:,.1f}  RMSE={m['RMSE']:,.1f}  "
          f"R²={m['R2']:.4f}  MAPE={m['MAPE_%']:.2f}%")
    return m


# ── Data loading + splitting ───────────────────────────────────────────────────

def load_and_split():
    print("Loading data…")
    df = pd.read_parquet(PROCESSED)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    # Target encoding — computed from train split ONLY (no leakage)
    arr_enc = compute_arr_target_enc(df_train)
    global_mean = float(df_train["prix_m2"].mean())

    X_train, y_train = prepare_features_v2(df_train, arr_enc, global_mean)
    X_test,  y_test  = prepare_features_v2(df_test,  arr_enc, global_mean)

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {len(FEATURE_COLS_V2)}")
    return X_train, X_test, y_train, y_test, arr_enc, global_mean


# ── Optuna objective (LightGBM) ──────────────────────────────────────────────

def lgb_objective(trial, X_cv: pd.DataFrame, y_cv: pd.Series) -> float:
    import lightgbm as lgb

    params = {
        "objective":          "regression_l1",  # optimise MAE directly
        "metric":             "mae",
        "n_estimators":       trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples":  trial.suggest_int("min_child_samples", 5, 80),
        "feature_fraction":   trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":   trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":       1,
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "verbose":            -1,
        "random_state":       RANDOM_STATE,
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    maes = []
    for tr_idx, val_idx in kf.split(X_cv):
        X_tr, X_val = X_cv.iloc[tr_idx], X_cv.iloc[val_idx]
        y_tr, y_val = y_cv.iloc[tr_idx], y_cv.iloc[val_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        maes.append(mean_absolute_error(y_val, m.predict(X_val)))

    return float(np.mean(maes))


# ── LightGBM training ────────────────────────────────────────────────────────

def train_lightgbm(X_train, y_train, X_test, y_test) -> tuple[object, dict]:
    import lightgbm as lgb

    print(f"\nOptuna LightGBM ({N_OPTUNA_TRIALS} trials, {CV_FOLDS}-fold CV)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: lgb_objective(t, X_train, y_train),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    best = study.best_params
    print(f"  Best CV MAE: {study.best_value:,.1f}  params: {best}")

    # Re-train on full train split with best params
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        verbose=-1,
        random_state=RANDOM_STATE,
        **best,
    )
    model.fit(X_train, y_train)
    metrics = evaluate(y_test.values, model.predict(X_test), "LightGBM_v2")
    return model, metrics


# ── XGBoost training ─────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test) -> tuple[object, dict]:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  XGBoost not installed — skipping")
        return None, {}

    print("\nTraining XGBoost…")
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:absoluteerror",
        eval_metric="mae",
        random_state=RANDOM_STATE,
        verbosity=0,
        early_stopping_rounds=50,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    metrics = evaluate(y_test.values, model.predict(X_test), "XGBoost_v2")
    return model, metrics


# ── Ensemble ─────────────────────────────────────────────────────────────────

def ensemble_predict(models: list, X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict(X) for m in models if m is not None]
    return np.mean(preds, axis=0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FairSquare — Improved Model Training")
    print("=" * 60)

    X_train, X_test, y_train, y_test, arr_enc, global_mean = load_and_split()

    print("\n[1/3] LightGBM with Optuna")
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)

    print("\n[2/3] XGBoost")
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)

    # ── Ensemble ──────────────────────────────────────────────────────
    print("\n[3/3] Ensemble (LGB + XGB average)")
    active_models = [m for m in [lgb_model, xgb_model] if m is not None]
    if len(active_models) > 1:
        ens_pred = ensemble_predict(active_models, X_test)
        ens_metrics = evaluate(y_test.values, ens_pred, "Ensemble_v2")
    else:
        ens_metrics = {}

    # ── Save best model ────────────────────────────────────────────────
    # Pick model with lowest MAE
    candidates = {
        "lightgbm_v2": (lgb_model, lgb_metrics.get("MAE", 9999)),
    }
    if xgb_model is not None:
        candidates["xgboost_v2"] = (xgb_model, xgb_metrics.get("MAE", 9999))
    if ens_metrics:
        candidates["ensemble_v2"] = (active_models, ens_metrics.get("MAE", 9999))

    best_name = min(candidates, key=lambda k: candidates[k][1])
    best_obj  = candidates[best_name][0]
    best_mae  = candidates[best_name][1]

    print(f"\nBest model: {best_name} (MAE={best_mae:,.1f})")

    # Save artifact
    artifact = {
        "model":       best_obj,
        "arr_enc":     arr_enc,
        "global_mean": global_mean,
        "feature_cols": FEATURE_COLS_V2,
    }
    artifact_path = ARTIFACTS_DIR / "best_model.pkl"
    joblib.dump(artifact, artifact_path)
    print(f"Saved -> {artifact_path}")

    # ── Update metrics.json (keep old + add new) ──────────────────────
    old_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else []
    old_names   = {m["model"] for m in old_metrics}

    new_metrics = [m for m in [lgb_metrics, xgb_metrics, ens_metrics] if m]
    for nm in new_metrics:
        if nm.get("model") not in old_names:
            old_metrics.append(nm)
        else:
            # Overwrite with updated result
            old_metrics = [nm if m["model"] == nm["model"] else m for m in old_metrics]

    METRICS_PATH.write_text(json.dumps(old_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated -> {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
