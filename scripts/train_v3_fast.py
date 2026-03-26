"""
FairSquare — Model Training v3 (fast, no Optuna)
=================================================
Fixed params calibrated from v2 Optuna runs.
New features: voie_target_enc, grid_target_enc, lat2/lon2/latxlon.
Dataset: 2023-2025 (67k rows).

Run: python scripts/train_v3_fast.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.features_v2 import (
    FEATURE_COLS_V2,
    compute_target_encodings,
    prepare_features_v2,
)

PROCESSED     = ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"
METRICS_PATH  = ROOT / "data" / "outputs" / "ml" / "metrics.json"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# -- Helpers ----------------------------------------------------------

def _mape(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred, name):
    m = {
        "model":  name,
        "MAE":    round(float(mean_absolute_error(y_true, y_pred)), 1),
        "RMSE":   round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "R2":     round(float(r2_score(y_true, y_pred)), 4),
        "MAPE_%": round(_mape(y_true, y_pred), 2),
    }
    print(f"  {name:20s}  MAE={m['MAE']:7,.1f}  R2={m['R2']:.4f}  MAPE={m['MAPE_%']:.2f}%")
    return m


# -- Data -------------------------------------------------------------

def load_and_split():
    print("Loading 2023-2025 dataset…")
    df = pd.read_parquet(PROCESSED)
    print(f"  {len(df):,} rows · {df.shape[1]} cols")

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    print("  Computing target encodings (arr + street + grid)…")
    encs = compute_target_encodings(df_train)
    print(f"  arr: 20  street: {len(encs['voie_enc']):,}  grid: {len(encs['grid_enc']):,}")

    kw = dict(
        arr_target_enc=encs["arr_enc"],
        global_mean=encs["global_mean"],
        voie_enc=encs["voie_enc"],
        grid_enc=encs["grid_enc"],
    )
    X_tr, y_tr = prepare_features_v2(df_train, **kw)
    X_te, y_te = prepare_features_v2(df_test,  **kw)
    print(f"  Train: {len(X_tr):,} · Test: {len(X_te):,} · Features: {len(FEATURE_COLS_V2)}")
    return X_tr, X_te, y_tr, y_te, encs


# -- Models -----------------------------------------------------------

def train_lgb(X_tr, y_tr, X_te, y_te, X_val, y_val):
    import lightgbm as lgb
    print("\n[LightGBM] training…")

    # Params calibrated from v2 Optuna best + extra capacity for larger dataset
    params = dict(
        objective="regression_l1",
        metric="mae",
        n_estimators=3000,          # high ceiling — early stopping will cap it
        learning_rate=0.03,
        num_leaves=255,
        max_depth=10,
        min_child_samples=20,
        feature_fraction=0.75,
        bagging_fraction=0.80,
        bagging_freq=1,
        reg_alpha=0.05,
        reg_lambda=0.1,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    return model, evaluate(y_te.values, model.predict(X_te), "LightGBM_v3")


def train_xgb(X_tr, y_tr, X_te, y_te, X_val, y_val):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  XGBoost not installed — skip"); return None, {}

    print("\n[XGBoost] training…")
    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=10,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.5,
        objective="reg:absoluteerror",
        eval_metric="mae",
        early_stopping_rounds=80,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {model.best_iteration}")
    return model, evaluate(y_te.values, model.predict(X_te), "XGBoost_v3")


def train_cat(X_tr, y_tr, X_te, y_te, X_val, y_val):
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("  CatBoost not installed — skip"); return None, {}

    print("\n[CatBoost] training…")
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        loss_function="MAE",
        eval_metric="MAE",
        early_stopping_rounds=80,
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    print(f"  Best iteration: {model.best_iteration_}")
    return model, evaluate(y_te.values, model.predict(X_te), "CatBoost_v3")


# -- Main -------------------------------------------------------------

def main():
    print("=" * 62)
    print("FairSquare  —  Training v3  (fixed params, fast)")
    print("  Features: arr + street + 500m-grid + lat2 + lon2 + latxlon")
    print("  Dataset : 2023-2025 (67 k rows)")
    print("=" * 62)

    X_tr, X_te, y_tr, y_te, encs = load_and_split()

    # Single val split for early stopping (10 % of training data)
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.10, random_state=RANDOM_STATE
    )

    print("\n--- Model Results -------------------------------------------")
    lgb_m, lgb_met = train_lgb(X_tr2, y_tr2, X_te, y_te, X_val, y_val)
    xgb_m, xgb_met = train_xgb(X_tr2, y_tr2, X_te, y_te, X_val, y_val)
    cat_m, cat_met = train_cat(X_tr2, y_tr2, X_te, y_te, X_val, y_val)

    # Retrain winners on FULL training data (no val split)
    # LightGBM re-fit at best_iteration on full X_tr
    import lightgbm as lgb
    lgb_final_params = dict(
        objective="regression_l1", metric="mae",
        n_estimators=lgb_m.best_iteration_,
        learning_rate=0.03, num_leaves=255, max_depth=10,
        min_child_samples=20, feature_fraction=0.75,
        bagging_fraction=0.80, bagging_freq=1,
        reg_alpha=0.05, reg_lambda=0.1,
        verbose=-1, random_state=RANDOM_STATE,
    )
    print("\n  Refitting LightGBM on full train…")
    lgb_full = lgb.LGBMRegressor(**lgb_final_params)
    lgb_full.fit(X_tr, y_tr)

    # Ensemble (all non-None models, using final LGB)
    active = [(lgb_full, "LGB"), (xgb_m, "XGB"), (cat_m, "CAT")]
    active = [(m, n) for m, n in active if m is not None]

    print("\n--- Ensemble ------------------------------------------------")
    ens_pred = np.mean([m.predict(X_te) for m, _ in active], axis=0)
    ens_met  = evaluate(y_te.values, ens_pred, "Ensemble_v3")

    # -- Summary -------------------------------------------------------
    print("\n--- Summary -------------------------------------------------")
    all_results = {}
    if lgb_met:  all_results["LightGBM_v3"] = (lgb_full,  lgb_met)
    if xgb_met:  all_results["XGBoost_v3"]  = (xgb_m,    xgb_met)
    if cat_met:  all_results["CatBoost_v3"] = (cat_m,    cat_met)
    if ens_met:  all_results["Ensemble_v3"] = ([m for m, _ in active], ens_met)

    best_name = min(all_results, key=lambda k: all_results[k][1].get("MAE", 9999))
    best_obj, best_met = all_results[best_name]
    print(f"\n  Best: {best_name}  ->  MAE={best_met['MAE']:,.1f}  R2={best_met['R2']:.4f}")

    mae_ok = best_met["MAE"] < 1800
    r2_ok  = best_met["R2"]  > 0.50
    print(f"  MAE < 1800 : {'OK' if mae_ok else 'FAIL'}  ({best_met['MAE']:.0f})")
    print(f"  R2  > 0.50 : {'OK' if r2_ok  else 'FAIL'}  ({best_met['R2']:.4f})")

    # -- Save artifact -------------------------------------------------
    artifact = {
        "model":        best_obj,
        "arr_enc":      encs["arr_enc"],
        "voie_enc":     encs["voie_enc"],
        "grid_enc":     encs["grid_enc"],
        "global_mean":  encs["global_mean"],
        "feature_cols": FEATURE_COLS_V2,
    }
    artifact_path = ARTIFACTS_DIR / "best_model.pkl"
    joblib.dump(artifact, artifact_path)
    print(f"\n  Saved -> {artifact_path}")

    # -- Update metrics.json -------------------------------------------
    old = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else []
    old_names = {m["model"] for m in old}
    new_entries = [m for m in [lgb_met, xgb_met, cat_met, ens_met] if m]
    for nm in new_entries:
        if nm.get("model") not in old_names:
            old.append(nm)
        else:
            old = [nm if m["model"] == nm["model"] else m for m in old]
    METRICS_PATH.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Updated -> {METRICS_PATH}")

    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()
