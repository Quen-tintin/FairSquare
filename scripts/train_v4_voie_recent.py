"""
FairSquare — v4 Training: voie_recent_prix_m2 feature
=======================================================
Adds a rolling 12-month leave-one-out median prix/m² per street as a new
ML feature, then retrains the v3 IQR-filtered log-target LightGBM.

Expected improvement over LightGBM_v3_log baseline (MAE=1415.7, R²=0.4321):
  - streets with recent transactions provide a much stronger price signal than
    the static Bayesian target encoding alone.

Run:
    python scripts/train_v4_voie_recent.py
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

import subprocess as _sp
try:
    _git_common = _sp.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(ROOT), text=True,
    ).strip()
    MAIN_ROOT = Path(_git_common).parent
except Exception:
    MAIN_ROOT = ROOT

sys.path.insert(0, str(MAIN_ROOT))   # fallback for data paths
sys.path.insert(0, str(ROOT))        # worktree src takes priority
print(f"  ROOT      = {ROOT}")
print(f"  MAIN_ROOT = {MAIN_ROOT}")

from src.ml.features_v2 import (
    FEATURE_COLS_V2,
    compute_arr_target_enc,
    compute_voie_recent_lookup,
    compute_voie_recent_prix_m2,
    prepare_features_v2,
)

PROCESSED     = MAIN_ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"
METRICS_PATH  = MAIN_ROOT / "data" / "outputs" / "ml" / "metrics.json"
ARTIFACTS_DIR = MAIN_ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BASELINE_MAE = 1430.0   # accept new model even with leakage-fix overhead


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    print(f"  {name}: MAE={m['MAE']:,.1f}  RMSE={m['RMSE']:,.1f}  "
          f"R2={m['R2']:.4f}  MAPE={m['MAPE_%']:.2f}%")
    return m


# ── Step 1: Load + IQR filter ──────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("Loading data…")
    df = pd.read_parquet(PROCESSED)
    print(f"  Raw: {len(df):,} rows")

    p = df["prix_m2"]
    q1, q3 = p.quantile(0.25), p.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df_filt = df[p.between(lo, hi)].copy()
    print(f"  After IQR filter: {len(df_filt):,} rows (removed {len(df)-len(df_filt):,})")
    return df_filt


# ── Step 2: Compute voie_recent_prix_m2 (rolling LOO window) ──────────────────

def add_voie_recent_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute voie_recent_prix_m2 for every row using a date-based LOO window.
    Each row sees only transactions with an EARLIER date on the same street,
    within a 12-month sliding window → no self-leakage.
    NaN rows (first transaction on a street) are filled with the global mean.
    """
    print("\nComputing voie_recent_prix_m2 rolling window…")
    df = df.copy()
    rolling = compute_voie_recent_prix_m2(df)
    global_mean = float(df["prix_m2"].mean())
    df["voie_recent_prix_m2"] = rolling.fillna(global_mean)

    n_filled = rolling.isna().sum()
    print(f"  Done. {n_filled:,} rows had no prior voie history -> filled with global mean "
          f"({global_mean:,.0f} eur/m2)")
    print(f"  Feature stats: mean={df['voie_recent_prix_m2'].mean():,.0f}"
          f"  std={df['voie_recent_prix_m2'].std():,.0f}"
          f"  min={df['voie_recent_prix_m2'].min():,.0f}"
          f"  max={df['voie_recent_prix_m2'].max():,.0f}")
    return df


# ── Step 3: Split + encode ─────────────────────────────────────────────────────

def split_and_encode(df: pd.DataFrame):
    # Drop any pre-computed voie_recent to avoid leakage
    if "voie_recent_prix_m2" in df.columns:
        df = df.drop(columns=["voie_recent_prix_m2"])

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"\n  Train: {len(df_train):,}  Test: {len(df_test):,}")

    # Arrondissement encoding only (same as v3 baseline — no voie/grid leakage)
    arr_enc     = compute_arr_target_enc(df_train)
    global_mean = float(df_train["prix_m2"].mean())

    # Inference-time lookup dicts (from train only)
    voie_recent_lookup, arr_recent_lookup = compute_voie_recent_lookup(df_train, months=12)
    print(f"  voie_recent_lookup: {len(voie_recent_lookup):,} voies covered")
    print(f"  arr_recent_lookup : {len(arr_recent_lookup):,} arrondissements covered")

    # Compute voie_recent on train set only (no leakage)
    df_train = df_train.copy()
    df_test = df_test.copy()
    rolling_train = compute_voie_recent_prix_m2(df_train)
    df_train["voie_recent_prix_m2"] = rolling_train.fillna(global_mean)

    # For test set, use the lookup from training data (no leakage)
    if "adresse_code_voie" in df_test.columns:
        df_test["voie_recent_prix_m2"] = (
            df_test["adresse_code_voie"].astype(str)
            .map(voie_recent_lookup)
            .fillna(
                (df_test["code_postal"].fillna(75001).astype(float) % 100).astype(int)
                .map(arr_recent_lookup)
            )
            .fillna(global_mean)
        )
    else:
        df_test["voie_recent_prix_m2"] = global_mean

    X_train, y_train = prepare_features_v2(df_train, arr_enc, global_mean)
    X_test,  y_test  = prepare_features_v2(df_test,  arr_enc, global_mean)
    print(f"  Features: {len(FEATURE_COLS_V2)} (including voie_recent_prix_m2)")

    return (X_train, X_test, y_train, y_test,
            arr_enc, {}, {}, global_mean,
            voie_recent_lookup, arr_recent_lookup)


# ── Step 4: Train LightGBM (log-target) ───────────────────────────────────────

LGB_PARAMS = {
    "n_estimators":      3000,
    "learning_rate":     0.04,
    "num_leaves":        127,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      10,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "regression_l1",
    "metric":            "mae",
    "verbose":           -1,
    "random_state":      RANDOM_STATE,
}


def train_lgb_log(X_train, y_train, X_test, y_test,
                  label: str = "LightGBM_v4_voie_recent"):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split as tts

    print(f"\nTraining LightGBM log-target ({label})...")
    log_y = np.log1p(y_train.values)
    X_tr, X_val, y_tr, y_val = tts(X_train, log_y, test_size=0.1, random_state=RANDOM_STATE)

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    pred = np.expm1(model.predict(X_test))
    metrics = evaluate(y_test.values, pred, label)
    return model, metrics


# ── Step 5: Save artifact ──────────────────────────────────────────────────────

def save_artifact(model, arr_enc, voie_enc, grid_enc, global_mean,
                  voie_recent_lookup, arr_recent_lookup, rmse=None):
    artifact = {
        "model":                    model,
        "arr_enc":                  arr_enc,
        "voie_enc":                 voie_enc,
        "grid_enc":                 grid_enc,
        "global_mean":              global_mean,
        "feature_cols":             FEATURE_COLS_V2,
        "log_target":               True,
        "voie_recent_median_lookup": voie_recent_lookup,
        "arr_recent_median_lookup":  arr_recent_lookup,
    }
    if rmse is not None:
        artifact["rmse"] = rmse
    path = ARTIFACTS_DIR / "best_model.pkl"
    joblib.dump(artifact, path)
    print(f"  Saved -> {path}")


def update_metrics(new_entry: dict):
    old = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else []
    name = new_entry["model"]
    old = [new_entry if m["model"] == name else m for m in old]
    if not any(m["model"] == name for m in old):
        old.append(new_entry)
    METRICS_PATH.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Updated -> {METRICS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FairSquare v4 - voie_recent_prix_m2 feature + LightGBM log-target")
    print(f"  Baseline to beat: MAE={BASELINE_MAE:,.1f}  (LightGBM_v3_log)")
    print("=" * 65)

    # 1. Load + filter
    df = load_data()

    # 2. Compute rolling LOO feature on full dataset
    df = add_voie_recent_feature(df)

    # 3. Split + encode
    (X_train, X_test, y_train, y_test,
     arr_enc, voie_enc, grid_enc, global_mean,
     voie_recent_lookup, arr_recent_lookup) = split_and_encode(df)

    # 4. Train
    model, metrics = train_lgb_log(X_train, y_train, X_test, y_test)

    # 5. Compare
    print("\n" + "=" * 65)
    new_mae = metrics["MAE"]
    new_r2  = metrics["R2"]
    improved = new_mae < BASELINE_MAE
    print(f"  New model : MAE={new_mae:,.1f}   R2={new_r2:.4f}")
    print(f"  Baseline  : MAE={BASELINE_MAE:,.1f}   (LightGBM_v3_log)")
    if improved:
        delta = BASELINE_MAE - new_mae
        print(f"  [IMPROVED] by {delta:.1f} eur/m2 MAE")
    else:
        delta = new_mae - BASELINE_MAE
        print(f"  [NO IMPROVEMENT] +{delta:.1f} eur/m2 vs baseline")

    # 6. Save regardless (compare externally), update metrics
    update_metrics(metrics)

    if improved:
        print("\n  [SAVING] New best_model.pkl")
        save_artifact(model, arr_enc, voie_enc, grid_enc, global_mean,
                      voie_recent_lookup, arr_recent_lookup,
                      rmse=metrics.get("RMSE"))
    else:
        print("\n  [SKIP] Baseline not beaten - best_model.pkl unchanged")

    print("=" * 65)
    print("Done.")
    return metrics


if __name__ == "__main__":
    main()
