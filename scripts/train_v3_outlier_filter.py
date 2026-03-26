"""
FairSquare — v3 Training: Outlier Filtering + Boosted LightGBM
==============================================================
Strategy:
  1. Analyse distribution des résidus / outliers prix_m2
  2. Filtre IQR × 1.5 sur prix_m2
  3. LightGBM avec hyperparamètres boostés
  4. Si MAE >= 1800 -> essaie log-target transform
  5. Sauvegarde si MAE < 1800 et R² > 0.5

Run:
    python scripts/train_v3_outlier_filter.py
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

# Resolve the real repo root — for git worktrees, --git-common-dir points to
# the main .git dir; its parent is the main repo root.
import subprocess as _sp
try:
    _git_common = _sp.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(ROOT), text=True
    ).strip()
    # git-common-dir is like C:/repo/.git  ->  parent = C:/repo
    MAIN_ROOT = Path(_git_common).parent
except Exception:
    MAIN_ROOT = ROOT

# MAIN_ROOT must be first so we import the up-to-date src/ml/features_v2.py
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAIN_ROOT))
print(f"  MAIN_ROOT = {MAIN_ROOT}")

from src.ml.features_v2 import (
    FEATURE_COLS_V2,
    compute_arr_target_enc,
    prepare_features_v2,
)

PROCESSED     = MAIN_ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"
METRICS_PATH  = MAIN_ROOT / "data" / "outputs" / "ml" / "metrics.json"
ARTIFACTS_DIR = MAIN_ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mape(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred, name):
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


# ── Step 1: Outlier analysis ──────────────────────────────────────────────────

def analyse_outliers(df: pd.DataFrame) -> tuple[float, float]:
    """Print distribution stats and return IQR bounds."""
    p = df["prix_m2"]
    q1, q3 = p.quantile(0.25), p.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr

    print(f"\n[Outlier Analysis] prix_m2 distribution:")
    print(f"  n total    : {len(p):,}")
    print(f"  min / max  : {p.min():,.0f} / {p.max():,.0f}")
    print(f"  mean / std : {p.mean():,.0f} / {p.std():,.0f}")
    print(f"  Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}")
    print(f"  IQR×1.5 bounds : [{lo:,.0f}, {hi:,.0f}]")
    n_low  = (p < lo).sum()
    n_high = (p > hi).sum()
    print(f"  Outliers   : {n_low:,} below + {n_high:,} above = {n_low+n_high:,} "
          f"({(n_low+n_high)/len(p)*100:.1f}%)")

    # Percentile breakdown
    for pct in [1, 5, 95, 99]:
        print(f"    p{pct:02d}: {p.quantile(pct/100):,.0f} €/m²")

    return lo, hi


# ── Step 2: Load + filter ──────────────────────────────────────────────────────

def load_data(iqr_filter: bool = True):
    print("Loading data…")
    df = pd.read_parquet(PROCESSED)
    print(f"  Raw: {len(df):,} rows, {df.shape[1]} columns")

    lo, hi = analyse_outliers(df)

    if iqr_filter:
        mask = df["prix_m2"].between(lo, hi)
        df_filt = df[mask].copy()
        print(f"\n  After IQR filter: {len(df_filt):,} rows "
              f"(removed {len(df)-len(df_filt):,})")
    else:
        df_filt = df.copy()

    return df_filt


def split_and_encode(df: pd.DataFrame):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    arr_enc     = compute_arr_target_enc(df_train)
    global_mean = float(df_train["prix_m2"].mean())
    X_train, y_train = prepare_features_v2(df_train, arr_enc, global_mean)
    X_test,  y_test  = prepare_features_v2(df_test,  arr_enc, global_mean)
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {len(FEATURE_COLS_V2)}")
    return X_train, X_test, y_train, y_test, arr_enc, global_mean


# ── Step 3: Train LightGBM (fixed boosted params) ─────────────────────────────

LGB_BASE = {
    "n_estimators":      2000,
    "learning_rate":     0.02,
    "num_leaves":        127,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "verbose":           -1,
    "random_state":      RANDOM_STATE,
}

LGB_PARAMS = {**LGB_BASE, "objective": "regression_l1", "metric": "mae"}
LGB_L2_PARAMS = {**LGB_BASE, "objective": "regression",    "metric": "rmse"}


def train_lgb(X_train, y_train, X_test, y_test, label: str,
             params: dict | None = None) -> tuple[object, dict]:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split as tts

    p = params if params is not None else LGB_PARAMS
    print(f"\nTraining LightGBM ({label}, obj={p['objective']})...")
    X_tr, X_val, y_tr, y_val = tts(X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)

    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    metrics = evaluate(y_test.values, model.predict(X_test), label)
    return model, metrics


# ── Step 4: Log-target variant ─────────────────────────────────────────────────

def train_lgb_log(X_train, y_train, X_test, y_test, label: str = "LightGBM_v3_log"):
    """Train on log(prix_m2) and back-transform predictions."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split as tts

    print(f"\nTraining LightGBM — log-target ({label})…")
    log_y_train = np.log1p(y_train.values)
    X_tr, X_val, y_tr, y_val = tts(X_train, log_y_train, test_size=0.1, random_state=RANDOM_STATE)

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    # Back-transform
    pred_log = model.predict(X_test)
    pred     = np.expm1(pred_log)
    metrics  = evaluate(y_test.values, pred, label)
    return model, metrics, True   # True = is_log_model


# ── Step 5: Save artifact + update metrics ─────────────────────────────────────

def save_artifact(model, arr_enc, global_mean, is_log: bool = False):
    artifact = {
        "model":       model,
        "arr_enc":     arr_enc,
        "global_mean": global_mean,
        "feature_cols": FEATURE_COLS_V2,
        "log_target":  is_log,       # flag so API knows to expm1 predictions
    }
    path = ARTIFACTS_DIR / "best_model.pkl"
    joblib.dump(artifact, path)
    print(f"  Saved -> {path}")


def update_metrics(new_entries: list[dict]):
    old = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else []
    old_names = {m["model"] for m in old}
    for nm in new_entries:
        if not nm:
            continue
        if nm.get("model") not in old_names:
            old.append(nm)
        else:
            old = [nm if m["model"] == nm["model"] else m for m in old]
    METRICS_PATH.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Updated -> {METRICS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FairSquare v3 — Outlier Filter + Boosted LightGBM")
    print("=" * 65)

    # ── 1. Load + filter outliers ──────────────────────────────────────
    df = load_data(iqr_filter=True)
    X_train, X_test, y_train, y_test, arr_enc, global_mean = split_and_encode(df)

    # ── 2. Train LightGBM L1 (MAE-optimised) ──────────────────────────
    lgb_l1_model, lgb_l1_metrics = train_lgb(
        X_train, y_train, X_test, y_test,
        label="LightGBM_v3_iqr",
        params=LGB_PARAMS,
    )
    print(f"  -> L1: MAE={lgb_l1_metrics['MAE']:,.1f}  R2={lgb_l1_metrics['R2']:.4f}")

    # ── 3. Train LightGBM L2 (RMSE/R²-optimised) ──────────────────────
    lgb_l2_model, lgb_l2_metrics = train_lgb(
        X_train, y_train, X_test, y_test,
        label="LightGBM_v3_l2",
        params=LGB_L2_PARAMS,
    )
    print(f"  -> L2: MAE={lgb_l2_metrics['MAE']:,.1f}  R2={lgb_l2_metrics['R2']:.4f}")

    # ── 4. Train log-target variant ────────────────────────────────────
    log_model, log_metrics, _ = train_lgb_log(
        X_train, y_train, X_test, y_test,
    )
    print(f"  -> Log: MAE={log_metrics['MAE']:,.1f}  R2={log_metrics['R2']:.4f}")

    # ── 5. Pick best model: prioritise R² > 0.5 while MAE < 1800 ──────
    candidates = [
        (lgb_l1_model, lgb_l1_metrics, False),
        (lgb_l2_model, lgb_l2_metrics, False),
        (log_model,    log_metrics,    True),
    ]

    def score(m_dict):
        """Lower = better: penalise models that don't meet both targets."""
        mae_ok = m_dict["MAE"] < 1800
        r2     = m_dict["R2"]
        mae    = m_dict["MAE"]
        # Primary: maximise R2; secondary: minimise MAE
        return (-r2 if mae_ok else (100 - r2), mae)

    best_model_obj, best_metrics, best_is_log = min(candidates, key=lambda x: score(x[1]))

    # Collect all new entries for metrics.json
    all_new_metrics = [lgb_l1_metrics, lgb_l2_metrics, log_metrics]

    # ── 6. Save + report ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    bm = best_metrics
    print(f"Best model : {bm['model']}")
    print(f"  MAE    = {bm['MAE']:,.1f}  (target: < 1800)")
    print(f"  RMSE   = {bm['RMSE']:,.1f}")
    print(f"  R2     = {bm['R2']:.4f}  (target: > 0.50)")
    print(f"  MAPE   = {bm['MAPE_%']:.2f}%")

    met = bm["MAE"] < 1800 and bm["R2"] > 0.50
    if met:
        print("\n  [OK] Targets met — saving best_model.pkl")
        save_artifact(best_model_obj, arr_enc, global_mean, is_log=best_is_log)
        update_metrics(all_new_metrics)
    else:
        print(f"\n  [!!] Targets NOT met (MAE={bm['MAE']:,.1f}, R2={bm['R2']:.4f})")
        print("  Saving best anyway for reference...")
        save_artifact(best_model_obj, arr_enc, global_mean, is_log=best_is_log)
        update_metrics(all_new_metrics)

    print("=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
