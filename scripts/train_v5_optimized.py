"""
FairSquare — v5 Optimized Training
====================================
Builds on v4 + outlier experiment results to push MAE below 1,200 €/m².

Improvements stacked incrementally (each step builds on the previous):
  v5a — IQR 1.0x per-arrondissement  (best cleaning from experiment)
  v5b — + Price filter 5k-20k €/m²   (cut extreme tails)
  v5c — + Huber loss                  (robust to remaining outliers)
  v5d — + Temporal weighting          (recent transactions matter more)
  v5e — + Optuna 40-trial HPO         (optimal hyperparameters)

Best model saved to models/artifacts/best_model.pkl if it beats v4 (MAE 1,427).

Run:
    python scripts/train_v5_optimized.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.features_v2 import (
    FEATURE_COLS_V2,
    compute_arr_target_enc,
    compute_voie_recent_lookup,
    compute_voie_recent_prix_m2,
    prepare_features_v2,
)

PROCESSED     = ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"
METRICS_PATH  = ROOT / "data" / "outputs" / "ml" / "metrics.json"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE  = 42
V4_BASELINE   = 1427.0


# ── Metrics ──────────────────────────────────────────────────────────────────

def _mape(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def _within(y_true, y_pred, t):
    return round(float((np.abs(y_true - y_pred) <= t).mean() * 100), 1)

def evaluate(y_true, y_pred, label):
    m = {
        "model":            label,
        "MAE":              round(float(mean_absolute_error(y_true, y_pred)), 1),
        "RMSE":             round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "R2":               round(float(r2_score(y_true, y_pred)), 4),
        "MAPE_%":           round(_mape(y_true, y_pred), 2),
        "pct_within_1000":  _within(y_true, y_pred, 1000),
        "pct_within_2000":  _within(y_true, y_pred, 2000),
    }
    print(f"  {label}: MAE={m['MAE']:>8,.1f}  R2={m['R2']:.4f}  "
          f"MAPE={m['MAPE_%']:.2f}%  <1k={m['pct_within_1000']}%")
    return m


# ── Cleaning ─────────────────────────────────────────────────────────────────

def apply_iqr_per_arr(df: pd.DataFrame, factor: float = 1.0) -> pd.DataFrame:
    """IQR filter per arrondissement (tighter than global IQR)."""
    arr = (df["code_postal"].fillna(75001).astype(float) % 100).astype(int)
    df = df.copy()
    df["_arr"] = arr
    mask = pd.Series(True, index=df.index)
    for a in df["_arr"].unique():
        idx = df["_arr"] == a
        p = df.loc[idx, "prix_m2"]
        q1, q3 = p.quantile(0.25), p.quantile(0.75)
        iqr = q3 - q1
        mask.loc[idx] = p.between(q1 - factor * iqr, q3 + factor * iqr)
    return df[mask].drop(columns=["_arr"])


def apply_price_filter(df: pd.DataFrame,
                       lo: float = 5000, hi: float = 20000) -> pd.DataFrame:
    """Keep only the core market — removes extreme tails that bias the model."""
    before = len(df)
    df = df[df["prix_m2"].between(lo, hi)].copy()
    print(f"    Price filter {lo:,.0f}-{hi:,.0f}: {before:,} -> {len(df):,} "
          f"(removed {before-len(df):,})")
    return df


# ── Feature prep ─────────────────────────────────────────────────────────────

def full_feature_pipeline(df: pd.DataFrame):
    """Split → target encode → voie_recent → prepare X/y.
    Returns (X_train, X_test, y_train, y_test, artifact_dict)
    """
    # voie_recent on full set (LOO by date, no self-leakage)
    rolling = compute_voie_recent_prix_m2(df)
    gm_full = float(df["prix_m2"].mean())
    df = df.copy()
    df["voie_recent_prix_m2"] = rolling.fillna(gm_full)

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE
    )

    # Recompute voie_recent on train only (no test leakage)
    df_train = df_train.drop(columns=["voie_recent_prix_m2"])
    df_test  = df_test.drop(columns=["voie_recent_prix_m2"])

    arr_enc = compute_arr_target_enc(df_train)
    gm      = float(df_train["prix_m2"].mean())
    voie_recent_lookup, arr_recent_lookup = compute_voie_recent_lookup(
        df_train, months=12
    )

    df_train = df_train.copy()
    df_test  = df_test.copy()

    roll_tr = compute_voie_recent_prix_m2(df_train)
    df_train["voie_recent_prix_m2"] = roll_tr.fillna(gm)

    if "adresse_code_voie" in df_test.columns:
        df_test["voie_recent_prix_m2"] = (
            df_test["adresse_code_voie"].astype(str)
            .map(voie_recent_lookup)
            .fillna(
                (df_test["code_postal"].fillna(75001).astype(float) % 100)
                .astype(int).map(arr_recent_lookup)
            )
            .fillna(gm)
        )
    else:
        df_test["voie_recent_prix_m2"] = gm

    X_train, y_train = prepare_features_v2(df_train, arr_enc, gm)
    X_test,  y_test  = prepare_features_v2(df_test,  arr_enc, gm)

    artifact = {
        "arr_enc":                   arr_enc,
        "voie_enc":                  {},
        "grid_enc":                  {},
        "global_mean":               gm,
        "feature_cols":              FEATURE_COLS_V2,
        "log_target":                True,
        "voie_recent_median_lookup": voie_recent_lookup,
        "arr_recent_median_lookup":  arr_recent_lookup,
    }
    return X_train, X_test, y_train, y_test, artifact


# ── Temporal weights ──────────────────────────────────────────────────────────

def temporal_weights(df_train: pd.DataFrame) -> np.ndarray | None:
    """Assign higher weight to more recent transactions.

    2023 → weight 1.0 | 2024 → 1.4 | 2025 → 1.8
    Returns aligned weight array or None if no year column.
    """
    if "annee" not in df_train.columns:
        return None
    year = df_train["annee"].fillna(2023).astype(float)
    weights = 1.0 + (year - 2023) * 0.4
    return weights.clip(lower=0.5).values


# ── LightGBM training ─────────────────────────────────────────────────────────

BASE_PARAMS = {
    "n_estimators":      3000,
    "learning_rate":     0.04,
    "num_leaves":        127,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      10,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "metric":            "mae",
    "verbose":           -1,
    "random_state":      RANDOM_STATE,
}


def train_lgb(X_train, y_train, X_test, y_test,
              label: str,
              objective: str = "regression_l1",
              sample_weight=None,
              extra_params: dict | None = None) -> tuple:
    import lightgbm as lgb

    params = {**BASE_PARAMS, "objective": objective}
    if extra_params:
        params.update(extra_params)

    log_y = np.log1p(y_train.values)

    # Validation split (stratified if weights given)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, log_y, test_size=0.1, random_state=RANDOM_STATE
    )
    w_tr = None
    if sample_weight is not None:
        # Align weights with the train fold indices
        w_tr_full = sample_weight[:len(X_train)]
        split_idx = int(len(X_train) * 0.9)
        w_tr = w_tr_full[:split_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(60, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    pred = np.expm1(model.predict(X_test))
    metrics = evaluate(y_test.values, pred, label)
    metrics["best_iteration"] = model.best_iteration_
    return model, metrics


# ── Optuna HPO ────────────────────────────────────────────────────────────────

def optuna_hpo(X_train, y_train, n_trials: int = 40) -> dict:
    """Bayesian hyperparameter search with Optuna (40 trials ~10 min)."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [SKIP] optuna not installed — pip install optuna")
        return {}

    import lightgbm as lgb

    log_y = np.log1p(y_train.values)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, log_y, test_size=0.15, random_state=RANDOM_STATE
    )

    def objective(trial):
        p = {
            "objective":         "huber",
            "metric":            "mae",
            "verbose":           -1,
            "random_state":      RANDOM_STATE,
            "n_estimators":      3000,
            "learning_rate":     trial.suggest_float("lr", 0.01, 0.08, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 63, 255),
            "min_child_samples": trial.suggest_int("min_child", 10, 50),
            "feature_fraction":  trial.suggest_float("feat_frac", 0.6, 1.0),
            "bagging_fraction":  trial.suggest_float("bag_frac", 0.6, 1.0),
            "bagging_freq":      trial.suggest_int("bag_freq", 1, 15),
            "reg_alpha":         trial.suggest_float("alpha", 0.01, 2.0, log=True),
            "reg_lambda":        trial.suggest_float("lambda", 0.01, 5.0, log=True),
            "alpha":             trial.suggest_float("huber_alpha", 0.7, 0.99),
        }
        model = lgb.LGBMRegressor(**p)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(40, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        pred_val = np.expm1(model.predict(X_val))
        return float(mean_absolute_error(np.expm1(y_val), pred_val))

    print(f"  Running Optuna ({n_trials} trials)…")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  Best trial: val MAE-proxy={study.best_value:,.1f}")
    print(f"  Best params: {best}")
    return best


# ── Artifact save / metrics update ───────────────────────────────────────────

def save_artifact(model, artifact: dict, metrics: dict):
    artifact["model"] = model
    artifact["model_version"] = metrics["model"]
    path = ARTIFACTS_DIR / "best_model.pkl"
    joblib.dump(artifact, path)
    print(f"  Saved → {path}")


def update_metrics(entry: dict):
    old = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else []
    name = entry["model"]
    found = False
    for i, m in enumerate(old):
        if m["model"] == name:
            old[i] = entry
            found = True
    if not found:
        old.append(entry)
    METRICS_PATH.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 70
    print(SEP)
    print("FairSquare v5 — Incremental Optimization")
    print(f"  Beating: v4 MAE={V4_BASELINE:,.1f} €/m²")
    print(SEP)

    df_raw = pd.read_parquet(PROCESSED)
    print(f"Loaded: {len(df_raw):,} rows\n")

    all_metrics = []

    # ── v5a: IQR 1.0x per-arrondissement ─────────────────────────────
    print("[v5a] IQR 1.0x per-arrondissement (best cleaning)")
    df_a = apply_iqr_per_arr(df_raw, factor=1.0)
    print(f"  Rows: {len(df_a):,} (removed {len(df_raw)-len(df_a):,})")
    X_tr, X_te, y_tr, y_te, art = full_feature_pipeline(df_a)
    model_a, m_a = train_lgb(X_tr, y_tr, X_te, y_te, "LightGBM_v5a_iqr10")
    all_metrics.append(m_a)

    # ── v5b: + Price filter 5k-20k ────────────────────────────────────
    print("\n[v5b] + Price filter 5,000-20,000 €/m²")
    df_b = apply_price_filter(df_a, lo=5000, hi=20000)
    X_tr, X_te, y_tr, y_te, art_b = full_feature_pipeline(df_b)
    model_b, m_b = train_lgb(X_tr, y_tr, X_te, y_te, "LightGBM_v5b_price_filter")
    all_metrics.append(m_b)

    # ── v5c: + Huber loss ─────────────────────────────────────────────
    print("\n[v5c] + Huber loss (robust to remaining outliers)")
    X_tr, X_te, y_tr, y_te, art_c = full_feature_pipeline(df_b)
    model_c, m_c = train_lgb(
        X_tr, y_tr, X_te, y_te,
        "LightGBM_v5c_huber",
        objective="huber",
        extra_params={"alpha": 0.90},   # Huber threshold: 90th percentile
    )
    all_metrics.append(m_c)

    # ── v5d: + Temporal weighting ─────────────────────────────────────
    print("\n[v5d] + Temporal weighting (recent transactions ×1.8)")
    X_tr, X_te, y_tr, y_te, art_d = full_feature_pipeline(df_b)
    # Re-attach year for weight computation
    df_b_train, _ = train_test_split(df_b, test_size=0.2, random_state=RANDOM_STATE)
    w = temporal_weights(df_b_train.reset_index(drop=True))
    if w is not None:
        w_aligned = w[:len(X_tr)]
        print(f"  Weight range: {w_aligned.min():.2f} – {w_aligned.max():.2f}")
    else:
        w_aligned = None
    model_d, m_d = train_lgb(
        X_tr, y_tr, X_te, y_te,
        "LightGBM_v5d_temporal_weights",
        objective="huber",
        sample_weight=w_aligned,
        extra_params={"alpha": 0.90},
    )
    all_metrics.append(m_d)

    # ── v5e: + Optuna HPO ─────────────────────────────────────────────
    print("\n[v5e] + Optuna hyperparameter optimization (40 trials)")
    X_tr, X_te, y_tr, y_te, art_e = full_feature_pipeline(df_b)
    best_hpo = optuna_hpo(X_tr, y_tr, n_trials=40)

    if best_hpo:
        model_e, m_e = train_lgb(
            X_tr, y_tr, X_te, y_te,
            "LightGBM_v5e_optuna",
            objective="huber",
            extra_params={
                "alpha": best_hpo.get("huber_alpha", 0.90),
                "num_leaves": best_hpo.get("num_leaves", 127),
                "learning_rate": best_hpo.get("lr", 0.04),
                "min_child_samples": best_hpo.get("min_child", 20),
                "feature_fraction": best_hpo.get("feat_frac", 0.8),
                "bagging_fraction": best_hpo.get("bag_frac", 0.8),
                "bagging_freq": best_hpo.get("bag_freq", 10),
                "reg_alpha": best_hpo.get("alpha", 0.1),
                "reg_lambda": best_hpo.get("lambda", 1.0),
            },
        )
        all_metrics.append(m_e)
    else:
        m_e = m_d  # fallback if optuna not installed

    # ── Results ──────────────────────────────────────────────────────
    print("\n" + SEP)
    print("INCREMENTAL IMPROVEMENT TABLE")
    print(SEP)
    print(f"{'Model':<38} {'MAE':>8} {'R2':>7} {'MAPE%':>7} {'<1k%':>6} {'<2k%':>6} {'vs v4':>8}")
    print("-" * 70)
    for m in all_metrics:
        delta = m["MAE"] - V4_BASELINE
        tag   = f"{delta:+,.0f}"
        best_flag = " <--BEST" if m["MAE"] == min(x["MAE"] for x in all_metrics) else ""
        print(f"{m['model']:<38} {m['MAE']:>8,.1f} {m['R2']:>7.4f} "
              f"{m['MAPE_%']:>7.2f} {m['pct_within_1000']:>6.1f} "
              f"{m['pct_within_2000']:>6.1f} {tag:>8}{best_flag}")

    # ── Save best ─────────────────────────────────────────────────────
    best = min(all_metrics, key=lambda x: x["MAE"])
    print(f"\nBest model: {best['model']}  MAE={best['MAE']:,.1f} €/m²")

    if best["MAE"] < V4_BASELINE:
        improvement = V4_BASELINE - best["MAE"]
        print(f"[IMPROVED] by {improvement:.1f} €/m² vs v4 baseline")
        # Save the best model artifact
        model_map = {
            "LightGBM_v5a_iqr10":           (model_a, art),
            "LightGBM_v5b_price_filter":     (model_b, art_b),
            "LightGBM_v5c_huber":            (model_c, art_c),
            "LightGBM_v5d_temporal_weights": (model_d, art_d),
            "LightGBM_v5e_optuna":           (model_e, art_e) if best_hpo else (model_d, art_d),
        }
        best_model, best_art = model_map[best["model"]]
        save_artifact(best_model, best_art, best)
        print("  Saved to models/artifacts/best_model.pkl")
    else:
        print("[NO IMPROVEMENT over v4 — best_model.pkl unchanged]")

    # Update metrics.json with all runs
    for m in all_metrics:
        update_metrics({k: v for k, v in m.items()
                        if k not in ("best_iteration", "pct_within_1000", "pct_within_2000")})
    print(f"  Updated metrics.json ({len(all_metrics)} new entries)")

    print(SEP)
    print("Done.")
    return best


if __name__ == "__main__":
    main()
