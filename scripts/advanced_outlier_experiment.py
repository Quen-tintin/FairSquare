"""
FairSquare — Advanced Outlier Analysis Experiment
===================================================
Addresses instructor feedback on data quality, Rare vs Bad Label distinction,
and the impact of multivariate outlier detection on MAE.

Experiments:
  A. IQR 1.5× (current baseline)
  B. IQR 1.5× + Isolation Forest (contamination=0.03)
  C. IQR 1.5× + Isolation Forest (contamination=0.05)
  D. Tighter IQR 1.0×
  E. Tighter price bounds (3,000–25,000 €/m²)

For each strategy → retrain LightGBM → compare MAE/R²/MAPE on same test set.

Run:
    python scripts/advanced_outlier_experiment.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
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

PROCESSED = ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"
RESULTS_PATH = ROOT / "data" / "outputs" / "ml" / "outlier_experiment_results.json"
RANDOM_STATE = 42


# ── Metrics ──────────────────────────────────────────────────────────────────

def _mape(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _pct_within(y_true, y_pred, threshold):
    return float((np.abs(y_true - y_pred) <= threshold).mean() * 100)


def evaluate(y_true, y_pred):
    return {
        "MAE":    round(float(mean_absolute_error(y_true, y_pred)), 1),
        "RMSE":   round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "R2":     round(float(r2_score(y_true, y_pred)), 4),
        "MAPE_%": round(_mape(y_true, y_pred), 2),
        "pct_within_1000": round(_pct_within(y_true, y_pred, 1000), 1),
        "pct_within_2000": round(_pct_within(y_true, y_pred, 2000), 1),
    }


# ── Data loading ─────────────────────────────────────────────────────────────

def load_raw():
    """Load cleaned parquet (post dvf_cleaner, pre-IQR)."""
    df = pd.read_parquet(PROCESSED)
    print(f"Loaded: {len(df):,} rows")
    return df


# ── Cleaning strategies ──────────────────────────────────────────────────────

def apply_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """IQR filter per arrondissement."""
    df = df.copy()
    arr = (df["code_postal"].fillna(75001).astype(float) % 100).astype(int)
    df["_arr"] = arr
    mask = pd.Series(True, index=df.index)

    for a in df["_arr"].unique():
        idx = df["_arr"] == a
        p = df.loc[idx, "prix_m2"]
        q1, q3 = p.quantile(0.25), p.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        mask.loc[idx] = p.between(lo, hi)

    result = df[mask].drop(columns=["_arr"])
    return result


def apply_isolation_forest(df: pd.DataFrame, contamination: float = 0.03) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multivariate outlier detection using Isolation Forest.

    Returns:
        (clean_df, outliers_df)
    """
    features = ["surface_reelle_bati", "prix_m2", "latitude", "longitude"]
    # Add arrondissement
    arr = (df["code_postal"].fillna(75001).astype(float) % 100).astype(int)
    X = df[features].copy()
    X["arrondissement"] = arr.values

    clf = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_estimators=200,
        n_jobs=-1,
    )
    labels = clf.fit_predict(X)

    clean = df[labels == 1].copy()
    outliers = df[labels == -1].copy()
    return clean, outliers


def apply_tight_price_bounds(df: pd.DataFrame,
                              min_pm2: float = 3000,
                              max_pm2: float = 25000) -> pd.DataFrame:
    """Tighter price/m2 bounds."""
    return df[df["prix_m2"].between(min_pm2, max_pm2)].copy()


# ── Outlier categorization (Rare vs Bad Label) ──────────────────────────────

def categorize_outliers(outliers: pd.DataFrame, global_mean: float) -> dict:
    """Classify Isolation Forest outliers into Rare Label vs Bad Label.

    Rare Label: genuine extreme but plausible (luxury penthouse, micro-studio)
    Bad Label:  data entry error, family transfer, impossible combination
    """
    if len(outliers) == 0:
        return {"rare_labels": [], "bad_labels": [], "summary": {}}

    arr = (outliers["code_postal"].fillna(75001).astype(float) % 100).astype(int)
    outliers = outliers.copy()
    outliers["_arr"] = arr

    bad_mask = (
        # Total price impossibly low for Paris (< 50k for any apartment)
        (outliers["valeur_fonciere"] < 50_000) |
        # Price/m2 below 3,000 in premium arrondissements (1-8, 16)
        ((outliers["prix_m2"] < 3000) & (outliers["_arr"].isin({1,2,3,4,6,7,8,16}))) |
        # Surface < 9m2 with > 2 rooms (data error)
        ((outliers["surface_reelle_bati"] < 9) & (outliers["nombre_pieces_principales"] > 2)) |
        # Price/m2 below 1,500 anywhere in Paris (almost certainly a family transfer)
        (outliers["prix_m2"] < 1500)
    )

    bad_labels = outliers[bad_mask]
    rare_labels = outliers[~bad_mask]

    def _sample(df, n=5):
        rows = df.head(n)
        return [
            {
                "arr": int((r.get("code_postal", 75001) or 75001) % 100),
                "surface": round(float(r["surface_reelle_bati"]), 1),
                "prix_m2": round(float(r["prix_m2"]), 0),
                "prix_total": round(float(r["valeur_fonciere"]), 0),
            }
            for _, r in rows.iterrows()
        ]

    return {
        "n_bad": len(bad_labels),
        "n_rare": len(rare_labels),
        "bad_examples": _sample(bad_labels.sort_values("prix_m2")),
        "rare_examples": _sample(rare_labels.sort_values("prix_m2", ascending=False)),
        "summary": {
            "bad_mean_prix_m2": round(float(bad_labels["prix_m2"].mean()), 0) if len(bad_labels) > 0 else 0,
            "rare_mean_prix_m2": round(float(rare_labels["prix_m2"].mean()), 0) if len(rare_labels) > 0 else 0,
        }
    }


# ── Training pipeline (simplified from train_v4) ────────────────────────────

def train_and_evaluate(df: pd.DataFrame, label: str) -> dict:
    """Full pipeline: voie_recent → split → encode → train → evaluate."""
    import lightgbm as lgb

    # Compute voie_recent on full df before split (LOO by date = no leakage)
    rolling = compute_voie_recent_prix_m2(df)
    global_mean = float(df["prix_m2"].mean())
    df = df.copy()
    df["voie_recent_prix_m2"] = rolling.fillna(global_mean)

    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    # Drop voie_recent before re-computing on train only
    df_train = df_train.drop(columns=["voie_recent_prix_m2"])
    df_test = df_test.drop(columns=["voie_recent_prix_m2"])

    # Target encoding from train only
    arr_enc = compute_arr_target_enc(df_train)
    gm = float(df_train["prix_m2"].mean())
    voie_recent_lookup, arr_recent_lookup = compute_voie_recent_lookup(df_train, months=12)

    # voie_recent for train (LOO)
    df_train = df_train.copy()
    df_test = df_test.copy()
    roll_train = compute_voie_recent_prix_m2(df_train)
    df_train["voie_recent_prix_m2"] = roll_train.fillna(gm)

    # voie_recent for test (lookup from train)
    if "adresse_code_voie" in df_test.columns:
        df_test["voie_recent_prix_m2"] = (
            df_test["adresse_code_voie"].astype(str)
            .map(voie_recent_lookup)
            .fillna(
                (df_test["code_postal"].fillna(75001).astype(float) % 100).astype(int)
                .map(arr_recent_lookup)
            )
            .fillna(gm)
        )
    else:
        df_test["voie_recent_prix_m2"] = gm

    # Prepare features
    X_train, y_train = prepare_features_v2(df_train, arr_enc, gm)
    X_test, y_test = prepare_features_v2(df_test, arr_enc, gm)

    # Train LightGBM log-target
    log_y = np.log1p(y_train.values)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, log_y, test_size=0.1, random_state=RANDOM_STATE
    )

    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.04,
        num_leaves=127,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="regression_l1",
        metric="mae",
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    pred = np.expm1(model.predict(X_test))
    metrics = evaluate(y_test.values, pred)
    metrics["label"] = label
    metrics["n_train"] = len(df_train)
    metrics["n_test"] = len(df_test)
    metrics["best_iteration"] = model.best_iteration_

    return metrics


# ── Residual analysis by feature ─────────────────────────────────────────────

def residual_analysis_summary(df: pd.DataFrame, label: str) -> dict:
    """Quick residual analysis: which arrondissements have highest errors."""
    import lightgbm as lgb

    rolling = compute_voie_recent_prix_m2(df)
    global_mean = float(df["prix_m2"].mean())
    df = df.copy()
    df["voie_recent_prix_m2"] = rolling.fillna(global_mean)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    arr_enc = compute_arr_target_enc(df_train)
    gm = float(df_train["prix_m2"].mean())

    df_train = df_train.copy()
    df_test = df_test.copy()

    # Quick encoding
    voie_lookup, arr_lookup = compute_voie_recent_lookup(df_train, months=12)
    df_train_copy = df_train.drop(columns=["voie_recent_prix_m2"], errors="ignore")
    df_test_copy = df_test.drop(columns=["voie_recent_prix_m2"], errors="ignore")

    roll_tr = compute_voie_recent_prix_m2(df_train_copy)
    df_train_copy["voie_recent_prix_m2"] = roll_tr.fillna(gm)

    if "adresse_code_voie" in df_test_copy.columns:
        df_test_copy["voie_recent_prix_m2"] = (
            df_test_copy["adresse_code_voie"].astype(str)
            .map(voie_lookup).fillna(gm)
        )
    else:
        df_test_copy["voie_recent_prix_m2"] = gm

    X_train, y_train = prepare_features_v2(df_train_copy, arr_enc, gm)
    X_test, y_test = prepare_features_v2(df_test_copy, arr_enc, gm)

    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.04, num_leaves=127,
        min_child_samples=20, verbose=-1, random_state=RANDOM_STATE,
        objective="regression_l1",
    )
    log_y = np.log1p(y_train.values)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, log_y, test_size=0.1, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    pred = np.expm1(model.predict(X_test))
    residuals = pred - y_test.values

    # Residuals by arrondissement
    arr_test = X_test["arrondissement"].values.astype(int)
    by_arr = {}
    for a in sorted(set(arr_test)):
        mask = arr_test == a
        by_arr[int(a)] = {
            "MAE": round(float(np.mean(np.abs(residuals[mask]))), 1),
            "bias": round(float(np.mean(residuals[mask])), 1),
            "n": int(mask.sum()),
        }

    # Residuals by surface bucket
    surface_test = X_test["surface_reelle_bati"].values
    by_surface = {}
    for lo, hi, lbl in [(0,30,"<30m2"), (30,60,"30-60m2"), (60,100,"60-100m2"), (100,9999,">100m2")]:
        mask = (surface_test >= lo) & (surface_test < hi)
        if mask.sum() > 0:
            by_surface[lbl] = {
                "MAE": round(float(np.mean(np.abs(residuals[mask]))), 1),
                "bias": round(float(np.mean(residuals[mask])), 1),
                "n": int(mask.sum()),
            }

    # Residuals by price tier
    by_price = {}
    for lo, hi, lbl in [(0,7000,"<7k"), (7000,10000,"7-10k"), (10000,14000,"10-14k"), (14000,99999,">14k")]:
        mask = (y_test.values >= lo) & (y_test.values < hi)
        if mask.sum() > 0:
            by_price[lbl] = {
                "MAE": round(float(np.mean(np.abs(residuals[mask]))), 1),
                "bias": round(float(np.mean(residuals[mask])), 1),
                "n": int(mask.sum()),
            }

    return {
        "by_arrondissement": by_arr,
        "by_surface": by_surface,
        "by_price_tier": by_price,
    }


# ── Main experiment ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FairSquare — Advanced Outlier Experiment")
    print("=" * 70)

    df_raw = load_raw()

    # ── Strategy A: IQR 1.5x (current baseline) ──────────────────────
    print("\n[A] IQR 1.5x per arrondissement (current baseline)")
    df_a = apply_iqr(df_raw, factor=1.5)
    print(f"    Rows: {len(df_a):,} (removed {len(df_raw)-len(df_a):,})")
    metrics_a = train_and_evaluate(df_a, "A_IQR_1.5x")
    print(f"    MAE={metrics_a['MAE']:,.1f}  R2={metrics_a['R2']:.4f}  MAPE={metrics_a['MAPE_%']:.2f}%")

    # ── Strategy B: IQR 1.5x + Isolation Forest 3% ──────────────────
    print("\n[B] IQR 1.5x + Isolation Forest (contamination=3%)")
    df_b_pre = apply_iqr(df_raw, factor=1.5)
    df_b, outliers_b = apply_isolation_forest(df_b_pre, contamination=0.03)
    print(f"    Rows: {len(df_b):,} (IF removed {len(outliers_b):,} additional)")
    cat_b = categorize_outliers(outliers_b, float(df_b_pre["prix_m2"].mean()))
    print(f"    Bad Labels: {cat_b['n_bad']}  |  Rare Labels: {cat_b['n_rare']}")
    if cat_b["bad_examples"]:
        print(f"    Bad example:  arr={cat_b['bad_examples'][0]['arr']}, "
              f"{cat_b['bad_examples'][0]['surface']}m2, "
              f"{cat_b['bad_examples'][0]['prix_m2']:,.0f} eur/m2")
    if cat_b["rare_examples"]:
        print(f"    Rare example: arr={cat_b['rare_examples'][0]['arr']}, "
              f"{cat_b['rare_examples'][0]['surface']}m2, "
              f"{cat_b['rare_examples'][0]['prix_m2']:,.0f} eur/m2")
    metrics_b = train_and_evaluate(df_b, "B_IQR+IF_3pct")
    print(f"    MAE={metrics_b['MAE']:,.1f}  R2={metrics_b['R2']:.4f}  MAPE={metrics_b['MAPE_%']:.2f}%")

    # ── Strategy C: IQR 1.5x + Isolation Forest 5% ──────────────────
    print("\n[C] IQR 1.5x + Isolation Forest (contamination=5%)")
    df_c, outliers_c = apply_isolation_forest(df_b_pre, contamination=0.05)
    print(f"    Rows: {len(df_c):,} (IF removed {len(outliers_c):,} additional)")
    cat_c = categorize_outliers(outliers_c, float(df_b_pre["prix_m2"].mean()))
    print(f"    Bad Labels: {cat_c['n_bad']}  |  Rare Labels: {cat_c['n_rare']}")
    metrics_c = train_and_evaluate(df_c, "C_IQR+IF_5pct")
    print(f"    MAE={metrics_c['MAE']:,.1f}  R2={metrics_c['R2']:.4f}  MAPE={metrics_c['MAPE_%']:.2f}%")

    # ── Strategy D: Tighter IQR 1.0x ─────────────────────────────────
    print("\n[D] Tighter IQR 1.0x per arrondissement")
    df_d = apply_iqr(df_raw, factor=1.0)
    print(f"    Rows: {len(df_d):,} (removed {len(df_raw)-len(df_d):,})")
    metrics_d = train_and_evaluate(df_d, "D_IQR_1.0x")
    print(f"    MAE={metrics_d['MAE']:,.1f}  R2={metrics_d['R2']:.4f}  MAPE={metrics_d['MAPE_%']:.2f}%")

    # ── Strategy E: Tighter price bounds ─────────────────────────────
    print("\n[E] IQR 1.5x + Tight price bounds (3k-25k eur/m2)")
    df_e = apply_tight_price_bounds(df_a, min_pm2=3000, max_pm2=25000)
    print(f"    Rows: {len(df_e):,} (removed {len(df_a)-len(df_e):,} from bounds)")
    metrics_e = train_and_evaluate(df_e, "E_IQR+tight_bounds")
    print(f"    MAE={metrics_e['MAE']:,.1f}  R2={metrics_e['R2']:.4f}  MAPE={metrics_e['MAPE_%']:.2f}%")

    # ── Comparison table ─────────────────────────────────────────────
    all_metrics = [metrics_a, metrics_b, metrics_c, metrics_d, metrics_e]

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Strategy':<25} {'N_train':>8} {'MAE':>8} {'R2':>7} {'MAPE%':>7} {'<1k%':>6} {'<2k%':>6}")
    print("-" * 70)
    for m in all_metrics:
        tag = " <-- BEST" if m["MAE"] == min(x["MAE"] for x in all_metrics) else ""
        print(f"{m['label']:<25} {m['n_train']:>8,} {m['MAE']:>8,.1f} "
              f"{m['R2']:>7.4f} {m['MAPE_%']:>7.2f} "
              f"{m['pct_within_1000']:>6.1f} {m['pct_within_2000']:>6.1f}{tag}")

    # ── Residual deep-dive on best strategy ──────────────────────────
    best = min(all_metrics, key=lambda x: x["MAE"])
    best_label = best["label"]
    print(f"\n{'=' * 70}")
    print(f"RESIDUAL ANALYSIS — Best strategy: {best_label}")
    print(f"{'=' * 70}")

    # Pick the right df
    df_best = {"A_IQR_1.5x": df_a, "B_IQR+IF_3pct": df_b, "C_IQR+IF_5pct": df_c,
               "D_IQR_1.0x": df_d, "E_IQR+tight_bounds": df_e}[best_label]

    residuals = residual_analysis_summary(df_best, best_label)

    print("\nBy Arrondissement (top 5 worst MAE):")
    by_arr = sorted(residuals["by_arrondissement"].items(),
                    key=lambda x: x[1]["MAE"], reverse=True)[:5]
    for arr, r in by_arr:
        print(f"  {arr:>2}e:  MAE={r['MAE']:>8,.1f}  bias={r['bias']:>+8,.1f}  n={r['n']:>5}")

    print("\nBy Surface:")
    for lbl, r in residuals["by_surface"].items():
        print(f"  {lbl:<10}  MAE={r['MAE']:>8,.1f}  bias={r['bias']:>+8,.1f}  n={r['n']:>5}")

    print("\nBy Price Tier (eur/m2):")
    for lbl, r in residuals["by_price_tier"].items():
        print(f"  {lbl:<10}  MAE={r['MAE']:>8,.1f}  bias={r['bias']:>+8,.1f}  n={r['n']:>5}")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "experiments": all_metrics,
        "best_strategy": best_label,
        "outlier_categorization_3pct": cat_b,
        "outlier_categorization_5pct": cat_c,
        "residual_analysis": residuals,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}")

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
