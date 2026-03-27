"""
Test: do OSM features improve the LightGBM model?
Run from repo root: python scripts/test_osm_features.py
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.features_v2 import (
    FEATURE_COLS_V2, compute_target_encodings, add_features, prepare_features_v2
)

MODEL_PATH   = ROOT / "models" / "artifacts" / "best_model.pkl"
OSM_PARQUET  = ROOT / "data" / "outputs" / "dvf_paris_osm_enriched.parquet"
MAIN_MODEL   = Path("C:/Users/Quent/Documents/AI Capstone/FairSquare/models/artifacts/best_model.pkl")

OSM_EXTRA_COLS = [
    "nb_restaurants", "nb_cafes", "nb_bars", "nb_transport",
    "nb_parks", "nb_pharmacies", "nb_supermarches", "nb_ecoles",
    "dist_metro_m", "walkability_score",
]

# ─────────────────────────────────────────────────────────────
# 1. Inspect current best_model.pkl
# ─────────────────────────────────────────────────────────────
def inspect_model():
    src = MODEL_PATH if MODEL_PATH.exists() else MAIN_MODEL
    if not src.exists():
        print("❌  best_model.pkl not found — skipping inspection")
        return None
    with open(src, "rb") as f:
        art = pickle.load(f)
    feature_cols = art.get("feature_cols", FEATURE_COLS_V2)
    osm_in_model = [c for c in feature_cols if c in OSM_EXTRA_COLS]
    print(f"\n{'='*60}")
    print("CURRENT best_model.pkl FEATURES")
    print(f"{'='*60}")
    print(f"  Total features : {len(feature_cols)}")
    print(f"  Feature list   : {feature_cols}")
    print(f"  OSM features   : {osm_in_model if osm_in_model else 'NONE — model does NOT use OSM'}")
    print(f"  global_mean    : {art.get('global_mean', 'N/A')}")
    # Copy to worktree if needed
    if not MODEL_PATH.exists() and MAIN_MODEL.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(MAIN_MODEL, MODEL_PATH)
        print(f"  [OK] Copied model to worktree: {MODEL_PATH}")
    return art


# ─────────────────────────────────────────────────────────────
# 2. Train both models on same train/test split
# ─────────────────────────────────────────────────────────────
def train_and_compare():
    print(f"\n{'='*60}")
    print("LOADING OSM-ENRICHED DATASET")
    print(f"{'='*60}")
    df = pd.read_parquet(OSM_PARQUET)
    print(f"  Rows: {len(df):,}   Cols: {len(df.columns)}")

    # Filter valid rows
    df = df[
        df["prix_m2"].between(2_000, 25_000) &
        df["surface_reelle_bati"].between(5, 300) &
        df["latitude"].notna() &
        df["longitude"].notna()
    ].copy()
    print(f"  After filtering: {len(df):,} rows")

    # Fill OSM NaNs with median
    for col in OSM_EXTRA_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Train/test split (same seed = fair comparison)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  Train: {len(train_df):,}   Test: {len(test_df):,}")

    # Compute target encodings from train set only
    enc = compute_target_encodings(train_df)
    enc_kwargs = dict(
        arr_target_enc=enc["arr_enc"],
        voie_enc=enc["voie_enc"],
        grid_enc=enc["grid_enc"],
        global_mean=enc["global_mean"],
    )

    # ── Model A: v2 baseline (no OSM) ──────────────────────────
    print(f"\n{'='*60}")
    print("MODEL A — v2 baseline (no OSM features)")
    print(f"{'='*60}")
    X_train_a, y_train_a = prepare_features_v2(train_df, **enc_kwargs)
    X_test_a,  y_test_a  = prepare_features_v2(test_df,  **enc_kwargs)

    lgb_params = dict(
        n_estimators=500, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=42,
    )
    model_a = lgb.LGBMRegressor(**lgb_params)
    model_a.fit(X_train_a, y_train_a,
                eval_set=[(X_test_a, y_test_a)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    pred_a = model_a.predict(X_test_a)
    mae_a = mean_absolute_error(y_test_a, pred_a)
    r2_a  = r2_score(y_test_a, pred_a)
    print(f"  MAE = {mae_a:,.1f} €/m²   R² = {r2_a:.4f}")

    # ── Model B: v2 + OSM ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("MODEL B — v2 + OSM features")
    print(f"{'='*60}")

    feature_cols_osm = FEATURE_COLS_V2 + [c for c in OSM_EXTRA_COLS if c in df.columns]

    def prepare_osm(split_df):
        data = add_features(split_df, **enc_kwargs)
        avail = [c for c in feature_cols_osm if c in data.columns]
        X = data[avail].copy()
        y = data["prix_m2"].copy()
        mask = X.notna().all(axis=1) & y.notna()
        return X[mask].astype(float).reset_index(drop=True), y[mask].reset_index(drop=True), avail

    X_train_b, y_train_b, cols_b = prepare_osm(train_df)
    X_test_b,  y_test_b,  _      = prepare_osm(test_df)

    model_b = lgb.LGBMRegressor(**lgb_params)
    model_b.fit(X_train_b, y_train_b,
                eval_set=[(X_test_b, y_test_b)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    pred_b = model_b.predict(X_test_b)
    mae_b = mean_absolute_error(y_test_b, pred_b)
    r2_b  = r2_score(y_test_b, pred_b)
    print(f"  MAE = {mae_b:,.1f} €/m²   R² = {r2_b:.4f}")
    print(f"  OSM features added: {[c for c in OSM_EXTRA_COLS if c in df.columns]}")

    # ── Verdict ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    print(f"  Baseline (v2)   : MAE={mae_a:,.1f}  R²={r2_a:.4f}")
    print(f"  v2 + OSM        : MAE={mae_b:,.1f}  R²={r2_b:.4f}")
    delta_mae = mae_a - mae_b
    delta_r2  = r2_b - r2_a
    print(f"  Δ MAE           : {delta_mae:+.1f} €/m² (positive = OSM helps)")
    print(f"  Δ R²            : {delta_r2:+.4f}  (positive = OSM helps)")

    osm_helps = delta_mae > 10 or delta_r2 > 0.005
    if osm_helps:
        print("\n  ✅  OSM AIDE — saving v2+OSM as best_model.pkl")
        best_art = {
            "model":       model_b,
            "feature_cols": cols_b,
            "arr_enc":     enc["arr_enc"],
            "voie_enc":    enc["voie_enc"],
            "grid_enc":    enc["grid_enc"],
            "global_mean": enc["global_mean"],
            "mae":         mae_b,
            "r2":          r2_b,
            "osm_features": [c for c in OSM_EXTRA_COLS if c in df.columns],
        }
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(best_art, f)
        # Also save to main repo
        if MAIN_MODEL.parent.exists():
            with open(MAIN_MODEL, "wb") as f:
                pickle.dump(best_art, f)
        print(f"  Saved → {MODEL_PATH}")
    else:
        print("\n  ⚠️  OSM N'AIDE PAS SIGNIFICATIVEMENT — garder best_model.pkl actuel (v2 baseline)")
        # Still save the baseline for the worktree
        best_art = {
            "model":       model_a,
            "feature_cols": FEATURE_COLS_V2,
            "arr_enc":     enc["arr_enc"],
            "voie_enc":    enc["voie_enc"],
            "grid_enc":    enc["grid_enc"],
            "global_mean": enc["global_mean"],
            "mae":         mae_a,
            "r2":          r2_a,
            "osm_features": [],
        }
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(best_art, f)
        print(f"  Saved baseline → {MODEL_PATH}")

    return best_art, osm_helps


if __name__ == "__main__":
    inspect_model()
    train_and_compare()
