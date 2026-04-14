"""
GET /model/errors — compute and return prediction error analysis.
Recomputes residuals from the model on the test set (cached after first call).
"""
from __future__ import annotations

import logging
import sys
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

ARTIFACT_PATH = ROOT / "models" / "artifacts" / "best_model.pkl"
PARQUET_PATH  = ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"

router = APIRouter(tags=["model"])
logger = logging.getLogger(__name__)

# Cached result (populated on first request, never evicted)
_CACHE: dict | None = None
_CACHE_ERROR: str | None = None


def _build_errors() -> dict:
    import joblib
    from sklearn.model_selection import train_test_split
    from src.ml.features_v2 import add_features, FEATURE_COLS_V2

    # ── 1. Load artifact ──────────────────────────────────────────────────
    artifact    = joblib.load(ARTIFACT_PATH)
    model       = artifact["model"]
    feature_cols = artifact.get("feature_cols", FEATURE_COLS_V2)
    log_target  = artifact.get("log_target", True)

    # ── 2. Load + IQR filter (identical to training) ──────────────────────
    df = pd.read_parquet(PARQUET_PATH)

    p      = df["prix_m2"]
    q1, q3 = p.quantile(0.25), p.quantile(0.75)
    iqr    = q3 - q1
    df     = df[p.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)].copy()

    # ── 3. Derive arrondissement ──────────────────────────────────────────
    if "arrondissement" not in df.columns:
        col = "code_postal" if "code_postal" in df.columns else "code_commune"
        df["arrondissement"] = df[col].fillna(0).astype(int) % 100

    # ── 4. Feature engineering ────────────────────────────────────────────
    df = add_features(df, artifact)

    # ── 5. Same 80/20 split as training ──────────────────────────────────
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df = test_df.reset_index(drop=True)

    # ── 6. Predict ────────────────────────────────────────────────────────
    X_test = test_df.reindex(columns=feature_cols, fill_value=0)
    preds_raw = model.predict(X_test)
    preds  = np.expm1(preds_raw) if log_target else preds_raw

    actual = test_df["prix_m2"].values
    errors = preds - actual
    abs_err = np.abs(errors)
    arr_col = test_df["arrondissement"].values

    # ── 7. Sample scatter (500 pts) ───────────────────────────────────────
    n   = len(actual)
    idx = np.random.RandomState(42).choice(n, size=min(500, n), replace=False)
    scatter = [
        {
            "actual":        round(float(actual[i]),  1),
            "predicted":     round(float(preds[i]),   1),
            "error":         round(float(errors[i]),  1),
            "arrondissement": int(arr_col[i]),
        }
        for i in idx
    ]

    # ── 8. Residual histogram ─────────────────────────────────────────────
    bins = np.arange(-3000, 3500, 500)
    counts, edges = np.histogram(errors, bins=bins)
    residual_dist = [
        {"range": f"{int(edges[i])} to {int(edges[i+1])}", "count": int(counts[i])}
        for i in range(len(counts))
    ]

    # ── 9. Error by arrondissement ────────────────────────────────────────
    error_by_arr = []
    for arr in sorted(set(arr_col.tolist())):
        mask = arr_col == arr
        if mask.sum() > 0:
            error_by_arr.append({
                "arrondissement": int(arr),
                "MAE":           round(float(abs_err[mask].mean()), 1),
                "median_error":  round(float(np.median(abs_err[mask])), 1),
                "count":         int(mask.sum()),
                "bias":          round(float(errors[mask].mean()), 1),
            })

    # ── 10. Global metrics ────────────────────────────────────────────────
    metrics = {
        "MAE":             round(float(abs_err.mean()), 1),
        "RMSE":            round(float(np.sqrt((errors ** 2).mean())), 1),
        "median_AE":       round(float(np.median(abs_err)), 1),
        "R2":              round(float(1 - np.sum(errors**2) /
                                       np.sum((actual - actual.mean())**2)), 4),
        "n_test":          int(n),
        "pct_within_1000": round(float((abs_err < 1000).mean() * 100), 1),
        "pct_within_2000": round(float((abs_err < 2000).mean() * 100), 1),
    }

    return {
        "scatter_data":           scatter,
        "residual_distribution":  residual_dist,
        "error_by_arrondissement": error_by_arr,
        "metrics":                metrics,
    }


@router.get("/model/errors", summary="Prediction error analysis on test set")
async def get_model_errors():
    global _CACHE, _CACHE_ERROR

    # Return cached success
    if _CACHE is not None:
        return _CACHE

    # Retry if previous attempt failed (or first run)
    try:
        _CACHE = _build_errors()
        _CACHE_ERROR = None
        return _CACHE
    except Exception as exc:
        _CACHE_ERROR = traceback.format_exc()
        logger.exception("Failed to compute error analysis")
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "traceback": _CACHE_ERROR[-2000:]}
        )
