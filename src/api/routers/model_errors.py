"""
GET /model/errors — compute and return prediction error analysis.
Recomputes residuals from the model on the test set (cached after first call).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter

ROOT = Path(__file__).resolve().parents[3]

import sys
sys.path.insert(0, str(ROOT))

ARTIFACT_PATH = ROOT / "models" / "artifacts" / "best_model.pkl"
PARQUET_PATH = ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"

router = APIRouter(tags=["model"])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _compute_errors() -> dict:
    """Load model + data, split identically to training, compute residuals."""
    try:
        import joblib
        from sklearn.model_selection import train_test_split
        from src.ml.features_v2 import add_features, FEATURE_COLS_V2

        artifact = joblib.load(ARTIFACT_PATH)
        model = artifact["model"]
        feature_cols = artifact.get("feature_cols", FEATURE_COLS_V2)
        log_target = artifact.get("log_target", True)

        df = pd.read_parquet(PARQUET_PATH)

        # Derive arrondissement
        if "arrondissement" not in df.columns:
            if "code_postal" in df.columns:
                df["arrondissement"] = df["code_postal"].fillna(0).astype(int) % 100
            elif "code_commune" in df.columns:
                df["arrondissement"] = df["code_commune"].fillna(0).astype(int) % 100

        # Add features (same as training)
        df = add_features(df, artifact)

        # Same split as training (random_state=42, 80/20)
        _, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Predict
        available = [c for c in feature_cols if c in test_df.columns]
        X_test = test_df[available].copy()

        # Fill NaN with 0 for missing features
        for c in feature_cols:
            if c not in X_test.columns:
                X_test[c] = 0
        X_test = X_test[feature_cols]

        preds_raw = model.predict(X_test)
        if log_target:
            preds = np.expm1(preds_raw)
        else:
            preds = preds_raw

        actual = test_df["prix_m2"].values
        errors = preds - actual
        abs_errors = np.abs(errors)

        # Scatter data (sample up to 500 for performance)
        n = len(actual)
        idx = np.random.RandomState(42).choice(n, size=min(500, n), replace=False)
        scatter = [
            {
                "actual": round(float(actual[i]), 1),
                "predicted": round(float(preds[i]), 1),
                "error": round(float(errors[i]), 1),
                "arrondissement": int(test_df.iloc[i].get("arrondissement", 0)),
            }
            for i in idx
        ]

        # Residual distribution histogram
        bins = np.arange(-3000, 3500, 500)
        counts, edges = np.histogram(errors, bins=bins)
        residual_dist = [
            {"range": f"{int(edges[i])} to {int(edges[i+1])}", "count": int(counts[i])}
            for i in range(len(counts))
        ]

        # Error by arrondissement
        arr_col = test_df["arrondissement"].values
        error_by_arr = []
        for arr in sorted(set(arr_col)):
            mask = arr_col == arr
            if mask.sum() > 0:
                error_by_arr.append({
                    "arrondissement": int(arr),
                    "MAE": round(float(abs_errors[mask].mean()), 1),
                    "median_error": round(float(np.median(abs_errors[mask])), 1),
                    "count": int(mask.sum()),
                    "bias": round(float(errors[mask].mean()), 1),
                })

        # Global metrics
        metrics = {
            "MAE": round(float(abs_errors.mean()), 1),
            "RMSE": round(float(np.sqrt((errors ** 2).mean())), 1),
            "median_AE": round(float(np.median(abs_errors)), 1),
            "R2": round(float(1 - np.sum(errors**2) / np.sum((actual - actual.mean())**2)), 4),
            "n_test": int(n),
            "pct_within_1000": round(float((abs_errors < 1000).mean() * 100), 1),
            "pct_within_2000": round(float((abs_errors < 2000).mean() * 100), 1),
        }

        return {
            "scatter_data": scatter,
            "residual_distribution": residual_dist,
            "error_by_arrondissement": error_by_arr,
            "metrics": metrics,
        }

    except Exception:
        logger.exception("Failed to compute error analysis")
        return {
            "scatter_data": [],
            "residual_distribution": [],
            "error_by_arrondissement": [],
            "metrics": {},
        }


@router.get("/model/errors", summary="Prediction error analysis on test set")
async def get_model_errors():
    return _compute_errors()
