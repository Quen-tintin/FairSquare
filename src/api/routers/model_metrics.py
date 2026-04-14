"""
GET /model/metrics — return model evolution history + feature importance.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter

ROOT = Path(__file__).resolve().parents[3]
METRICS_PATH = ROOT / "data" / "outputs" / "ml" / "metrics.json"
ARTIFACT_PATH = ROOT / "models" / "artifacts" / "best_model.pkl"

router = APIRouter(tags=["model"])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_metrics() -> list[dict]:
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        logger.warning("metrics.json not found at %s", METRICS_PATH)
        return []


@lru_cache(maxsize=1)
def _load_feature_importance() -> list[dict]:
    """Extract feature importance from the trained LightGBM model."""
    try:
        import joblib
        artifact = joblib.load(ARTIFACT_PATH)
        model = artifact["model"]
        feature_cols = artifact.get("feature_cols", [])
        # sklearn API: feature_importances_ (not feature_importance())
        importance = model.feature_importances_
        pairs = sorted(
            zip(feature_cols, importance),
            key=lambda x: x[1],
            reverse=True,
        )
        total = sum(v for _, v in pairs) or 1.0
        return [
            {"feature": name, "importance": round(val / total, 4)}
            for name, val in pairs[:15]
        ]
    except Exception:
        logger.exception("Failed to extract feature importance")
        return []


@router.get("/model/metrics", summary="Model evolution history + feature importance")
async def get_model_metrics():
    metrics = _load_metrics()
    importance = _load_feature_importance()

    # Training data stats from the artifact
    try:
        import joblib
        artifact = joblib.load(ARTIFACT_PATH)
        n_features = len(artifact.get("feature_cols", []))
        model_version = artifact.get("model_version", "LightGBM_v4")
        log_target = artifact.get("log_target", True)
    except Exception:
        n_features = 0
        model_version = "unknown"
        log_target = False

    # Current best model (last entry)
    current = metrics[-1] if metrics else {}

    return {
        "model_history": metrics,
        "feature_importance": importance,
        "current_model": {
            "name": current.get("model", model_version),
            "MAE": current.get("MAE"),
            "R2": current.get("R2"),
            "RMSE": current.get("RMSE"),
            "MAPE": current.get("MAPE_%"),
            "n_features": n_features,
            "log_target": log_target,
        },
        "training_stats": {
            "total_transactions": 43173,
            "time_horizon_months": 24,
            "coverage": "Paris (75)",
            "data_source": "DVF Open Data 2023-2025",
        },
    }
