"""
GET /model/errors — return precomputed prediction error analysis.
Results are precomputed locally and stored as a static JSON file.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[3]
ERROR_ANALYSIS_PATH = ROOT / "data" / "outputs" / "ml" / "error_analysis.json"

router = APIRouter(tags=["model"])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_error_analysis() -> dict:
    try:
        with open(ERROR_ANALYSIS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load error_analysis.json from %s", ERROR_ANALYSIS_PATH)
        raise


@router.get("/model/errors", summary="Prediction error analysis on test set")
async def get_model_errors():
    try:
        return _load_error_analysis()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error analysis file not found: {exc}")
