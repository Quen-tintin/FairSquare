"""
POST /analyze_url — scrape a real-estate listing URL, run ML prediction + Vision analysis.
Wraps the existing analyze_listing_url() function as an API endpoint.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Lazy load — url_analyzer is heavy (imports ML model, Vision, etc.)
_analyzer = None

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])


class AnalyzeURLRequest(BaseModel):
    url: str = Field(..., min_length=10, description="URL de l'annonce (SeLoger, LeBonCoin, PAP, BienIci)")
    manual_overrides: dict | None = Field(
        default=None,
        description="Overrides manuels: {prix, surface, pieces, arrondissement}",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "url": "https://www.seloger.com/annonces/achat/appartement/paris-11eme-75/264544373.htm",
        }
    }}


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from src.frontend.url_analyzer import analyze_listing_url
        _analyzer = analyze_listing_url
    return _analyzer


@router.post("/analyze_url", summary="Analyser une annonce depuis son URL")
async def analyze_url(req: AnalyzeURLRequest):
    """
    Scrape une annonce immobilière, prédit le prix via LightGBM,
    applique les corrections (étage, DPE, rénovation via Gemini Vision),
    et retourne l'analyse complète avec gem score.
    """
    try:
        analyze = _get_analyzer()
        result = analyze(req.url, manual_overrides=req.manual_overrides)
    except Exception as exc:
        logger.exception("analyze_url failed for %s", req.url[:60])
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse: {exc}")

    # If scraping returned partial data, return 422 with details
    if result.get("status") == "needs_manual_input":
        raise HTTPException(
            status_code=422,
            detail={
                "message": result.get("message", "Données incomplètes"),
                "partial": result.get("partial", {}),
            },
        )

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Analyse échouée"),
        )

    return result
