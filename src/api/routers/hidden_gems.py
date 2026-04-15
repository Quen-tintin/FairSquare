"""
GET /hidden_gems — return scored hidden gem listings.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Query

ROOT = Path(__file__).resolve().parents[3]
GEMS_PATH = ROOT / "src" / "frontend" / "live_listings_scored.json"

router = APIRouter(tags=["recommendations"])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_gems() -> dict:
    """Load and cache the scored listings JSON from disk.

    Returns:
        dict: Parsed JSON with keys ``metadata`` and ``gems``.
              Falls back to an empty structure on I/O error.
    """
    try:
        with open(GEMS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load gems from %s", GEMS_PATH)
        return {"metadata": {}, "gems": []}


@router.get("/hidden_gems", summary="Scored hidden gem listings")
async def get_hidden_gems(
    min_gem_score: float = Query(0.0, ge=0, le=1),
    arrondissement: int | None = Query(None, ge=1, le=20),
    min_surface: float | None = Query(None, ge=0),
    max_price: float | None = Query(None),
):
    data = _load_gems()
    gems = data.get("gems", [])

    # Apply filters
    filtered = []
    for g in gems:
        if g.get("gem_score", 0) < min_gem_score:
            continue
        if arrondissement and g.get("arrondissement") != arrondissement:
            continue
        if min_surface and g.get("surface", 0) < min_surface:
            continue
        if max_price and g.get("prix_annonce", 0) > max_price:
            continue
        filtered.append(g)

    # Sort by gem_score descending
    filtered.sort(key=lambda x: x.get("gem_score", 0), reverse=True)

    return {
        "metadata": data.get("metadata", {}),
        "gems": filtered,
        "total": len(filtered),
    }
