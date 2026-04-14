"""
GET /dvf/transactions — return real DVF transactions with filtering.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

ROOT = Path(__file__).resolve().parents[3]
PARQUET_PATH = ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"

router = APIRouter(tags=["data"])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_dvf() -> pd.DataFrame:
    """Load DVF parquet once and cache."""
    try:
        df = pd.read_parquet(PARQUET_PATH)
        # Derive arrondissement from code_postal (e.g. 75011.0 → 11)
        if "arrondissement" not in df.columns and "code_postal" in df.columns:
            df["arrondissement"] = df["code_postal"].fillna(0).astype(int) % 100
        elif "arrondissement" not in df.columns and "code_commune" in df.columns:
            df["arrondissement"] = df["code_commune"].fillna(0).astype(int) % 100
        return df
    except Exception:
        logger.exception("Failed to load DVF parquet from %s", PARQUET_PATH)
        return pd.DataFrame()


@router.get("/dvf/transactions", summary="Filtered DVF transactions")
async def get_transactions(
    arrondissement: int | None = Query(None, ge=1, le=20),
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None),
    min_surface: float | None = Query(None, ge=0),
    max_surface: float | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    df = _load_dvf()
    if df.empty:
        return {"transactions": [], "total": 0, "stats": {}}

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if arrondissement is not None:
        mask &= df["arrondissement"] == arrondissement
    if min_price is not None:
        mask &= df["valeur_fonciere"] >= min_price
    if max_price is not None:
        mask &= df["valeur_fonciere"] <= max_price
    if min_surface is not None:
        mask &= df["surface_reelle_bati"] >= min_surface
    if max_surface is not None:
        mask &= df["surface_reelle_bati"] <= max_surface

    filtered = df[mask]
    total = len(filtered)

    # Stats for the filtered set
    stats = {}
    if total > 0:
        stats = {
            "avg_prix_m2": round(filtered["prix_m2"].mean(), 1),
            "median_prix_m2": round(filtered["prix_m2"].median(), 1),
            "avg_surface": round(filtered["surface_reelle_bati"].mean(), 1),
            "avg_price": round(filtered["valeur_fonciere"].mean(), 0),
            "total_count": total,
        }
        # Per-arrondissement breakdown
        arr_stats = (
            filtered.groupby("arrondissement")
            .agg(
                count=("prix_m2", "size"),
                avg_prix_m2=("prix_m2", "mean"),
                median_prix_m2=("prix_m2", "median"),
            )
            .round(1)
            .reset_index()
            .to_dict("records")
        )
        stats["by_arrondissement"] = arr_stats

    # Paginate
    page = filtered.sort_values("date_mutation", ascending=False).iloc[offset : offset + limit]

    # Build response rows
    cols = [
        "date_mutation", "valeur_fonciere", "surface_reelle_bati",
        "nombre_pieces_principales", "type_local", "prix_m2",
        "latitude", "longitude", "arrondissement",
    ]
    available_cols = [c for c in cols if c in page.columns]
    transactions = page[available_cols].replace({np.nan: None}).to_dict("records")

    # Format dates
    for t in transactions:
        if t.get("date_mutation") is not None:
            t["date_mutation"] = str(t["date_mutation"])[:10]

    return {
        "transactions": transactions,
        "total": total,
        "stats": stats,
    }
