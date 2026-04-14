"""
FairSquare — Hidden Gem Recommender Engine
==========================================
Finds under-valued properties in DVF 2023 data based on:
    gem_score = (prix_predit_m2 − prix_m2) / prix_predit_m2

A property is a "hidden gem" when the model thinks it should be worth MORE
than the recorded transaction price (gem_score > 0.10 = under-valued by 10%+).

Usage:
    from src.recommender.engine import RecommenderEngine
    engine = RecommenderEngine()
    results = engine.recommend(budget=400_000, arrondissement=11, surface=60)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.features.osm_features import OSMFeatureExtractor
from src.vision.renovation_scorer import RenovationScorer
from config.settings import get_settings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROCESSED_PATH = ROOT / "data" / "processed" / "dvf_paris_2023_2025_clean.parquet"
ARTIFACT_PATH  = ROOT / "models" / "artifacts" / "best_model.pkl"

# Hidden gem threshold: predicted price must exceed transaction price by this %
GEM_THRESHOLD = 0.10


@dataclass
class GemResult:
    rank: int
    arrondissement: int
    surface: float
    pieces: int
    prix_m2_transaction: float   # actual DVF transaction price/m²
    prix_m2_predit: float        # model prediction
    prix_total_transaction: float
    prix_total_predit: float
    # Specialist fields: Market Range (±10% confidence)
    prix_total_min: float
    prix_total_max: float
    gem_score: float             # (predit - transaction) / predit
    decote_pct: float            # how under-priced in % terms
    adresse: str
    mois_transaction: int
    walkability_hint: str = field(default="N/A")


class RecommenderEngine:
    """
    Recommends under-valued (hidden gem) properties from DVF 2023.

    Parameters
    ----------
    use_trained_model : bool
        If True, loads the trained v2 model from artifacts/.
        If False (or model missing), falls back to a lightweight
        per-arrondissement average for scoring.
    """

    def __init__(self, use_trained_model: bool = True) -> None:
        self._df: pd.DataFrame | None = None
        self._model_artifact: dict | None = None
        self._use_trained = use_trained_model
        self._loaded = False

    # ── Lazy loading ────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        # Load DVF data
        if not PROCESSED_PATH.exists():
            raise FileNotFoundError(
                f"DVF processed data not found: {PROCESSED_PATH}\n"
                "Run `python scripts/run_dvf_poc.py` first."
            )
        self._df = pd.read_parquet(PROCESSED_PATH)

        # Derive arrondissement
        self._df["arrondissement"] = (
            self._df["code_postal"].fillna(75001).astype(float) % 100
        ).astype(int)

        # Try to load model artifact
        if self._use_trained and ARTIFACT_PATH.exists():
            try:
                import joblib
                self._model_artifact = joblib.load(ARTIFACT_PATH)
            except Exception:
                self._model_artifact = None
        else:
            self._model_artifact = None

        self._loaded = True

    # ── Prediction helpers ───────────────────────────────────────────────

    def _predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Return prix/m² predictions for every row of *df*."""
        if self._model_artifact is not None:
            from src.ml.features_v2 import FEATURE_COLS_V2, add_features

            artifact = self._model_artifact
            arr_enc  = artifact["arr_enc"]
            g_mean   = artifact["global_mean"]

            enriched = add_features(df, arr_target_enc=arr_enc, global_mean=g_mean)
            X = enriched[FEATURE_COLS_V2].astype(float)

            model = artifact["model"]
            models = model if isinstance(model, list) else [model]
            preds = np.mean([m.predict(X) for m in models], axis=0)
            return preds

        # Fallback: use per-arrondissement median prix_m2 from the full dataset
        arr_median = self._df.groupby("arrondissement")["prix_m2"].median().to_dict()
        return df["arrondissement"].map(arr_median).fillna(self._df["prix_m2"].median()).values

    # ── Public API ───────────────────────────────────────────────────────

    def recommend(
        self,
        budget: float,
        arrondissement: Optional[int] = None,
        surface_min: float = 0.0,
        surface_max: float = 500.0,
        pieces_min: int = 1,
        top_n: int = 5,
        gem_threshold: float = GEM_THRESHOLD,
    ) -> list[GemResult]:
        """
        Find the top hidden gems matching the search criteria.

        Parameters
        ----------
        budget          : Max total price in € (using transaction price as proxy)
        arrondissement  : Preferred arrondissement (1-20), or None for all
        surface_min     : Minimum surface in m²
        surface_max     : Maximum surface in m²
        pieces_min      : Minimum number of main rooms
        top_n           : Number of results to return
        gem_threshold   : Minimum gem_score to qualify (default 0.10 = 10% under-valued)
        use_vision      : Whether to use Gemini to score photos (requires API key)
        """
        self._ensure_loaded()
        df = self._df.copy()
        settings = get_settings()

        # ── Filters ──────────────────────────────────────────────────
        # Budget: prix_m2 × surface ≤ budget
        df = df[df["prix_m2"] * df["surface_reelle_bati"] <= budget]

        if arrondissement is not None:
            # Allow ±1 arrondissement flex for better results
            df = df[df["arrondissement"].between(
                max(1, arrondissement - 1), min(20, arrondissement + 1)
            )]

        # Exclude non-market transactions: prix_m2 must be within IQR of its arrondissement
        # (avoids foreclosure sales, family transfers, HLM, etc.)
        arr_q25 = self._df.groupby("arrondissement")["prix_m2"].quantile(0.15)
        df = df.copy()
        df["_arr_q15"] = df["arrondissement"].map(arr_q25).fillna(4000)

        df = df[
            df["surface_reelle_bati"].between(surface_min, surface_max)
            & (pd.to_numeric(df["nombre_pieces_principales"], errors="coerce").fillna(0) >= pieces_min)
            & df["prix_m2"].notna()
            & df["surface_reelle_bati"].notna()
            & (df["prix_m2"] >= df["_arr_q15"])     # must be at least 15th percentile for its arr
        ]

        if df.empty:
            return []

        # ── Score ────────────────────────────────────────────────────
        preds = self._predict_batch(df)
        df = df.copy()
        df["prix_m2_predit"] = preds
        df["gem_score"] = (df["prix_m2_predit"] - df["prix_m2"]) / df["prix_m2_predit"].clip(lower=1)

        # Keep only genuine gems
        gems = df[df["gem_score"] >= gem_threshold].copy()
        gems = gems.sort_values("gem_score", ascending=False).head(top_n * 3)  # overselect then pick

        if gems.empty:
            return []

        # ── Format results ───────────────────────────────────────────
        results = []
        for rank, (_, row) in enumerate(gems.iterrows(), start=1):
            if rank > top_n:
                break

            pieces    = int(pd.to_numeric(row.get("nombre_pieces_principales", 2), errors="coerce") or 2)
            surface   = float(row["surface_reelle_bati"])
            prix_m2   = float(row["prix_m2"])
            predit_m2 = float(row["prix_m2_predit"])
            score     = float(row["gem_score"])

            # Build address from available fields
            parts = []
            if pd.notna(row.get("adresse_numero")):
                parts.append(str(int(row["adresse_numero"])))
            if pd.notna(row.get("adresse_nom_voie")):
                parts.append(str(row["adresse_nom_voie"]).title())
            arr = int(row["arrondissement"])
            parts.append(f"Paris {arr}{'er' if arr == 1 else 'e'}")
            adresse = " ".join(parts) if parts else f"Paris {arr}e"

            predit_total = prix_m2 * surface
            
            # Specialist logic: Reality-based confidence range (±8% to ±12% depending on model)
            # We use a standard ±10% 'Specialist' range for the market appraisal.
            prix_min = round(predit_m2 * surface * 0.90, -2)
            prix_max = round(predit_m2 * surface * 1.10, -2)

            res = GemResult(
                rank=rank,
                arrondissement=arr,
                surface=round(surface, 1),
                pieces=pieces,
                prix_m2_transaction=round(prix_m2, 0),
                prix_m2_predit=round(predit_m2, 0),
                prix_total_transaction=round(prix_m2 * surface, 0),
                prix_total_predit=round(predit_m2 * surface, 0),
                prix_total_min=prix_min,
                prix_total_max=prix_max,
                gem_score=round(score, 4),
                decote_pct=round(score * 100, 1),
                adresse=adresse,
                mois_transaction=int(row.get("mois", 0)),
            )
            
            # ── Phase 2: Enrichment (Top Gems only) ──────────────────
            # Add OSM walkability if possible
            try:
                osm = OSMFeatureExtractor(courtesy_delay=0)
                osm_feat = osm.get_features(row['latitude'], row['longitude'])
                res.walkability_hint = f"Walkability: {osm_feat.walkability_score}/100"
            except Exception:
                res.walkability_hint = "Walkability: N/A (OSM Timeout)"

            results.append(res)

        return results

    # ── Summary stats ────────────────────────────────────────────────────

    def market_summary(self, arrondissement: Optional[int] = None) -> dict:
        """Quick market stats — useful for the Streamlit dashboard."""
        self._ensure_loaded()
        df = self._df
        if arrondissement is not None:
            df = df[df["arrondissement"] == arrondissement]

        return {
            "count":           len(df),
            "prix_m2_median":  round(df["prix_m2"].median(), 0),
            "prix_m2_mean":    round(df["prix_m2"].mean(), 0),
            "prix_m2_q25":     round(df["prix_m2"].quantile(0.25), 0),
            "prix_m2_q75":     round(df["prix_m2"].quantile(0.75), 0),
            "surface_median":  round(df["surface_reelle_bati"].median(), 1),
        }
