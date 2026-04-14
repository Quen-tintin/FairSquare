"""
POST /predict — load trained model, compute price prediction + SHAP.
Falls back to a rule-based estimate if no model artifact exists.
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    ShapContribution,
)
from src.ml.features_v2 import FEATURE_COLS_V2, add_features

router = APIRouter(tags=["predictions"])

ARTIFACT_PATH = ROOT / "models" / "artifacts" / "best_model.pkl"

# ── OSM feature columns (available in dvf_paris_osm_enriched.parquet) ─
_OSM_COLS = ["nb_restaurants", "nb_transport", "nb_parks", "walkability_score", "dist_metro_m", "nb_ecoles"]

# Paris-wide medians used as defaults when parquet join misses a cell
_OSM_DEFAULTS = {
    "nb_restaurants":  102.0,
    "nb_transport":     28.0,
    "nb_parks":          6.0,
    "walkability_score": 43.0,
    "dist_metro_m":    214.0,
    "nb_ecoles":         9.0,
}

# V4 feature set extends V2 with OSM features (used when a v4 model is available)
FEATURE_COLS_V4 = FEATURE_COLS_V2 + _OSM_COLS

# ── Confidence interval width (€/m²) — default, overridden by artifact RMSE if available
_CI_HALF_WIDTH_DEFAULT = 1500.0


@lru_cache(maxsize=1)
def _load_artifact() -> dict[str, Any] | None:
    """Load model artifact once and cache. Returns None if not found."""
    try:
        import joblib
        return joblib.load(ARTIFACT_PATH)
    except Exception:
        logging.exception("Failed to load model artifact from %s", ARTIFACT_PATH)
        return None


def _build_input_df(req: PredictionRequest) -> pd.DataFrame:
    """Convert a PredictionRequest into a single-row DataFrame."""
    import datetime
    annee = req.annee if req.annee is not None else datetime.datetime.now().year
    df = pd.DataFrame([{
        "code_postal":               75000 + req.arrondissement,
        "surface_reelle_bati":       req.surface,
        "nombre_pieces_principales": req.pieces,
        "latitude":                  req.latitude,
        "longitude":                 req.longitude,
        "annee":                     annee,
        "mois":                      req.mois,
        "trimestre":                 req.trimestre,
        "nombre_lots":               req.nombre_lots,
        # carrez columns — unknown from API input, use NaN (→ ratio=1.0)
        "lot1_surface_carrez":       np.nan,
        "lot2_surface_carrez":       np.nan,
        "lot3_surface_carrez":       np.nan,
        "lot4_surface_carrez":       np.nan,
        "lot5_surface_carrez":       np.nan,
    }])

    # OSM enrichment (best-effort, falls back to Paris medians)
    for col, val in _OSM_DEFAULTS.items():
        df[col] = val   # pre-fill with medians; overwritten below if parquet available
    try:
        osm_path = ROOT / "data" / "outputs" / "dvf_paris_osm_enriched.parquet"
        if osm_path.exists():
            osm_df = pd.read_parquet(osm_path, columns=["lat_grid", "lon_grid"] + _OSM_COLS)
            grid_lat = round(req.latitude / 0.005) * 0.005
            grid_lon = round(req.longitude / 0.005) * 0.005
            osm_row = osm_df[(osm_df.lat_grid == grid_lat) & (osm_df.lon_grid == grid_lon)]
            if not osm_row.empty:
                for col in _OSM_COLS:
                    df[col] = float(osm_row[col].values[0])
    except Exception:
        logging.warning("OSM enrichment failed — using Paris-wide medians", exc_info=True)

    return df


def _shap_contributions(model: Any, X: pd.DataFrame,
                        prix_m2: float, feat_cols: list[str]) -> list[ShapContribution]:
    """Compute SHAP values and return top contributions sorted by |value|.

    Model is trained on log1p(prix_m2) → SHAP values are in log-space.
    Multiply by prix_m2 to convert to approximate €/m² contributions.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)[0]  # single row → 1-D array
        # Convert log-space SHAP → €/m²  (delta_log ≈ delta_price / price)
        sv_eur = sv * prix_m2
        labels = {
            "log_surface":             "Surface (m²)",
            "surface_reelle_bati":     "Surface brute",
            "nombre_pieces_principales": "Nb de pièces",
            "pieces_per_m2":           "Densité pièces/m²",
            "surface_per_piece":       "Surface/pièce",
            "carrez_ratio":            "Ratio Carrez",
            "arrondissement":          "Arrondissement",
            "arr_target_enc":          "Moyenne arr.",
            "is_premium_arr":          "Arr. premium",
            "dist_center_km":          "Distance centre",
            "voie_target_enc":         "Rue / voie",
            "grid_target_enc":         "Zone géographique fine",
            "voie_recent_prix_m2":     "Prix récents (même rue)",
            "longitude":               "Longitude",
            "latitude":                "Latitude",
            "lat_sq":                  "Coord. lat²",
            "lon_sq":                  "Coord. lon²",
            "lat_lon_cross":           "Coord. lat×lon",
            "arr_price_x_log_surface": "Arr. × surface",
            "premium_x_log_surface":   "Prestige × surface",
            "premium_x_dist_center":   "Prestige × distance",
            "voie_x_density":          "Rue × densité",
            "annee":                   "Année transaction",
            "mois":                    "Mois",
            "trimestre":               "Trimestre",
            "nombre_lots":             "Nb lots",
        }
        contribs = []
        for feat, val in zip(feat_cols, sv_eur):
            contribs.append(ShapContribution(
                feature=feat,
                value=round(float(val), 0),
                display=labels.get(feat, feat),
            ))
        contribs.sort(key=lambda c: abs(c.value), reverse=True)
        return contribs[:8]
    except Exception:
        logging.warning("SHAP computation failed", exc_info=True)
        return []


def _xai_text(prix_m2: float, shap_list: list[ShapContribution],
              arrondissement: int, dist_center: float) -> str:
    top_pos = [c for c in shap_list if c.value > 0][:2]
    top_neg = [c for c in shap_list if c.value < 0][:2]

    pos_txt = ", ".join(f"**{c.display}** (+{c.value:,.0f} €/m²)" for c in top_pos)
    neg_txt = ", ".join(f"**{c.display}** ({c.value:,.0f} €/m²)" for c in top_neg)

    arr_str = f"{arrondissement}{'er' if arrondissement == 1 else 'e'} arrondissement"
    parts = [f"Ce bien dans le {arr_str} (à {dist_center:.1f} km du centre) "
             f"est estimé à **{prix_m2:,.0f} €/m²**."]
    if pos_txt:
        parts.append(f"Facteurs valorisants : {pos_txt}.")
    if neg_txt:
        parts.append(f"Facteurs pénalisants : {neg_txt}.")
    return " ".join(parts)


# Approximate centroids per arrondissement (lat, lon)
_ARR_CENTROIDS = {
    1:  (48.8603, 2.3477),  2:  (48.8666, 2.3504),
    3:  (48.8630, 2.3601),  4:  (48.8533, 2.3526),
    5:  (48.8462, 2.3500),  6:  (48.8490, 2.3340),
    7:  (48.8562, 2.3187),  8:  (48.8745, 2.3084),
    9:  (48.8777, 2.3358),  10: (48.8759, 2.3622),
    11: (48.8589, 2.3796),  12: (48.8427, 2.3946),
    13: (48.8315, 2.3626),  14: (48.8280, 2.3259),
    15: (48.8420, 2.3014),  16: (48.8636, 2.2735),
    17: (48.8905, 2.3139),  18: (48.8926, 2.3474),
    19: (48.8847, 2.3799),  20: (48.8646, 2.3979),
}
_MAX_ARR_DIST_KM = 3.0  # max plausible distance between lat/lon and arr centroid


def _validate_coordinates(req: "PredictionRequest") -> tuple[float, float]:
    """
    Check that lat/lon is geographically consistent with the declared arrondissement.
    If the distance exceeds _MAX_ARR_DIST_KM, snap to the arrondissement centroid
    and log a warning (scraping artefact rather than hard error).
    Returns (lat, lon) — possibly corrected.
    """
    centroid = _ARR_CENTROIDS.get(req.arrondissement)
    if centroid is None:
        return req.latitude, req.longitude

    clat, clon = centroid
    R = 6371.0
    dlat = np.radians(clat - req.latitude)
    dlon = np.radians(clon - req.longitude)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(req.latitude)) * np.cos(np.radians(clat))
         * np.sin(dlon / 2) ** 2)
    dist_km = R * 2 * np.arcsin(np.sqrt(a))

    if dist_km > _MAX_ARR_DIST_KM:
        logging.warning(
            "lat/lon (%.4f, %.4f) is %.1f km from arr %d centroid — snapping to centroid",
            req.latitude, req.longitude, dist_km, req.arrondissement,
        )
        return clat, clon
    return req.latitude, req.longitude


@router.post("/predict", response_model=PredictionResponse, summary="Prédire le prix d'un bien")
async def predict(req: PredictionRequest):
    artifact = _load_artifact()

    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Lancez scripts/train_improved_model.py d'abord.",
        )

    model     = artifact["model"]
    arr_enc   = artifact["arr_enc"]
    global_m  = artifact["global_mean"]

    # Handle ensemble (list of models) vs single model
    models_list = model if isinstance(model, list) else [model]

    # Validate + correct lat/lon vs declared arrondissement
    lat, lon = _validate_coordinates(req)
    req = req.model_copy(update={"latitude": lat, "longitude": lon})

    # Build input
    df_in = _build_input_df(req)

    # voie_recent_prix_m2: use voie-level lookup if adresse_code_voie provided,
    # otherwise fallback to arrondissement-level recent median
    voie_recent_lookup = artifact.get("voie_recent_median_lookup", {})
    arr_recent = artifact.get("arr_recent_median_lookup", {})
    if req.adresse_code_voie and req.adresse_code_voie in voie_recent_lookup:
        df_in["voie_recent_prix_m2"] = float(voie_recent_lookup[req.adresse_code_voie])
        df_in["adresse_code_voie"] = req.adresse_code_voie
    else:
        df_in["voie_recent_prix_m2"] = float(arr_recent.get(req.arrondissement, global_m))

    df_in = add_features(df_in, arr_target_enc=arr_enc, global_mean=global_m)
    feat_cols = artifact.get("feature_cols", FEATURE_COLS_V2)
    avail_cols = [c for c in feat_cols if c in df_in.columns]
    X = df_in[avail_cols].astype(float)

    # Predict — handle log-target transformation
    log_target = artifact.get("log_target", False)
    preds = np.array([m.predict(X)[0] for m in models_list])
    raw_pred = float(np.mean(preds))
    prix_m2 = float(np.expm1(raw_pred)) if log_target else raw_pred
    prix_total = prix_m2 * req.surface

    # SHAP (use first model) — pass prix_m2 for log→€/m² conversion
    shap_list = _shap_contributions(models_list[0], X, prix_m2, avail_cols)

    # Hidden gem score — with negotiation margin correction
    # DVF = final sale price, annonce = asking price (~7% higher on average)
    _MARGE_NEGOCIATION = 0.07
    gem_score = None
    is_gem = False
    if req.prix_affiche is not None:
        prix_estime_vente = req.prix_affiche * (1 - _MARGE_NEGOCIATION)
        prix_estime_vente_m2 = prix_estime_vente / req.surface
        gem_score = round((prix_m2 - prix_estime_vente_m2) / prix_m2, 4)
        is_gem = gem_score > 0.10

    # Confidence interval — use RMSE from artifact if available
    _ci_half = float(artifact.get("rmse", _CI_HALF_WIDTH_DEFAULT))

    # XAI text
    dist_c = float(df_in["dist_center_km"].iloc[0])
    xai = _xai_text(prix_m2, shap_list, req.arrondissement, dist_c)

    return PredictionResponse(
        prix_predit_m2=round(prix_m2, 1),
        prix_predit_total=round(prix_total, 0),
        confidence_low=round(prix_m2 - _ci_half, 1),
        confidence_high=round(prix_m2 + _ci_half, 1),
        shap_contributions=shap_list,
        hidden_gem_score=gem_score,
        is_hidden_gem=is_gem,
        xai_summary=xai,
        model_version="v2",
    )
