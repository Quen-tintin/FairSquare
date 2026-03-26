"""
POST /predict — load trained model, compute price prediction + SHAP.
Falls back to a rule-based estimate if no model artifact exists.
"""
from __future__ import annotations

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

# ── Confidence interval width (€/m²) — approximate from training residuals
_CI_HALF_WIDTH = 1500.0


@lru_cache(maxsize=1)
def _load_artifact() -> dict[str, Any] | None:
    """Load model artifact once and cache. Returns None if not found."""
    try:
        import joblib
        return joblib.load(ARTIFACT_PATH)
    except Exception:
        return None


def _build_input_df(req: PredictionRequest) -> pd.DataFrame:
    """Convert a PredictionRequest into a single-row DataFrame."""
    return pd.DataFrame([{
        "code_postal":               75000 + req.arrondissement,
        "surface_reelle_bati":       req.surface,
        "nombre_pieces_principales": req.pieces,
        "latitude":                  req.latitude,
        "longitude":                 req.longitude,
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


def _shap_contributions(model: Any, X: pd.DataFrame) -> list[ShapContribution]:
    """Compute SHAP values and return top contributions sorted by |value|."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)[0]  # single row → 1-D array
        contribs = []
        labels = {
            "log_surface":             "Surface (m²)",
            "surface_reelle_bati":     "Surface brute",
            "nombre_pieces_principales": "Nb de pièces",
            "pieces_per_m2":           "Densité pièces/m²",
            "surface_per_piece":       "Surface/pièce",
            "carrez_ratio":            "Ratio Carrez",
            "arrondissement":          "Arrondissement",
            "arr_target_enc":          "Moyenne arr.",
            "arr_price_x_log_surface": "Arr. × surface",
            "is_premium_arr":          "Arr. premium",
            "dist_center_km":          "Distance centre",
            "longitude":               "Longitude",
            "latitude":                "Latitude",
            "mois":                    "Mois",
            "trimestre":               "Trimestre",
            "nombre_lots":             "Nb lots",
        }
        for feat, val in zip(FEATURE_COLS_V2, sv):
            contribs.append(ShapContribution(
                feature=feat,
                value=round(float(val), 1),
                display=labels.get(feat, feat),
            ))
        contribs.sort(key=lambda c: abs(c.value), reverse=True)
        return contribs[:8]
    except Exception:
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

    # Build input
    df_in = _build_input_df(req)
    df_in = add_features(df_in, arr_target_enc=arr_enc, global_mean=global_m)
    X = df_in[FEATURE_COLS_V2].astype(float)

    # Predict
    preds = np.array([m.predict(X)[0] for m in models_list])
    prix_m2 = float(np.mean(preds))
    prix_total = prix_m2 * req.surface

    # SHAP (use first model)
    shap_list = _shap_contributions(models_list[0], X)

    # Hidden gem score
    gem_score = None
    is_gem = False
    if req.prix_affiche is not None:
        prix_affiche_m2 = req.prix_affiche / req.surface
        gem_score = round((prix_m2 - prix_affiche_m2) / prix_m2, 4)
        is_gem = gem_score > 0.10

    # XAI text
    dist_c = float(df_in["dist_center_km"].iloc[0])
    xai = _xai_text(prix_m2, shap_list, req.arrondissement, dist_c)

    return PredictionResponse(
        prix_predit_m2=round(prix_m2, 1),
        prix_predit_total=round(prix_total, 0),
        confidence_low=round(prix_m2 - _CI_HALF_WIDTH, 1),
        confidence_high=round(prix_m2 + _CI_HALF_WIDTH, 1),
        shap_contributions=shap_list,
        hidden_gem_score=gem_score,
        is_hidden_gem=is_gem,
        xai_summary=xai,
        model_version="v2",
    )
