"""Pydantic schemas for the /predict endpoint."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    surface: float = Field(..., gt=0, le=1000, description="Surface réelle bâtie (m²)")
    pieces: int = Field(..., ge=1, le=20, description="Nombre de pièces principales")
    arrondissement: int = Field(..., ge=1, le=20, description="Arrondissement Paris (1-20)")
    latitude: float = Field(..., ge=48.7, le=49.0, description="Latitude WGS84")
    longitude: float = Field(..., ge=2.0, le=2.7, description="Longitude WGS84")
    annee: int | None = Field(default=None, ge=2020, le=2030, description="Année de la transaction (défaut: année courante)")
    mois: int = Field(default=6, ge=1, le=12, description="Mois de la transaction (1-12)")
    trimestre: int = Field(default=2, ge=1, le=4, description="Trimestre (1-4)")
    nombre_lots: int = Field(default=1, ge=1, le=10, description="Nombre de lots")
    prix_affiche: float | None = Field(
        default=None, gt=0,
        description="Prix affiché en annonce (€) — si fourni, calcule le score Hidden Gem",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "surface": 65,
            "pieces": 3,
            "arrondissement": 11,
            "latitude": 48.853,
            "longitude": 2.3698,
            "mois": 6,
            "trimestre": 2,
            "nombre_lots": 1,
            "prix_affiche": 380000,
        }
    }}


class ShapContribution(BaseModel):
    feature: str
    value: float = Field(description="Contribution SHAP en €/m²")
    display: str = Field(description="Libellé lisible pour l'UI")


class PredictionResponse(BaseModel):
    prix_predit_m2: float = Field(description="Prix prédit par m² (€/m²)")
    prix_predit_total: float = Field(description="Prix total prédit (€)")
    confidence_low: float = Field(description="Borne basse intervalle de confiance 90% (€/m²)")
    confidence_high: float = Field(description="Borne haute intervalle de confiance 90% (€/m²)")
    shap_contributions: list[ShapContribution] = Field(
        description="Top contributions SHAP (les plus impactantes en premier)"
    )
    hidden_gem_score: float | None = Field(
        default=None,
        description="Score Hidden Gem — (prédit − affiché) / prédit. Positif = sous-évalué.",
    )
    is_hidden_gem: bool = Field(
        default=False,
        description="True si hidden_gem_score > 0.10 (sous-évalué de plus de 10%)",
    )
    xai_summary: str = Field(description="Explication en langage naturel (français)")
    model_version: str = Field(default="v2", description="Version du modèle utilisé")
