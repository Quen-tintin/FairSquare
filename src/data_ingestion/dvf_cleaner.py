"""
DVF Data Cleaner & EDA
======================
Transforme le DataFrame brut DVF en un DataFrame propre, typé et enrichi,
prêt pour le feature engineering.

Colonnes clés conservées / renommées :
  - id_mutation       : identifiant unique de la transaction
  - date_mutation     : date de la vente
  - valeur_fonciere   : prix de vente (€)
  - surface_reelle_bati : surface habitable (m²)
  - nombre_pieces_principales : nombre de pièces
  - type_local        : Appartement / Maison / etc.
  - latitude / longitude
  - code_departement, code_commune
  - prix_m2           : feature dérivée (valeur / surface)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
#  Colonnes brutes DVF → noms normalisés                              #
# ------------------------------------------------------------------ #
_RENAME_MAP: dict[str, str] = {
    "idmutation": "id_mutation",
    "datemutation": "date_mutation",
    "valeurfonc": "valeur_fonciere",
    "sbati": "surface_reelle_bati",
    "nbpprinc": "nombre_pieces_principales",
    "libtyplocmut": "type_local",
    "lat": "latitude",
    "lon": "longitude",
    "codedep": "code_departement",
    "codecommunedep": "code_commune",
}

# Types locaux à conserver (on exclut les terrains, parkings seuls, etc.)
_TYPES_LOCAUX_CIBLES = {"Appartement", "Maison"}

# Seuils de filtrage (valeurs aberrantes)
_PRIX_MIN = 10_000          # € — en dessous : sûrement une erreur
_PRIX_MAX = 30_000_000      # € — ventes atypiques / immeubles entiers
_SURFACE_MIN = 5            # m²
_SURFACE_MAX = 1_500        # m² — inclut les grands apparts légitimes
_PRIX_M2_MIN = 2_000        # €/m² — élimine les valeurs aberrantes basses
_PRIX_M2_MAX = 50_000       # €/m²


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de nettoyage DVF.

    Étapes :
        1. Renommage des colonnes
        2. Conversion des types
        3. Filtrage métier (type de bien, seuils)
        4. Suppression des doublons
        5. Création de features dérivées basiques

    Args:
        df: DataFrame brut issu de DVFClient.

    Returns:
        DataFrame propre, prêt pour le feature engineering.
    """
    logger.info("Starting DVF cleaning — %d rows in", len(df))
    df = df.copy()

    df = _rename_columns(df)
    df = _cast_types(df)
    df = _filter_type_local(df)
    df = _filter_outliers(df)
    df = _drop_duplicates(df)
    df = _add_derived_features(df)

    logger.info("DVF cleaning done — %d rows out", len(df))
    return df


def eda_summary(df: pd.DataFrame) -> dict:
    """
    Génère un résumé EDA rapide : stats descriptives, valeurs manquantes,
    distribution par type de bien et par département.

    Returns:
        Dict structuré — facile à afficher ou à passer à un LLM.
    """
    numeric_cols = ["valeur_fonciere", "surface_reelle_bati", "prix_m2", "nombre_pieces_principales"]
    available = [c for c in numeric_cols if c in df.columns]

    summary = {
        "shape": df.shape,
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "descriptive_stats": df[available].describe().round(2).to_dict(),
        "type_local_counts": df["type_local"].value_counts().to_dict() if "type_local" in df.columns else {},
        "dept_counts": df["code_departement"].value_counts().to_dict() if "code_departement" in df.columns else {},
        "year_counts": (
            df["date_mutation"].dt.year.value_counts().sort_index().to_dict()
            if "date_mutation" in df.columns
            else {}
        ),
    }
    return summary


# ------------------------------------------------------------------ #
#  Étapes internes                                                     #
# ------------------------------------------------------------------ #

def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes en minuscules puis applique le mapping."""
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in _RENAME_MAP.items() if k in df.columns})
    return df


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes vers leurs types Python/Pandas corrects."""
    if "date_mutation" in df.columns:
        df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")

    for col in ["valeur_fonciere", "surface_reelle_bati", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "nombre_pieces_principales" in df.columns:
        df["nombre_pieces_principales"] = pd.to_numeric(
            df["nombre_pieces_principales"], errors="coerce"
        ).astype("Int64")  # nullable integer

    return df


def _filter_type_local(df: pd.DataFrame) -> pd.DataFrame:
    """Garde uniquement les appartements et maisons."""
    if "type_local" not in df.columns:
        logger.warning("Column 'type_local' not found — skipping type filter")
        return df
    before = len(df)
    df = df[df["type_local"].isin(_TYPES_LOCAUX_CIBLES)].copy()
    logger.debug("Type filter: %d → %d rows", before, len(df))
    return df


def _filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Élimine les valeurs aberrantes sur prix et surface."""
    before = len(df)
    mask = pd.Series(True, index=df.index)

    if "valeur_fonciere" in df.columns:
        mask &= df["valeur_fonciere"].between(_PRIX_MIN, _PRIX_MAX)

    if "surface_reelle_bati" in df.columns:
        mask &= df["surface_reelle_bati"].between(_SURFACE_MIN, _SURFACE_MAX)
        # Filtre prix/m² si les deux colonnes sont disponibles
        if "valeur_fonciere" in df.columns:
            prix_m2 = df["valeur_fonciere"] / df["surface_reelle_bati"]
            mask &= prix_m2.between(_PRIX_M2_MIN, _PRIX_M2_MAX)

    df = df[mask].copy()
    logger.debug("Outlier filter: %d → %d rows", before, len(df))
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons sur id_mutation si la colonne existe."""
    if "id_mutation" not in df.columns:
        return df
    before = len(df)
    df = df.drop_duplicates(subset=["id_mutation"]).copy()
    logger.debug("Dedup: %d → %d rows", before, len(df))
    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les premières features dérivées basiques."""
    if "valeur_fonciere" in df.columns and "surface_reelle_bati" in df.columns:
        df["prix_m2"] = (df["valeur_fonciere"] / df["surface_reelle_bati"]).round(2)

    if "date_mutation" in df.columns:
        df["annee"] = df["date_mutation"].dt.year
        df["mois"] = df["date_mutation"].dt.month
        df["trimestre"] = df["date_mutation"].dt.quarter

    return df
