"""Feature preparation for ML models."""
from __future__ import annotations

import pandas as pd

FEATURE_COLS = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "arrondissement",
    "longitude",
    "latitude",
    "mois",
    "nombre_lots",
]
TARGET = "prix_m2"

FEATURE_LABELS = {
    "surface_reelle_bati": "Surface (m2)",
    "nombre_pieces_principales": "Nb pieces",
    "arrondissement": "Arrondissement",
    "longitude": "Longitude",
    "latitude": "Latitude",
    "mois": "Mois",
    "nombre_lots": "Nb lots",
}


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select and clean features. Returns (X, y) with no NaN rows."""
    data = df.copy()

    # Derive arrondissement from code_postal (75001 -> 1, 75016 -> 16)
    data["arrondissement"] = (data["code_postal"].fillna(75001) % 100).astype(float)

    # Nullable Int64 -> float
    data["nombre_pieces_principales"] = pd.to_numeric(
        data["nombre_pieces_principales"], errors="coerce"
    )
    data["nombre_lots"] = pd.to_numeric(data["nombre_lots"], errors="coerce").fillna(1)

    X = data[FEATURE_COLS].copy()
    y = data[TARGET].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].astype(float).reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    return X, y
