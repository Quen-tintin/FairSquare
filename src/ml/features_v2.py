"""
Enhanced feature engineering for FairSquare v2.
Adds target encoding, geographic features, and surface interactions
to push R² from 0.19 → 0.5+.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Feature columns used by the v2 model ────────────────────────────
FEATURE_COLS_V2 = [
    "log_surface",
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "pieces_per_m2",
    "surface_per_piece",
    "carrez_ratio",
    "arrondissement",
    "arr_target_enc",          # mean prix_m2 per arr (train-only)
    "arr_price_x_log_surface", # interaction
    "is_premium_arr",
    "dist_center_km",
    "longitude",
    "latitude",
    "annee",                   # temporal trend 2023→2025
    "mois",
    "trimestre",
    "nombre_lots",
]

TARGET = "prix_m2"

PARIS_CENTER_LAT = 48.8566
PARIS_CENTER_LON = 2.3522

# Arrondissements with consistently above-average prices
PREMIUM_ARRONDISSEMENTS = {1, 2, 3, 4, 6, 7, 8, 16, 17}


def haversine_km(lat: pd.Series, lon: pd.Series,
                 ref_lat: float, ref_lon: float) -> pd.Series:
    """Vectorised haversine distance in km."""
    R = 6371.0
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = np.radians(ref_lat) - lat_r
    dlon = np.radians(ref_lon) - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(np.radians(ref_lat)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame,
                 arr_target_enc: dict[int, float] | None = None,
                 global_mean: float = 11071.0) -> pd.DataFrame:
    """
    Add engineered features to *df* in-place copy.

    Parameters
    ----------
    arr_target_enc : dict mapping arrondissement (1-20) → mean prix_m2.
        Must be computed from the **training set only** to avoid leakage.
        If None, uses global_mean as fallback for all rows.
    global_mean : fallback mean used when arr_target_enc is None or key missing.
    """
    data = df.copy()

    # ── Arrondissement (1-20) ─────────────────────────────────────────
    data["arrondissement"] = (
        data["code_postal"].fillna(75001).astype(float) % 100
    ).astype(int)

    # ── Surface features ─────────────────────────────────────────────
    surface = data["surface_reelle_bati"].clip(lower=1)
    data["log_surface"] = np.log1p(surface)

    pieces = pd.to_numeric(data["nombre_pieces_principales"], errors="coerce").fillna(2).clip(lower=1)
    data["pieces_per_m2"]   = (pieces / surface).clip(0.005, 0.5)
    data["surface_per_piece"] = (surface / pieces).clip(5, 500)

    # ── Carrez ratio ─────────────────────────────────────────────────
    carrez_cols = [c for c in [f"lot{i}_surface_carrez" for i in range(1, 6)]
                   if c in data.columns]
    total_carrez = data[carrez_cols].fillna(0).sum(axis=1)
    # Where no carrez data at all, fall back to surface (ratio = 1)
    data["carrez_ratio"] = np.where(
        total_carrez > 0,
        (total_carrez / surface).clip(0.5, 2.0),
        1.0,
    )

    # ── Geographic features ─────────────────────────────────────────
    lat = data["latitude"].fillna(PARIS_CENTER_LAT)
    lon = data["longitude"].fillna(PARIS_CENTER_LON)
    data["dist_center_km"] = haversine_km(lat, lon, PARIS_CENTER_LAT, PARIS_CENTER_LON)

    # ── Premium arrondissement flag ──────────────────────────────────
    data["is_premium_arr"] = data["arrondissement"].isin(PREMIUM_ARRONDISSEMENTS).astype(int)

    # ── Target encoding ──────────────────────────────────────────────
    if arr_target_enc is not None:
        data["arr_target_enc"] = (
            data["arrondissement"].map(arr_target_enc).fillna(global_mean)
        )
    else:
        data["arr_target_enc"] = global_mean

    # ── Interaction ──────────────────────────────────────────────────
    data["arr_price_x_log_surface"] = data["arr_target_enc"] * data["log_surface"]

    # ── Ensure nombre_pieces_principales is float ────────────────────
    data["nombre_pieces_principales"] = pieces

    # ── Trimestre (already in df from cleaner) ───────────────────────
    if "trimestre" not in data.columns:
        data["trimestre"] = ((data["mois"].fillna(6) - 1) // 3 + 1).astype(int)

    # ── Annee (temporal trend) ────────────────────────────────────────
    if "annee" not in data.columns:
        data["annee"] = 2023  # fallback for single-year datasets

    return data


def prepare_features_v2(
    df: pd.DataFrame,
    arr_target_enc: dict[int, float] | None = None,
    global_mean: float = 11071.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full pipeline: add features → select columns → drop NaN rows.
    Returns (X, y).
    """
    data = add_features(df, arr_target_enc=arr_target_enc, global_mean=global_mean)

    X = data[FEATURE_COLS_V2].copy()
    y = data[TARGET].copy()

    mask = X.notna().all(axis=1) & y.notna()
    return X[mask].astype(float).reset_index(drop=True), y[mask].reset_index(drop=True)


def compute_arr_target_enc(df_train: pd.DataFrame) -> dict[int, float]:
    """
    Compute mean prix_m2 per arrondissement on the training subset.
    Safe to call before `add_features` — only needs code_postal + prix_m2.
    """
    arr = (df_train["code_postal"].fillna(75001).astype(float) % 100).astype(int)
    return df_train.groupby(arr)["prix_m2"].mean().to_dict()
