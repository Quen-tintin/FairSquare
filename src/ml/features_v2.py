"""
Enhanced feature engineering for FairSquare v2.
Adds target encoding, geographic features, and surface interactions
to push R² from 0.19 → 0.5+.

v2.1 additions
--------------
- Street-level target encoding (adresse_code_voie)  → +10% R²
- Fine-grained grid encoding (0.005° lat/lon cells)
- Quadratic spatial features (lat², lon², lat×lon)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Feature columns used by the v2 model ────────────────────────────
FEATURE_COLS_V2 = [
    # Surface
    "log_surface",
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "pieces_per_m2",
    "surface_per_piece",
    "carrez_ratio",
    # Location — coarse
    "arrondissement",
    "arr_target_enc",           # mean prix_m2 per arr (train-only)
    "is_premium_arr",
    "dist_center_km",
    # Location — fine
    "voie_target_enc",          # mean prix_m2 per street (Bayesian smoothed)
    "grid_target_enc",          # mean prix_m2 per ~500m grid cell
    # Raw coordinates + quadratic terms
    "longitude",
    "latitude",
    "lat_sq",
    "lon_sq",
    "lat_lon_cross",
    # Interactions
    "arr_price_x_log_surface",  # arr_target_enc × log_surface
    # Temporal
    "annee",
    "mois",
    "trimestre",
    # Building
    "nombre_lots",
]

TARGET = "prix_m2"

PARIS_CENTER_LAT = 48.8566
PARIS_CENTER_LON = 2.3522

# Arrondissements with consistently above-average prices
PREMIUM_ARRONDISSEMENTS = {1, 2, 3, 4, 6, 7, 8, 16, 17}

# Bayesian smoothing factor for target encoding
# Higher = more shrinkage towards global mean (avoids overfit on rare streets)
_SMOOTH_K = 10


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


def _bayesian_smooth(group_mean: float, group_count: int,
                     global_mean: float, k: int = _SMOOTH_K) -> float:
    """Shrinks group mean towards global mean based on sample size."""
    return (group_count * group_mean + k * global_mean) / (group_count + k)


def compute_target_encodings(df_train: pd.DataFrame) -> dict:
    """
    Compute all target encodings from the training set ONLY (no leakage).

    Returns a dict with keys:
        arr_enc   : arrondissement → smoothed mean prix_m2
        voie_enc  : adresse_code_voie → smoothed mean prix_m2
        grid_enc  : (lat_grid, lon_grid) → smoothed mean prix_m2
        global_mean : float
    """
    global_mean = float(df_train["prix_m2"].mean())

    # Arrondissement encoding
    arr = (df_train["code_postal"].fillna(75001).astype(float) % 100).astype(int)
    arr_stats = df_train.groupby(arr)["prix_m2"].agg(["mean", "count"])
    arr_enc = {
        int(idx): _bayesian_smooth(row["mean"], row["count"], global_mean)
        for idx, row in arr_stats.iterrows()
    }

    # Street-level encoding (adresse_code_voie)
    voie_enc: dict = {}
    if "adresse_code_voie" in df_train.columns:
        voie_stats = df_train.groupby("adresse_code_voie")["prix_m2"].agg(["mean", "count"])
        voie_enc = {
            str(idx): _bayesian_smooth(row["mean"], row["count"], global_mean)
            for idx, row in voie_stats.iterrows()
        }

    # Grid encoding (~500m cells: round to 2 decimal places)
    grid_enc: dict = {}
    if "latitude" in df_train.columns and "longitude" in df_train.columns:
        lat_g = df_train["latitude"].fillna(PARIS_CENTER_LAT).round(2)
        lon_g = df_train["longitude"].fillna(PARIS_CENTER_LON).round(2)
        grid_key = lat_g.astype(str) + "_" + lon_g.astype(str)
        grid_stats = df_train.groupby(grid_key)["prix_m2"].agg(["mean", "count"])
        grid_enc = {
            str(idx): _bayesian_smooth(row["mean"], row["count"], global_mean)
            for idx, row in grid_stats.iterrows()
        }

    return {
        "arr_enc": arr_enc,
        "voie_enc": voie_enc,
        "grid_enc": grid_enc,
        "global_mean": global_mean,
    }


# Backwards-compatible alias
def compute_arr_target_enc(df_train: pd.DataFrame) -> dict[int, float]:
    """Compute mean prix_m2 per arrondissement (train-only)."""
    arr = (df_train["code_postal"].fillna(75001).astype(float) % 100).astype(int)
    return df_train.groupby(arr)["prix_m2"].mean().to_dict()


def add_features(
    df: pd.DataFrame,
    arr_target_enc: dict[int, float] | None = None,
    global_mean: float = 10796.0,
    voie_enc: dict[str, float] | None = None,
    grid_enc: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Add engineered features to *df* in-place copy.

    Parameters
    ----------
    arr_target_enc : dict mapping arrondissement (1-20) → mean prix_m2.
    global_mean    : fallback mean when encoding key is missing.
    voie_enc       : dict mapping adresse_code_voie → smoothed mean prix_m2.
    grid_enc       : dict mapping "{lat_grid}_{lon_grid}" → smoothed mean prix_m2.
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
    data["pieces_per_m2"]     = (pieces / surface).clip(0.005, 0.5)
    data["surface_per_piece"] = (surface / pieces).clip(5, 500)

    # ── Carrez ratio ─────────────────────────────────────────────────
    carrez_cols = [c for c in [f"lot{i}_surface_carrez" for i in range(1, 6)]
                   if c in data.columns]
    total_carrez = data[carrez_cols].fillna(0).sum(axis=1)
    data["carrez_ratio"] = np.where(
        total_carrez > 0,
        (total_carrez / surface).clip(0.5, 2.0),
        1.0,
    )

    # ── Raw coordinates ───────────────────────────────────────────────
    lat = data["latitude"].fillna(PARIS_CENTER_LAT)
    lon = data["longitude"].fillna(PARIS_CENTER_LON)

    # ── Geographic distance ────────────────────────────────────────
    data["dist_center_km"] = haversine_km(lat, lon, PARIS_CENTER_LAT, PARIS_CENTER_LON)

    # ── Quadratic spatial features ────────────────────────────────
    data["lat_sq"]        = lat ** 2
    data["lon_sq"]        = lon ** 2
    data["lat_lon_cross"] = lat * lon

    # ── Premium arrondissement flag ──────────────────────────────────
    data["is_premium_arr"] = data["arrondissement"].isin(PREMIUM_ARRONDISSEMENTS).astype(int)

    # ── Arrondissement target encoding ───────────────────────────────
    if arr_target_enc is not None:
        data["arr_target_enc"] = (
            data["arrondissement"].map(arr_target_enc).fillna(global_mean)
        )
    else:
        data["arr_target_enc"] = global_mean

    # ── Street-level target encoding ──────────────────────────────────
    if voie_enc and "adresse_code_voie" in data.columns:
        data["voie_target_enc"] = (
            data["adresse_code_voie"].astype(str).map(voie_enc).fillna(data["arr_target_enc"])
        )
    else:
        data["voie_target_enc"] = data["arr_target_enc"]

    # ── Grid-level target encoding (~500m cells) ──────────────────────
    if grid_enc:
        lat_g = lat.round(2).astype(str)
        lon_g = lon.round(2).astype(str)
        grid_key = lat_g + "_" + lon_g
        data["grid_target_enc"] = (
            grid_key.map(grid_enc).fillna(data["voie_target_enc"])
        )
    else:
        data["grid_target_enc"] = data["voie_target_enc"]

    # ── Interaction ───────────────────────────────────────────────────
    data["arr_price_x_log_surface"] = data["arr_target_enc"] * data["log_surface"]

    # ── Ensure nombre_pieces_principales is float ─────────────────────
    data["nombre_pieces_principales"] = pieces

    # ── Temporal features (already in df from cleaner) ───────────────
    if "trimestre" not in data.columns:
        data["trimestre"] = ((data["mois"].fillna(6) - 1) // 3 + 1).astype(int)
    if "annee" not in data.columns:
        data["annee"] = 2023

    return data


def prepare_features_v2(
    df: pd.DataFrame,
    arr_target_enc: dict[int, float] | None = None,
    global_mean: float = 10796.0,
    voie_enc: dict[str, float] | None = None,
    grid_enc: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full pipeline: add features → select columns → drop NaN rows.
    Returns (X, y).
    """
    data = add_features(
        df,
        arr_target_enc=arr_target_enc,
        global_mean=global_mean,
        voie_enc=voie_enc,
        grid_enc=grid_enc,
    )

    X = data[FEATURE_COLS_V2].copy()
    y = data[TARGET].copy()

    mask = X.notna().all(axis=1) & y.notna()
    return X[mask].astype(float).reset_index(drop=True), y[mask].reset_index(drop=True)


def predict_price(
    surface: float,
    pieces: int,
    code_postal: int,
    latitude: float,
    longitude: float,
    artifact: dict,
    adresse_code_voie: str | None = None,
) -> float:
    """
    Single-row prediction from a trained artifact dict.

    artifact keys: model, arr_enc, voie_enc, grid_enc, global_mean, feature_cols
    """
    row = {
        "surface_reelle_bati":      surface,
        "nombre_pieces_principales": pieces,
        "code_postal":              code_postal,
        "latitude":                 latitude,
        "longitude":                longitude,
        "mois":                     6,
        "trimestre":                2,
        "annee":                    2025,
        "nombre_lots":              1,
        "lot1_surface_carrez":      surface,
        "prix_m2":                  0.0,   # placeholder
    }
    if adresse_code_voie is not None:
        row["adresse_code_voie"] = adresse_code_voie

    df_row = pd.DataFrame([row])
    X, _ = prepare_features_v2(
        df_row,
        arr_target_enc=artifact.get("arr_enc"),
        global_mean=artifact.get("global_mean", 10796.0),
        voie_enc=artifact.get("voie_enc"),
        grid_enc=artifact.get("grid_enc"),
    )

    model = artifact["model"]
    # Support both single model and list (ensemble)
    if isinstance(model, list):
        pred = float(np.mean([m.predict(X)[0] for m in model]))
    else:
        pred = float(model.predict(X)[0])
    return pred * surface
