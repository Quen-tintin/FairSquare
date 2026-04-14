"""Unit tests — feature engineering (features_v2.py)."""
import numpy as np
import pandas as pd
import pytest

from src.ml.features_v2 import add_features, FEATURE_COLS_V2, haversine_km


_BASE_ROW = {
    "code_postal": 75011,
    "surface_reelle_bati": 65.0,
    "nombre_pieces_principales": 3,
    "latitude": 48.853,
    "longitude": 2.3698,
    "annee": 2024,
    "mois": 6,
    "trimestre": 2,
    "nombre_lots": 1,
    "lot1_surface_carrez": np.nan,
    "lot2_surface_carrez": np.nan,
    "lot3_surface_carrez": np.nan,
    "lot4_surface_carrez": np.nan,
    "lot5_surface_carrez": np.nan,
    "prix_m2": 10000.0,
}


def _make_df(**overrides):
    row = {**_BASE_ROW, **overrides}
    return pd.DataFrame([row])


# ── haversine ────────────────────────────────────────────────────────────────

def test_haversine_zero():
    s = pd.Series([48.8566])
    assert haversine_km(s, pd.Series([2.3522]), 48.8566, 2.3522).iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_paris_london():
    # Paris → London ~340 km
    s_lat = pd.Series([51.5074])
    s_lon = pd.Series([-0.1278])
    dist = haversine_km(s_lat, s_lon, 48.8566, 2.3522).iloc[0]
    assert 330 < dist < 350


# ── add_features ─────────────────────────────────────────────────────────────

def test_add_features_basic_columns():
    df = add_features(_make_df())
    for col in FEATURE_COLS_V2:
        assert col in df.columns, f"Missing feature: {col}"


def test_log_surface():
    df = add_features(_make_df(surface_reelle_bati=65.0))
    assert df["log_surface"].iloc[0] == pytest.approx(np.log1p(65.0), rel=1e-6)


def test_dist_center_km_11e():
    df = add_features(_make_df())
    # 11e is ~1-2 km from Paris center
    assert 0.5 < df["dist_center_km"].iloc[0] < 3.0


def test_arrondissement_extracted():
    df = add_features(_make_df(code_postal=75016))
    assert df["arrondissement"].iloc[0] == 16


def test_premium_arr_flag():
    df_prem = add_features(_make_df(code_postal=75006))  # 6e is premium
    df_norm = add_features(_make_df(code_postal=75013))  # 13e is not
    assert df_prem["is_premium_arr"].iloc[0] == 1
    assert df_norm["is_premium_arr"].iloc[0] == 0


def test_carrez_ratio_no_carrez():
    """When no carrez surface → ratio defaults to 1.0."""
    df = add_features(_make_df())
    assert df["carrez_ratio"].iloc[0] == pytest.approx(1.0)


def test_carrez_ratio_with_carrez():
    df = add_features(_make_df(lot1_surface_carrez=60.0))
    expected = min(max(60.0 / 65.0, 0.5), 2.0)
    assert df["carrez_ratio"].iloc[0] == pytest.approx(expected, rel=1e-4)


def test_interaction_features_present():
    df = add_features(_make_df(), arr_target_enc={11: 10500.0}, global_mean=10000.0)
    assert "premium_x_log_surface" in df.columns
    assert "premium_x_dist_center" in df.columns
    assert "voie_x_density" in df.columns


def test_arr_target_enc_applied():
    enc = {11: 10500.0}
    df = add_features(_make_df(), arr_target_enc=enc, global_mean=9000.0)
    assert df["arr_target_enc"].iloc[0] == pytest.approx(10500.0)


def test_arr_target_enc_fallback():
    """Unknown arrondissement falls back to global_mean."""
    df = add_features(_make_df(code_postal=75099), arr_target_enc={}, global_mean=9500.0)
    assert df["arr_target_enc"].iloc[0] == pytest.approx(9500.0)


def test_voie_recent_fallback_to_global_mean():
    """No voie_recent column and no lookup → filled with global_mean."""
    df = add_features(_make_df(), global_mean=9999.0)
    assert df["voie_recent_prix_m2"].iloc[0] == pytest.approx(9999.0)


def test_osm_defaults_filled():
    """OSM columns absent from input → filled with Paris-wide medians."""
    df = add_features(_make_df())
    assert df["dist_metro_m"].iloc[0] == pytest.approx(214.0)
    assert df["walkability_score"].iloc[0] == pytest.approx(43.0)


def test_pieces_per_m2_clipped():
    """pieces_per_m2 should be clipped to [0.005, 0.5]."""
    df = add_features(_make_df(surface_reelle_bati=1.0, nombre_pieces_principales=100))
    assert df["pieces_per_m2"].iloc[0] <= 0.5
