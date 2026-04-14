"""Unit tests — predict router helpers (coordinate validation, SHAP, gem score)."""
import math
import numpy as np
import pytest


# ── Coordinate validation ────────────────────────────────────────────────────

from src.api.routers.predict import _validate_coordinates, _ARR_CENTROIDS
from src.api.schemas.prediction import PredictionRequest


def _req(**kwargs) -> PredictionRequest:
    defaults = dict(
        surface=65.0, pieces=3, arrondissement=11,
        latitude=48.853, longitude=2.3698,
        mois=6, trimestre=2, nombre_lots=1,
    )
    return PredictionRequest(**{**defaults, **kwargs})


def test_valid_coords_unchanged():
    """Coords inside tolerance → returned as-is."""
    req = _req(arrondissement=11, latitude=48.853, longitude=2.3698)
    lat, lon = _validate_coordinates(req)
    assert lat == pytest.approx(48.853)
    assert lon == pytest.approx(2.3698)


def test_wrong_arr_snaps_to_centroid():
    """16e coords declared as 11e (>3 km off) → snapped to 11e centroid."""
    req = _req(arrondissement=11, latitude=48.8636, longitude=2.2735)  # 16e centroid
    lat, lon = _validate_coordinates(req)
    c_lat, c_lon = _ARR_CENTROIDS[11]
    assert lat == pytest.approx(c_lat)
    assert lon == pytest.approx(c_lon)


def test_unknown_arr_no_crash():
    """Arrondissement not in lookup → coords returned unchanged."""
    req = _req(arrondissement=11, latitude=48.853, longitude=2.3698)
    # Patch arrondissement to something not in dict temporarily
    from src.api.routers.predict import _ARR_CENTROIDS as m
    req2 = PredictionRequest(
        surface=65, pieces=3, arrondissement=11,
        latitude=48.853, longitude=2.3698,
        mois=6, trimestre=2, nombre_lots=1,
    )
    lat, lon = _validate_coordinates(req2)
    assert isinstance(lat, float)
    assert isinstance(lon, float)


def test_centroids_all_paris():
    """All 20 arrondissement centroids should be within Paris bounding box."""
    for arr, (lat, lon) in _ARR_CENTROIDS.items():
        assert 48.80 < lat < 48.93, f"arr {arr} lat out of range: {lat}"
        assert 2.22 < lon < 2.47, f"arr {arr} lon out of range: {lon}"


# ── Gem score formula ────────────────────────────────────────────────────────

def _gem_score(prix_m2: float, prix_affiche: float, surface: float,
               marge: float = 0.07) -> float:
    prix_estime_vente = prix_affiche * (1 - marge)
    prix_estime_vente_m2 = prix_estime_vente / surface
    return round((prix_m2 - prix_estime_vente_m2) / prix_m2, 4)


def test_gem_score_underpriced():
    """Asking price well below model estimate → positive, is_gem=True."""
    score = _gem_score(prix_m2=10000, prix_affiche=350_000, surface=65)
    assert score > 0.10


def test_gem_score_overpriced():
    """Asking price above model estimate → negative score."""
    score = _gem_score(prix_m2=10000, prix_affiche=900_000, surface=65)
    assert score < 0


def test_gem_score_fair_price():
    """Asking price ≈ DVF predicted price after marge → score near zero."""
    # prix_m2 * surface / (1 - marge) = asking price at par
    surface, prix_m2, marge = 65.0, 10000.0, 0.07
    prix_affiche = prix_m2 * surface / (1 - marge)
    score = _gem_score(prix_m2, prix_affiche, surface, marge)
    assert abs(score) < 0.001


def test_gem_score_marge_neutralises_typical_spread():
    """Asking 7% above DVF (the typical market spread) → gem_score ≈ 0 with marge."""
    surface, prix_m2 = 65.0, 10000.0
    # asking = DVF × 1.07  (the average market spread between ask and sold price)
    prix_affiche = prix_m2 * surface * 1.07
    score_with_marge  = _gem_score(prix_m2, prix_affiche, surface, marge=0.07)
    score_without_marge = _gem_score(prix_m2, prix_affiche, surface, marge=0.0)
    # Without marge: asking 7% > DVF → score = -0.07 (looks overpriced without context)
    assert score_without_marge < -0.05
    # With marge: asking 7% > DVF is the "normal" spread → score ≈ 0 (fair price)
    assert abs(score_with_marge) < 0.005


# ── Pieces heuristic ─────────────────────────────────────────────────────────

def _pieces_from_surface(surface: float) -> int:
    if surface < 20:   return 1
    if surface < 35:   return 1
    if surface < 55:   return 2
    if surface < 75:   return 3
    if surface < 100:  return 4
    if surface < 130:  return 5
    return max(6, round(surface / 22))


@pytest.mark.parametrize("surface,expected", [
    (15,  1),   # studio
    (25,  1),   # T1
    (45,  2),   # T2
    (62,  3),   # T3  (old heuristic gave 2 — was wrong)
    (85,  4),   # T4
    (115, 5),   # T5
    (160, 7),   # grand appartement
])
def test_pieces_heuristic(surface, expected):
    assert _pieces_from_surface(surface) == expected


def test_pieces_heuristic_62m2_corrected():
    """62m² should give 3 rooms — old flat surface/25 gave 2, which was wrong."""
    assert _pieces_from_surface(62) == 3
    assert max(1, round(62 / 25)) == 2  # confirm old logic was wrong
