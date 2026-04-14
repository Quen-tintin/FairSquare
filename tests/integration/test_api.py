"""Integration tests — /predict API endpoint (requires model artifact)."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

_BASE = dict(
    surface=65, pieces=3, arrondissement=11,
    latitude=48.853, longitude=2.3698,
    mois=6, trimestre=2, nombre_lots=1,
)


# ── Health ───────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Predict — basic ──────────────────────────────────────────────────────────

def test_predict_returns_200():
    r = client.post("/predict", json=_BASE)
    assert r.status_code == 200


def test_predict_response_shape():
    r = client.post("/predict", json=_BASE)
    d = r.json()
    assert "prix_predit_m2" in d
    assert "prix_predit_total" in d
    assert "confidence_low" in d
    assert "confidence_high" in d
    assert "shap_contributions" in d
    assert "hidden_gem_score" in d
    assert "is_hidden_gem" in d
    assert "xai_summary" in d


def test_predict_price_plausible_11e():
    """11e prices historically 8 000–13 000 €/m²."""
    r = client.post("/predict", json=_BASE)
    prix = r.json()["prix_predit_m2"]
    assert 7_000 < prix < 15_000, f"prix_m2 out of range: {prix}"


def test_predict_total_consistent():
    r = client.post("/predict", json=_BASE)
    d = r.json()
    assert d["prix_predit_total"] == pytest.approx(
        d["prix_predit_m2"] * _BASE["surface"], rel=0.01
    )


def test_predict_ci_uses_rmse():
    """CI half-width should equal model RMSE (~1954), not old hardcoded 1500."""
    r = client.post("/predict", json=_BASE)
    d = r.json()
    half = d["confidence_high"] - d["prix_predit_m2"]
    assert half > 1500, f"CI half-width {half:.0f} looks like old hardcoded 1500"


def test_predict_shap_nonzero():
    """After log→€/m² fix, SHAP values should be substantial (not ~0)."""
    r = client.post("/predict", json=_BASE)
    shap = r.json()["shap_contributions"]
    assert len(shap) > 0
    max_abs = max(abs(c["value"]) for c in shap)
    assert max_abs > 50, f"SHAP values suspiciously small: max={max_abs}"


def test_predict_no_gem_score_without_prix_affiche():
    r = client.post("/predict", json=_BASE)
    d = r.json()
    assert d["hidden_gem_score"] is None
    assert d["is_hidden_gem"] is False


# ── Predict — gem score ──────────────────────────────────────────────────────

def test_predict_gem_score_underpriced():
    r = client.post("/predict", json={**_BASE, "prix_affiche": 200_000})
    d = r.json()
    assert d["hidden_gem_score"] is not None
    assert d["hidden_gem_score"] > 0.10
    assert d["is_hidden_gem"] is True


def test_predict_gem_score_overpriced():
    r = client.post("/predict", json={**_BASE, "prix_affiche": 2_000_000})
    d = r.json()
    assert d["hidden_gem_score"] < 0
    assert d["is_hidden_gem"] is False


def test_predict_gem_marge_applied():
    """At asking = DVF price, gem_score ≈ +7% (buyer negotiates 7% off → pays below DVF)."""
    r0 = client.post("/predict", json=_BASE)
    prix_m2 = r0.json()["prix_predit_m2"]
    # Asking = exact DVF predicted price
    prix_affiche = prix_m2 * _BASE["surface"]
    r = client.post("/predict", json={**_BASE, "prix_affiche": prix_affiche})
    # With marge: estimated_vente = ask × 0.93 < DVF → gem_score ≈ +0.07
    score = r.json()["hidden_gem_score"]
    assert 0.05 < score < 0.10


# ── Predict — adresse_code_voie ──────────────────────────────────────────────

def test_predict_with_unknown_voie():
    """Unknown voie code → fallback to arr-level, no crash."""
    r = client.post("/predict", json={**_BASE, "adresse_code_voie": "ZZZZ"})
    assert r.status_code == 200


# ── Predict — coordinate validation ─────────────────────────────────────────

def test_predict_wrong_coords_snapped():
    """16e coords + arr=11 → snapped, price should match 11e range not 16e."""
    r_correct = client.post("/predict", json=_BASE)
    r_wrong   = client.post("/predict", json={
        **_BASE,
        "latitude": 48.8636, "longitude": 2.2735,  # 16e centroid
    })
    # Both should return valid 200 responses
    assert r_correct.status_code == 200
    assert r_wrong.status_code == 200
    # Wrong coords get snapped — price should be close to correct (both in 11e range)
    p_correct = r_correct.json()["prix_predit_m2"]
    p_wrong   = r_wrong.json()["prix_predit_m2"]
    assert abs(p_correct - p_wrong) < 1000, (
        f"Snapping failed: correct={p_correct:.0f}, snapped={p_wrong:.0f}"
    )


# ── Predict — validation errors ──────────────────────────────────────────────

def test_predict_invalid_surface():
    r = client.post("/predict", json={**_BASE, "surface": -10})
    assert r.status_code == 422


def test_predict_invalid_arrondissement():
    r = client.post("/predict", json={**_BASE, "arrondissement": 25})
    assert r.status_code == 422


def test_predict_invalid_lat():
    r = client.post("/predict", json={**_BASE, "latitude": 40.0})  # outside Paris
    assert r.status_code == 422


# ── Predict — premium arrondissement ─────────────────────────────────────────

def test_predict_6e_higher_than_20e():
    """6e (premium) should predict higher €/m² than 20e."""
    r6  = client.post("/predict", json={**_BASE, "arrondissement": 6,
                                        "latitude": 48.8490, "longitude": 2.3340})
    r20 = client.post("/predict", json={**_BASE, "arrondissement": 20,
                                        "latitude": 48.8646, "longitude": 2.3979})
    assert r6.json()["prix_predit_m2"] > r20.json()["prix_predit_m2"]
