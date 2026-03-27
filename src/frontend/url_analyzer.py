"""
FairSquare URL Analyzer
========================
Scrape une annonce immobilière depuis son URL et retourne une analyse FairSquare.

Supporte : SeLoger, LeBonCoin, PAP, BienIci
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.features_v2 import add_features, FEATURE_COLS_V2

MODEL_PATH = ROOT / "models" / "artifacts" / "best_model.pkl"

# ── Centroides des arrondissements parisiens (lat, lon) ──────────────
ARR_CENTROIDS: dict[int, tuple[float, float]] = {
    1:  (48.8603, 2.3477),
    2:  (48.8666, 2.3504),
    3:  (48.8630, 2.3601),
    4:  (48.8533, 2.3526),
    5:  (48.8462, 2.3500),
    6:  (48.8490, 2.3340),
    7:  (48.8562, 2.3187),
    8:  (48.8745, 2.3084),
    9:  (48.8777, 2.3358),
    10: (48.8759, 2.3622),
    11: (48.8589, 2.3796),
    12: (48.8427, 2.3946),
    13: (48.8315, 2.3626),
    14: (48.8280, 2.3259),
    15: (48.8420, 2.3014),
    16: (48.8636, 2.2735),
    17: (48.8905, 2.3139),
    18: (48.8926, 2.3474),
    19: (48.8847, 2.3799),
    20: (48.8646, 2.3979),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(str(v).replace(" ", "").replace(",", ".").replace("\xa0", "")))
    except Exception:
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(str(v).replace(" ", "").replace(",", ".").replace("\xa0", ""))
    except Exception:
        return default


def _extract_number(text: str) -> float | None:
    """Extract first number from a string."""
    m = re.search(r"[\d\s]+[,.]?\d*", text.replace("\xa0", "").replace(" ", ""))
    if m:
        try:
            return float(m.group().replace(",", "."))
        except Exception:
            pass
    return None


def _arr_from_postal(cp: Any) -> int:
    """Extract arrondissement (1-20) from a Paris postal code."""
    try:
        return int(float(str(cp))) % 100
    except Exception:
        return 0


def _fetch_html(url: str, timeout: int = 15) -> tuple[str, BeautifulSoup]:
    """Fetch URL and return (raw_text, soup). Forces UTF-8 decoding."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    # Always decode as UTF-8 (SeLoger, LBC are UTF-8 despite headers sometimes lying)
    text = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(resp.content, "html.parser", from_encoding="utf-8")
    return text, soup


# ────────────────────────────────────────────────────────────────────
# SeLoger scraper
# ────────────────────────────────────────────────────────────────────

def _scrape_seloger(url: str) -> dict:
    raw, soup = _fetch_html(url)

    result: dict = {"source": "SeLoger"}

    # ── 1. JSON-LD ──────────────────────────────────────────────
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                data = data[0]
            if not isinstance(data, dict):
                continue
            if data.get("@type") in ("Apartment", "House", "RealEstateListing", "Product"):
                if "name" in data and not result.get("titre"):
                    result["titre"] = data["name"]
                if "offers" in data:
                    offers = data["offers"]
                    if isinstance(offers, list):
                        offers = offers[0]
                    if "price" in offers and not result.get("prix"):
                        result["prix"] = _to_int(offers["price"])
                if "floorSize" in data and not result.get("surface"):
                    fs = data["floorSize"]
                    if isinstance(fs, dict):
                        result["surface"] = _to_float(fs.get("value", 0))
                    else:
                        result["surface"] = _to_float(fs)
                if "numberOfRooms" in data and not result.get("pieces"):
                    result["pieces"] = _to_int(data["numberOfRooms"])
                if "address" in data:
                    addr = data["address"]
                    if isinstance(addr, dict):
                        cp = addr.get("postalCode", "")
                        result["code_postal"] = str(cp)
                        arr = _arr_from_postal(cp)
                        if arr:
                            result["arrondissement"] = arr
        except Exception:
            pass

    # ── 2. __PRELOADED_STATE__ or window.INITIAL_STATE ───────────
    for pattern in [
        r'window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});?\s*(?:window|</script>)',
        r'window\.DATA_LAYER\s*=\s*(\{.*?\});',
        r'"listing"\s*:\s*(\{[^}]{20,}?\})',
    ]:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                _dig_seloger_json(obj, result)
                break
            except Exception:
                pass

    # ── 3. Meta tags + title tag (SeLoger varies between server/client render) ─
    def meta(name: str, prop: str | None = None) -> str:
        tag = (soup.find("meta", {"property": prop}) if prop else None) \
            or soup.find("meta", {"name": name})
        return tag.get("content", "").strip() if tag else ""

    og_title  = meta("og:title", "og:title")
    title_tag = soup.title.string.strip() if soup.title and soup.title.string else ""
    # Decode HTML entities in title (e.g. &#x2F; → /)
    import html as _html
    title_tag = _html.unescape(title_tag)
    meta_desc = meta("description") or meta("og:description", "og:description") or og_title or title_tag

    if not result.get("titre"):
        result["titre"] = og_title or title_tag

    # SeLoger meta: "Appartement à vendre T3/F3 44 m² 510000 € Paris (75017)"
    # Parse price + surface from meta — extract all numbers and heuristically assign
    if not result.get("prix") and meta_desc:
        # Price is typically the largest number > 50000
        nums = [int(n) for n in re.findall(r"\d+", meta_desc.replace("\xa0", "").replace(" ", ""))
                if int(n) > 10_000 and int(n) < 10_000_000]
        if nums:
            result["prix"] = max(nums)

    # Parse surface from meta description/title  (number before m² or m2 or just "m")
    if not result.get("surface") and meta_desc:
        # Match patterns: "44 m²", "44m²", "44 m2", "44m2"
        m = re.search(r"(\d+[\.,]?\d*)\s*m[²2\xb2]", meta_desc)
        if not m:
            # Fallback: match "44 m" followed by non-digit
            m = re.search(r"\b(\d{2,3})\s+m\b", meta_desc)
        if m:
            result["surface"] = _to_float(m.group(1))

    # Parse pieces from og:title "T3/F3" or "3 pièces"
    if not result.get("pieces"):
        if og_title:
            m = re.search(r"[TF](\d)", og_title)
            if m:
                result["pieces"] = int(m.group(1))
        if not result.get("pieces") and meta_desc:
            m = re.search(r"(\d+)\s*pi[eè]ces?", meta_desc, re.IGNORECASE)
            if m:
                result["pieces"] = _to_int(m.group(1))

    # Parse arrondissement from meta content or URL
    if not result.get("arrondissement"):
        for text_src in [meta_desc, og_title]:
            if not text_src:
                continue
            m = re.search(r"Paris\s*\(75(\d{2,3})\)", text_src)
            if m:
                arr = int(m.group(1)) % 100
                if 1 <= arr <= 20:
                    result["arrondissement"] = arr
                    break
            m = re.search(r"Paris\s+(\d{1,2})(?:er|e|ème|eme)", text_src, re.IGNORECASE)
            if m:
                result["arrondissement"] = int(m.group(1))
                break
        if not result.get("arrondissement"):
            mu = re.search(r"paris-(\d{1,2})(?:er|e|eme|ème)?-75", url, re.IGNORECASE)
            if mu:
                result["arrondissement"] = int(mu.group(1))

    # ── 4. window["__UFRN_LIFECYCLE_SERVERREQUEST__"] deep JSON ─────
    if not result.get("prix") or not result.get("surface"):
        m = re.search(r'window\["__UFRN_LIFECYCLE_SERVERREQUEST__"\]\s*=\s*JSON\.parse\("(.*?)"\);\s*\n', raw)
        if m:
            try:
                inner = m.group(1).encode().decode("unicode_escape")
                obj = json.loads(inner)
                classified = _find_nested(obj, ["app_cldp", "data", "classified"]) or {}
                if not result.get("prix"):
                    price_val = classified.get("price", classified.get("Prix"))
                    if price_val:
                        result["prix"] = _to_int(price_val)
                if not result.get("surface"):
                    surf_val = classified.get("surface", classified.get("Surface"))
                    if surf_val:
                        result["surface"] = _to_float(surf_val)
                if not result.get("pieces"):
                    rooms = classified.get("rooms", classified.get("nbPieces"))
                    if rooms:
                        result["pieces"] = _to_int(rooms)
            except Exception:
                pass

    return result


def _dig_seloger_json(obj: dict, result: dict) -> None:
    """Recursively search a JSON object for listing fields."""
    if not isinstance(obj, dict):
        return
    for key, val in obj.items():
        lkey = key.lower()
        if lkey in ("price", "prix") and not result.get("prix") and isinstance(val, (int, float)):
            result["prix"] = int(val)
        elif lkey in ("surface", "area") and not result.get("surface") and isinstance(val, (int, float)):
            result["surface"] = float(val)
        elif lkey in ("rooms", "pieces", "numberofrooms") and not result.get("pieces") and isinstance(val, (int, float)):
            result["pieces"] = int(val)
        elif lkey in ("postalcode", "codepostal", "cp") and not result.get("arrondissement") and isinstance(val, (str, int)):
            arr = _arr_from_postal(val)
            if arr:
                result["arrondissement"] = arr
        elif isinstance(val, dict):
            _dig_seloger_json(val, result)
        elif isinstance(val, list):
            for item in val[:5]:
                if isinstance(item, dict):
                    _dig_seloger_json(item, result)


# ────────────────────────────────────────────────────────────────────
# LeBonCoin scraper (__NEXT_DATA__)
# ────────────────────────────────────────────────────────────────────

def _scrape_leboncoin(url: str) -> dict:
    raw, soup = _fetch_html(url)
    result: dict = {"source": "LeBonCoin"}

    tag = soup.find("script", id="__NEXT_DATA__")
    if tag and tag.string:
        try:
            nd = json.loads(tag.string)
            ad = _find_nested(nd, ["props", "pageProps", "ad"]) or {}
            result["titre"]  = ad.get("subject", "")
            result["prix"]   = _to_int(ad.get("price", [0])[0] if isinstance(ad.get("price"), list) else ad.get("price", 0))
            attrs = {a.get("key"): a.get("value_label", a.get("value", ""))
                     for a in ad.get("attributes", [])}
            result["surface"] = _to_float(attrs.get("square", 0))
            result["pieces"]  = _to_int(attrs.get("rooms", 0))
            loc = ad.get("location", {})
            result["code_postal"]   = str(loc.get("zipcode", ""))
            result["arrondissement"] = _arr_from_postal(loc.get("zipcode", 0))
            if loc.get("lat"):
                result["latitude"]  = float(loc["lat"])
                result["longitude"] = float(loc["lng"])
        except Exception:
            pass

    if not result.get("titre"):
        result["titre"] = soup.title.string if soup.title else ""
    return result


# ────────────────────────────────────────────────────────────────────
# PAP scraper
# ────────────────────────────────────────────────────────────────────

def _scrape_pap(url: str) -> dict:
    raw, soup = _fetch_html(url)
    result: dict = {"source": "PAP"}

    # JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                data = data[0]
            if not isinstance(data, dict):
                continue
            if not result.get("titre"):
                result["titre"] = data.get("name", "")
            if not result.get("prix") and "offers" in data:
                off = data["offers"]
                if isinstance(off, list):
                    off = off[0]
                result["prix"] = _to_int(off.get("price", 0))
        except Exception:
            pass

    # Specific PAP HTML structure
    def meta(name: str) -> str:
        tag = soup.find("meta", {"property": name}) or soup.find("meta", {"name": name})
        return tag.get("content", "").strip() if tag else ""

    if not result.get("titre"):
        result["titre"] = meta("og:title") or (soup.title.string if soup.title else "")

    text = soup.get_text(" ", strip=True)

    if not result.get("prix"):
        m = re.search(r"([\d\s]{3,})\s*€", text)
        if m:
            result["prix"] = _to_int(m.group(1))

    if not result.get("surface"):
        m = re.search(r"([\d,\.]+)\s*m[²2]", text)
        if m:
            result["surface"] = _to_float(m.group(1))

    if not result.get("pieces"):
        m = re.search(r"(\d+)\s*pi[eè]ces?", text, re.IGNORECASE)
        if m:
            result["pieces"] = _to_int(m.group(1))

    if not result.get("arrondissement"):
        m = re.search(r"Paris\s+(\d{1,2})(?:er|e|ème|eme)", text, re.IGNORECASE)
        if m:
            result["arrondissement"] = int(m.group(1))
        else:
            m = re.search(r"750(\d{2})", text)
            if m:
                result["arrondissement"] = int(m.group(1))

    return result


# ────────────────────────────────────────────────────────────────────
# BienIci scraper
# ────────────────────────────────────────────────────────────────────

def _scrape_bienici(url: str) -> dict:
    raw, soup = _fetch_html(url)
    result: dict = {"source": "BienIci"}

    # __NEXT_DATA__ or window.__NEXT_DATA__
    tag = soup.find("script", id="__NEXT_DATA__")
    if tag and tag.string:
        try:
            nd = json.loads(tag.string)
            listing = _find_nested(nd, ["props", "pageProps", "listingData"]) or \
                      _find_nested(nd, ["props", "pageProps", "ad"]) or {}
            if listing:
                result["titre"]   = listing.get("title", listing.get("name", ""))
                result["prix"]    = _to_int(listing.get("price", 0))
                result["surface"] = _to_float(listing.get("surfaceArea", listing.get("surface", 0)))
                result["pieces"]  = _to_int(listing.get("roomsQuantity", listing.get("rooms", 0)))
                cp = listing.get("postalCode", listing.get("zipCode", ""))
                result["code_postal"]    = str(cp)
                result["arrondissement"] = _arr_from_postal(cp)
                if listing.get("blurInfo", {}).get("position", {}).get("lat"):
                    result["latitude"]  = float(listing["blurInfo"]["position"]["lat"])
                    result["longitude"] = float(listing["blurInfo"]["position"]["lon"])
        except Exception:
            pass

    # Fallback: meta tags
    def meta(name: str) -> str:
        tag2 = soup.find("meta", {"property": name}) or soup.find("meta", {"name": name})
        return tag2.get("content", "").strip() if tag2 else ""

    if not result.get("titre"):
        result["titre"] = meta("og:title") or (soup.title.string if soup.title else "")

    text = soup.get_text(" ", strip=True)
    if not result.get("prix"):
        m = re.search(r"([\d\s]{3,})\s*€", text)
        if m:
            result["prix"] = _to_int(m.group(1))
    if not result.get("surface"):
        m = re.search(r"([\d,\.]+)\s*m[²2]", text)
        if m:
            result["surface"] = _to_float(m.group(1))
    if not result.get("pieces"):
        m = re.search(r"(\d+)\s*pi[eè]ces?", text, re.IGNORECASE)
        if m:
            result["pieces"] = _to_int(m.group(1))

    return result


def _find_nested(obj: Any, keys: list[str]) -> Any:
    """Navigate nested dict with a list of keys."""
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


# ────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ────────────────────────────────────────────────────────────────────

_cached_artifact: dict | None = None


def _load_artifact() -> dict | None:
    global _cached_artifact
    if _cached_artifact is not None:
        return _cached_artifact
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        _cached_artifact = pickle.load(f)
    return _cached_artifact


# ────────────────────────────────────────────────────────────────────
# Prediction + SHAP
# ────────────────────────────────────────────────────────────────────

def _predict_and_explain(
    surface: float,
    pieces: int,
    arrondissement: int,
    latitude: float,
    longitude: float,
) -> tuple[float, float, list[dict]]:
    """Returns (prix_predit_m2, prix_predit_total, shap_top3)."""
    art = _load_artifact()
    if art is None:
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    row = {
        "surface_reelle_bati":       surface,
        "nombre_pieces_principales": pieces,
        "code_postal":               75000 + arrondissement,
        "latitude":                  latitude,
        "longitude":                 longitude,
        "mois":      6,
        "trimestre": 2,
        "annee":     2025,
        "nombre_lots": 1,
        "lot1_surface_carrez": surface,
        "prix_m2": 0.0,
        "adresse_code_voie": None,
    }
    df_row = pd.DataFrame([row])
    feat_cols = art.get("feature_cols", FEATURE_COLS_V2)

    # Add OSM features if model uses them (fill with 0 = neutral)
    osm_feats = art.get("osm_features", [])
    for col in osm_feats:
        df_row[col] = 0.0

    df_feat = add_features(
        df_row,
        arr_target_enc=art.get("arr_enc"),
        global_mean=art.get("global_mean", 10015.0),
        voie_enc=art.get("voie_enc"),
        grid_enc=art.get("grid_enc"),
    )
    # Add OSM cols if needed
    for col in osm_feats:
        if col not in df_feat.columns:
            df_feat[col] = 0.0

    avail_cols = [c for c in feat_cols if c in df_feat.columns]
    X = df_feat[avail_cols].astype(float)

    model = art["model"]
    if isinstance(model, list):
        prix_m2 = float(np.mean([m.predict(X)[0] for m in model]))
        lgb_model = model[0]
    else:
        prix_m2 = float(model.predict(X)[0])
        lgb_model = model

    prix_total = int(prix_m2 * surface)

    # SHAP top-3
    shap_top3: list[dict] = []
    try:
        import shap as _shap
        explainer = _shap.TreeExplainer(lgb_model)
        sv = explainer.shap_values(X)[0]

        _LABELS = {
            "arr_target_enc":            f"Localisation {arrondissement}e arr.",
            "grid_target_enc":           "Zone géographique fine",
            "voie_target_enc":           "Rue / voie",
            "dist_center_km":            "Distance centre Paris",
            "log_surface":               f"Surface {surface:.0f} m²",
            "surface_reelle_bati":       f"Surface {surface:.0f} m²",
            "nombre_pieces_principales": f"{int(pieces)} pièces",
            "arrondissement":            f"{arrondissement}e arrondissement",
            "is_premium_arr":            "Arr. premium",
            "arr_price_x_log_surface":   "Surface x localisation",
            "dist_metro_m":              "Distance métro",
            "walkability_score":         "Score walkabilité",
            "nb_transport":              "Transports à proximité",
        }
        sv_items = [(avail_cols[i], float(sv[i])) for i in range(len(avail_cols))]
        top3 = sorted(sv_items, key=lambda x: abs(x[1]), reverse=True)[:3]
        shap_top3 = [
            {"feature": _LABELS.get(k, k), "impact": round(v)}
            for k, v in top3
        ]
    except Exception:
        pass

    return prix_m2, prix_total, shap_top3


# ────────────────────────────────────────────────────────────────────
# Main public API
# ────────────────────────────────────────────────────────────────────

def analyze_listing_url(url: str) -> dict:
    """
    Scrape une annonce immobilière depuis son URL et retourne l'analyse FairSquare.

    Returns:
        {
            "success": bool,
            "titre": str,
            "prix_annonce": int,
            "surface": float,
            "pieces": int,
            "arrondissement": int,
            "prix_predit_m2": float,
            "prix_predit_total": int,
            "gem_score": float,       # (prix_predit_m2 - prix_affiche_m2) / prix_predit_m2
            "gain_potentiel": int,    # en €
            "is_hidden_gem": bool,    # gem_score > 0.08
            "shap_top3": list,        # top 3 facteurs SHAP
            "error": str | None
        }
    """
    url = url.strip()

    # ── 1. Routing ────────────────────────────────────────────────
    try:
        domain = url.lower()
        if "seloger.com" in domain:
            raw_data = _scrape_seloger(url)
        elif "leboncoin.fr" in domain:
            raw_data = _scrape_leboncoin(url)
        elif "pap.fr" in domain:
            raw_data = _scrape_pap(url)
        elif "bienici.com" in domain:
            raw_data = _scrape_bienici(url)
        else:
            # Generic fallback: try SeLoger-style parsing
            raw_data = _scrape_seloger(url)
            raw_data["source"] = "Inconnu"
    except requests.exceptions.HTTPError as exc:
        return {
            "success": False, "error": f"Erreur HTTP {exc.response.status_code} — l'annonce est peut-être expirée ou bloquée par le site.",
            "titre": "", "prix_annonce": 0, "surface": 0, "pieces": 0, "arrondissement": 0,
            "prix_predit_m2": 0, "prix_predit_total": 0, "gem_score": 0,
            "gain_potentiel": 0, "is_hidden_gem": False, "shap_top3": [],
        }
    except Exception as exc:
        return {
            "success": False, "error": f"Impossible de récupérer la page : {exc}",
            "titre": "", "prix_annonce": 0, "surface": 0, "pieces": 0, "arrondissement": 0,
            "prix_predit_m2": 0, "prix_predit_total": 0, "gem_score": 0,
            "gain_potentiel": 0, "is_hidden_gem": False, "shap_top3": [],
        }

    # ── 2. Validate extracted data ────────────────────────────────
    prix      = raw_data.get("prix", 0) or 0
    surface   = raw_data.get("surface", 0.0) or 0.0
    pieces    = raw_data.get("pieces", 0) or 0
    arr       = raw_data.get("arrondissement", 0) or 0
    titre     = raw_data.get("titre", "") or ""

    missing = []
    if prix <= 0:      missing.append("prix")
    if surface <= 0:   missing.append("surface")
    if not (1 <= arr <= 20): missing.append("arrondissement")

    if missing:
        return {
            "success": False,
            "error": (
                f"Données incomplètes extraites ({raw_data.get('source','?')}) — "
                f"champs manquants : {', '.join(missing)}. "
                f"Le site utilise probablement du JavaScript dynamique (anti-scraping).\n"
                f"Extrait : prix={prix}, surface={surface}m², arr={arr}"
            ),
            "titre": titre, "prix_annonce": prix, "surface": surface,
            "pieces": pieces, "arrondissement": arr,
            "prix_predit_m2": 0, "prix_predit_total": 0, "gem_score": 0,
            "gain_potentiel": 0, "is_hidden_gem": False, "shap_top3": [],
        }

    # ── 3. Lat/lon : use scraped if available, else centroid ──────
    lat = raw_data.get("latitude")
    lon = raw_data.get("longitude")
    if not lat or not lon:
        lat, lon = ARR_CENTROIDS.get(arr, (48.8566, 2.3522))

    if pieces <= 0:
        pieces = max(1, round(surface / 25))

    # ── 4. Predict ────────────────────────────────────────────────
    try:
        prix_m2, prix_total, shap_top3 = _predict_and_explain(
            surface=surface, pieces=pieces,
            arrondissement=arr, latitude=lat, longitude=lon,
        )
    except Exception as exc:
        return {
            "success": False, "error": f"Erreur modèle : {exc}",
            "titre": titre, "prix_annonce": prix, "surface": surface,
            "pieces": pieces, "arrondissement": arr,
            "prix_predit_m2": 0, "prix_predit_total": 0, "gem_score": 0,
            "gain_potentiel": 0, "is_hidden_gem": False, "shap_top3": [],
        }

    # ── 5. Gem score ─────────────────────────────────────────────
    prix_affiche_m2 = prix / surface
    # gem_score > 0 = bien sous-évalué (opportunité)
    gem_score = (prix_m2 - prix_affiche_m2) / prix_m2 if prix_m2 > 0 else 0.0
    gain_potentiel = max(0, int(prix_total - prix))

    return {
        "success":         True,
        "source":          raw_data.get("source", ""),
        "titre":           titre,
        "prix_annonce":    int(prix),
        "surface":         float(surface),
        "pieces":          int(pieces),
        "arrondissement":  int(arr),
        "latitude":        float(lat),
        "longitude":       float(lon),
        "prix_affiche_m2": int(prix_affiche_m2),
        "prix_predit_m2":  round(prix_m2, 0),
        "prix_predit_total": int(prix_total),
        "gem_score":       round(gem_score, 4),
        "gain_potentiel":  gain_potentiel,
        "is_hidden_gem":   gem_score > 0.08,
        "shap_top3":       shap_top3,
        "error":           None,
    }


# ────────────────────────────────────────────────────────────────────
# CLI quick-test
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_urls = [
        "https://www.seloger.com/annonces/achat/appartement/paris-17eme-75/batignolles-cardinet/263831889.htm",
        "https://www.seloger.com/annonces/achat/appartement/paris-15eme-75/georges-brassens/263255041.htm",
        "https://www.seloger.com/annonces/achat/appartement/paris-18eme-75/clignancourt-jules-joffrin/264544373.htm",
        "https://www.seloger.com/annonces/achat/appartement/paris-6eme-75/saint-placide/260085101.htm",
        "https://www.seloger.com/annonces/achat/appartement/paris-15eme-75/dupleix-motte-picquet/264496169.htm",
    ]
    for url in test_urls:
        print(f"\n{'='*65}")
        print(f"URL: {url[-60:]}")
        r = analyze_listing_url(url)
        if r["success"]:
            print(f"  Titre     : {r['titre'][:60]}")
            print(f"  Surface   : {r['surface']} m2  /  {r['pieces']} pieces")
            print(f"  Arr.      : {r['arrondissement']}e")
            print(f"  Prix ann. : {r['prix_annonce']:,} EUR  ({r['prix_affiche_m2']:,} EUR/m2)")
            print(f"  Prix pred.: {r['prix_predit_total']:,} EUR  ({r['prix_predit_m2']:,.0f} EUR/m2)")
            badge = "[HIDDEN GEM]" if r["is_hidden_gem"] else ("[SURCOTE]" if r["gem_score"] < 0 else "[CORRECT]")
            print(f"  Gem score : {r['gem_score']*100:.1f}%  {badge}")
            print(f"  Gain pot. : {r['gain_potentiel']:,} EUR")
            for s in r["shap_top3"]:
                sign = "+" if s["impact"] > 0 else ""
                print(f"  SHAP      : {s['feature']} = {sign}{s['impact']:,} EUR/m2")
        else:
            print(f"  ERREUR: {r['error']}")
