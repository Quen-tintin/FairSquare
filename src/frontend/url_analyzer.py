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
from datetime import datetime

try:
    import cloudscraper as _cloudscraper
    _HAS_CLOUDSCRAPER = True
except ImportError:
    _HAS_CLOUDSCRAPER = False

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.features_v2 import add_features, FEATURE_COLS_V2

MODEL_PATH = ROOT / "models" / "artifacts" / "best_model.pkl"

# ── Correction tables ─────────────────────────────────────────────────
FLOOR_CORRECTIONS: dict[int, float] = {
    0: 0.93,  # RDC: -7%
    1: 0.96,  # 1er: -4%
    2: 0.99,  # 2ème: -1%
    3: 1.00,  # 3ème: référence
    4: 1.02,  # 4ème: +2%
    5: 1.04,  # 5ème+: +4%
}
DPE_CORRECTIONS: dict[str, float] = {
    'A': 1.05, 'B': 1.03, 'C': 1.01, 'D': 1.00,
    'E': 0.97, 'F': 0.94, 'G': 0.90,
}
RENOVATION_KEYWORDS = [
    "rénov", "refait à neuf", "travaux récents", "parquet",
    "cave", "gardien", "ascenseur", "digicode",
]
QUALITY_KEYWORDS = [
    "standing", "ancien", "haussmannien", "lumineux",
    "belle vue", "beaux volumes", "luminosité",
]

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
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
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


def _fetch_html(url: str, timeout: int = 20) -> tuple[str, BeautifulSoup]:
    """
    Fetch URL with anti-bot strategy:
      Strategy A (priority): cloudscraper — handles Cloudflare JS challenges
      Strategy B (fallback): requests with full browser headers
    Returns (raw_text, soup).
    """
    # Strategy A — cloudscraper bypasses Cloudflare
    if _HAS_CLOUDSCRAPER:
        try:
            scraper = _cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
            resp = scraper.get(url, timeout=timeout)
            resp.raise_for_status()
            text = resp.content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(resp.content, "lxml")
            return text, soup
        except Exception:
            pass  # fall through to Strategy B

    # Strategy B — requests with realistic browser headers
    resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    text = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(resp.content, "lxml")
    return text, soup


# ────────────────────────────────────────────────────────────────────
# Photo extractor (shared by all scrapers)
# ────────────────────────────────────────────────────────────────────

def _extract_photo_url(soup: BeautifulSoup) -> str | None:
    """
    Extract the first listing photo URL, in priority order:
      1. og:image meta tag  — direct CDN URL, always the hero photo
      2. JSON-LD image[]    — structured data, very reliable
      3. First <img> with a CDN-looking src
    Returns None if nothing found.
    """
    # 1. og:image — best option
    og = soup.find("meta", {"property": "og:image"})
    if og:
        src = og.get("content", "").strip()
        if src.startswith("http"):
            return src

    # 2. JSON-LD image field
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, list):
                data = data[0]
            if not isinstance(data, dict):
                continue
            imgs = data.get("image", [])
            if isinstance(imgs, str) and imgs.startswith("http"):
                return imgs
            if isinstance(imgs, list) and imgs:
                first = imgs[0]
                candidate = first if isinstance(first, str) else (first.get("url") or first.get("contentUrl") or "")
                if candidate.startswith("http"):
                    return candidate
        except Exception:
            pass

    # 3. First <img> with a CDN/photo URL
    CDN_HINTS = ("photo", "cdn", "seloger", "leboncoin", "img", "annonce", "media")
    for img in soup.find_all("img", src=True):
        src = img.get("src", "")
        if src.startswith("http") and any(h in src.lower() for h in CDN_HINTS):
            return src

    return None


# ────────────────────────────────────────────────────────────────────
# SeLoger scraper
# ────────────────────────────────────────────────────────────────────

def _scrape_seloger(url: str) -> dict:
    raw, soup = _fetch_html(url)

    result: dict = {"source": "SeLoger", "_raw": raw, "_soup": soup}

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

    # ── 5. Photo ──────────────────────────────────────────────────
    if not result.get("photo_url"):
        result["photo_url"] = _extract_photo_url(soup)

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
    result: dict = {"source": "LeBonCoin", "_raw": raw, "_soup": soup}

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

    # Photo
    if not result.get("photo_url"):
        # LeBonCoin: images in __NEXT_DATA__ ad.images[0].urls.large
        try:
            if tag and tag.string:
                nd = json.loads(tag.string)
                ad = _find_nested(nd, ["props", "pageProps", "ad"]) or {}
                imgs = ad.get("images", [])
                if isinstance(imgs, list) and imgs:
                    url_map = imgs[0].get("urls", {})
                    candidate = url_map.get("large") or url_map.get("medium") or url_map.get("thumb", "")
                    if candidate.startswith("http"):
                        result["photo_url"] = candidate
        except Exception:
            pass
    if not result.get("photo_url"):
        result["photo_url"] = _extract_photo_url(soup)

    return result


# ────────────────────────────────────────────────────────────────────
# PAP scraper
# ────────────────────────────────────────────────────────────────────

def _scrape_pap(url: str) -> dict:
    raw, soup = _fetch_html(url)
    result: dict = {"source": "PAP", "_raw": raw, "_soup": soup}

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

    # Photo
    if not result.get("photo_url"):
        result["photo_url"] = _extract_photo_url(soup)

    return result


# ────────────────────────────────────────────────────────────────────
# BienIci scraper
# ────────────────────────────────────────────────────────────────────

def _scrape_bienici(url: str) -> dict:
    raw, soup = _fetch_html(url)
    result: dict = {"source": "BienIci", "_raw": raw, "_soup": soup}

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

    # Photo
    if not result.get("photo_url"):
        result["photo_url"] = _extract_photo_url(soup)

    return result


def _find_nested(obj: Any, keys: list[str]) -> Any:
    """Navigate nested dict with a list of keys."""
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


# ────────────────────────────────────────────────────────────────────
# Listing feature extractor (floor, DPE, renovation)
# ────────────────────────────────────────────────────────────────────

def extract_listing_features(html: str, soup: BeautifulSoup) -> dict:
    """
    Extract extra qualitative features from a listing page HTML.

    Returns:
        {
          etage: int | None,            # 0=RDC, None=unknown
          nb_etages: int | None,        # total floors in building
          dpe_classe: str | None,       # A-G
          renovation_score: float,      # 0-1
          quality_score: float,         # 0-1
          charges_mensuelles: int|None, # €/mois
          features_found: list[str],    # human-readable detected items
        }
    """
    text = soup.get_text(" ", strip=True)
    text_lower = text.lower()
    features_found: list[str] = []

    # ── 1. Floor (étage) ──────────────────────────────────────────
    etage: int | None = None
    nb_etages: int | None = None

    if re.search(r'\brdc\b|rez[- ]de[- ]chauss', text, re.IGNORECASE):
        etage = 0
        features_found.append("RDC")

    # "dernier étage" — treat as 5+ for correction purposes
    if re.search(r'dernier\s+[eé]tage', text, re.IGNORECASE) and etage is None:
        # Try to get the actual floor number from patterns like "5e et dernier"
        m = re.search(r'(\d+)\s*(?:er|[eè]me?)\s+et\s+dernier', text, re.IGNORECASE)
        if m:
            etage = int(m.group(1))
            features_found.append(f"Dernier étage ({etage}e)")
        else:
            etage = 6  # dernier étage → at least +4% bucket
            features_found.append("Dernier étage")

    # "Xème étage sur Y" or "Xème étage"
    if etage is None:
        m = re.search(
            r'(\d+)\s*(?:er|[eè]me?)\s*[eé]tage(?:\s+sur\s+(\d+))?',
            text, re.IGNORECASE,
        )
        if m:
            etage = int(m.group(1))
            if m.group(2):
                nb_etages = int(m.group(2))
                features_found.append(f"Étage {etage}/{nb_etages}")
            else:
                features_found.append(f"Étage {etage}")

    # ── 2. DPE class ─────────────────────────────────────────────
    dpe_classe: str | None = None

    # Plain-text patterns
    m = re.search(
        r'(?:classe\s+[eé]nergie|dpe|[eé]tiquette\s+[eé]nerg[eé]tique)\s*:?\s*([A-G])\b',
        text, re.IGNORECASE,
    )
    if m:
        dpe_classe = m.group(1).upper()
        features_found.append(f"DPE {dpe_classe}")

    # Fallback: look in elements with DPE-related class/id names
    if not dpe_classe:
        for tag in soup.find_all(
            True,
            class_=re.compile(r'dpe|energy|energie|diagnostic', re.IGNORECASE),
        ):
            tag_text = tag.get_text(" ", strip=True)
            m2 = re.search(r'\b([A-G])\b', tag_text)
            if m2:
                dpe_classe = m2.group(1).upper()
                features_found.append(f"DPE {dpe_classe}")
                break

    # ── 3. Renovation & quality keywords ─────────────────────────
    reno_hits: list[str] = [kw for kw in RENOVATION_KEYWORDS if kw.lower() in text_lower]
    quality_hits: list[str] = [kw for kw in QUALITY_KEYWORDS if kw.lower() in text_lower]
    features_found.extend(reno_hits)
    features_found.extend(quality_hits)

    renovation_score = len(reno_hits) / len(RENOVATION_KEYWORDS)
    quality_score = len(quality_hits) / len(QUALITY_KEYWORDS)

    # ── 4. Monthly charges ───────────────────────────────────────
    charges_mensuelles: int | None = None
    for pattern in [
        r'charges?\s*:?\s*([\d\s\xa0]+)\s*€',
        r'([\d\s\xa0]+)\s*€\s*/?\s*mois\s+de\s+charges?',
        r'charges?\s+(?:de\s+)?([\d\s\xa0]+)\s*€\s*/?\s*mois',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                val = int(m.group(1).replace(" ", "").replace("\xa0", ""))
                if 10 < val < 5000:  # sanity check
                    charges_mensuelles = val
                    features_found.append(f"Charges {val} €/mois")
                    break
            except Exception:
                pass

    return {
        "etage": etage,
        "nb_etages": nb_etages,
        "dpe_classe": dpe_classe,
        "renovation_score": round(renovation_score, 3),
        "quality_score": round(quality_score, 3),
        "charges_mensuelles": charges_mensuelles,
        "features_found": features_found,
    }


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

    _global_mean = art.get("global_mean", 10015.0)
    _arr_recent  = art.get("arr_recent_median_lookup", {})
    _recent_val  = float(_arr_recent.get(arrondissement, _global_mean))

    now = datetime.now()
    row = {
        "surface_reelle_bati":       surface,
        "nombre_pieces_principales": pieces,
        "code_postal":               75000 + arrondissement,
        "latitude":                  latitude,
        "longitude":                 longitude,
        "mois":      now.month,
        "trimestre": (now.month - 1) // 3 + 1,
        "annee":     now.year,
        "nombre_lots": 1,
        "lot1_surface_carrez": surface,
        "prix_m2": 0.0,
        "adresse_code_voie": None,
        "voie_recent_prix_m2": _recent_val,
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
        prix_m2 = float(np.expm1(np.mean([m.predict(X)[0] for m in model])))
        lgb_model = model[0]
    else:
        prix_m2 = float(np.expm1(model.predict(X)[0]))
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
# URL pattern extractor (no HTTP needed)
# ────────────────────────────────────────────────────────────────────

def _extract_from_url_pattern(url: str) -> dict:
    """Extract listing metadata from URL structure without any HTTP request."""
    result: dict = {}

    # SeLoger arrondissement: paris-11eme-75, paris-8e-75, paris-1er-75
    m = re.search(r'paris-(\d{1,2})(?:er|e|eme|ème)?-75', url, re.IGNORECASE)
    if m:
        arr = int(m.group(1))
        if 1 <= arr <= 20:
            result["arrondissement"] = arr

    # Listing ID: last long number before .htm (or end of path segment)
    m = re.search(r'/(\d{6,12})(?:\.htm)?(?:[?#/]|$)', url)
    if m:
        result["listing_id"] = m.group(1)

    # Property type
    url_lower = url.lower()
    if "appartement" in url_lower:
        result["type_bien"] = "appartement"
    elif "maison" in url_lower:
        result["type_bien"] = "maison"

    return result


# ────────────────────────────────────────────────────────────────────
# SeLoger lightweight API (faster than full HTML scraping)
# ────────────────────────────────────────────────────────────────────

def _try_seloger_api(listing_id: str) -> dict | None:
    """
    Try SeLoger internal API endpoints before falling back to HTML scraping.
    Returns a scraped-data dict (same shape as _scrape_seloger), or None.
    """
    mobile_headers = {
        "User-Agent": "SeLoger/2023 CFNetwork/1474 Darwin/23.0.0",
        "x-device-type": "phone",
        "x-platform": "ios",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "fr-FR,fr;q=0.9",
    }

    # Endpoint 1: classic JSON caracteristique endpoint
    try:
        api_url = (
            f"https://www.seloger.com/detail,json,caracteristique_bien.json"
            f"?idAnnonce={listing_id}"
        )
        resp = requests.get(api_url, headers=mobile_headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            result: dict = {"source": "SeLoger"}
            if isinstance(data, dict):
                _dig_seloger_json(data, result)
            if result.get("prix") or result.get("surface"):
                result.setdefault("photo_url", None)
                return result
    except Exception:
        pass

    # Endpoint 2: mobile REST API
    try:
        api_url = f"https://api.seloger.com/api/v1/listings/{listing_id}"
        resp = requests.get(api_url, headers=mobile_headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            result = {"source": "SeLoger"}
            if isinstance(data, dict):
                _dig_seloger_json(data, result)
            if result.get("prix") or result.get("surface"):
                result.setdefault("photo_url", None)
                return result
    except Exception:
        pass

    return None


# ────────────────────────────────────────────────────────────────────
# HTML paste parser (user copies page source manually)
# ────────────────────────────────────────────────────────────────────

def _parse_pasted_html(html_text: str, url: str = "") -> dict:
    """
    Parse HTML pasted by the user (e.g. from browser View Source).
    Uses the same extraction strategies as the site-specific scrapers.
    """
    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:
        soup = BeautifulSoup(html_text, "html.parser")

    result: dict = {"source": "Collé", "_raw": html_text, "_soup": soup}
    url_lower = url.lower()

    # ── LeBonCoin: __NEXT_DATA__ ─────────────────────────────────────
    if "leboncoin" in url_lower:
        tag = soup.find("script", id="__NEXT_DATA__")
        if tag and tag.string:
            try:
                nd = json.loads(tag.string)
                ad = _find_nested(nd, ["props", "pageProps", "ad"]) or {}
                if ad:
                    result["titre"] = ad.get("subject", "")
                    price_raw = ad.get("price", [0])
                    result["prix"] = _to_int(price_raw[0] if isinstance(price_raw, list) else price_raw)
                    attrs = {a.get("key"): a.get("value_label", a.get("value", ""))
                             for a in ad.get("attributes", [])}
                    result["surface"] = _to_float(attrs.get("square", 0))
                    result["pieces"] = _to_int(attrs.get("rooms", 0))
                    loc = ad.get("location", {})
                    result["code_postal"] = str(loc.get("zipcode", ""))
                    result["arrondissement"] = _arr_from_postal(loc.get("zipcode", 0))
            except Exception:
                pass

    # ── JSON-LD (SeLoger, PAP, BienIci) ─────────────────────────────
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
                if "offers" in data and not result.get("prix"):
                    offers = data["offers"]
                    if isinstance(offers, list):
                        offers = offers[0]
                    result["prix"] = _to_int(offers.get("price", 0))
                if "floorSize" in data and not result.get("surface"):
                    fs = data["floorSize"]
                    result["surface"] = _to_float(fs.get("value", 0) if isinstance(fs, dict) else fs)
                if "numberOfRooms" in data and not result.get("pieces"):
                    result["pieces"] = _to_int(data["numberOfRooms"])
                if "address" in data and not result.get("arrondissement"):
                    addr = data["address"]
                    if isinstance(addr, dict):
                        cp = addr.get("postalCode", "")
                        result["code_postal"] = str(cp)
                        arr = _arr_from_postal(cp)
                        if arr:
                            result["arrondissement"] = arr
        except Exception:
            pass

    # ── window.__PRELOADED_STATE__ / DATA_LAYER ──────────────────────
    for pattern in [
        r'window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});?\s*(?:window|</script>)',
        r'window\.DATA_LAYER\s*=\s*(\{.*?\});',
    ]:
        m = re.search(pattern, html_text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                _dig_seloger_json(obj, result)
                break
            except Exception:
                pass

    # ── Meta tags fallback ───────────────────────────────────────────
    import html as _html

    def meta(name: str, prop: str | None = None) -> str:
        tag = (soup.find("meta", {"property": prop}) if prop else None) \
            or soup.find("meta", {"name": name})
        return tag.get("content", "").strip() if tag else ""

    og_title  = meta("og:title", "og:title")
    title_tag = _html.unescape(soup.title.string.strip() if soup.title and soup.title.string else "")
    meta_desc = meta("description") or meta("og:description", "og:description") or og_title or title_tag

    if not result.get("titre"):
        result["titre"] = og_title or title_tag

    if not result.get("prix") and meta_desc:
        nums = [int(n) for n in re.findall(r"\d+", meta_desc.replace("\xa0", "").replace(" ", ""))
                if int(n) > 10_000 and int(n) < 10_000_000]
        if nums:
            result["prix"] = max(nums)

    if not result.get("surface") and meta_desc:
        m2 = re.search(r"(\d+[\.,]?\d*)\s*m[²2\xb2]", meta_desc)
        if not m2:
            m2 = re.search(r"\b(\d{2,3})\s+m\b", meta_desc)
        if m2:
            result["surface"] = _to_float(m2.group(1))

    if not result.get("pieces"):
        for text_src in [og_title, meta_desc]:
            if not text_src:
                continue
            m2 = re.search(r"[TF](\d)", text_src)
            if m2:
                result["pieces"] = int(m2.group(1))
                break
        if not result.get("pieces") and meta_desc:
            m2 = re.search(r"(\d+)\s*pi[eè]ces?", meta_desc, re.IGNORECASE)
            if m2:
                result["pieces"] = _to_int(m2.group(1))

    if not result.get("arrondissement"):
        for text_src in [meta_desc, og_title]:
            if not text_src:
                continue
            m2 = re.search(r"Paris\s*\(75(\d{2,3})\)", text_src)
            if m2:
                arr = int(m2.group(1)) % 100
                if 1 <= arr <= 20:
                    result["arrondissement"] = arr
                    break
            m2 = re.search(r"Paris\s+(\d{1,2})(?:er|e|ème|eme)", text_src, re.IGNORECASE)
            if m2:
                result["arrondissement"] = int(m2.group(1))
                break
        if not result.get("arrondissement") and url:
            mu = re.search(r"paris-(\d{1,2})(?:er|e|eme|ème)?-75", url, re.IGNORECASE)
            if mu:
                result["arrondissement"] = int(mu.group(1))

    # ── Photo ────────────────────────────────────────────────────────
    if not result.get("photo_url"):
        result["photo_url"] = _extract_photo_url(soup)

    return result


# ────────────────────────────────────────────────────────────────────
# Main public API
# ────────────────────────────────────────────────────────────────────

def analyze_listing_url(
    url: str,
    manual_overrides: dict | None = None,
    pasted_html: str | None = None,
) -> dict:
    """
    Analyse une annonce immobilière depuis son URL.

    Args:
        url: URL de l'annonce.
        manual_overrides: Si fourni, skip le scraping et utilise ces valeurs directement.
            Clés attendues: prix (int), surface (float), pieces (int), arrondissement (int).
        pasted_html: Si fourni, parse ce HTML au lieu de faire une requête HTTP.

    Returns:
        Résultat complet {"success": True, ...}
        ou {"status": "needs_manual_input", "partial": {...}, "message": str}
        ou {"success": False, "error": str, ...}
    """
    url = url.strip()

    # Step 1: Always extract what we can from the URL (free, no HTTP)
    url_data = _extract_from_url_pattern(url)
    listing_id = url_data.get("listing_id")

    # Shared variables populated by each path below
    prix: int = 0
    surface: float = 0.0
    pieces: int = 0
    arr: int = 0
    titre: str = ""
    photo_url: str | None = None
    source: str = ""
    listing_extras: dict = {}
    floor_corr = dpe_corr = reno_corr = 1.0
    lat: float | None = None
    lon: float | None = None

    # ── Path A: manual overrides (skip all scraping) ──────────────────
    if manual_overrides:
        prix    = _to_int(manual_overrides.get("prix", 0))
        surface = _to_float(manual_overrides.get("surface", 0.0))
        pieces  = _to_int(manual_overrides.get("pieces", 0))
        arr     = _to_int(manual_overrides.get("arrondissement", url_data.get("arrondissement", 0)))
        source  = "Manuel"

    # ── Path B: parse user-pasted HTML ───────────────────────────────
    elif pasted_html:
        raw_data = _parse_pasted_html(pasted_html, url)
        prix     = raw_data.get("prix", 0) or 0
        surface  = raw_data.get("surface", 0.0) or 0.0
        pieces   = raw_data.get("pieces", 0) or 0
        arr      = raw_data.get("arrondissement", 0) or url_data.get("arrondissement", 0)
        titre    = raw_data.get("titre", "") or ""
        photo_url = raw_data.get("photo_url")
        source   = raw_data.get("source", "Collé")

        if prix <= 0 or surface <= 0 or not (1 <= arr <= 20):
            missing = [f for f, v in [("prix", prix), ("surface", surface)]
                       if not v] + ([] if 1 <= arr <= 20 else ["arrondissement"])
            return {
                "success": False,
                "error": f"HTML collé — données manquantes : {', '.join(missing)}",
                "titre": titre, "prix_annonce": prix, "surface": surface,
                "pieces": pieces, "arrondissement": arr,
                "prix_predit_m2": 0, "prix_predit_m2_brut": 0, "prix_predit_total": 0,
                "gem_score": 0, "gain_potentiel": 0, "is_hidden_gem": False,
                "shap_top3": [], "listing_extras": {}, "corrections": {},
                "photo_url": photo_url,
            }

        _raw_html = raw_data.pop("_raw", "") or ""
        _soup_obj = raw_data.pop("_soup", None)
        try:
            if _soup_obj is not None:
                listing_extras = extract_listing_features(_raw_html, _soup_obj)
                etage_val = listing_extras.get("etage")
                dpe_val   = listing_extras.get("dpe_classe")
                reno_val  = listing_extras.get("renovation_score", 0.0)
                if etage_val is not None:
                    floor_corr = FLOOR_CORRECTIONS.get(min(etage_val, 5), 1.04)
                if dpe_val:
                    dpe_corr = DPE_CORRECTIONS.get(dpe_val, 1.0)
                reno_corr = 0.95 + reno_val * 0.13
        except Exception:
            listing_extras = {}

    # ── Path C: normal scraping (API → HTML) ─────────────────────────
    else:
        # Try lightweight API first (avoids heavy HTML fetch)
        raw_data = None
        if listing_id and "seloger.com" in url.lower():
            raw_data = _try_seloger_api(listing_id)

        # Fall back to full HTML scraping
        if not raw_data:
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
                    raw_data = _scrape_seloger(url)
                    raw_data["source"] = "Inconnu"
            except requests.exceptions.HTTPError as exc:
                # IP-based blocking (403/429) → ask user to fill manually
                return {
                    "status": "needs_manual_input",
                    "partial": url_data,
                    "message": f"Erreur HTTP {exc.response.status_code}",
                }
            except Exception:
                return {
                    "status": "needs_manual_input",
                    "partial": url_data,
                    "message": "Impossible de récupérer la page",
                }

        prix     = raw_data.get("prix", 0) or 0
        surface  = raw_data.get("surface", 0.0) or 0.0
        pieces   = raw_data.get("pieces", 0) or 0
        arr      = raw_data.get("arrondissement", 0) or url_data.get("arrondissement", 0)
        titre    = raw_data.get("titre", "") or ""
        photo_url = raw_data.get("photo_url")
        source   = raw_data.get("source", "")
        lat      = raw_data.get("latitude")
        lon      = raw_data.get("longitude")

        # Incomplete data → ask user to fill the gaps
        if prix <= 0 or surface <= 0 or not (1 <= arr <= 20):
            partial = {**url_data}
            if prix > 0:         partial["prix"] = prix
            if surface > 0:      partial["surface"] = surface
            if 1 <= arr <= 20:   partial["arrondissement"] = arr
            if pieces > 0:       partial["pieces"] = pieces
            return {
                "status": "needs_manual_input",
                "partial": partial,
                "message": f"Données incomplètes (prix={prix}, surface={surface}m², arr={arr})",
            }

        # Extract qualitative features (floor, DPE, renovation)
        _raw_html: str = raw_data.pop("_raw", "") or ""
        _soup_obj: BeautifulSoup | None = raw_data.pop("_soup", None)
        try:
            if _soup_obj is not None or _raw_html:
                if _soup_obj is None:
                    _soup_obj = BeautifulSoup(_raw_html, "html.parser")
                listing_extras = extract_listing_features(_raw_html, _soup_obj)
                etage_val = listing_extras.get("etage")
                dpe_val   = listing_extras.get("dpe_classe")
                reno_val  = listing_extras.get("renovation_score", 0.0)
                if etage_val is not None:
                    floor_corr = FLOOR_CORRECTIONS.get(min(etage_val, 5), 1.04)
                if dpe_val:
                    dpe_corr = DPE_CORRECTIONS.get(dpe_val, 1.0)
                reno_corr = 0.95 + reno_val * 0.13
        except Exception:
            listing_extras = {}

    # ── Common path: lat/lon → predict → gem score ────────────────────

    if not lat or not lon:
        lat, lon = ARR_CENTROIDS.get(arr if 1 <= arr <= 20 else 11, (48.8566, 2.3522))

    if pieces <= 0:
        pieces = max(1, round(surface / 25))

    safe_arr = arr if 1 <= arr <= 20 else 11

    try:
        prix_m2_lgbm, _, shap_top3 = _predict_and_explain(
            surface=surface, pieces=pieces,
            arrondissement=safe_arr, latitude=lat, longitude=lon,
        )
    except Exception as exc:
        return {
            "success": False, "error": f"Erreur modèle : {exc}",
            "titre": titre, "prix_annonce": prix, "surface": surface,
            "pieces": pieces, "arrondissement": arr,
            "prix_predit_m2": 0, "prix_predit_m2_brut": 0, "prix_predit_total": 0, "gem_score": 0,
            "gain_potentiel": 0, "is_hidden_gem": False, "shap_top3": [],
            "listing_extras": {}, "corrections": {}, "photo_url": photo_url,
        }

    prix_m2    = prix_m2_lgbm * floor_corr * dpe_corr * reno_corr
    prix_total = int(prix_m2 * surface)

    prix_affiche_m2 = prix / surface
    gem_score       = (prix_m2 - prix_affiche_m2) / prix_m2 if prix_m2 > 0 else 0.0
    gain_potentiel  = max(0, int(prix_total - prix))

    return {
        "success":             True,
        "source":              source,
        "titre":               titre,
        "prix_annonce":        int(prix),
        "surface":             float(surface),
        "pieces":              int(pieces),
        "arrondissement":      int(safe_arr),
        "latitude":            float(lat),
        "longitude":           float(lon),
        "prix_affiche_m2":     int(prix_affiche_m2),
        "prix_predit_m2_brut": round(prix_m2_lgbm, 0),
        "prix_predit_m2":      round(prix_m2, 0),
        "prix_predit_total":   prix_total,
        "gem_score":           round(gem_score, 4),
        "gain_potentiel":      gain_potentiel,
        "is_hidden_gem":       gem_score > 0.08,
        "shap_top3":           shap_top3,
        "listing_extras":      listing_extras,
        "corrections": {
            "floor_corr": round(floor_corr, 4),
            "dpe_corr":   round(dpe_corr, 4),
            "reno_corr":  round(reno_corr, 4),
            "total_corr": round(floor_corr * dpe_corr * reno_corr, 4),
        },
        "photo_url": photo_url,
        "error":     None,
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
            print(f"  Prix brut : {r['prix_predit_m2_brut']:,.0f} EUR/m2  (LightGBM)")
            print(f"  Prix pred.: {r['prix_predit_total']:,} EUR  ({r['prix_predit_m2']:,.0f} EUR/m2)  [après corrections]")
            badge = "[HIDDEN GEM]" if r["is_hidden_gem"] else ("[SURCOTE]" if r["gem_score"] < 0 else "[CORRECT]")
            print(f"  Gem score : {r['gem_score']*100:.1f}%  {badge}")
            print(f"  Gain pot. : {r['gain_potentiel']:,} EUR")
            ex = r.get("listing_extras", {})
            if ex.get("features_found"):
                print(f"  Détectés  : {', '.join(ex['features_found'])}")
            c = r.get("corrections", {})
            if c.get("total_corr", 1.0) != 1.0:
                print(f"  Corrections: étage×{c['floor_corr']} DPE×{c['dpe_corr']} réno×{c['reno_corr']:.3f} = ×{c['total_corr']:.3f}")
            for s in r["shap_top3"]:
                sign = "+" if s["impact"] > 0 else ""
                print(f"  SHAP      : {s['feature']} = {sign}{s['impact']:,} EUR/m2")
        else:
            print(f"  ERREUR: {r['error']}")
