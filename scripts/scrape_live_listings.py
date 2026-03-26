"""
FairSquare — Live Listings Scraper + Scorer
=============================================
Tries SeLoger → PAP → fallback hardcoded.
Applies best_model.pkl, saves scored gems.

Run:
    python scripts/scrape_live_listings.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

import subprocess as _sp
try:
    _git_common = _sp.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(ROOT), text=True
    ).strip()
    MAIN_ROOT = Path(_git_common).parent
except Exception:
    MAIN_ROOT = ROOT

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MAIN_ROOT))

import numpy as np
import requests
from bs4 import BeautifulSoup

ARTIFACTS_DIR = MAIN_ROOT / "models" / "artifacts"
FRONTEND_DIR  = ROOT / "src" / "frontend"
OUT_RAW       = FRONTEND_DIR / "live_listings.json"
OUT_SCORED    = FRONTEND_DIR / "live_listings_scored.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.fr/",
}

# ── Realistic fallback listings (DVF 2024-2025 price levels) ──────────────────
FALLBACK_LISTINGS = [
    {
        "titre": "Appartement 3 pièces — Nation / Bastille",
        "arrondissement": 11,
        "surface": 62,
        "pieces": 3,
        "prix_annonce": 485_000,
        "latitude": 48.8528,
        "longitude": 2.3695,
        "url": "https://www.seloger.com/annonces/achat/appartement/paris-11eme-75/",
        "source": "Fallback",
        "description": "Traversant lumineux 3e étage, parquet, cuisine équipée, cave.",
    },
    {
        "titre": "Studio — Montmartre / Abbesses",
        "arrondissement": 18,
        "surface": 28,
        "pieces": 1,
        "prix_annonce": 235_000,
        "latitude": 48.8847,
        "longitude": 2.3381,
        "url": "https://www.seloger.com/annonces/achat/appartement/paris-18eme-75/",
        "source": "Fallback",
        "description": "Studio avec vue dégagée, refait à neuf, DPE C.",
    },
    {
        "titre": "Appartement 2 pièces — Belleville / Ménilmontant",
        "arrondissement": 20,
        "surface": 44,
        "pieces": 2,
        "prix_annonce": 319_000,
        "latitude": 48.8672,
        "longitude": 2.3943,
        "url": "https://www.seloger.com/annonces/achat/appartement/paris-20eme-75/",
        "source": "Fallback",
        "description": "Beau 2 pièces rénové, double exposition, calme.",
    },
    {
        "titre": "Appartement 4 pièces — Bercy / Gare de Lyon",
        "arrondissement": 12,
        "surface": 88,
        "pieces": 4,
        "prix_annonce": 695_000,
        "latitude": 48.8462,
        "longitude": 2.3722,
        "url": "https://www.pap.fr/annonce/ventes-appartements-paris-12eme-75/",
        "source": "Fallback",
        "description": "Grand appartement familial, balcon, parking, digicode.",
    },
    {
        "titre": "Appartement 2 pièces — Buttes-Chaumont",
        "arrondissement": 19,
        "surface": 38,
        "pieces": 2,
        "prix_annonce": 272_000,
        "latitude": 48.8777,
        "longitude": 2.3836,
        "url": "https://www.pap.fr/annonce/ventes-appartements-paris-19eme-75/",
        "source": "Fallback",
        "description": "Cosy 2p proche parc, bon état général, cave.",
    },
    {
        "titre": "Appartement 3 pièces — Ivry-sur-Seine / Place d'Italie",
        "arrondissement": 13,
        "surface": 67,
        "pieces": 3,
        "prix_annonce": 478_000,
        "latitude": 48.8302,
        "longitude": 2.3564,
        "url": "https://www.seloger.com/annonces/achat/appartement/paris-13eme-75/",
        "source": "Fallback",
        "description": "Traversant calme, cuisine séparée, SDB, cave. DPE D.",
    },
    {
        "titre": "Studio — Batignolles / Clichy",
        "arrondissement": 17,
        "surface": 31,
        "pieces": 1,
        "prix_annonce": 285_000,
        "latitude": 48.8869,
        "longitude": 2.3179,
        "url": "https://www.seloger.com/annonces/achat/appartement/paris-17eme-75/",
        "source": "Fallback",
        "description": "Studio traversant lumineux, rénovation récente, proche métro.",
    },
    {
        "titre": "Appartement 4 pièces — Oberkampf / République",
        "arrondissement": 11,
        "surface": 92,
        "pieces": 4,
        "prix_annonce": 749_000,
        "latitude": 48.8644,
        "longitude": 2.3736,
        "url": "https://www.pap.fr/annonce/ventes-appartements-paris-11eme-75/",
        "source": "Fallback",
        "description": "Haussmannien rénové, moulures, parquet, 3 chambres, cave box.",
    },
]


def try_seloger() -> list[dict]:
    """Attempt to scrape SeLoger listings."""
    print("  [SeLoger] Tentative de scraping…")
    try:
        url = "https://www.seloger.com/achat/appartements/paris-75/"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        print(f"  [SeLoger] HTTP {resp.status_code}")
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")

        listings = []
        # SeLoger uses various class names — try common patterns
        cards = (
            soup.find_all("article", {"data-test": "card"})
            or soup.find_all("li", class_=lambda c: c and "listing" in c.lower())
            or soup.find_all("div", class_=lambda c: c and "card" in c.lower())
        )
        print(f"  [SeLoger] {len(cards)} cards trouvées")

        for card in cards[:10]:
            try:
                title_el = card.find(["h2", "h3", "span"], class_=lambda c: c and "title" in c.lower())
                price_el = card.find(["span", "div", "p"], class_=lambda c: c and "price" in c.lower())
                surface_el = card.find(text=lambda t: t and "m²" in t)

                if not (title_el and price_el):
                    continue

                price_text = price_el.get_text(strip=True).replace(" ", "").replace("\u202f", "").replace("€", "")
                price = int("".join(c for c in price_text if c.isdigit()) or "0")
                if price < 50_000:
                    continue

                surface_text = str(surface_el or "50")
                surface = float("".join(c for c in surface_text.split("m²")[0] if c.isdigit() or c == ".") or "50")

                listings.append({
                    "titre": title_el.get_text(strip=True)[:80],
                    "arrondissement": 11,
                    "surface": surface,
                    "pieces": 2,
                    "prix_annonce": price,
                    "latitude": 48.8566,
                    "longitude": 2.3522,
                    "url": "https://www.seloger.com/",
                    "source": "SeLoger",
                    "description": "",
                })
            except Exception:
                continue

        print(f"  [SeLoger] {len(listings)} annonces extraites")
        return listings
    except Exception as exc:
        print(f"  [SeLoger] Erreur: {exc}")
        return []


def try_pap() -> list[dict]:
    """Attempt to scrape PAP listings."""
    print("  [PAP] Tentative de scraping…")
    try:
        url = "https://www.pap.fr/annonce/ventes-appartements-paris-75g439-?tri=prix-croissant"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        print(f"  [PAP] HTTP {resp.status_code}")
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")

        listings = []
        # PAP uses search-list-item or similar
        cards = (
            soup.find_all("li", class_=lambda c: c and "search-list" in c.lower())
            or soup.find_all("article", class_=lambda c: c and "annonce" in c.lower())
            or soup.find_all("div", class_=lambda c: c and "ad-" in (c or ""))
            or soup.find_all("div", attrs={"data-id": True})
        )
        print(f"  [PAP] {len(cards)} cards trouvées")

        for card in cards[:10]:
            try:
                title_el = card.find(["h2", "h3", "a"])
                price_els = card.find_all(text=lambda t: t and "€" in t)
                surface_els = card.find_all(text=lambda t: t and "m²" in t)

                if not price_els:
                    continue

                price_text = price_els[0].replace(" ", "").replace("\u202f", "").replace("€", "")
                price = int("".join(c for c in price_text if c.isdigit()) or "0")
                if price < 50_000:
                    continue

                surface = 50.0
                if surface_els:
                    st = str(surface_els[0]).split("m²")[0]
                    surface = float("".join(c for c in st if c.isdigit() or c == ".") or "50")

                arr = 11
                title = title_el.get_text(strip=True) if title_el else "Appartement Paris"
                for i in range(1, 21):
                    if f"{i}e" in title or f"{i}ème" in title.lower() or f"75{i:03d}" in title:
                        arr = i
                        break

                listings.append({
                    "titre": title[:80],
                    "arrondissement": arr,
                    "surface": surface,
                    "pieces": 2,
                    "prix_annonce": price,
                    "latitude": 48.8566,
                    "longitude": 2.3522,
                    "url": "https://www.pap.fr/",
                    "source": "PAP",
                    "description": "",
                })
            except Exception:
                continue

        print(f"  [PAP] {len(listings)} annonces extraites")
        return listings
    except Exception as exc:
        print(f"  [PAP] Erreur: {exc}")
        return []


def get_listings() -> list[dict]:
    """Try scraping in order, fall back to hardcoded if needed."""
    listings = []

    # Option A: SeLoger
    listings = try_seloger()
    if len(listings) >= 4:
        print(f"[OK] SeLoger: {len(listings)} annonces récupérées")
        return listings

    # Option B: PAP
    pap = try_pap()
    listings.extend(pap)
    if len(listings) >= 4:
        print(f"[OK] PAP: {len(pap)} annonces récupérées ({len(listings)} total)")
        return listings

    # Option D: Fallback hardcoded
    print(f"[FALLBACK] Scraping insuffisant ({len(listings)} annonces). Utilisation du fallback.")
    # Merge scraped + fallback to reach 8 total
    needed = max(0, 8 - len(listings))
    listings.extend(FALLBACK_LISTINGS[:needed])
    print(f"[FALLBACK] {len(listings)} annonces au total (dont {len(listings) - needed} scrapées + {needed} fallback)")
    return listings


def load_artifact():
    """Load the best model artifact from disk."""
    pkl_path = ARTIFACTS_DIR / "best_model.pkl"
    if not pkl_path.exists():
        print(f"[WARN] Modèle introuvable : {pkl_path}")
        return None
    import joblib
    artifact = joblib.load(pkl_path)
    print(f"[OK] Modèle chargé depuis {pkl_path}")
    return artifact


def predict_price_safe(surface, pieces, code_postal, lat, lon, artifact) -> float:
    """Predict prix total (€) handling log_target transformation."""
    import pandas as pd
    from src.ml.features_v2 import prepare_features_v2

    row = {
        "surface_reelle_bati":       surface,
        "nombre_pieces_principales":  pieces,
        "code_postal":               code_postal,
        "latitude":                  lat,
        "longitude":                 lon,
        "mois":                      6,
        "trimestre":                 2,
        "annee":                     2025,
        "nombre_lots":               1,
        "lot1_surface_carrez":       surface,
        "prix_m2":                   0.0,
    }
    df_row = pd.DataFrame([row])
    X, _ = prepare_features_v2(
        df_row,
        arr_target_enc=artifact.get("arr_enc"),
        global_mean=artifact.get("global_mean", 10015.0),
        voie_enc=artifact.get("voie_enc"),
        grid_enc=artifact.get("grid_enc"),
    )

    model = artifact["model"]
    if isinstance(model, list):
        raw_pred = float(np.mean([m.predict(X)[0] for m in model]))
    else:
        raw_pred = float(model.predict(X)[0])

    # Handle log-target: model predicts log(prix_m2)
    if artifact.get("log_target"):
        prix_m2 = float(np.exp(raw_pred))
    else:
        prix_m2 = raw_pred

    return prix_m2 * surface


def score_listings(listings: list[dict], artifact) -> list[dict]:
    """Apply model, calculate gem_score, filter and sort."""
    # Arrondissement -> centroid lat/lon mapping for Paris
    ARR_COORDS = {
        1:  (48.8602, 2.3477), 2:  (48.8662, 2.3471), 3:  (48.8638, 2.3566),
        4:  (48.8549, 2.3523), 5:  (48.8468, 2.3480), 6:  (48.8489, 2.3334),
        7:  (48.8565, 2.3084), 8:  (48.8741, 2.3082), 9:  (48.8762, 2.3373),
        10: (48.8759, 2.3605), 11: (48.8590, 2.3790), 12: (48.8410, 2.3877),
        13: (48.8272, 2.3579), 14: (48.8262, 2.3256), 15: (48.8374, 2.2984),
        16: (48.8620, 2.2726), 17: (48.8891, 2.3112), 18: (48.8924, 2.3444),
        19: (48.8826, 2.3756), 20: (48.8640, 2.3999),
    }

    scored = []
    for item in listings:
        arr = item.get("arrondissement", 11)
        surface = float(item.get("surface", 50))
        pieces = int(item.get("pieces", 2))
        prix_annonce = float(item.get("prix_annonce", 0))
        lat = item.get("latitude") or ARR_COORDS.get(arr, (48.8566, 2.3522))[0]
        lon = item.get("longitude") or ARR_COORDS.get(arr, (48.8566, 2.3522))[1]

        code_postal = 75000 + arr

        if artifact is None:
            # No model: use simple heuristic based on Paris median prices
            median_m2 = {
                1: 14500, 2: 13200, 3: 13000, 4: 13800, 5: 12500,
                6: 15500, 7: 16000, 8: 14000, 9: 11500, 10: 10500,
                11: 10800, 12: 9800, 13: 9200, 14: 10200, 15: 10500,
                16: 13800, 17: 11200, 18: 10200, 19: 9000, 20: 9100,
            }
            prix_predit_m2 = median_m2.get(arr, 10800)
            prix_predit = prix_predit_m2 * surface
        else:
            try:
                prix_predit = predict_price_safe(
                    surface=surface,
                    pieces=pieces,
                    code_postal=code_postal,
                    lat=lat,
                    lon=lon,
                    artifact=artifact,
                )
                prix_predit_m2 = prix_predit / surface
            except Exception as exc:
                print(f"  [WARN] Prediction echouee pour {item.get('titre', '?')}: {exc}")
                continue

        prix_affiche_m2 = prix_annonce / surface
        gem_score = (prix_predit_m2 - prix_affiche_m2) / prix_predit_m2
        gain_potentiel = prix_predit - prix_annonce

        scored_item = {
            **item,
            "prix_predit": round(prix_predit),
            "prix_predit_m2": round(prix_predit_m2),
            "prix_affiche_m2": round(prix_affiche_m2),
            "gem_score": round(gem_score, 4),
            "gain_potentiel": round(gain_potentiel),
            "sous_evaluation_pct": round(gem_score * 100, 1),
        }
        scored.append(scored_item)

        label = item.get("titre", "?")[:40]
        print(f"  {label} predit {prix_predit_m2:,.0f}e/m2 "
              f"vs affiche {prix_affiche_m2:,.0f}e/m2 "
              f"gem_score={gem_score:.3f}")

    # Filter: gem_score > 0.08 (under-valued by 8%)
    gems = [s for s in scored if s["gem_score"] > 0.08]
    gems.sort(key=lambda x: x["gem_score"], reverse=True)
    print(f"\n[RESULTATS] {len(scored)} biens scores, {len(gems)} pepites (gem_score > 8%)")
    return gems


def main():
    print("=" * 60)
    print("FairSquare — Scraper & Scorer")
    print("=" * 60)

    # Step 1: Get listings
    print("\n[ÉTAPE 1] Récupération des annonces…")
    listings = get_listings()

    # Save raw listings
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RAW.write_text(json.dumps(listings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {len(listings)} annonces sauvegardees")

    # Step 2: Score
    print("\n[ÉTAPE 2] Application du modèle FairSquare…")
    artifact = load_artifact()
    gems = score_listings(listings, artifact)

    # Step 3: Save scored
    import datetime
    output = {
        "metadata": {
            "date_mise_a_jour": datetime.datetime.now().strftime("%Y-%m-%d"),
            "nb_annonces_analysees": len(listings),
            "nb_pepites": len(gems),
            "seuil_gem_score": 0.08,
        },
        "gems": gems,
    }
    OUT_SCORED.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {len(gems)} pepites sauvegardees")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
