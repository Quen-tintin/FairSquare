"""
Test complet FairSquare — Validation de tous les modules PoC
Lancer : python scripts/run_full_test.py
"""

import sys
import time
import traceback
from pathlib import Path

# ───────────────────────────────────────────────
#  Helpers d'affichage
# ───────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(title: str):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def ok(msg: str):
    print(f"  {GREEN}[OK]{RESET}  {msg}")

def fail(msg: str):
    print(f"  {RED}[FAIL]{RESET} {msg}")

def info(msg: str):
    print(f"  {YELLOW}[INFO]{RESET} {msg}")

def sep():
    print(f"  {CYAN}{'-'*50}{RESET}")

results: dict[str, bool] = {}


# ───────────────────────────────────────────────
#  TEST 1 — Config / Env
# ───────────────────────────────────────────────
header("TEST 1 · Configuration & Environnement")
try:
    from config.settings import get_settings
    s = get_settings()
    ok(f"Settings loaded")
    ok(f"Python {sys.version.split()[0]}")
    if s.google_api_key:
        ok(f"GOOGLE_API_KEY présente ({s.google_api_key[:8]}...)")
    else:
        fail("GOOGLE_API_KEY absente !")
    results["config"] = bool(s.google_api_key)
except Exception as e:
    fail(f"Config error: {e}")
    results["config"] = False


# ───────────────────────────────────────────────
#  TEST 2 — DVF : lecture des Parquets sauvegardés
# ───────────────────────────────────────────────
header("TEST 2 · DVF — Données & Pipeline de nettoyage")
try:
    import pandas as pd
    from src.data_ingestion.dvf_cleaner import clean, eda_summary

    raw_path   = Path("data/raw/dvf/dvf_paris_2023_poc.parquet")
    clean_path = Path("data/processed/dvf_paris_2023_clean.parquet")

    if raw_path.exists():
        df_raw = pd.read_parquet(raw_path)
        ok(f"Raw Parquet chargé  : {len(df_raw):,} lignes × {df_raw.shape[1]} colonnes")
    else:
        info("Parquet raw absent — téléchargement DVF Paris 75 (2023)...")
        from src.data_ingestion.dvf_client import DVFClient
        client = DVFClient()
        df_raw = client.fetch_department("75", year=2023)
        client.save_raw(df_raw, "dvf_paris_2023_poc.parquet")
        ok(f"Téléchargement OK   : {len(df_raw):,} lignes")

    if clean_path.exists():
        df_clean = pd.read_parquet(clean_path)
        ok(f"Clean Parquet chargé: {len(df_clean):,} lignes")
    else:
        df_clean = clean(df_raw)
        df_clean.to_parquet(clean_path, index=False)
        ok(f"Nettoyage OK        : {len(df_clean):,} lignes")

    summary = eda_summary(df_clean)
    sep()
    ok(f"Shape après nettoyage       : {summary['shape']}")
    ok(f"Prix/m² médian              : {summary['descriptive_stats'].get('prix_m2', {}).get('50%', 'N/A'):.0f} €/m²")
    ok(f"Prix médian                 : {summary['descriptive_stats'].get('valeur_fonciere', {}).get('50%', 'N/A'):.0f} €")
    ok(f"Types de locaux             : {summary['type_local_counts']}")
    ok(f"Départements présents       : {list(summary['dept_counts'].keys())}")
    ok(f"Années présentes            : {list(summary['year_counts'].keys())}")

    # Quelques stats prix/m2 par arrondissement
    if "code_postal" in df_clean.columns and "prix_m2" in df_clean.columns:
        sep()
        top = (
            df_clean.groupby("code_postal")["prix_m2"]
            .median()
            .sort_values(ascending=False)
            .head(5)
        )
        info("Top 5 arrondissements par prix/m² médian :")
        for cp, pm2 in top.items():
            print(f"       {cp} : {pm2:,.0f} EUR/m2")

    results["dvf"] = True
except Exception as e:
    fail(f"DVF error: {e}")
    traceback.print_exc()
    results["dvf"] = False


# ───────────────────────────────────────────────
#  TEST 3 — OSM : un seul point (République)
# ───────────────────────────────────────────────
header("TEST 3 · OSM — Feature Engineering géospatial")
try:
    from src.features.osm_features import OSMFeatureExtractor

    extractor = OSMFeatureExtractor(radius_m=1000, courtesy_delay=0)

    info("Requête Overpass pour Place de la République (48.8673, 2.3630)...")
    t0 = time.time()
    feat = extractor.get_features(lat=48.8673, lon=2.3630)
    elapsed = time.time() - t0

    sep()
    ok(f"Requête en {elapsed:.1f}s")
    ok(f"dist_metro_m         : {feat.dist_metro_m} m")
    ok(f"dist_park_m          : {feat.dist_park_m} m")
    ok(f"dist_school_m        : {feat.dist_school_m} m")
    ok(f"dist_supermarket_m   : {feat.dist_supermarket_m} m")
    ok(f"transit_count_500m   : {feat.transit_count_500m} stations")
    ok(f"park_area_500m_m2    : {feat.park_area_500m_m2:,.0f} m²")
    ok(f"walkability_score    : {feat.walkability_score}/100")

    if feat.errors:
        for err in feat.errors:
            fail(f"Erreur OSM: {err}")
        results["osm"] = False
    else:
        results["osm"] = True
except Exception as e:
    fail(f"OSM error: {e}")
    traceback.print_exc()
    results["osm"] = False


# ───────────────────────────────────────────────
#  TEST 4 — Vision : image locale
# ───────────────────────────────────────────────
header("TEST 4 · Vision — Renovation Scorer (Gemini)")
try:
    from src.vision.renovation_scorer import RenovationScorer

    test_img = Path("data/raw/test_apartment.jpg")
    if not test_img.exists():
        info("Image test absente — génération d'une image synthétique...")
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (640, 480), color=(220, 200, 180))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 100, 590, 400], fill=(240, 230, 210), outline=(150, 130, 110), width=3)
        draw.rectangle([200, 200, 440, 400], fill=(180, 160, 140), outline=(120, 100, 80), width=2)
        draw.rectangle([100, 150, 200, 300], fill=(200, 220, 240), outline=(100, 120, 140), width=2)
        draw.rectangle([440, 150, 540, 300], fill=(200, 220, 240), outline=(100, 120, 140), width=2)
        img.save(test_img, "JPEG")
        ok("Image synthétique créée")

    scorer = RenovationScorer()
    info(f"Modèle Gemini : {scorer._model_name}")
    info("Envoi de l'image au modèle...")

    t0 = time.time()
    result = scorer.score_from_file(test_img)
    elapsed = time.time() - t0

    sep()
    ok(f"Réponse en {elapsed:.1f}s")
    ok(f"renovation_score  : {result.renovation_score}/5")
    ok(f"space_category    : {result.space_category}")
    ok(f"reasoning         : {result.reasoning}")
    ok(f"model_used        : {result.model_used}")

    results["vision"] = True
except Exception as e:
    fail(f"Vision error: {e}")
    traceback.print_exc()
    results["vision"] = False


# ───────────────────────────────────────────────
#  TEST 5 — Imports des modules ML (structure)
# ───────────────────────────────────────────────
header("TEST 5 · ML — Vérification de la structure (modules)")
ml_modules = [
    ("src.ml", "Package ML"),
    ("src.ml.models", "Sous-package models"),
    ("src.ml.xai", "Sous-package XAI"),
    ("src.recommender", "Package recommender"),
    ("src.api.main", "FastAPI app"),
]
ml_ok = True
for mod, label in ml_modules:
    try:
        __import__(mod)
        ok(f"{label} ({mod})")
    except ImportError as e:
        fail(f"{label}: {e}")
        ml_ok = False
results["ml_structure"] = ml_ok


# ───────────────────────────────────────────────
#  RAPPORT FINAL
# ───────────────────────────────────────────────
header("RAPPORT FINAL")
all_ok = all(results.values())
for name, status in results.items():
    icon = f"{GREEN}PASS{RESET}" if status else f"{RED}FAIL{RESET}"
    print(f"  [{icon}]  {name}")

print()
if all_ok:
    print(f"  {BOLD}{GREEN}Tous les tests passent — FairSquare PoC operationnel !{RESET}")
else:
    failed = [k for k, v in results.items() if not v]
    print(f"  {BOLD}{RED}Echecs : {', '.join(failed)}{RESET}")

print()
