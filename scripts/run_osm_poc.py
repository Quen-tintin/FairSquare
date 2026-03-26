"""
PoC OSM — Features géospatiales pour 3 points en IDF
Lancer : python scripts/run_osm_poc.py
"""

from src.features.osm_features import OSMFeatureExtractor
import json

# Coordonnées de test : République, Vincennes, Boulogne-Billancourt
TEST_POINTS = [
    (48.8673, 2.3629, "République (Paris 11)"),
    (48.8479, 2.4395, "Vincennes"),
    (48.8350, 2.2385, "Boulogne-Billancourt"),
]

def main():
    extractor = OSMFeatureExtractor(radius_m=1000)

    for lat, lon, label in TEST_POINTS:
        print(f"\n=== {label} ({lat}, {lon}) ===")
        features = extractor.get_features(lat=lat, lon=lon)
        data = features.to_dict()
        for k, v in data.items():
            print(f"  {k:30s} : {v}")
        if features.errors:
            print(f"  ERRORS: {features.errors}")

if __name__ == "__main__":
    main()
