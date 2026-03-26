"""
PoC Vision — Score de rénovation sur des photos d'annonces
Lancer : python scripts/run_vision_poc.py
Requiert GOOGLE_API_KEY dans .env

Modèle utilisé : gemini-3.1-flash-lite-preview (500 RPD gratuit, support vision)
Pour passer à un modèle plus puissant : RenovationScorer(model="models/gemini-2.5-flash")
"""

from pathlib import Path
from src.vision.renovation_scorer import RenovationScorer

def main():
    scorer = RenovationScorer()  # utilise DEFAULT_MODEL

    # Test 1 : image locale (toujours disponible)
    test_img = Path("data/raw/test_apartment.jpg")
    if test_img.exists():
        print("=== Test image locale ===")
        result = scorer.score_from_file(test_img)
        print(f"  Renovation score  : {result.renovation_score}/5")
        print(f"  Space category    : {result.space_category}")
        print(f"  Reasoning         : {result.reasoning}")
        print(f"  Model             : {result.model_used}")

    # Test 2 : URL publique (remplacer par une vraie photo d'annonce)
    # Décommentez et remplacez l'URL par une photo d'annonce réelle :
    # print("\n=== Test URL annonce ===")
    # result = scorer.score_from_url("https://cdn.seloger.com/votre-photo.jpg")
    # print(result.to_dict())

if __name__ == "__main__":
    main()
