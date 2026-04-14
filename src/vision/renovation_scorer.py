"""
Computer Vision — Renovation Scorer (Google Gemini via google-genai SDK)
=========================================================================
Utilise Gemini (Google AI Studio) pour analyser une photo d'annonce
immobilière et produire deux sorties structurées :

  1. renovation_score (int 1–5)
       1 = Très bon état / rénové récemment
       2 = Bon état, quelques travaux cosmétiques
       3 = État correct, travaux moyens attendus
       4 = Mauvais état, gros travaux nécessaires
       5 = Inhabitable / à démolir

  2. space_category (str)
       "Étroit" | "Standard" | "Spacieux"

Quota recommandé : gemini-2.5-flash-lite (20 RPD)
→ Pour plus de requêtes : gemini-2.5-flash-lite-preview ou gemini-2.0-flash-lite

Usage:
    scorer = RenovationScorer()
    result = scorer.score_from_url("https://...")
    result = scorer.score_from_file("/path/to/photo.jpg")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import httpx
from google import genai
from google.genai import types
from PIL import Image

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

SpaceCategory = Literal["Étroit", "Standard", "Spacieux"]

# Gemini 3.1 Flash Lite — 500 RPD, 6 RPM (meilleur quota dispo pour la vision)
DEFAULT_MODEL = "gemini-2.5-flash"

# ------------------------------------------------------------------ #
#  Prompt                                                              #
# ------------------------------------------------------------------ #
_PROMPT = """
Tu es un expert immobilier et architecte d'intérieur.
Analyse cette photo d'annonce immobilière.
Réponds UNIQUEMENT en JSON valide, sans markdown, sans texte autour.

Format attendu :
{
  "renovation_score": <entier de 1 à 5>,
  "space_category": "<Étroit|Standard|Spacieux>",
  "luminosite": "<Sombre|Moyenne|Lumineuse>",
  "hauteur_plafond": "<Basse|Standard|Haute>",
  "has_outdoor_space": <true|false>,
  "qualite_generale": "<Basique|Standard|Premium>",
  "reasoning": "<justification concise en 1-2 phrases>"
}

Grille renovation_score :
  1 = Très bon état / rénové récemment (cuisine neuve, parquet, peintures fraîches)
  2 = Bon état, petits travaux cosmétiques (repeindre, joints salle de bain)
  3 = État correct, travaux moyens (cuisine à refaire, carrelage daté)
  4 = Mauvais état, gros travaux (humidité visible, électricité ancienne)
  5 = Inhabitable / travaux majeurs de structure

luminosite : évaluation de la lumière naturelle visible dans la pièce.
hauteur_plafond : Basse < 2.5m, Standard 2.5–3m, Haute > 3m (haussmannien typique).
has_outdoor_space : true si on voit un balcon, terrasse ou jardin accessible.
qualite_generale : impression globale des matériaux et finitions.
""".strip()


# ------------------------------------------------------------------ #
#  Structures de données                                               #
# ------------------------------------------------------------------ #

@dataclass
class VisionResult:
    """Résultat d'une analyse Vision pour une photo."""
    renovation_score: int
    space_category: SpaceCategory
    luminosite: str          # "Sombre" | "Moyenne" | "Lumineuse"
    hauteur_plafond: str     # "Basse" | "Standard" | "Haute"
    has_outdoor_space: bool
    qualite_generale: str    # "Basique" | "Standard" | "Premium"
    reasoning: str
    model_used: str
    raw_response: str

    def to_dict(self) -> dict:
        return {
            "renovation_score": self.renovation_score,
            "space_category": self.space_category,
            "luminosite": self.luminosite,
            "hauteur_plafond": self.hauteur_plafond,
            "has_outdoor_space": self.has_outdoor_space,
            "qualite_generale": self.qualite_generale,
            "reasoning": self.reasoning,
            "model_used": self.model_used,
        }


# ------------------------------------------------------------------ #
#  Scorer principal                                                    #
# ------------------------------------------------------------------ #

class RenovationScorer:
    """
    Wrapper autour de l'API Gemini (google-genai SDK) pour scorer des photos.

    Args:
        model:      Identifiant du modèle Gemini multimodal.
        max_tokens: Limite de tokens pour la réponse.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
    ) -> None:
        settings = get_settings()
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY manquante dans le fichier .env")
        self._client = genai.Client(api_key=settings.google_api_key)
        self._model_name = model
        # Disable thinking for gemini-2.5 models — structured JSON extraction
        # doesn't benefit from chain-of-thought, and thinking tokens eat into
        # max_output_tokens, truncating the JSON response.
        _config_kwargs: dict = {"max_output_tokens": max_tokens, "temperature": 0.1}
        try:
            _config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except AttributeError:
            pass  # older SDK versions don't have ThinkingConfig
        self._generate_config = types.GenerateContentConfig(**_config_kwargs)

    # ------------------------------------------------------------------ #
    #  Interface publique                                                  #
    # ------------------------------------------------------------------ #

    _MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB

    def score_from_uri(self, image_url: str) -> VisionResult:
        """Pass image URL directly to Gemini — no local download, avoids CDN auth issues."""
        logger.info("Scoring via URI: %s", image_url[:80])
        url_lower = image_url.lower().split("?")[0]
        if url_lower.endswith(".png"):
            mime = "image/png"
        elif url_lower.endswith(".webp"):
            mime = "image/webp"
        elif url_lower.endswith(".gif"):
            mime = "image/gif"
        else:
            mime = "image/jpeg"
        image_part = types.Part.from_uri(file_uri=image_url, mime_type=mime)
        return self._call_gemini_raw([_PROMPT, image_part])

    def score_from_url(self, image_url: str) -> VisionResult:
        """Télécharge l'image depuis une URL et l'analyse (fallback avec Referer)."""
        logger.info("Downloading image: %s", image_url[:80])
        from urllib.parse import urlparse
        parsed = urlparse(image_url)
        referer = f"{parsed.scheme}://{parsed.netloc}/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": referer,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        response = httpx.get(image_url, follow_redirects=True, timeout=15.0, headers=headers)
        response.raise_for_status()
        if len(response.content) > self._MAX_IMAGE_BYTES:
            raise ValueError(f"Image trop volumineuse ({len(response.content) / 1e6:.1f} MB > 20 MB)")
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return self._call_gemini(image)

    def score_from_file(self, file_path: str | Path) -> VisionResult:
        """Analyse une image locale."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        logger.info("Scoring image from file: %s", path.name)
        image = Image.open(path).convert("RGB")
        return self._call_gemini(image)

    def score_batch(self, image_urls: list[str]) -> list[VisionResult]:
        """Analyse plusieurs images en séquence. Tries native URI approach first."""
        results = []
        for i, url in enumerate(image_urls):
            logger.info("Batch scoring — image %d/%d", i + 1, len(image_urls))
            try:
                results.append(self.score_from_uri(url))
            except Exception as exc_uri:
                logger.warning("URI scoring failed (%s), falling back to download", exc_uri)
                try:
                    results.append(self.score_from_url(url))
                except Exception as exc:
                    logger.error("Failed to score image %s: %s", url[:60], exc)
                    results.append(_fallback_result(str(exc), self._model_name))
        return results

    # ------------------------------------------------------------------ #
    #  Appel Gemini                                                        #
    # ------------------------------------------------------------------ #

    def _call_gemini_raw(self, contents: list) -> VisionResult:
        """Core Gemini call — contents can be [prompt_str, PIL.Image] or [prompt_str, Part]."""
        logger.debug("Calling Gemini model: %s", self._model_name)
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=self._generate_config,
        )
        raw = response.text.strip()
        logger.debug("Raw Gemini response: %s", raw)
        return _parse_response(raw, self._model_name)

    def _call_gemini(self, image: Image.Image) -> VisionResult:
        """Envoie une image PIL + prompt à Gemini."""
        return self._call_gemini_raw([_PROMPT, image])


# ------------------------------------------------------------------ #
#  Fonctions utilitaires                                               #
# ------------------------------------------------------------------ #

def _parse_response(raw: str, model: str) -> VisionResult:
    """Parse et valide la réponse JSON du LLM."""
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(clean[start:end])
        else:
            raise ValueError(f"Réponse non-parseable : {raw[:200]}")

    try:
        score = max(1, min(5, int(data.get("renovation_score", 3))))
    except (ValueError, TypeError):
        logger.warning("renovation_score non-numérique: %r → 3", data.get("renovation_score"))
        score = 3
    category = data.get("space_category", "Standard")
    if category not in ("Étroit", "Standard", "Spacieux"):
        logger.warning("Catégorie inconnue '%s' → 'Standard'", category)
        category = "Standard"

    luminosite = data.get("luminosite", "Moyenne")
    if luminosite not in ("Sombre", "Moyenne", "Lumineuse"):
        luminosite = "Moyenne"

    hauteur_plafond = data.get("hauteur_plafond", "Standard")
    if hauteur_plafond not in ("Basse", "Standard", "Haute"):
        hauteur_plafond = "Standard"

    has_outdoor = bool(data.get("has_outdoor_space", False))

    qualite = data.get("qualite_generale", "Standard")
    if qualite not in ("Basique", "Standard", "Premium"):
        qualite = "Standard"

    return VisionResult(
        renovation_score=score,
        space_category=category,
        luminosite=luminosite,
        hauteur_plafond=hauteur_plafond,
        has_outdoor_space=has_outdoor,
        qualite_generale=qualite,
        reasoning=data.get("reasoning", ""),
        model_used=model,
        raw_response=raw,
    )


def _fallback_result(error_msg: str, model: str) -> VisionResult:
    return VisionResult(
        renovation_score=3,
        space_category="Standard",
        luminosite="Moyenne",
        hauteur_plafond="Standard",
        has_outdoor_space=False,
        qualite_generale="Standard",
        reasoning=f"Erreur d'analyse : {error_msg}",
        model_used=model,
        raw_response="",
    )
