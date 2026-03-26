"""
DVF Client — Fichiers CSV géo-enrichis (data.gouv.fr)
=======================================================
Télécharge les fichiers DVF (Demandes de Valeurs Foncières) géolocalisés
depuis data.gouv.fr. Ces fichiers sont la source officielle, disponibles
sans authentification, déjà géocodés (lat/lon inclus).

Source : https://files.data.gouv.fr/geo-dvf/latest/csv/
Format : {annee}/departements/{code_dept}.csv.gz

Départements IDF : 75, 77, 78, 91, 92, 93, 94, 95
Années disponibles : 2018 → 2024 (mise à jour annuelle)
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pandas as pd

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

IDF_DEPARTMENTS: list[str] = ["75", "77", "78", "91", "92", "93", "94", "95"]

_BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv"


class DVFClient:
    """
    Client pour les CSV DVF géolocalisés publiés sur data.gouv.fr.

    Usage:
        client = DVFClient()
        df = client.fetch_department("75", year=2023)
        df_idf = client.fetch_idf(year=2023)
    """

    def __init__(self, timeout: float = 120.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    #  Interface publique                                                  #
    # ------------------------------------------------------------------ #

    def fetch_department(self, dep_code: str, year: int = 2023) -> pd.DataFrame:
        """
        Télécharge le CSV DVF d'un département pour une année donnée.

        Args:
            dep_code: Code département (ex: "75").
            year:     Année des transactions (2018–2024).

        Returns:
            DataFrame brut avec toutes les colonnes du fichier DVF.
        """
        url = f"{_BASE_URL}/{year}/departements/{dep_code}.csv.gz"
        logger.info("Downloading DVF CSV — dept=%s year=%d → %s", dep_code, year, url)

        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()

        df = pd.read_csv(
            io.BytesIO(response.content),
            compression="gzip",
            low_memory=False,
        )
        logger.info("Dept %s/%d — %d rows downloaded", dep_code, year, len(df))
        return df

    def fetch_idf(self, year: int = 2023) -> pd.DataFrame:
        """
        Télécharge et concatène les CSV DVF pour toute l'Île-de-France.

        Args:
            year: Année des transactions.

        Returns:
            DataFrame concaténé (tous les 8 départements IDF).
        """
        frames: list[pd.DataFrame] = []
        for dep in IDF_DEPARTMENTS:
            try:
                df = self.fetch_department(dep, year=year)
                df["code_departement_source"] = dep
                frames.append(df)
            except httpx.HTTPStatusError as exc:
                logger.warning("Skipping dept %s — HTTP %s", dep, exc.response.status_code)

        if not frames:
            logger.warning("No DVF data retrieved for IDF year=%d", year)
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        logger.info("IDF %d — %d total rows", year, len(result))
        return result

    def save_raw(self, df: pd.DataFrame, filename: str) -> Path:
        """Sauvegarde le DataFrame brut en Parquet dans data/raw/dvf/."""
        settings = get_settings()
        out_dir = settings.data_raw_dir / "dvf"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Saved → %s (%d rows)", out_path, len(df))
        return out_path
