"""
Centralised configuration — reads from environment variables / .env file.
All modules import from here; never hardcode secrets or paths elsewhere.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "fairsquare"
    postgres_user: str = "fairsquare_user"
    postgres_password: str = "changeme"

    # APIs
    openai_api_key: str = ""
    google_api_key: str = ""          # Google AI Studio (Gemini)
    firecrawl_api_key: str = ""
    dvf_base_url: str = "https://apidf.datafoncier.cerema.fr"

    # App
    environment: str = "development"
    log_level: str = "INFO"

    # Derived paths (not from env)
    @property
    def data_raw_dir(self) -> Path:
        return ROOT_DIR / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return ROOT_DIR / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return ROOT_DIR / "models"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Singleton — import and call this everywhere."""
    return Settings()
