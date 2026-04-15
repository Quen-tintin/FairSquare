"""Health-check router."""
from pathlib import Path

from fastapi import APIRouter

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "models" / "artifacts" / "best_model.pkl"

router = APIRouter(tags=["ops"])


@router.get("/health", summary="Service health check")
async def health_check() -> dict:
    """Return service liveness and model artifact availability.

    Returns:
        dict: Keys ``status`` (str), ``model_ready`` (bool), ``model_path`` (str | None).
    """
    return {
        "status": "ok",
        "model_ready": ARTIFACT.exists(),
        "model_path": str(ARTIFACT) if ARTIFACT.exists() else None,
    }
