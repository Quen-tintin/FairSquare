"""Health-check router."""
from pathlib import Path

from fastapi import APIRouter

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "models" / "artifacts" / "best_model.pkl"

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_ready": ARTIFACT.exists(),
        "model_path": str(ARTIFACT) if ARTIFACT.exists() else None,
    }
