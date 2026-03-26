"""
FairSquare — FastAPI entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from src.api.routers.health import router as health_router
from src.api.routers.predict import router as predict_router

settings = get_settings()

app = FastAPI(
    title="FairSquare API",
    description="Hidden Gems detection for Île-de-France real-estate",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
