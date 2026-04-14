"""
FairSquare — FastAPI entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from src.api.routers.health import router as health_router
from src.api.routers.predict import router as predict_router
from src.api.routers.analyze_url import router as analyze_url_router
from src.api.routers.model_metrics import router as model_metrics_router
from src.api.routers.dvf_transactions import router as dvf_transactions_router
from src.api.routers.hidden_gems import router as hidden_gems_router
from src.api.routers.model_errors import router as model_errors_router

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
    allow_origins=[
        "http://localhost:8501",
        "http://localhost:3000",
        "http://localhost:5173",
        # AI Studio Build / Cloud Run / Vercel — accept any HTTPS origin
        "*",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(analyze_url_router)
app.include_router(model_metrics_router)
app.include_router(dvf_transactions_router)
app.include_router(hidden_gems_router)
app.include_router(model_errors_router)
