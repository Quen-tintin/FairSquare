# FairSquare

**AI-powered real-estate valuation and Hidden Gem detection for Paris**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-brightgreen)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![API](https://img.shields.io/badge/API-live%20on%20Render-46C2CB?logo=render)](https://fairsquare-api.onrender.com/docs)
[![App](https://img.shields.io/badge/App-live%20on%20Vercel-black?logo=vercel)](https://fairsqure-app.vercel.app)

FairSquare estimates the fair market price per m² of Parisian apartments using a LightGBM v5 model trained on **67,000+ DVF transactions** (2023–2025) and surfaces *Hidden Gems* — listings priced more than 10 % below the predicted market value. It exposes a **FastAPI** backend consumed by a **React/Vite** dashboard deployed on Vercel.

---

## Live Demo

| Service | URL |
|---------|-----|
| React dashboard | <https://fairsqure-app.vercel.app> |
| FastAPI + Swagger UI | <https://fairsquare-api.onrender.com/docs> |

---

## Table of Contents

1. [Architecture](#architecture)
2. [Model Performance](#model-performance)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
   - [pip (standard)](#pip-standard)
   - [conda (recommended on Windows)](#conda-recommended-on-windows)
5. [Environment Variables](#environment-variables)
6. [Running the API locally](#running-the-api-locally)
7. [Running the Streamlit demo](#running-the-streamlit-demo)
8. [Training the model](#training-the-model)
9. [Docker](#docker)
10. [Running tests](#running-tests)
11. [Project layout](#project-layout)

---

## Architecture

```
FairSquare/
├── config/                    # Centralised Pydantic settings (loaded from .env)
│   └── settings.py
├── data/
│   ├── raw/                   # DVF CSV files downloaded from data.gouv.fr
│   ├── processed/             # Cleaned DVF parquet (~67 k transactions)
│   └── outputs/ml/            # metrics.json · error_analysis.json · SHAP figures
├── docker/
│   ├── Dockerfile.api         # FastAPI image (python:3.11-slim + GDAL)
│   └── Dockerfile.frontend    # Streamlit image
├── docker-compose.yml         # API + Frontend + PostgreSQL (PostGIS)
├── models/artifacts/
│   └── best_model.pkl         # LightGBM v5b (~6 MB, joblib)
├── scripts/                   # Training, evaluation, scraping, report generation
│   ├── train_v5_optimized.py  # v5 incremental training (5 steps + Optuna)
│   ├── advanced_outlier_experiment.py  # 5-strategy outlier comparison
│   ├── train_v4_voie_recent.py
│   ├── scrape_live_listings.py
│   └── generate_pdf.py        # FINAL_REPORT.md -> FINAL_REPORT.pdf
├── src/
│   ├── api/                   # FastAPI application
│   │   ├── main.py
│   │   ├── routers/           # predict · hidden_gems · dvf · model_metrics · model_errors · health
│   │   └── schemas/           # Pydantic request/response models
│   ├── data_ingestion/        # DVF download & cleaning pipeline
│   ├── features/              # OpenStreetMap proximity feature extraction
│   ├── ml/                    # Feature engineering v2, model tournament, SHAP
│   ├── recommender/           # Hidden Gem scoring engine
│   ├── utils/                 # Centralised logger
│   └── vision/                # Google Gemini renovation scorer
├── tests/                     # Unit + integration tests (pytest)
├── requirements.txt           # Production dependencies
├── pyproject.toml             # Build config + dev extras
└── .env.example               # Environment variable template
```

---

## Model Performance

FairSquare went through **22 modelling iterations**. Below is the high-level progression:

| Version | Key change | MAE (€/m²) | R² | MAPE |
|---------|-----------|:-----------:|:---:|:----:|
| Linear Regression | Baseline | 2,558 | 0.12 | 35.3 % |
| GAM | + non-linear terms | 2,459 | 0.16 | 34.4 % |
| LightGBM v1 | Gradient boosting | 2,417 | 0.19 | 33.5 % |
| LightGBM v2 | Log target + IQR filter | 1,612 | 0.34 | 21.4 % |
| LightGBM v3 | + arrondissement encoding | 1,416 | 0.43 | 15.8 % |
| LightGBM v4 | + voie_recent street encoding | 1,427 | 0.43 | 15.8 % |
| **LightGBM v5b** | **+ IQR per-arr (1.0x) + price filter 5–20 k** | **1,255** | **0.53** | **12.7 %** |

> Full 22-run table, outlier experiment results, and residual analysis by arrondissement / surface / price tier are documented in [`FINAL_REPORT.md`](FINAL_REPORT.md).

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | 3.12 also works |
| GDAL (system) | any recent | Only needed for `geopandas`. See [installation note](#pip-standard). |
| Docker + Compose | 24+ | Only for the containerised setup |

---

## Installation

### pip (standard)

```bash
# 1. Clone the repository
git clone https://github.com/Quen-tintin/FairSquare.git
cd FairSquare

# 2. Create and activate a virtual environment
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install GDAL system library (required by geopandas)
#   Linux:  sudo apt-get install -y libgdal-dev gdal-bin
#   macOS:  brew install gdal
#   Windows: skip — use the conda route below instead

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. (Optional) Install in editable mode with dev extras
pip install -e ".[dev]"
```

### conda (recommended on Windows)

Conda ships pre-compiled GDAL binaries, which avoids the Windows GDAL build pain entirely.

```bash
# 1. Create a dedicated environment
conda create -n fairsquare python=3.11 -y
conda activate fairsquare

# 2. Install geospatial stack via conda-forge (handles GDAL automatically)
conda install -c conda-forge geopandas shapely pyarrow -y

# 3. Install the remaining dependencies via pip
pip install -r requirements.txt
```

---

## Environment Variables

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|:--------:|-------------|
| `GOOGLE_API_KEY` | Yes (vision) | Google AI Studio key — used by the Gemini renovation scorer |
| `FIRECRAWL_API_KEY` | Yes (scraping) | Firecrawl API key — used by `url_analyzer.py` to scrape listings |
| `POSTGRES_HOST` | No | PostgreSQL host (default: `localhost`) |
| `POSTGRES_PORT` | No | PostgreSQL port (default: `5432`) |
| `POSTGRES_DB` | No | Database name (default: `fairsquare`) |
| `POSTGRES_USER` | No | Database user (default: `fairsquare_user`) |
| `POSTGRES_PASSWORD` | No | Database password (default: `changeme`) |
| `ENVIRONMENT` | No | `development` or `production` (default: `development`) |
| `LOG_LEVEL` | No | Logging verbosity (default: `INFO`) |

> The API works without PostgreSQL — it reads directly from the parquet files and cached JSON artefacts in `data/outputs/ml/`.

---

## Running the API locally

```bash
uvicorn src.api.main:app --reload --port 8000
```

Swagger UI is available at **<http://localhost:8000/docs>**.

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check + model artefact status |
| `POST` | `/predict` | Price prediction + confidence interval + SHAP explanation |
| `POST` | `/analyze_url` | Scrape a listing URL (SeLoger, PAP…) and predict its price |
| `GET` | `/hidden_gems` | Filtered list of under-valued listings (gem_score > 10 %) |
| `GET` | `/dvf/transactions` | Browse DVF transactions with arrondissement / surface filters |
| `GET` | `/model/metrics` | Full model-evolution history (22 training runs) |
| `GET` | `/model/errors` | Pre-computed error analysis (scatter, residuals, MAE by arrondissement) |

---

## Running the Streamlit demo

```bash
streamlit run src/frontend/app.py
```

Opens at **<http://localhost:8501>**. The demo lets you input an address, surface and number of rooms, then returns the predicted price, SHAP waterfall chart, and an interactive map of nearby Hidden Gems.

---

## Training the model

All training scripts are in `scripts/`. They read from `data/processed/` (parquet) and write artefacts to `models/artifacts/` and `data/outputs/ml/`.

```bash
# Recommended: v5 incremental pipeline (IQR per-arr, Huber loss, Optuna HPO)
python scripts/train_v5_optimized.py

# Outlier experiment — compares 5 cleaning strategies, saves JSON results
python scripts/advanced_outlier_experiment.py

# v4 reference training (voie_recent street encoding)
python scripts/train_v4_voie_recent.py

# Model tournament (LinearRegression / GAM / LightGBM baseline)
python scripts/run_ml_pipeline.py

# Download and clean fresh DVF data from data.gouv.fr / datafoncier API
python scripts/run_dvf_poc.py

# Regenerate FINAL_REPORT.pdf from FINAL_REPORT.md
python scripts/generate_pdf.py
```

### Training pipeline summary

```
run_dvf_poc.py              # 1. Download DVF (2023-2025), ~120 k raw rows
    -> dvf_cleaner.py       # 2. Clean + deduplicate -> ~67 k rows (parquet)
    -> features_v2.py       # 3. Feature engineering (target enc, voie_recent, OSM)
    -> train_v5_optimized.py# 4. IQR filter -> price filter -> Huber -> temporal weights -> Optuna
    -> best_model.pkl       # 5. Serialise best model (v5b, MAE 1,255 €/m²)
    -> metrics.json         # 6. Append run metrics
```

---

## Docker

The `docker-compose.yml` starts three services: **PostgreSQL** (PostGIS), the **FastAPI backend**, and the **Streamlit frontend**.

```bash
# Build and start all services
docker-compose up --build

# Run only the API (no DB required for basic predictions)
docker build -f docker/Dockerfile.api -t fairsquare-api .
docker run -p 8000:8000 --env-file .env fairsquare-api

# Run only the Streamlit frontend
docker build -f docker/Dockerfile.frontend -t fairsquare-frontend .
docker run -p 8501:8501 --env-file .env fairsquare-frontend
```

Service URLs after `docker-compose up`:

| Service | URL |
|---------|-----|
| FastAPI + Swagger | <http://localhost:8000/docs> |
| Streamlit demo | <http://localhost:8501> |
| PostgreSQL | `localhost:5432` (user: `fairsquare_user`) |

> **Production note:** The public API runs on Render (free tier, cold start ~30 s). Configuration is in [`render.yaml`](render.yaml).

---

## Running tests

```bash
# Run the full test suite with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_predict.py -v
```

---

## Project layout (key files)

```
src/api/routers/
    predict.py          # POST /predict — LightGBM inference + SHAP
    hidden_gems.py      # GET  /hidden_gems — gem_score > 10 % filter
    model_metrics.py    # GET  /model/metrics — reads metrics.json
    model_errors.py     # GET  /model/errors — reads error_analysis.json
    health.py           # GET  /health

src/ml/
    features_v2.py      # Full feature pipeline (target enc, voie_recent, interactions)
    tournament.py       # LinearReg / GAM / LightGBM comparison helper

src/features/
    osm_features.py     # OpenStreetMap Overpass API — metro, parks, schools

src/recommender/
    engine.py           # Hidden Gem scoring (gem_score + negotiation margin)

src/vision/
    renovation_scorer.py # Google Gemini image analysis (0–10 renovation score)

scripts/
    train_v5_optimized.py           # v5 training: 5 incremental steps + Optuna
    advanced_outlier_experiment.py  # Isolation Forest + 4 other strategies
    generate_pdf.py                 # FINAL_REPORT.md -> PDF via xhtml2pdf
```

---

## License

MIT — see [`LICENSE`](LICENSE) for details.
