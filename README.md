# FairSquare 🏠

**AI-powered real-estate valuation and Hidden Gem detection for Paris (Île-de-France)**

FairSquare predicts the fair market price per m² of Parisian apartments using a LightGBM model trained on 67,000+ DVF transactions (2023–2025), enriched with OpenStreetMap proximity features. It exposes a FastAPI backend consumed by a React dashboard that surfaces under-valued listings — *Hidden Gems* — in real time.

---

## Live Demo

| Service | URL |
|---------|-----|
| React dashboard | https://fairsqure-app.vercel.app |
| FastAPI backend | https://fairsquare-api.onrender.com/docs |

---

## Architecture

```
FairSquare/
├── config/                  # Centralised Pydantic settings (env vars)
├── data/
│   ├── processed/           # Cleaned DVF parquet (~67k transactions)
│   └── outputs/ml/          # metrics.json, error_analysis.json, SHAP figures
├── docker/                  # Dockerfile.api + Dockerfile.frontend
├── models/artifacts/        # best_model.pkl (LightGBM v4, ~6 MB)
├── notebooks/               # (empty — all analysis lives in scripts/)
├── scripts/                 # Training, evaluation, and scraping scripts
├── src/
│   ├── api/                 # FastAPI app + 7 routers
│   ├── data_ingestion/      # DVF download & cleaning pipeline
│   ├── features/            # OpenStreetMap feature extraction
│   ├── ml/                  # Feature engineering v2, model tournament
│   ├── recommender/         # Hidden Gem engine
│   ├── utils/               # Centralised logger
│   └── vision/              # Google Gemini renovation scorer
└── tests/                   # Unit + integration tests
```

---

## Prerequisites

- Python 3.11+
- GDAL system library (for `geopandas`) — install via `apt-get install libgdal-dev` on Linux
- A `.env` file (copy from `.env.example`)

---

## Installation

```bash
# 1. Clone
git clone https://github.com/Quen-tintin/FairSquare.git
cd FairSquare

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and fill in GOOGLE_API_KEY, FIRECRAWL_API_KEY, etc.
```

---

## Running the API locally

```bash
uvicorn src.api.main:app --reload --port 8000
```

The interactive docs are available at **http://localhost:8000/docs**.

### Key endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check + model artifact status |
| `POST` | `/predict` | Price prediction + SHAP explanation |
| `POST` | `/analyze_url` | Scrape a listing URL and predict its price |
| `GET`  | `/hidden_gems` | Filtered list of under-valued listings |
| `GET`  | `/dvf/transactions` | Browse raw DVF transactions with filters |
| `GET`  | `/model/metrics` | Full model-evolution history (17 runs) |
| `GET`  | `/model/errors` | Precomputed error analysis (scatter, residuals, MAE by arrondissement) |

---

## Training the model

```bash
# Full v4 training pipeline (IQR filter + voie_recent encoding)
python scripts/train_v4_voie_recent.py

# Model tournament (LinearRegression / GAM / LightGBM baseline comparison)
python scripts/run_ml_pipeline.py

# Download and clean fresh DVF data from data.gouv.fr
python scripts/run_dvf_poc.py
```

---

## Running the Streamlit demo

```bash
streamlit run src/frontend/app.py
```

---

## Docker

```bash
# Build and start both services (API + Streamlit frontend)
docker-compose up --build

# API only
docker build -f docker/Dockerfile.api -t fairsquare-api .
docker run -p 8000:8000 --env-file .env fairsquare-api
```

---

## Running tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes (vision) | Google AI Studio key for Gemini Vision |
| `FIRECRAWL_API_KEY` | Yes (scraping) | Firecrawl API key for URL analysis |
| `POSTGRES_HOST` | No | PostgreSQL host (defaults to localhost) |
| `POSTGRES_PASSWORD` | No | PostgreSQL password (defaults to `changeme`) |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |

---

## Model Performance Summary

| Model | MAE (€/m²) | R² | MAPE |
|-------|-----------|-----|------|
| Linear Regression (baseline) | 2,558 | 0.12 | 35.3% |
| GAM | 2,459 | 0.16 | 34.4% |
| LightGBM v1 | 2,417 | 0.19 | 33.5% |
| LightGBM v3 (IQR + log target) | **1,416** | **0.43** | **15.8%** |
| LightGBM v4 (+ voie_recent) | 1,427 | 0.43 | 15.8% |

---

## License

MIT — see `LICENSE` for details.
