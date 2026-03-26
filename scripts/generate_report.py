"""
PDF Report Generator — FairSquare PoC
======================================
Run after run_ml_pipeline.py:
    python scripts/generate_report.py
Output: data/outputs/FairSquare_PoC_Report.pdf
"""
import json
from datetime import date
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

FIG     = Path("data/outputs/ml/figures")
METRICS = Path("data/outputs/ml/metrics.json")
OUT_PDF = Path("data/outputs/FairSquare_PoC_Report.pdf")

ARIAL   = "C:/Windows/Fonts/arial.ttf"
ARIALB  = "C:/Windows/Fonts/arialbd.ttf"
ARIALI  = "C:/Windows/Fonts/ariali.ttf"

BLUE     = (30, 60, 120)
WHITE    = (255, 255, 255)
LGRAY    = (240, 248, 255)
DGRAY    = (50, 50, 50)
GREEN_HL = (210, 245, 210)


# ── PDF class ──────────────────────────────────────────────────────────────
class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Arial", "I", 8)
        self.set_text_color(160, 160, 160)
        self.set_y(6)
        self.cell(95, 5, "FairSquare — Capstone AI PoC Report 2026")
        self.cell(0, 5, f"Page {self.page_no()}", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(220, 220, 220)
        self.line(10, 12, 200, 12)
        self.ln(4)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("Arial", "I", 7)
        self.set_text_color(180, 180, 180)
        self.cell(0, 5,
                  f"Generated on {date.today().strftime('%B %d, %Y')} | "
                  "Python 3.12 | LightGBM | pygam | Google Gemini",
                  align="C")

    def load_fonts(self):
        self.add_font("Arial",  fname=ARIAL)
        self.add_font("Arial",  style="B", fname=ARIALB)
        self.add_font("Arial",  style="I", fname=ARIALI)


# ── Helpers ────────────────────────────────────────────────────────────────
def section_title(pdf, num, title, subtitle=""):
    pdf.ln(3)
    pdf.set_fill_color(*BLUE)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, f"  {num}. {title}", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if subtitle:
        pdf.set_text_color(100, 100, 100)
        pdf.set_font("Arial", "I", 9)
        pdf.cell(0, 5, f"  {subtitle}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_text_color(*DGRAY)


def body(pdf, text, size=10):
    pdf.set_text_color(*DGRAY)
    pdf.set_font("Arial", "", size)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(2)


def bullet(pdf, items, size=10):
    pdf.set_text_color(*DGRAY)
    pdf.set_font("Arial", "", size)
    for item in items:
        pdf.set_x(18)
        pdf.cell(5, 5.5, "-")
        pdf.multi_cell(0, 5.5, item)
    pdf.ln(2)


def add_figure(pdf, path, caption, w=175):
    if path.exists():
        x = (210 - w) / 2
        pdf.image(str(path), x=x, w=w)
        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 5, caption, align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)
    else:
        pdf.set_font("Arial", "I", 9)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 5, f"[Missing figure: {path.name}]",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)


def kv(pdf, key, value, key_w=52):
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(*BLUE)
    pdf.cell(key_w, 6, key + ":")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(*DGRAY)
    pdf.multi_cell(0, 6, value)


def metrics_table(pdf, data):
    headers = ["Model", "MAE (EUR/m2)", "RMSE (EUR/m2)", "R2", "MAPE (%)"]
    widths  = [52, 35, 35, 25, 28]

    pdf.set_fill_color(*BLUE)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Arial", "B", 10)
    for h, w in zip(headers, widths):
        pdf.cell(w, 8, h, border=1, fill=True, align="C")
    pdf.ln()

    best_r2 = max(row.get("R2", 0) for row in data)

    for i, row in enumerate(data):
        is_best = row.get("R2") == best_r2
        if is_best:
            pdf.set_fill_color(*GREEN_HL)
        elif i % 2 == 0:
            pdf.set_fill_color(*LGRAY)
        else:
            pdf.set_fill_color(*WHITE)

        pdf.set_text_color(*DGRAY)
        pdf.set_font("Arial", "B" if is_best else "", 10)
        vals = [
            row.get("model", "?"),
            str(row.get("MAE", "?")),
            str(row.get("RMSE", "?")),
            str(row.get("R2", "?")),
            str(row.get("MAPE_%", "?")),
        ]
        for v, w in zip(vals, widths):
            pdf.cell(w, 7, v, border=1, fill=True, align="C")
        pdf.ln()

    pdf.ln(2)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 4, "Green = best model by R2",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)


# ── Pages ──────────────────────────────────────────────────────────────────
def page_cover(pdf):
    pdf.add_page()

    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 0, 210, 65, "F")

    pdf.set_y(12)
    pdf.set_font("Arial", "B", 38)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 18, "FairSquare", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Arial", "", 15)
    pdf.cell(0, 10, "Proof of Concept Report", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 8, "Capstone AI 2025-2026", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(78)
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 8,
             "AI-Powered Real Estate Valuation Platform — Ile-de-France",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(10)
    pdf.set_text_color(*DGRAY)

    modules = [
        ("DVF Data",
         "80,964 real estate transactions — Paris 2023 (data.gouv.fr)"),
        ("Feature Engineering",
         "Geospatial OSM: walkability, transit, parks, schools"),
        ("Computer Vision",
         "Renovation score via Google Gemini 3.1 Flash Lite"),
        ("ML Pipeline",
         "Model tournament: LinearReg / GAM / LightGBM + SHAP XAI"),
    ]

    for title, desc in modules:
        pdf.set_x(25)
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(*BLUE)
        pdf.cell(52, 7, title)
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(*DGRAY)
        pdf.multi_cell(0, 7, desc)

    pdf.set_y(255)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6,
             f"Generated on {date.today().strftime('%B %d, %Y')} | Python 3.12 / Anaconda",
             align="C")


def page_context(pdf):
    pdf.add_page()
    section_title(pdf, "1", "Project Context",
                  "FairSquare — AI for the Parisian real estate market")

    body(pdf,
         "FairSquare is a Proof of Concept platform developed as part of a Capstone AI project. "
         "The goal is to provide intelligent, explainable real estate price estimates for the "
         "Ile-de-France region, by combining multiple data sources and machine learning approaches.\n\n"
         "Tech stack: Python 3.12 / Anaconda, LightGBM 4.5, pygam 0.12, "
         "scikit-learn 1.6, SHAP 0.51, Google Gemini API, FastAPI, Streamlit, "
         "PostgreSQL, Docker.")

    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 7, "System Architecture:",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    components = [
        ("DVF Ingestion",       "HTTP client to data.gouv.fr, strict cleaning, Parquet output"),
        ("OSM Features",        "Overpass extractor (1 query/point), 7 geospatial features"),
        ("AI Vision",           "RenovationScorer via Gemini, score 1-5 + space category"),
        ("ML Pipeline",         "3-model tournament, standardized metrics, SHAP XAI"),
        ("FastAPI",             "Endpoint /predict (in progress), Pydantic schemas"),
        ("Streamlit Frontend",  "Interactive dashboard (Phase 3)"),
    ]

    for comp, desc in components:
        pdf.set_x(15)
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(*BLUE)
        pdf.cell(45, 6, comp)
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(*DGRAY)
        pdf.multi_cell(0, 6, desc)
    pdf.ln(3)


def page_dataset(pdf):
    pdf.add_page()
    section_title(pdf, "2", "DVF Dataset — Paris 2023",
                  "Demandes de Valeurs Foncieres — official government source")

    stats = [
        ("Source",          "data.gouv.fr — geocoded DVF (geo-dvf/latest)"),
        ("Coverage",        "Paris intramuros (dept. 75), year 2023"),
        ("Raw rows",        "80,964 transactions"),
        ("After cleaning",  "29,412 rows (apartments + houses)"),
        ("Total features",  "44 columns including 4 derived (prix_m2, annee, mois, trimestre)"),
        ("Median price/m2", "10,286 EUR/m2"),
        ("Median price",    "420,000 EUR"),
        ("Property types",  "29,253 apartments | 159 houses"),
        ("Storage format",  "Parquet (raw + clean) — optimized I/O"),
    ]

    for k, v in stats:
        kv(pdf, k, v)
        pdf.ln(1)

    pdf.ln(3)
    body(pdf,
         "The DVF cleaning pipeline applies the following steps in order: column renaming, "
         "type casting, filtering on type_local (Apartment/House only), "
         "outlier filtering (price 10k-30M EUR, area 5-1000 m2, price/m2 500-50,000), "
         "deduplication, then addition of derived features.")

    add_figure(pdf, FIG / "01_prix_m2_distribution.png",
               "Fig. 1 - Price/m2 distribution: raw (left) and log-transformed (right)")
    add_figure(pdf, FIG / "03_prix_by_arrondissement.png",
               "Fig. 2 - Median price/m2 by Parisian arrondissement (75001-75020)")


def page_features(pdf):
    pdf.add_page()
    section_title(pdf, "3", "Feature Engineering",
                  "7 numerical features — ML pipeline")

    features_info = [
        ("surface_reelle_bati",        "Property area (m2)",        "Continuous",    "Main predictor"),
        ("nombre_pieces_principales",  "Number of rooms",           "Continuous",    "Size proxy"),
        ("arrondissement",             "Arrondissement (1-20)",     "Categorical",   "Location"),
        ("longitude",                  "GPS Longitude",             "Continuous",    "East-West position"),
        ("latitude",                   "GPS Latitude",              "Continuous",    "North-South position"),
        ("mois",                       "Month (1-12)",              "Cyclical",      "Seasonality"),
        ("nombre_lots",                "Number of lots / mutation", "Continuous",    "Price/m2 quality"),
    ]

    pdf.set_fill_color(*BLUE)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Arial", "B", 10)
    for h, w in zip(["Feature", "Description", "Type", "Interest"], [48, 52, 30, 45]):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    for i, (feat, desc, typ, note) in enumerate(features_info):
        pdf.set_fill_color(*(LGRAY if i % 2 == 0 else WHITE))
        pdf.set_text_color(*DGRAY)
        pdf.set_font("Arial", "", 9)
        for val, w in zip([feat, desc, typ, note], [48, 52, 30, 45]):
            pdf.cell(w, 6, val, border=1, fill=True)
        pdf.ln()
    pdf.ln(4)

    body(pdf,
         "Target variable: prix_m2 = valeur_fonciere / surface_reelle_bati (EUR/m2). "
         "Train/test split: 80% / 20% (random_state=42). "
         "Normalization: StandardScaler applied only for LinearRegression (sklearn Pipeline). "
         "GAM and LightGBM are scale-invariant.")

    add_figure(pdf, FIG / "02_feature_correlations.png",
               "Fig. 3 - Correlation matrix (features + target prix/m2)")


def page_tournament(pdf, metrics_data):
    pdf.add_page()
    section_title(pdf, "4", "Model Tournament",
                  "LinearRegression (baseline) | GAM (interpretable) | LightGBM (performant)")

    body(pdf,
         "Three models were trained and evaluated on the same train/test split "
         "(23,498 / 5,875 observations):")

    models_desc = [
        ("LinearRegression (sklearn)",
         "Linear baseline with StandardScaler normalization. "
         "Limited for non-linear relationships (high prices in certain arrondissements, "
         "area threshold effects, etc.)."),
        ("LinearGAM (pygam 0.12)",
         "Generalized Additive Model: each feature is modeled independently "
         "by cubic splines or factor/linear terms. "
         "Good balance between interpretability and performance."),
        ("LightGBM (gradient boosting)",
         "Most expressive model: 500 trees, learning_rate=0.05, num_leaves=63. "
         "Captures complex non-linear interactions. "
         "Fast training even on 23k observations."),
    ]

    for title, desc in models_desc:
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(*BLUE)
        pdf.set_x(15)
        pdf.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(22)
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(*DGRAY)
        pdf.multi_cell(0, 5.5, desc)
        pdf.ln(1)

    pdf.ln(3)
    metrics_table(pdf, metrics_data)

    add_figure(pdf, FIG / "04_model_comparison.png",
               "Fig. 4 - MAE / RMSE / R2 comparison (gold bar = best score)")

    pdf.set_fill_color(255, 248, 220)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(120, 80, 0)
    pdf.multi_cell(0, 5,
                   "Note: The R2 score (0.19) reflects the inherent limitations of DVF data. "
                   "This fiscal dataset lacks qualitative features "
                   "(floor, condition, exposure, balcony) that explain the majority of "
                   "intra-arrondissement price variance. Academic studies on DVF report R2=0.5-0.8 "
                   "but with enriched datasets (DPE, listings, etc.). "
                   "FairSquare's added value is precisely to add these features "
                   "via OSM, AI Vision, and listing scraping.")
    pdf.ln(3)


def page_avp(pdf):
    pdf.add_page()
    section_title(pdf, "4b", "Residual Analysis",
                  "Predicted vs Actual on the test set")

    add_figure(pdf, FIG / "05_actual_vs_predicted.png",
               "Fig. 5 - Predictions vs Actual values (diagonal = perfect prediction)")

    body(pdf,
         "LightGBM is the most concentrated around the perfect diagonal. "
         "All models show underestimation for properties > 15k EUR/m2 "
         "and overdispersion in the 8k-14k EUR/m2 range — a zone where variance depends "
         "on uncaptured characteristics (floor, renovation status, etc.).")


def page_shap(pdf):
    pdf.add_page()
    section_title(pdf, "5", "Explainability — SHAP (LightGBM)",
                  "SHapley Additive exPlanations: impact of each feature on predictions")

    body(pdf,
         "SHAP quantifies the marginal contribution of each feature to each individual prediction. "
         "TreeExplainer is used (exact computation for tree-based models). "
         "SHAP values are computed on 500 observations from the test set.\n\n"
         "Interpretation: SHAP > 0 increases the predicted price/m2, SHAP < 0 decreases it.")

    add_figure(pdf, FIG / "06_shap_summary.png",
               "Fig. 6 - SHAP Summary Plot (beeswarm): each point = 1 transaction")
    add_figure(pdf, FIG / "07_shap_importance.png",
               "Fig. 7 - SHAP Feature Importance: mean absolute SHAP per feature")
    add_figure(pdf, FIG / "08_shap_surface.png",
               "Fig. 8 - SHAP Dependence Plot: non-linear effect of property area")

    body(pdf,
         "Latitude and longitude dominate SHAP importance — geographic location is the #1 "
         "driver of price/m2 in Paris. "
         "Area has a negative SHAP effect for large properties (>100 m2): "
         "larger units have a lower price/m2, which is consistent with market reality. "
         "Arrondissement captures a portion of geography complementary to lat/lon.")


def page_osm_vision(pdf):
    pdf.add_page()
    section_title(pdf, "6", "Complementary Features",
                  "OSM Geospatial + AI Computer Vision")

    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 7, "6.1 OpenStreetMap — Geospatial Feature Engineering",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    body(pdf,
         "The OSM extractor queries the Overpass API in a single request per point "
         "(anti rate-limit 429 strategy). It combines all POI types in one call "
         "(transit, parks, schools, shops) and computes features locally.")

    bullet(pdf, [
        "dist_metro_m: distance to nearest metro/RER station (m)",
        "dist_park_m: distance to nearest park (m)",
        "dist_school_m: distance to nearest school (m)",
        "dist_supermarket_m: distance to nearest supermarket (m)",
        "transit_count_500m: number of transit stops within 500m",
        "park_area_500m_m2: total park area within 500m (m2)",
        "walkability_score: composite score 0-100 (weighted transit/park/school/shop)",
    ])

    body(pdf,
         "Validated example — Place de la Republique (48.8673, 2.3630):\n"
         "  Metro: 40m | Park: 343m | School: 247m | Shop: 130m\n"
         "  60 transit stops within 500m | Walkability = 76.38/100")

    pdf.ln(3)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 7, "6.2 Computer Vision — Renovation Score (Google Gemini)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    body(pdf,
         "The Vision module uses Google Gemini 3.1 Flash Lite Preview (500 RPD free tier) "
         "to analyze property listing photos. The structured prompt instructs the model to "
         "respond only in valid JSON with 3 fields:")

    bullet(pdf, [
        "renovation_score (int 1-5): 1=excellent condition/renovated, 5=uninhabitable",
        "space_category (str): Narrow | Standard | Spacious",
        "reasoning (str): justification in 1-2 sentences",
    ])

    body(pdf,
         "Validated test: score=1/5 (excellent condition), Standard, response time ~2-3s. "
         "Model selected after systematic quota testing "
         "(gemini-2.5-flash exhausted, gemini-2.0-flash: 429 rate-limited, etc.).")


def page_conclusions(pdf):
    pdf.add_page()
    section_title(pdf, "7", "Conclusions & Next Steps",
                  "Phase 2 — Towards production")

    conclusions = [
        ("DVF Pipeline",
         "Robust ingestion via data.gouv.fr (fallback from CEREMA API which was unreachable). "
         "80k -> 29k clean rows. Optimized Parquet format."),
        ("OSM Features",
         "Single-query Overpass strategy eliminates 429 rate-limiting. "
         "Walkability score validated on real Parisian locations."),
        ("AI Vision",
         "Gemini 3.1 Flash Lite working in production (500 RPD). "
         "Response time < 3s. Renovation score is coherent."),
        ("ML Pipeline",
         "LightGBM outperforms LinearRegression and GAM. "
         "SHAP confirms that geographic location (lat/lon) is the dominant factor. "
         "Limited R2 (0.19) expected on DVF alone — will improve with OSM + Vision features."),
    ]

    for title, text in conclusions:
        pdf.set_x(15)
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(*BLUE)
        pdf.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(22)
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(*DGRAY)
        pdf.multi_cell(0, 5.5, text)
        pdf.ln(2)

    pdf.ln(3)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 7, "Next steps (Phase 2):",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    bullet(pdf, [
        "Integrate OSM features (walkability, dist_metro) into ML pipeline -> expected R2 +0.10",
        "Integrate renovation_score (Gemini Vision) as an additional ML feature",
        "Expand to all 8 IDF departments (77, 78, 91, 92, 93, 94, 95)",
        "Scrape SeLoger/Leboncoin listings (floor, DPE energy rating, balcony...)",
        "Deploy FastAPI with /predict endpoint and Swagger documentation",
        "Build Streamlit dashboard with Leaflet map and interactive filters",
        "Build recommendation system (collaborative filtering + content-based)",
        "Containerize with Docker Compose (API + PostgreSQL + Streamlit)",
    ])


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    if METRICS.exists():
        metrics_data = json.loads(METRICS.read_text(encoding="utf-8"))
    else:
        metrics_data = []
        print("  [WARN] metrics.json not found — empty table")

    pdf = Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(12, 16, 12)
    pdf.load_fonts()

    page_cover(pdf)
    page_context(pdf)
    page_dataset(pdf)
    page_features(pdf)
    page_tournament(pdf, metrics_data)
    page_avp(pdf)
    page_shap(pdf)
    page_osm_vision(pdf)
    page_conclusions(pdf)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT_PDF))
    print(f"\n  PDF generated ({pdf.page_no()} pages): {OUT_PDF.resolve()}")
    return OUT_PDF


if __name__ == "__main__":
    main()
