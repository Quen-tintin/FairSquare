"""
PoC DVF — Récupération et nettoyage des données Paris (75)
Lancer : python scripts/run_dvf_poc.py
"""

from src.data_ingestion.dvf_client import DVFClient
from src.data_ingestion.dvf_cleaner import clean, eda_summary
import json

def main():
    client = DVFClient()

    print("=== Fetching DVF — Paris 75 (2023) ===")
    df_raw = client.fetch_department("75", year=2023)
    print(f"Raw rows: {len(df_raw)}")

    # Nettoyage
    df_clean = clean(df_raw)
    print(f"Clean rows: {len(df_clean)}")

    # EDA
    summary = eda_summary(df_clean)
    print("\n=== EDA Summary ===")
    print(json.dumps(summary, indent=2, default=str, ensure_ascii=False))

    # Sauvegarde
    client.save_raw(df_raw, "dvf_paris_2023_poc.parquet")
    df_clean.to_parquet("data/processed/dvf_paris_2023_clean.parquet", index=False)
    print("\nFiles saved.")

if __name__ == "__main__":
    main()
