import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion.dvf_client import DVFClient
from src.data_ingestion.dvf_cleaner import clean

def main():
    client = DVFClient()
    print("Fetching DVF data for Paris (75) - 2023...")
    try:
        df = client.fetch_department("75", year=2023)
        print(f"Downloaded {len(df)} rows.")
        
        df_clean = clean(df)
        print(f"Cleaned data: {len(df_clean)} rows.")
        
        # Save a sample for testing
        out_root = Path("data/raw/dvf")
        out_root.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet("data/raw/dvf/paris_2023_test.parquet")
        print("Saved to data/raw/dvf/paris_2023_test.parquet")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
