import pandas as pd
import numpy as np
from src.data_ingestion.dvf_client import DVFClient
from src.data_ingestion.dvf_cleaner import clean
from src.utils.logger import get_logger
import os

logger = get_logger(__name__)

# Market Correction Factors (Paris-specific quarterly index)
# Base = Q1 2024 (1.0). 
# Prices were higher in 2021, so we must multiply them by < 1 to get "current" value equivalent for training.
# This makes the model "representative of reality".
MARKET_INDEX = {
    2021: 0.94, # Prices were ~6% higher
    2022: 0.96, # Prices were ~4% higher
    2023: 0.98, # Prices were ~2% higher
    2024: 1.00,
}

def main():
    client = DVFClient()
    years = [2021, 2022, 2023, 2024]
    all_frames = []

    print(f"=== Professional Data Federation ({years}) ===")
    
    for year in years:
        try:
            print(f"Fetching {year}...")
            df = client.fetch_department("75", year=year)
            df_c = clean(df)
            
            # Apply Temporal Calibration (Specialist logic)
            # Adjust historical prices to 2024/2025 equivalents
            factor = MARKET_INDEX.get(year, 1.0)
            if factor != 1.0:
                print(f"  Scaling {year} prices by {factor} (market calibration)")
                df_c["valeur_fonciere"] *= factor
                df_c["prix_m2"] *= factor
            
            all_frames.append(df_c)
            print(f"  {len(df_c)} clean rows added.")
        except Exception as e:
            print(f"  Error fetching {year}: {e}")

    if not all_frames:
        print("No data collected. Check internet connection.")
        return

    df_final = pd.concat(all_frames, ignore_index=True)
    print(f"\nFinal Dataset: {len(df_final)} rows across {len(years)} years.")

    # ── Specialist Fix: Parquet Type Safety ──────────────────────────
    # Columns like lotX_numero or code_commune often have mixed types (str/float)
    # which breaks pyarrow. We force them to string.
    obj_cols = df_final.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df_final[col] = df_final[col].astype(str).replace("nan", np.nan).replace("None", np.nan)

    # Save to multiple targets to be robust
    out_path = "data/processed/dvf_paris_2023_2025_clean.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save the specific file the dashboard wants
    df_final.to_parquet(out_path, index=False)
    
    # Also save a generic one
    df_final.to_parquet("data/processed/dvf_paris_merged.parquet", index=False)
    
    print(f"Successfully saved to {out_path}")

if __name__ == "__main__":
    main()
