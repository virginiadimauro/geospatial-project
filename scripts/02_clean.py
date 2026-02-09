"""
02_clean.py
- Load raw data
- Cleaning/wrangling
- CRS checks + geometry validity checks
- Save cleaned data into data/processed/
"""

from pathlib import Path

def main():
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("TODO: cleaning + CRS checks (gdf.crs) + geometry validity (gdf.is_valid)")
    print(f"Processed data folder ready: {processed_dir.resolve()}")

if __name__ == "__main__":
    main()
