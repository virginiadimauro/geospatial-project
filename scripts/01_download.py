"""
01_download.py
- Download data (or document official sources if download is not permitted)
- Save raw data into data/raw/
"""

from pathlib import Path

def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("TODO: download data into data/raw/ or write instructions in docs/.")
    print(f"Raw data folder ready: {raw_dir.resolve()}")

if __name__ == "__main__":
    main()
