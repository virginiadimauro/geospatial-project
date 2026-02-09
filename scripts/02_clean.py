from pathlib import Path

def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    print("TODO: cleaning/wrangling + CRS checks + geometry validity.")

if __name__ == "__main__":
    main()
