from pathlib import Path

def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    print("TODO: download data into data/raw/ (or write instructions in docs/).")

if __name__ == "__main__":
    main()
