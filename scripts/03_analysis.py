from pathlib import Path

def main():
    Path("outputs").mkdir(parents=True, exist_ok=True)
    print("TODO: analysis + export outputs into outputs/.")

if __name__ == "__main__":
    main()
