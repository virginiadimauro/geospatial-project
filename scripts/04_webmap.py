from pathlib import Path

def main():
    Path("webmap").mkdir(parents=True, exist_ok=True)
    print("TODO: create interactive web map into webmap/.")

if __name__ == "__main__":
    main()
