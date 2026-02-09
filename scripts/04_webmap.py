"""
04_webmap.py
- Build an interactive web map (not decorative)
- Save into webmap/ (e.g., index.html)
"""

from pathlib import Path

def main():
    webmap_dir = Path("webmap")
    webmap_dir.mkdir(parents=True, exist_ok=True)

    print("TODO: generate interactive web map into webmap/")
    print(f"Webmap folder ready: {webmap_dir.resolve()}")

if __name__ == "__main__":
    main()
