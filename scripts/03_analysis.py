"""
03_analysis.py
- Analysis that answers the research question
- Produce tables/figures into outputs/
"""

from pathlib import Path

def main():
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("TODO: analysis + export outputs (tables/figures) into outputs/")
    print(f"Outputs folder ready: {outputs_dir.resolve()}")

if __name__ == "__main__":
    main()
