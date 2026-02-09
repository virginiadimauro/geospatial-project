# Reproducibility

## Environment
- Create env:
  micromamba env create -f environment/environment.yml
- Activate:
  micromamba activate geo

## Run pipeline
python scripts/01_download.py
python scripts/02_clean.py
python scripts/03_analysis.py
python scripts/04_webmap.py

## Notes
- Distances/areas must be computed in a projected CRS (meters), not EPSG:4326.
- Always check CRS, geometry validity, missing values and outliers.
