# Reproducibility (Source of Truth)

This document is the authoritative reference for reproducibility: environment constraints, required inputs, pipeline execution order, sample definitions, and CRS/unit policy.

## 1) Environment (pinned)

Create and activate exactly from `environment/environment.yml`:

```bash
micromamba env create -f environment/environment.yml
micromamba activate geo
```

Environment expectations:
- Python 3.12
- NumPy pinning is defined in `environment/environment.yml` for compatibility with the spatial stack (`numba`, `libpysal`, `esda`, `spreg`)
- Geospatial stack is fully conda-based via the environment file

Verification commands:

```bash
python -c "import numpy, numba; print('numpy', numpy.__version__, 'numba', numba.__version__)"
python -c "import libpysal, esda, spreg; print('PySAL OK')"
python -c "import geopandas, shapely, pyproj, fiona, rasterio; from osgeo import gdal; print('Geo stack OK')"
```

## 2) Raw inputs (official required)

Download all files from the Madrid snapshot and store them in `data/original/`.

Required snapshot files to download:
- `listings.csv.gz` (detailed listings)
- `calendar.csv.gz` (detailed calendar)
- `reviews.csv.gz` (detailed reviews)
- `listings.csv` (summary listings)
- `reviews.csv` (summary reviews)
- `neighbourhoods.csv`
- `neighbourhoods.geojson`

Required local filenames in this project (collision-safe naming):
- `listings.csv`, `calendar.csv`, `reviews.csv` (extracted detailed files)
- `listings_summary.csv`, `reviews_summary.csv` (summary files)
- `neighbourhoods.csv`, `neighbourhoods.geojson`

Source used in this project:
- https://insideairbnb.com/get-the-data/
- Madrid, Comunidad de Madrid, Spain — 14 September, 2025

Example download commands:

```bash
cd data/original
curl -L -O https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/data/listings.csv.gz
curl -L -O https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/data/calendar.csv.gz
curl -L -O https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/data/reviews.csv.gz
curl -L https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/data/listings.csv -o listings_summary.csv
curl -L https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/data/reviews.csv -o reviews_summary.csv
curl -L -O https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/visualisations/neighbourhoods.csv
curl -L -O https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/visualisations/neighbourhoods.geojson
gunzip -f listings.csv.gz calendar.csv.gz reviews.csv.gz
cd ../..
```

## 3) Authoritative pipeline order

Run commands from project root.

### Phase A — Data preparation

```bash
jupyter lab notebooks/05_final_pipeline.ipynb
```

Headless execution (no browser):

```bash
jupyter execute notebooks/05_final_pipeline.ipynb --inplace
```

Path note: when running from project root, use `notebooks/05_final_pipeline.ipynb` (not just `05_final_pipeline.ipynb`).

Expected outputs in `data/processed/` include:
- `listings_clean.parquet`
- `model_sample.parquet`
- `neighbourhoods_enriched.geojson`
- `calendar_enriched_with_neighbourhoods.parquet`
- `reviews_clean.parquet`

Skip Phase A only if you already ran it locally and `data/processed/` contains the listed outputs.

### Phase B — Analyses

```bash
python scripts/01_verify_spatial_data.py
python scripts/03_ols_price_analysis.py
python scripts/04_spatial_autocorr_morans_i.py
python scripts/05_lm_diagnostic_tests.py
python scripts/06_morans_i_subset_consistency_check.py
python scripts/07_spatial_models_sar_sem.py
python scripts/02_make_static_map_overview_inset.py
```

### Phase C — Web map (optional visualization)

```bash
bash webmap/run.sh
```

Alternative launch:

```bash
streamlit run webmap/app.py
```

## 4) Sample flow and analysis samples

Reference audit trail:
- `outputs/tables/sample_flow.csv`

Filtering flow:

| Stage | Action | N remaining | N dropped |
|---|---|---:|---:|
| 0 | Initial load | 24,987 | — |
| 1 | Price parsing (valid/non-missing) | 18,940 | 6,047 |
| 2 | Winsorization (0.5%–99.5%) | 18,765 | 175 |
| 3 | Complete covariates for models | 15,641 | 3,124 |

Sample definitions:

1. **Price-valid sample (N=18,940)**
   - Used for listing-level Moran’s I in `scripts/04_spatial_autocorr_morans_i.py`.

2. **Model-complete sample (N=15,641)**
   - Used in `scripts/03_ols_price_analysis.py`, `scripts/05_lm_diagnostic_tests.py`, `scripts/06_morans_i_subset_consistency_check.py`, `scripts/07_spatial_models_sar_sem.py`.

## 5) CRS and units policy

- Use projected CRS (meters) for metric operations (distance/area/weights).
- Do not use EPSG:4326 for metric computations.
- Check CRS consistency before spatial joins, weights, and model estimation.

## 6) Output folders policy

Project-wide policy:
- `outputs/` → tables and intermediate artifacts
- `reports/figures/` → final report figures

Related outputs:
- Static web map: `reports/maps/interactive_map.html`
