# Madrid Airbnb — Geospatial Analysis

This repository provides a reproducible geospatial workflow to study Airbnb prices in Madrid using Inside Airbnb data.
The analysis combines data preparation, OLS and spatial econometric models (SAR/SEM), and map-based inspection of residual patterns.
Research question: to what extent does location (accessibility + neighbourhood context) explain nightly prices beyond listing/host characteristics, and when are spatial models preferable to OLS?

## Quickstart

### 1) Create environment

```bash
micromamba env create -f environment/environment.yml
micromamba activate geo
```

### 2) Download data (`data/original/`)

Use Madrid snapshot **14 September, 2025** from Inside Airbnb.

Download all snapshot files (detailed + summary + neighbourhood files):
- `https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/data/listings.csv.gz`
- `https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/data/calendar.csv.gz`
- `https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/data/reviews.csv.gz`
- `https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/data/listings.csv`
- `https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/data/reviews.csv`
- `https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/visualisations/neighbourhoods.csv`
- `https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2025-09-14/visualisations/neighbourhoods.geojson`

Use this naming convention in `data/original/` to avoid collisions between detailed and summary files:
- detailed: `listings.csv.gz`, `calendar.csv.gz`, `reviews.csv.gz` → extracted as `listings.csv`, `calendar.csv`, `reviews.csv`
- summary: `listings_summary.csv`, `reviews_summary.csv`
- neighbourhood: `neighbourhoods.csv`, `neighbourhoods.geojson`

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

### 3) Run notebook (data preparation)

```bash
jupyter lab notebooks/05_final_pipeline.ipynb
```

Headless execution (no browser):

```bash
jupyter execute notebooks/05_final_pipeline.ipynb --inplace
```

Note: if you run from project root, include the `notebooks/` prefix in the path.

Skip this phase only if you already ran it locally and `data/processed/` contains the expected outputs.

### 4) Run scripts (analysis)

```bash
python scripts/01_verify_spatial_data.py
python scripts/03_ols_price_analysis.py
python scripts/04_spatial_autocorr_morans_i.py
python scripts/05_lm_diagnostic_tests.py
python scripts/06_morans_i_subset_consistency_check.py
python scripts/07_spatial_models_sar_sem.py
python scripts/02_make_static_map_overview_inset.py
```

## Web map

Launch:

```bash
bash webmap/run.sh
# or
streamlit run webmap/app.py
```

The web map supports inspection of residual patterns and filtering across layers.

## Repository structure (short)

```text
geospatial-project/
├── data/
│   ├── original/        # raw inputs
│   └── processed/       # generated datasets
├── environment/         # environment specification
├── notebooks/           # preparation notebooks
├── scripts/             # analysis entrypoints
├── src/                 # reusable modules
├── outputs/             # tables + intermediate artifacts
├── reports/figures/     # final report figures
└── webmap/              # Streamlit app
```

## Reproducibility details

For full reproducibility details (environment constraints, authoritative pipeline order, sample definitions, CRS/unit policy), see [docs/reproducibility.md](docs/reproducibility.md).