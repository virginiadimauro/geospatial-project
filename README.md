# Madrid Airbnb — Geospatial Analysis

This repository contains a reproducible pipeline for the spatial analysis of Inside Airbnb data in Madrid. The main objective is to assess how much location, captured through accessibility measures and neighbourhood context, explains Airbnb nightly prices beyond property and host characteristics, and whether spatial spillovers justify the use of spatial regression models (SAR/SEM) over OLS.

**Research Question (RQ)**

To what extent does location—captured through accessibility measures and neighbourhood context—explain Airbnb nightly prices in Madrid beyond property and host characteristics, and is there evidence of spatial spillovers that justify spatial regression models (SAR/SEM) over OLS?

**Project summary**

- **Data:** cleaning and enrichment of listings, integration with neighbourhood polygons (point-in-polygon), and computation of accessibility indicators.
- **Analytical objective:** estimate the effect of location on prices and test for spatial autocorrelation; compare OLS with spatial models (SAR, SEM).
- **Main outputs:** OLS results, SAR/SEM estimates, diagnostic tests for spatial autocorrelation (Moran's I), maps and figures for the report.

**Repository structure (short)**

geospatial-project/
- environment/                 — environment spec (conda/micromamba)
- data/                        — raw inputs and processed outputs
- notebooks/                   — exploratory analysis and final pipeline
- scripts/                     — reproducible entrypoints (01_, 02_, ...)
- src/                         — reusable Python modules
- reports/                     — static figures for the report

**Key data**

- `data/original/` — raw files from Inside Airbnb and supplementary tables (listings, calendar, reviews, neighbourhoods).
- `data/processed/` — cleaned and geo-enriched files used for analysis (e.g. `listings_points_enriched_sample.geojson`, `neighbourhoods_enriched.geojson`).

**Methods and analysis steps**

1. Data cleaning and spatial quality checks (`src/cleaning.py`, `scripts/01_verify_spatial_data.py`).
2. Spatial enrichment: point-in-polygon join, computation of accessibility measures and neighbourhood context variables (`src/spatial.py`).
3. OLS analysis as baseline, residual diagnostics and test for spatial autocorrelation (Moran's I) (`scripts/03_ols_price_analysis.py`, `scripts/05_lm_diagnostic_tests.py`).
4. Fit spatial regression models (SAR, SEM) and compare to OLS to assess spatial spillovers (`scripts/07_spatial_models_sar_sem.py`, `src/spatial.py`).

**Main scripts**

- `scripts/01_verify_spatial_data.py` — spatial quality checks and integrity.
- `scripts/02_make_static_map_overview_inset.py` — static map for the report.
- `scripts/03_ols_price_analysis.py` — OLS regressions and result export.
- `scripts/04_spatial_autocorr_morans_i.py` — Moran's I calculation and residual maps.
- `scripts/05_lm_diagnostic_tests.py` — linear model diagnostic tests.
- `scripts/07_spatial_models_sar_sem.py` — estimation and comparison of SAR/SEM models.

**Reproducibility (quick setup)**

1. Create and activate the environment (micromamba/conda):

```bash
micromamba env create -f environment/environment.yml
micromamba activate geo
```

2. Run the scripts in logical order (or use the notebooks in `notebooks/` to reproduce steps):

```bash
python scripts/01_verify_spatial_data.py
python scripts/03_ols_price_analysis.py
python scripts/04_spatial_autocorr_morans_i.py
python scripts/07_spatial_models_sar_sem.py
```

Note: keep the project root as the working directory so relative paths in `src/config.py` work.

**Outputs and results**

- Estimates and tables are saved in `outputs/tables/` (OLS vs spatial models) and figures in `outputs/figures/`.
- **Sample audit trail**: `outputs/tables/sample_flow.csv` documents the filtering steps (24,987 → 18,940 → 18,765 → 15,641) and distinguishes between "price-valid" (N=18,940) and "model-complete" (N=15,641) samples (see `docs/reproducibility.md`).
- Notebooks `notebooks/02_price_investigation.ipynb` and `notebooks/05_final_pipeline.ipynb` document the main analysis flow and summary results.

**Reproducing main results**

- To reproduce the OLS vs SAR/SEM comparison: run `scripts/03_ols_price_analysis.py` then `scripts/07_spatial_models_sar_sem.py`. CSV files with coefficients and test results will be in `outputs/tables/`.

