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

- **Official source:** Inside Airbnb — https://insideairbnb.com/get-the-data/
- **Dataset reference used in this project:** *Madrid, Comunidad de Madrid, Spain — 14 September, 2025* (download page section).

- `data/original/` — raw files from Inside Airbnb and supplementary tables (listings, calendar, reviews, neighbourhoods).
- `data/processed/` — cleaned and geo-enriched files used for analysis (e.g. `listings_points_enriched_sample.geojson`, `neighbourhoods_enriched.geojson`).

**How to download original data**

1. Open https://insideairbnb.com/get-the-data/
2. In the table, go to **Madrid, Comunidad de Madrid, Spain — 14 September, 2025**.
3. Download and place these files in `data/original/` with these names:
	- `listings.csv.gz` → `listings.csv`
	- `calendar.csv.gz` → `calendar.csv`
	- `reviews.csv.gz` → `reviews.csv`
	- `neighbourhoods.geojson`
4. (Optional) Download summary files if needed by exploratory checks:
	- `listings_summary.csv`
	- `reviews_summary.csv`

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

2. Reproduce the pipeline (two phases):

**Phase A: Data Preparation (Notebook-driven, generates `data/processed/` datasets)**

Skip this if `data/processed/` already contains the required files (see note below).

Execute the end-to-end cleaning and enrichment notebook:

```bash
jupyter notebook notebooks/05_final_pipeline.ipynb
```

This generates:
- `listings_clean.parquet` — cleaned listings with price parsing and QC
- `model_sample.parquet` — model-ready sample (N=15,641, all covariates complete)
- `neighbourhoods_enriched.geojson`, `neighbourhoods_enriched.parquet` — enriched neighbourhood polygons
- `calendar_enriched_with_neighbourhoods.parquet` — calendar data with spatial joins
- `reviews_clean.parquet` — cleaned reviews

Alternatively, individual steps can be explored in:
- `notebooks/02_price_investigation.ipynb` — price data exploration and outlier analysis

**Phase B: Analysis (Scripts, depends on Phase A outputs)**

Once `data/processed/` is populated, run the analysis scripts in order:

```bash
python scripts/01_verify_spatial_data.py          # QC: verify spatial integrity
python scripts/03_ols_price_analysis.py            # Baseline: OLS regression (Model A + B)
python scripts/04_spatial_autocorr_morans_i.py    # Test: Global Moran's I (price-valid sample)
python scripts/05_lm_diagnostic_tests.py          # Test: LM diagnostics for spatial dependence
python scripts/06_morans_i_subset_consistency_check.py  # Validation: consistency on model sample
python scripts/07_spatial_models_sar_sem.py       # Spatial models: SAR and SEM estimation
python scripts/02_make_static_map_overview_inset.py    # Visual: static map for report
```

**Note on `data/processed/` re-use**: If you have already run Phase A (or downloaded pre-processed data), you can skip the notebook and proceed directly to Phase B. The scripts will load the required parquet files from `data/processed/`.

---

**Important**: Keep the project root as the working directory so relative paths in `src/config.py` work correctly:

```bash
cd /path/to/geospatial-project
python scripts/03_ols_price_analysis.py  # ✓ Correct
```

(Not from a subdirectory)

**Outputs and results**

- Estimates and tables are saved in `outputs/tables/` (OLS vs spatial models) and figures in `outputs/figures/`.
- **Sample audit trail**: `outputs/tables/sample_flow.csv` documents the filtering steps (24,987 → 18,940 → 18,765 → 15,641) and distinguishes between "price-valid" (N=18,940) and "model-complete" (N=15,641) samples (see `docs/reproducibility.md`).
- Notebooks `notebooks/02_price_investigation.ipynb` and `notebooks/05_final_pipeline.ipynb` document the main analysis flow and summary results.

**Reproducing main results**

- To reproduce the OLS vs SAR/SEM comparison: run `scripts/03_ols_price_analysis.py` then `scripts/07_spatial_models_sar_sem.py`. CSV files with coefficients and test results will be in `outputs/tables/`.

---


## Interactive Web Map

An interactive Streamlit application visualizes spatial autocorrelation in model residuals and validates the superiority of SAR over OLS. The map transforms abstract statistical metrics (Moran's I, residuals) into actionable geographic intelligence.

### Quick Start

```bash
# Automated launch (with QA validation)
bash webmap/run.sh

# Manual launch
micromamba activate geo
streamlit run webmap/app.py

# Static HTML (no dependencies)
open reports/maps/interactive_map.html
```

**Dependencies:** All webmap requirements (streamlit, folium, etc.) are included in `environment/environment.yml`. Optional `webmap/runtime.txt` is for deployment platforms only.

### Interactive Features

**Sidebar controls:**
- **Price range slider** — Filter by nightly rate (€)
- **Room type multiselect** — Toggle private/hotel/shared rooms
- **Accommodates range** — Filter by guest capacity
- **Model choice radio** — Switch between OLS / SAR / Difference
- **Residual threshold** — Highlight high-error listings (|residual| > threshold)
- **Layer toggles** — Show/hide individual points or grid cells

**Map display:**
- Folium-based interactive map (OpenStreetMap basemap)
- Color-coded residuals: blue=overestimate, gray=fit, red=underestimate
- Clickable popups with listing details (price, rating, residuals)
- Two layers: 5,000 sample points + 23 neighborhood grid cells (~5-6 km²)
- Legend and interpretation guide embedded

**Summary statistics:** Filtered dataset size, price stats, residual stats by model

### Color Scale

| Color | Residual Range | Interpretation |
|-------|----------------|----------------|
| 🔵 Dark blue | ≤ -1.0 | Strong overestimate (OLS too high) |
| 🔵 Light blue | -0.5 to -0.1 | Mild overestimation |
| ⚪ Gray/White | -0.1 to +0.1 | Good model fit |
| 🟠 Light orange | +0.1 to +0.5 | Mild underestimation |
| 🔴 Dark red | ≥ +0.5 | Strong underestimate (missing factors) |

Diverging blue-gray-red palette is intuitive and colorblind-friendly; neutral white at zero.

### Research Validation

**Question 1: Does SAR reduce spatial autocorrelation?**
- **Visual answer:** OLS map shows systematic north-south clustering; SAR map shows dispersed residuals
- **Confirmed:** Moran's I reduction from 0.165 to 0.071 (57% improvement)

**Question 2: Where does OLS fail most?**
- **Central tourist zones** (Sol, Recoletos, Chamberí) show large positive OLS residuals
- Tourism premium not captured by baseline model; SAR captures via ρ=0.202 spatial lag term

**Question 3: Does SEM (spatial error) help?**
- SEM residuals nearly identical to OLS on map
- **Rejected:** No improvement in clustering patterns (Moran's I ~0.172)

### Regenerate Map Layers

If `data/processed/model_sample.parquet` changes:

```bash
# Step 1: Extract residuals from OLS/SAR/SEM models
python scripts/07b_extract_residuals.py
# → outputs/tables/residuals_for_map.csv (18,940 residuals)

# Step 2: Prepare GeoJSON layers for web map
python scripts/08_prepare_map_layers.py
# → data/processed/map_points_sample.geojson (5k sample, 2.4 MB)
# → data/processed/map_grid_cells.geojson (23 cells, 9 KB)

# Step 3: Validate and launch
python scripts/qa_webmap.py  # Quality checks (CRS, geometries, files)
streamlit run webmap/app.py
```

**Estimated time:** ~6 minutes (5 min extract + 1 min prep)

### Technical Details

**Key files:**
- `webmap/app.py` — Main Streamlit application
- `webmap/run.sh` — Automated launcher with QA validation
- `scripts/07b_extract_residuals.py` — Extract OLS/SAR/SEM residuals
- `scripts/08_prepare_map_layers.py` — Generate GeoJSON layers
- `scripts/qa_webmap.py` — Quality assurance (CRS EPSG:4326 web, EPSG:25830 UTM 30N spatial)
- `reports/maps/interactive_map.html` — Static export (5.9 MB)
- `webmap/runtime.txt` — Python version (optional, for deployment only)

**Data summary:**
- Total listings: 18,940 (N)
- Map sample: 5,000 (26%, random seed=42)
- Grid cells: 23 (0.05° resolution ≈ 5-6 km)
- HTML file: 5.9 MB (standalone, shareable)

**CRS consistency:**
- Web layers: EPSG:4326 (WGS84, degrees)
- Spatial models: EPSG:25830 (UTM Zone 30N, meters)
- Conversion: Automatic in scripts

**Visual evidence:**
- OLS map: Blue/red clusters visible (Moran's I=0.165)
- SAR map: Uniform colors (Moran's I=0.071, 57% reduction)
- SEM map: Similar to OLS (no improvement)
