# Madrid Airbnb - Geospatial Analysis Pipeline

This project implements a reproducible ETL pipeline for spatial analysis of Inside Airbnb data for Madrid. It cleans and enriches listings, integrates neighbourhood polygons through a point-in-polygon join, and produces web-ready GeoJSON layers plus static figures for the report.

```md
## Repository structure (short)

```text
geospatial-project/
├── environment/                 # conda/micromamba environment spec
├── data/                        # raw inputs and processed outputs
│   └── processed/
├── notebooks/                   # exploratory + final notebook
├── scripts/                     # reproducible entrypoints (01_, 02_, ...)
├── src/                         # reusable python modules
└── reports/figures/             # static figures for the report
```

## How to run (minimal CLI)

Prerequisites: create/activate the `geo` environment (see `environment/`):

```bash
# with micromamba/conda
micromamba env create -f environment/environment.yml
micromamba activate geo
```

1) QC spatial sanity checks

```bash
python scripts/01_verify_spatial_data.py
```

2) Generate the static report figure (overview + inset)

```bash
python scripts/02_make_static_map_overview_inset.py
```

Inputs (used by the above scripts):

- `data/processed/listings_clean.parquet`
- `data/processed/neighbourhoods_enriched.geojson`

Output:

- `reports/figures/fig_madrid_overview_inset_price.png`

## Notes

- Keep the project root as the working directory when running scripts so relative paths in `src/config.py` work.
- The `geo` environment pins Python 3.12 in `environment/environment.yml` for reproducibility.

## Troubleshooting

- If imports fail, ensure the `geo` env is active and packages were installed from `environment/environment.yml`.
- If plotting fails headless on a server, set `MPLBACKEND=Agg` before running.

---

For more details on the pipeline, see `notebooks/05_final_pipeline.ipynb` and the `src/` package functions.
