#!/bin/bash
# =============================================================================
# Script: webmap/run.sh
# Purpose:
#   Launch the Streamlit web map with reproducible pre-checks.
#
# Expected inputs:
#   - data/processed/model_sample.parquet
#   - outputs/tables/residuals_for_map.csv (auto-generated if missing)
#   - data/processed/map_points_sample.geojson (auto-generated if missing)
#   - data/processed/map_grid_cells.geojson (auto-generated if missing)
#
# Produced outputs (if regeneration is required):
#   - outputs/tables/residuals_for_map.csv
#   - data/processed/map_points_sample.geojson
#   - data/processed/map_grid_cells.geojson
#
# Usage:
#   bash webmap/run.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log_info() {
    echo "[INFO] $1"
}

log_warn() {
    echo "[WARNING] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

run_qa() {
    micromamba run -n geo python scripts/qa_webmap.py
}

log_info "Project root: $PROJECT_ROOT"

if ! command -v micromamba >/dev/null 2>&1; then
    log_error "micromamba not found in PATH. Install micromamba or update PATH."
    exit 1
fi

log_info "Step 1/2: Running quality assurance checks"
if ! run_qa; then
    log_warn "Quality assurance failed. Attempting deterministic regeneration of map artifacts."

    if [ ! -f "data/processed/model_sample.parquet" ]; then
        log_error "Missing required input: data/processed/model_sample.parquet"
        log_error "Run data preparation first: jupyter lab notebooks/05_final_pipeline.ipynb"
        exit 1
    fi

    log_info "Regenerating residuals table: outputs/tables/residuals_for_map.csv"
    micromamba run -n geo python scripts/07b_extract_residuals.py

    log_info "Regenerating map layers: data/processed/map_points_sample.geojson, data/processed/map_grid_cells.geojson"
    micromamba run -n geo python scripts/08_prepare_map_layers.py

    log_info "Re-running quality assurance checks"
    if ! run_qa; then
        log_error "Quality assurance still failing after regeneration. Check script outputs and logs above."
        exit 1
    fi
fi

log_info "Step 2/2: Launching Streamlit web map"
log_info "URL: http://localhost:8501"
micromamba run -n geo streamlit run webmap/app.py
