#!/usr/bin/env python
"""
Interactive web map for model residual inspection (Madrid Airbnb).

Purpose
-------
Provide an interactive and reproducible visualization of listing-level and
grid-level residual patterns from OLS/SAR/SEM outputs.

Expected inputs
---------------
- data/processed/model_sample.parquet
- outputs/tables/residuals_for_map.csv
- data/processed/map_points_sample.geojson (EPSG:4326)
- data/processed/map_grid_cells.geojson (EPSG:4326)

Output
------
Rendered Streamlit interface (no file writes).
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import numpy as np
from pathlib import Path
import os
import subprocess
import sys

# ============================================================================
# SETUP
# ============================================================================
st.set_page_config(page_title="Madrid Airbnb - Spatial Models", layout="wide")
st.markdown("# Interactive Map: Madrid Airbnb Spatial Analysis")
st.markdown("Residual maps for model diagnostics and spatial pattern inspection.")

def resolve_project_root() -> Path:
    env_root = os.getenv("GEOSPATIAL_PROJECT_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parent.parent)

    for root in candidates:
        if (root / "webmap" / "app.py").exists():
            return root
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = resolve_project_root()
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"


def get_required_paths() -> dict[str, Path]:
    return {
        "model sample": DATA_PROCESSED / "model_sample.parquet",
        "residuals table": OUTPUT_TABLES / "residuals_for_map.csv",
        "points layer": DATA_PROCESSED / "map_points_sample.geojson",
        "grid layer": DATA_PROCESSED / "map_grid_cells.geojson",
    }


def get_missing_paths(required_paths: dict[str, Path]) -> list[str]:
    return [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if not path.exists()
    ]


def bootstrap_missing_webmap_inputs(required_paths: dict[str, Path]) -> tuple[bool, str]:
    model_sample = required_paths["model sample"]
    if not model_sample.exists():
        return (
            False,
            "Missing model sample input. Expected file:\n"
            f"- {model_sample}\n"
            "Run the data preparation pipeline first: jupyter lab notebooks/05_final_pipeline.ipynb",
        )

    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    commands = [
        [sys.executable, "scripts/07b_extract_residuals.py"],
        [sys.executable, "scripts/08_prepare_map_layers.py"],
    ]
    for command in commands:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return (
                False,
                "Automatic regeneration failed.\n"
                f"Command: {' '.join(command)}\n"
                f"Exit code: {result.returncode}\n"
                f"Stdout:\n{result.stdout[-2000:]}\n"
                f"Stderr:\n{result.stderr[-2000:]}",
            )

    missing_after = get_missing_paths(required_paths)
    if missing_after:
        return (
            False,
            "Regeneration completed but required files are still missing:\n- "
            + "\n- ".join(missing_after),
        )

    return True, "Missing webmap inputs regenerated successfully."

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    required_paths = get_required_paths()
    missing_paths = get_missing_paths(required_paths)
    if missing_paths:
        raise FileNotFoundError(
            "Missing required webmap inputs.\n"
            f"Project root in use: {PROJECT_ROOT}\n"
            f"Current working directory: {Path.cwd()}\n"
            "Missing files:\n- "
            + "\n- ".join(missing_paths)
            + "\nRun: bash webmap/run.sh"
        )

    df = pd.read_parquet(required_paths["model sample"])
    resid = pd.read_csv(required_paths["residuals table"])
    points = gpd.read_file(required_paths["points layer"])
    grid = gpd.read_file(required_paths["grid layer"])

    required_df_cols = ["listing_id", "price_numeric", "accommodates"]
    required_resid_cols = ["listing_id", "residual_OLS", "residual_SAR"]
    required_points_cols = ["listing_id", "geometry"]
    required_grid_cols = ["geometry"]

    missing_df = [c for c in required_df_cols if c not in df.columns]
    missing_resid = [c for c in required_resid_cols if c not in resid.columns]
    missing_points = [c for c in required_points_cols if c not in points.columns]
    missing_grid = [c for c in required_grid_cols if c not in grid.columns]

    if missing_df:
        raise ValueError(f"Missing columns in model_sample.parquet: {missing_df}")
    if missing_resid:
        raise ValueError(f"Missing columns in residuals_for_map.csv: {missing_resid}")
    if missing_points:
        raise ValueError(f"Missing columns in map_points_sample.geojson: {missing_points}")
    if missing_grid:
        raise ValueError(f"Missing columns in map_grid_cells.geojson: {missing_grid}")

    if str(points.crs) != "EPSG:4326":
        st.warning(f"Points CRS is {points.crs}; EPSG:4326 is expected for web mapping.")
    if str(grid.crs) != "EPSG:4326":
        st.warning(f"Grid CRS is {grid.crs}; EPSG:4326 is expected for web mapping.")

    if points.geometry.isna().any() or (not points.geometry.is_valid.all()):
        st.warning("Points layer contains null/invalid geometries. Invalid rows may be skipped.")
    if grid.geometry.isna().any() or (not grid.geometry.is_valid.all()):
        st.warning("Grid layer contains null/invalid geometries. Invalid rows may be skipped.")

    return df, resid, points, grid

required_paths = get_required_paths()
initial_missing = get_missing_paths(required_paths)
if initial_missing:
    st.warning("Missing derived webmap files detected. Attempting automatic regeneration.")
    ok, bootstrap_message = bootstrap_missing_webmap_inputs(required_paths)
    if ok:
        st.info(bootstrap_message)
    else:
        st.error(
            "Error loading data: automatic regeneration did not complete successfully.\n"
            f"Project root in use: {PROJECT_ROOT}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Details:\n{bootstrap_message}"
        )
        st.stop()

try:
    df_full, resid_df, gdf_points, gdf_grid = load_data()
    st.info("Data loaded successfully.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


def first_existing_col(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.markdown("## Filters & Options")

# Price range slider
price_min, price_max = st.sidebar.slider(
    "Price range (€/night)",
    int(df_full['price_numeric'].min()),
    int(df_full['price_numeric'].max()),
    (50, 200),
    step=10
)

# Room type selector (from one-hot encoded columns)
room_cols = [c for c in df_full.columns if c.startswith('room_type_')]
room_labels = [c.replace('room_type_', '') for c in room_cols]
selected_rooms = st.sidebar.multiselect(
    "Room type",
    room_labels,
    default=room_labels[:2] if room_labels else []
)

# Accommodates slider
accommodates_range = st.sidebar.slider(
    "Accommodates (guests)",
    int(df_full['accommodates'].min()),
    int(df_full['accommodates'].max()),
    (1, 4),
    step=1
)

# Model comparison toggle
show_model = st.sidebar.radio(
    "Show residuals from:",
    ("OLS (baseline)", "SAR (spatial lag)", "Comparison (OLS - SAR)"),
    key="model_choice"
)

# Residual threshold
residual_threshold = st.sidebar.slider(
    "Highlight high absolute residuals (threshold)",
    0.0, 2.0,
    0.5,
    step=0.1
)

# Layer toggles
st.sidebar.markdown("## Layer toggles")
show_points = st.sidebar.checkbox("Show individual listings (5k sample)", value=True)
show_grid = st.sidebar.checkbox("Show neighborhood aggregates", value=True)

# ============================================================================
# FILTER DATA
# ============================================================================
# Apply room type filter
if selected_rooms:
    room_filter_cols = [f'room_type_{r}' for r in selected_rooms if f'room_type_{r}' in df_full.columns]
    if room_filter_cols:
        room_mask = pd.concat([df_full[[c]].astype(bool) for c in room_filter_cols], axis=1).any(axis=1)
    else:
        room_mask = pd.Series([True] * len(df_full), index=df_full.index)
else:
    room_mask = pd.Series([True] * len(df_full), index=df_full.index)

# Apply price filter
price_mask = (df_full['price_numeric'] >= price_min) & (df_full['price_numeric'] <= price_max)

# Apply accommodates filter
acc_mask = (df_full['accommodates'] >= accommodates_range[0]) & (df_full['accommodates'] <= accommodates_range[1])

# Combined filter
main_filter = room_mask & price_mask & acc_mask

df_filtered = df_full[main_filter].copy()
resid_filtered = resid_df[resid_df['listing_id'].isin(df_filtered['listing_id'])].copy()

st.sidebar.markdown(f"### Filtered results: {len(df_filtered)} / {len(df_full)} listings")

# ============================================================================
# CREATE MAP
# ============================================================================
# Map center (Madrid)
center_lat, center_lon = 40.4168, -3.7038

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles='OpenStreetMap'
)

# Color mapping for residuals
def get_residual_color(val):
    """Color residuals: negative (blue) to positive (red)"""
    if np.isnan(val):
        return '#999999'
    if val < -1:
        return '#08519c'  # dark blue
    if val < -0.5:
        return '#3182bd'  # medium blue
    if val < -0.1:
        return '#9ecae1'  # light blue
    if val < 0.1:
        return '#f7f7f7'  # neutral
    if val < 0.5:
        return '#fc9272'  # light orange
    if val < 1:
        return '#de2d26'  # medium red
    return '#a50f15'  # dark red

# ============================================================================
# LAYER 1: GRID CELLS (AGGREGATES)
# ============================================================================
if show_grid:
    grid_ols_col = first_existing_col(gdf_grid, ['residual_OLS_mean', 'residual_OLS', 'res_ols'])
    grid_sar_col = first_existing_col(gdf_grid, ['residual_SAR_mean', 'residual_SAR', 'res_sar'])

    if grid_ols_col is None or grid_sar_col is None:
        st.error(
            "Grid residual columns not found. Expected one of: "
            "(res_ols/res_sar) or (residual_OLS_mean/residual_SAR_mean). "
            "Regenerate map layers with scripts/07b_extract_residuals.py and scripts/08_prepare_map_layers.py."
        )
    else:
        if show_model == "OLS (baseline)":
            resid_col = grid_ols_col
            label_text = "OLS Residuals"
        elif show_model == "SAR (spatial lag)":
            resid_col = grid_sar_col
            label_text = "SAR Residuals"
        else:
            resid_col = 'res_diff'
            if 'res_diff' not in gdf_grid.columns:
                gdf_grid['res_diff'] = gdf_grid[grid_ols_col] - gdf_grid[grid_sar_col]
            label_text = "OLS - SAR (positive = OLS worse)"

        price_col = first_existing_col(gdf_grid, ['price_mean', 'mean_price', 'price_median'])
        count_col = first_existing_col(gdf_grid, ['count', 'n_listings'])
    
        for idx, row in gdf_grid.iterrows():
            residual_val = row.get(resid_col, np.nan)
            if pd.isna(residual_val):
                continue
            color = get_residual_color(residual_val)
            
            # Highlight high absolute residuals
            weight = 3 if abs(residual_val) > residual_threshold else 1
            opacity = 0.7 if abs(residual_val) > residual_threshold else 0.4
            
            price_val = row[price_col] if price_col is not None else np.nan
            count_val = row[count_col] if count_col is not None else np.nan
            
            popup_text = (
                f"<b>Grid Cell</b><br>"
                f"Mean Price: €{price_val:.0f}<br>" if pd.notna(price_val) else "<b>Grid Cell</b><br>"
            )
            if pd.notna(count_val):
                popup_text += f"Listings: {count_val:.0f}<br>"
            popup_text += f"{label_text}: {residual_val:.4f}<br>"
            if pd.notna(price_val):
                popup_text += f"Daily range: €{price_val:.0f}"
            
            folium.GeoJson(
                gpd.GeoSeries([row['geometry']]).__geo_interface__,
                style_function=lambda x, c=color, w=weight, op=opacity: {
                    'fillColor': c,
                    'color': c,
                    'weight': w,
                    'opacity': op,
                    'fillOpacity': 0.6
                },
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(m)

# ============================================================================
# LAYER 2: INDIVIDUAL POINTS
# ============================================================================
if show_points and len(resid_filtered) > 0:
    # Filter points to match filtered data
    gdf_pts_filtered = gdf_points[gdf_points['listing_id'].isin(resid_filtered['listing_id'])].copy()
    
    for idx, row in gdf_pts_filtered.iterrows():
        listing_id = row['listing_id']
        resid_row = resid_df[resid_df['listing_id'] == listing_id]
        
        if len(resid_row) == 0:
            continue
        
        if show_model == "OLS (baseline)":
            residual_val = resid_row['residual_OLS'].values[0]
            model_label = "OLS"
        elif show_model == "SAR (spatial lag)":
            residual_val = resid_row['residual_SAR'].values[0]
            model_label = "SAR"
        else:
            residual_ols = resid_row['residual_OLS'].values[0]
            residual_sar = resid_row['residual_SAR'].values[0]
            residual_val = residual_ols - residual_sar
            model_label = "OLS - SAR"
        
        color = get_residual_color(residual_val)
        
        # Only show points with high residuals if threshold > 0
        if abs(residual_val) < residual_threshold:
            opacity = 0.3
            radius = 3
        else:
            opacity = 0.8
            radius = 6
        
        popup_text = (
            f"<b>Listing {listing_id}</b><br>"
            f"Price: €{row['price_numeric']:.0f}/night<br>"
            f"log(Price): {row['log_price']:.2f}<br>"
            f"Accommodates: {row['accommodates']:.0f}<br>"
            f"{model_label} residual: {residual_val:.4f}<br>"
            f"Rating: {row['review_scores_rating']:.1f}"
        )
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=radius,
            popup=folium.Popup(popup_text, max_width=250),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=opacity,
            weight=1
        ).add_to(m)

# ============================================================================
# MAP DISPLAY
# ============================================================================
st.markdown("### Map")
st_folium(m, width=1400, height=600)

# ============================================================================
# INTERPRETATION
# ============================================================================
st.markdown("---")
st.markdown("## Interpretation Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Color legend")
    st.write("""
    - **Blue**: Negative residuals (model overestimates price)
    - **Gray**: Near-zero residuals (good fit)
    - **Red**: Positive residuals (model underestimates price)
    """)

with col2:
    st.markdown("### What you see")
    st.write(f"""
    Currently showing: **{show_model}**
    - {len(df_filtered)} listings match filters
    - {len(gdf_grid)} neighborhood cells
    - Highlighted: |residual| > {residual_threshold}
    """)

with col3:
    st.markdown("### Diagnostic scope")
    st.write("""
    The web map supports inspection of residual patterns and filtering.
    It is intended for exploratory diagnostics, not as standalone model evidence.
    Quantitative conclusions should be taken from saved tables and report outputs.
    """)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
st.markdown("---")
st.markdown("## Summary Statistics (Current Filter)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Price Statistics")
    if len(df_filtered) > 0:
        if 'log_price' in df_filtered.columns:
            st.metric("Mean Log Price", f"{df_filtered['log_price'].mean():.2f}")
        st.metric("Price Range", f"€{df_filtered['price_numeric'].min():.0f} - €{df_filtered['price_numeric'].max():.0f}")

with col2:
    st.markdown("### Residual Statistics")
    if len(resid_filtered) > 0:
        if show_model == "OLS (baseline)":
            col_use = 'residual_OLS'
        elif show_model == "SAR (spatial lag)":
            col_use = 'residual_SAR'
        else:
            resid_filtered = resid_filtered.copy()
            resid_filtered['residual_DIFF'] = resid_filtered['residual_OLS'] - resid_filtered['residual_SAR']
            col_use = 'residual_DIFF'
        
        st.metric("Mean residual", f"{resid_filtered[col_use].mean():.4f}")
        st.metric("Std residual", f"{resid_filtered[col_use].std():.4f}")
        st.metric("High |residual| (>0.5)", f"{(abs(resid_filtered[col_use]) > 0.5).sum()}")
