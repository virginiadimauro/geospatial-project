#!/usr/bin/env python
"""
Interactive Map: Madrid Airbnb Price Analysis
==============================================

Streamlit app for validating spatial econometric model conclusions.
Visualizes OLS vs SAR residuals and model fit quality across Madrid.
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import numpy as np
from pathlib import Path

# ============================================================================
# SETUP
# ============================================================================
st.set_page_config(page_title="Madrid Airbnb - Spatial Models", layout="wide")
st.markdown("# Interactive Map: Madrid Airbnb Spatial Analysis")
st.markdown("**Validating SAR vs OLS model conclusions through spatial visualization**")

PROJECT_ROOT = Path(__file__).parent.parent  # webmap/ -> project root
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PROCESSED / "model_sample.parquet")
    resid = pd.read_csv(OUTPUT_TABLES / "residuals_for_map.csv")
    points = gpd.read_file(DATA_PROCESSED / "map_points_sample.geojson")
    grid = gpd.read_file(DATA_PROCESSED / "map_grid_cells.geojson")
    return df, resid, points, grid

try:
    df_full, resid_df, gdf_points, gdf_grid = load_data()
    st.success("✓ Data loaded successfully")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

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
    default=room_labels[:2]
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
room_mask = pd.concat([df_full[[c]].astype(bool) for c in [f'room_type_{r}' for r in selected_rooms]], axis=1).any(axis=1) \
    if selected_rooms else pd.Series([True] * len(df_full))

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
    if show_model == "OLS (baseline)":
        resid_col = 'res_ols'
        label_text = "OLS Residuals"
    elif show_model == "SAR (spatial lag)":
        resid_col = 'res_sar'
        label_text = "SAR Residuals"
    else:
        resid_col = 'res_diff'
        if 'res_diff' not in gdf_grid.columns:
            gdf_grid['res_diff'] = gdf_grid['res_ols'] - gdf_grid['res_sar']
        label_text = "OLS - SAR (positive = OLS worse)"
    
    for idx, row in gdf_grid.iterrows():
        residual_val = row[resid_col]
        color = get_residual_color(residual_val)
        
        # Highlight high absolute residuals
        weight = 3 if abs(residual_val) > residual_threshold else 1
        opacity = 0.7 if abs(residual_val) > residual_threshold else 0.4
        
        popup_text = (
            f"<b>Grid Cell</b><br>"
            f"Mean Price: €{row['price_mean']:.0f}<br>"
            f"Listings: {row['count']:.0f}<br>"
            f"{label_text}: {residual_val:.4f}<br>"
            f"Daily range: €{row['price_mean']:.0f}"
        )
        
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
    - Highlighted: |residual| > ${residual_threshold}
    """)

with col3:
    st.markdown("### Key findings")
    st.write("""
    **SAR Model Advantage:**
    - ✓ Reduces residual autocorrelation (0.165 → 0.071)
    - ✓ Better spatial fit in clustered areas
    - ✓ Captures neighborhood spillovers
    
    **Use this map to:**
    - Find persistent problem areas
    - Validate clustering patterns
    - Identify structural omissions
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
        stats_price = df_filtered['log_price'].describe()
        st.metric("Mean Log Price", f"{df_filtered['log_price'].mean():.2f}")
        st.metric("Price Range", f"€{df_filtered['price_numeric'].min():.0f} - €{df_filtered['price_numeric'].max():.0f}")

with col2:
    st.markdown("### Residual Statistics")
    if len(resid_filtered) > 0:
        if show_model == "OLS (baseline)":
            col_use = 'residual_OLS'
        else:
            col_use = 'residual_SAR'
        
        st.metric("Mean residual", f"{resid_filtered[col_use].mean():.4f}")
        st.metric("Std residual", f"{resid_filtered[col_use].std():.4f}")
        st.metric("High |residual| (>0.5)", f"{(abs(resid_filtered[col_use]) > 0.5).sum()}")
