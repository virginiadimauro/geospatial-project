#!/usr/bin/env python3
"""Generate a static PNG map (overview + inset) of Madrid listings colored by price quantiles.

Outputs:
 - reports/figures/fig_madrid_overview_inset_price.png (300 dpi)

Notes:
 - Reads `data/processed/listings_clean.parquet` and
   `data/processed/neighbourhoods_enriched.geojson` (or .parquet if geojson missing).
 - Uses EPSG:4326 for final plotting. For metric buffering reprojects to EPSG:25830.
 - Optionally winsorizes prices at 1st/99th percentiles for stable quantile computation (documented in stdout).
"""
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm


def detect_neighbourhood_column(gdf):
    # Try common names
    candidates = [
        'neighbourhood', 'neighbourhood_name', 'neighbourhoods',
        'neighbourhood_group', 'name', 'NOMBRE', 'nombre'
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    # fallback: try any string column
    for c in gdf.columns:
        if gdf[c].dtype == object:
            return c
    return None


def main():
    base = Path(__file__).resolve().parents[1]
    listings_path = base / 'data' / 'processed' / 'listings_clean.parquet'
    nbhd_geojson = base / 'data' / 'processed' / 'neighbourhoods_enriched.geojson'
    nbhd_parquet = base / 'data' / 'processed' / 'neighbourhoods_enriched.parquet'
    out_path = base / 'reports' / 'figures' / 'fig_madrid_overview_inset_price.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not listings_path.exists():
        print(f'ERROR: listings file not found: {listings_path}', file=sys.stderr)
        sys.exit(1)

    # Load listings
    df = pd.read_parquet(listings_path)
    # Expect at least latitude, longitude, price
    lat_col = 'latitude' if 'latitude' in df.columns else ('lat' if 'lat' in df.columns else None)
    lon_col = 'longitude' if 'longitude' in df.columns else ('lon' if 'lon' in df.columns else None)
    if lat_col is None or lon_col is None or 'price' not in df.columns:
        print('ERROR: listings file missing required columns (latitude, longitude, price)', file=sys.stderr)
        sys.exit(1)

    # Create GeoDataFrame for listings
    gdf_listings = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs='EPSG:4326'
    )

    # Load neighbourhoods (geojson preferred)
    if nbhd_geojson.exists():
        gdf_nb = gpd.read_file(nbhd_geojson)
    elif nbhd_parquet.exists():
        gdf_nb = pd.read_parquet(nbhd_parquet)
        if 'geometry' in gdf_nb.columns:
            gdf_nb = gpd.GeoDataFrame(gdf_nb, geometry='geometry', crs='EPSG:4326')
    else:
        print('WARNING: neighbourhoods file not found; proceeding without polygons.', file=sys.stderr)
        gdf_nb = None

    # Document neighbourhood column
    nbhd_col = None
    if gdf_nb is not None:
        nbhd_col = detect_neighbourhood_column(gdf_nb)
        print(f'Detected neighbourhood/name column: {nbhd_col}')

    # Prepare price variable
    prices = gdf_listings['price'].astype(float).copy()

    # Winsorize at 1st-99th percentiles for stability (only for quantile calc)
    lower, upper = np.nanpercentile(prices.dropna(), [1, 99])
    print(f'Price winsorize bounds (1% - 99%): {lower:.2f} € - {upper:.2f} €')
    prices_w = prices.clip(lower=lower, upper=upper)

    # Compute quantile edges
    q_edges = prices_w.quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    # Ensure monotonic increasing -- if duplicates reduce bins
    unique_edges = np.unique(q_edges)
    if len(unique_edges) - 1 < 4:
        # fallback: use qcut with drops handled
        try:
            cats = pd.qcut(prices_w, q=4, labels=False, duplicates='drop')
            # convert to 1..k
            cats = cats + 1
        except Exception:
            cats = pd.cut(prices_w, bins=4, labels=False) + 1
    else:
        cats = pd.cut(prices_w, bins=unique_edges, include_lowest=True, labels=False) + 1

    # Attach class and colors
    gdf_listings['price_w'] = prices_w
    gdf_listings['price_class'] = cats.astype('Int64')

    # Colors (Q1->blue, Q2->light green, Q3->orange, Q4->red)
    color_hex = ['#1F78B4', '#B2DF8A', '#FDBF6F', '#E31A1C']
    # If fewer classes due to ties, use first K colors
    n_classes = int(gdf_listings['price_class'].dropna().unique().size)
    cmap = ListedColormap(color_hex[:max(1, n_classes)])
    bounds = list(range(1, 1 + n_classes + 1))
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    # Map class -> color hex column for deterministic plotting
    class_to_color = {i + 1: color_hex[i] for i in range(min(4, n_classes))}
    gdf_listings['color_hex'] = gdf_listings['price_class'].map(class_to_color)

    # Prepare figure
    fig = plt.figure(figsize=(10, 12))
    ax_main = fig.add_axes([0.05, 0.05, 0.9, 0.9])

    # Plot neighbourhoods as boundaries
    if gdf_nb is not None:
        gdf_nb.to_crs(epsg=4326).plot(ax=ax_main, facecolor='none', edgecolor='0.6', linewidth=0.8)

    # Plot listings
    # For visibility, plot smaller markers with some alpha
    # Plot by class to keep deterministic color order in legend
    plotted_classes = []
    for cls in sorted(gdf_listings['price_class'].dropna().unique()):
        cls = int(cls)
        subset = gdf_listings[gdf_listings['price_class'] == cls]
        subset.plot(ax=ax_main, markersize=6, color=class_to_color.get(cls, '#333333'), alpha=0.8, linewidth=0)
        plotted_classes.append(cls)

    ax_main.set_title('Madrid — Listings price quantiles (Q1–Q4)')
    ax_main.set_xlabel('Longitude')
    ax_main.set_ylabel('Latitude')

    # Legend: intervals and counts
    # Compute intervals using quantiles on price_w
    quantiles = prices_w.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
    labels = []
    counts = []
    for i in range(len(quantiles) - 1):
        lo = quantiles[i]
        hi = quantiles[i + 1]
        mask = (gdf_listings['price_w'] >= lo) & (gdf_listings['price_w'] <= hi) if i == len(quantiles) - 2 else (gdf_listings['price_w'] >= lo) & (gdf_listings['price_w'] < hi)
        cnt = int(mask.sum())
        counts.append(cnt)
        labels.append(f'€{lo:,.0f}–€{hi:,.0f} (n={cnt})')

    # Create legend patches in same order (Q1->Q4)
    patches = []
    for i, cls in enumerate(range(1, len(labels) + 1)):
        color = class_to_color.get(cls, '#777777')
        patches.append(mpatches.Patch(color=color, label=labels[i]))

    ax_main.legend(handles=patches, title='Price quantiles', loc='lower left')

    # Inset: zoom on neighbourhood 'Centro' if exists, otherwise centroid + buffer
    if gdf_nb is not None:
        try:
            # reproject to metric for buffering
            nb_metric = gdf_nb.to_crs(epsg=25830)
            found = None
            if nbhd_col is not None:
                # search for 'centro' case-insensitive
                matches = nb_metric[nb_metric[nbhd_col].str.contains('centro', case=False, na=False)]
                if len(matches) > 0:
                    found = matches.iloc[0]
            if found is None:
                # fallback to overall centroid of all neighbourhood polygons
                cent = nb_metric.unary_union.centroid
            else:
                cent = found.geometry.centroid

            # buffer in meters
            buffer_m = 1600  # between 1500-2000 m as suggested
            bbox_metric = cent.buffer(buffer_m).bounds  # minx, miny, maxx, maxy
            # Make a small GeoDataFrame from bbox and reproject to 4326
            minx, miny, maxx, maxy = bbox_metric
            # Convert bbox corners back to 4326
            from shapely.geometry import box

            bbox_geom = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=nb_metric.crs).to_crs(epsg=4326)
            minx4326, miny4326, maxx4326, maxy4326 = bbox_geom.total_bounds

            ax_inset = fig.add_axes([0.55, 0.6, 0.38, 0.35])
            if gdf_nb is not None:
                gdf_nb.to_crs(epsg=4326).plot(ax=ax_inset, facecolor='none', edgecolor='0.6', linewidth=0.6)
            for cls in sorted(gdf_listings['price_class'].dropna().unique()):
                cls = int(cls)
                subset = gdf_listings[gdf_listings['price_class'] == cls]
                subset.plot(ax=ax_inset, markersize=10, color=class_to_color.get(cls, '#333333'), alpha=0.9, linewidth=0)

            ax_inset.set_xlim(minx4326, maxx4326)
            ax_inset.set_ylim(miny4326, maxy4326)
            ax_inset.set_title('Zoom: Centro (buffer 1.6 km)')
        except Exception as e:
            warnings.warn(f'Could not create inset due to: {e}')

    # Save figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f'Saved figure to: {out_path}')


if __name__ == '__main__':
    main()
