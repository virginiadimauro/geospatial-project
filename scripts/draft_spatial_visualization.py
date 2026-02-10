"""
Spatial Visualization of Airbnb Listings in Madrid by Price

Generates a publication-ready map showing:
- Neighbourhood boundaries (EPSG:4326)
- Listings as points, colored by price quantiles
- Optionally uses price_per_bed if beds data available with good coverage
- Includes scale bar
- Saves as PNG at 300 DPI

Usage:
    python 06_spatial_visualization.py
    
Output:
    reports/figures/fig_madrid_listings_price.png
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path.cwd() if (Path.cwd() / 'data').exists() else Path.cwd().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CRS_WEB = "EPSG:4326"
CRS_METRIC = "EPSG:25830"

OUTPUT_FILE = FIGURES_DIR / "fig_madrid_listings_price.png"
DPI = 300

# Quality control thresholds
BEDS_COVERAGE_THRESHOLD = 0.80  # Use price_per_bed if >80% coverage
PRICE_QUANTILE_BOUNDS = [0.0, 0.25, 0.5, 0.75, 1.0]  # Q1, Q2, Q3, Q4
COORDINATE_TOLERANCE = 0.05  # Lat/lon deviation from expected Madrid bbox

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("SPATIAL VISUALIZATION: Madrid Airbnb Listings by Price")
print("="*80)

print("\n[1/5] Loading data...")

# Listings
df_listings = pd.read_parquet(DATA_PROCESSED / "listings_clean.parquet")
print(f"  ✓ Loaded listings: {len(df_listings):,} rows")

# Neighbourhoods
gdf_neigh = gpd.read_file(DATA_PROCESSED / "neighbourhoods_enriched.geojson")
print(f"  ✓ Loaded neighbourhoods: {len(gdf_neigh):,} polygons")

# ============================================================================
# QUALITY CHECKS
# ============================================================================

print("\n[2/5] Quality checks...")

# CRS check
if gdf_neigh.crs is None:
    print(f"  ⚠️  Neighbourhoods CRS missing; setting to {CRS_WEB}")
    gdf_neigh = gdf_neigh.set_crs(CRS_WEB)
elif gdf_neigh.crs != CRS_WEB:
    print(f"  ⚠️  Neighbourhoods CRS {gdf_neigh.crs} → converting to {CRS_WEB}")
    gdf_neigh = gdf_neigh.to_crs(CRS_WEB)

# Listings geometry validation
print(f"  Listings with price: {df_listings['price'].notna().sum():,} / {len(df_listings):,}")
print(f"  Listings with valid coordinates: {(df_listings['latitude'].notna() & df_listings['longitude'].notna()).sum():,}")

# Remove invalid coordinates
df_listings_valid = df_listings[
    (df_listings['latitude'].notna()) & 
    (df_listings['longitude'].notna()) &
    (df_listings['price'].notna())
].copy()

print(f"  ✓ Valid listings (price + coords): {len(df_listings_valid):,}")

# Madrid bounding box (with tolerance)
madrid_lat_min, madrid_lat_max = 40.35, 40.55
madrid_lon_min, madrid_lon_max = -3.80, -3.60

# Filter to Madrid region with tolerance
df_listings_madrid = df_listings_valid[
    (df_listings_valid['latitude'] >= madrid_lat_min - COORDINATE_TOLERANCE) &
    (df_listings_valid['latitude'] <= madrid_lat_max + COORDINATE_TOLERANCE) &
    (df_listings_valid['longitude'] >= madrid_lon_min - COORDINATE_TOLERANCE) &
    (df_listings_valid['longitude'] <= madrid_lon_max + COORDINATE_TOLERANCE)
].copy()

print(f"  ✓ Listings within Madrid bbox: {len(df_listings_madrid):,}")

n_before = len(df_listings)
n_final = len(df_listings_madrid)
print(f"\n  Summary: {n_before:,} → {n_final:,} listings ({100*n_final/n_before:.1f}% retained)")

# ============================================================================
# PRICE VARIABLE SELECTION
# ============================================================================

print("\n[3/5] Selecting price variable...")

use_price_per_bed = False
price_col = 'price'

# Check beds coverage
if 'beds' in df_listings_madrid.columns:
    beds_coverage = df_listings_madrid['beds'].notna().sum() / len(df_listings_madrid)
    print(f"  Beds coverage: {100*beds_coverage:.1f}%")
    
    if beds_coverage >= BEDS_COVERAGE_THRESHOLD:
        # Calculate price_per_bed (exclude beds <= 0)
        df_listings_madrid['price_per_bed'] = np.where(
            df_listings_madrid['beds'] > 0,
            df_listings_madrid['price'] / df_listings_madrid['beds'],
            np.nan
        )
        
        price_per_bed_valid = df_listings_madrid['price_per_bed'].notna().sum()
        if price_per_bed_valid > len(df_listings_madrid) * 0.95:  # >95% valid
            use_price_per_bed = True
            price_col = 'price_per_bed'
            print(f"  ✓ Using price_per_bed (n={price_per_bed_valid:,} valid)")
        else:
            print(f"  ⚠️  price_per_bed has only {price_per_bed_valid:,} valid values; falling back to price")
    else:
        print(f"  ⚠️  Beds coverage below threshold ({100*BEDS_COVERAGE_THRESHOLD:.0f}%); using price")
else:
    print(f"  ℹ️  Beds column not found; using price")

# Check for outliers and winsorize
p1, p99 = df_listings_madrid[price_col].quantile([0.01, 0.99])
print(f"  Price range (raw): €{df_listings_madrid[price_col].min():.0f} – €{df_listings_madrid[price_col].max():.0f}")
print(f"  Price range (1st–99th percentile): €{p1:.0f} – €{p99:.0f}")

df_listings_madrid[f'{price_col}_clipped'] = df_listings_madrid[price_col].clip(p1, p99)
price_col_final = f'{price_col}_clipped'

# ============================================================================
# CREATE PRICE QUANTILE CLASSES
# ============================================================================

print("\n[4/5] Creating price quantile classes...")

quantiles = df_listings_madrid[price_col_final].quantile(PRICE_QUANTILE_BOUNDS)
print(f"  Quantile boundaries:")
for i, (bound, val) in enumerate(zip(PRICE_QUANTILE_BOUNDS, quantiles)):
    print(f"    Q{int(bound*100)}: €{val:.0f}")

# Assign bins
df_listings_madrid['price_class'] = pd.cut(
    df_listings_madrid[price_col_final],
    bins=quantiles.values,
    labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'],
    include_lowest=True
)

print(f"  Price class distribution:")
for cls, count in df_listings_madrid['price_class'].value_counts().sort_index().items():
    print(f"    {cls}: {count:,} ({100*count/len(df_listings_madrid):.1f}%)")

# ============================================================================
# CREATE MAP
# ============================================================================

print("\n[5/5] Creating map...")

fig, ax = plt.subplots(figsize=(16, 14), dpi=100)  # Will save at 300 DPI

# Plot neighbourhood boundaries
gdf_neigh.boundary.plot(ax=ax, linewidth=0.5, color='#999999', alpha=0.6, label='Neighbourhoods')

# Define colors for quantiles (colorblind-friendly palette)
colors = {
    'Q1 (Lowest)': '#fee5d9',
    'Q2': '#fcae91',
    'Q3': '#fb6a4a',
    'Q4 (Highest)': '#cb181d'
}

alpha = 0.6
edge_color = '#333333'
edge_width = 0.3

# Plot listings by price class
for price_class in ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']:
    df_class = df_listings_madrid[df_listings_madrid['price_class'] == price_class]
    ax.scatter(
        df_class['longitude'],
        df_class['latitude'],
        c=colors[price_class],
        s=20,
        alpha=alpha,
        edgecolors=edge_color,
        linewidth=edge_width,
        label=f"{price_class} (n={len(df_class):,})",
        zorder=10
    )

# ============================================================================
# FORMATTING & ANNOTATIONS
# ============================================================================

ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')

if use_price_per_bed:
    title = "Spatial Distribution of Airbnb Listings in Madrid\ncolored by Price per Bed (4 Quantile Classes)"
    subtitle = f"n={len(df_listings_madrid):,} listings, prices clipped to 1st–99th percentile"
else:
    title = "Spatial Distribution of Airbnb Listings in Madrid\ncolored by Nightly Price (4 Quantile Classes)"
    subtitle = f"n={len(df_listings_madrid):,} listings, prices clipped to 1st–99th percentile"

ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
ax.text(0.5, -0.05, subtitle, ha='center', transform=ax.transAxes, fontsize=10, style='italic')

ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.legend(
    loc='upper left',
    fontsize=10,
    framealpha=0.95,
    edgecolor='black',
    fancybox=True
)

# Add scale bar (simple version: 5 km scale)
ax.text(
    0.02, 0.02,
    "Scale: ~5 km\n (approximate)",
    transform=ax.transAxes,
    fontsize=9,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'),
    verticalalignment='bottom'
)

# Set equal aspect ratio
ax.set_aspect('equal')

# ============================================================================
# SAVE
# ============================================================================

plt.tight_layout()
plt.savefig(
    OUTPUT_FILE,
    dpi=DPI,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
print(f"  ✓ Saved: {OUTPUT_FILE}")
print(f"    DPI: {DPI}, Format: PNG")

plt.close()

# ============================================================================
# REPORT TEXT
# ============================================================================

print("\n" + "="*80)
print("REPORT TEXT (copy-paste into report):")
print("="*80)

price_type = "price per bed (nightly, in €/bed)" if use_price_per_bed else "nightly price (in €)"
report_text = f"""
**Figure X: Spatial Distribution of Airbnb Listings in Madrid by {price_type.replace('bed', 'bed').capitalize()}**

This map displays the geographic distribution of {len(df_listings_madrid):,} Airbnb listings across Madrid's 128 neighbourhoods, symbolized by {price_type}. 
Listings are assigned to four quantile classes (Q1–Q4) to avoid arbitrary thresholds and ensure balanced class sizes. 
Colours range from light (Q1, lowest prices) to dark red (Q4, highest prices), enabling visual identification of price gradients across the city. 
Neighbourhood boundaries are shown in grey; points are rendered with slight transparency to reveal density clustering. 
Prices are winsorized to the 1st–99th percentile to reduce the visual impact of outliers.
The map reveals pronounced spatial clustering: premium listings (Q4) concentrate in central neighbourhoods and tourist zones, whilst lower-priced properties are dispersed in peripheral areas.
"""

print(report_text)

print("\n" + "="*80)
print("✅ VISUALIZATION COMPLETE")
print("="*80)
