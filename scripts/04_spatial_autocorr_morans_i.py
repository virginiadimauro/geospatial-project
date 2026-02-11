#!/usr/bin/env python
"""
Spatial Autocorrelation Analysis: Global Moran's I on OLS Model B Residuals

Two approaches:
(A) Listing-level: k-NN weights (k=8, k=12), coordinates in EPSG:25830
(B) Neighbourhood-level: Queen contiguity, aggregated residuals

Output:
- outputs/tables/morans_results.csv
- outputs/figures/morans_scatter_*.png
- outputs/figures/lisa_clusters_neighbourhood_queen.png (optional)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import warnings
import sys

# ensure repo root on path for `src` imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Spatial weights and autocorrelation
from libpysal.weights import KNN, Queen
from esda import Moran
from splot.esda import moran_scatterplot, lisa_cluster
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Stats
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from src.prep import load_listings_processed, build_model_df, get_y_X

# ============================================================================
# 0. SETUP
# ============================================================================
PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_ORIGINAL = PROJECT_ROOT / "data"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SPATIAL AUTOCORRELATION ANALYSIS: MORAN'S I (OLS Model B Residuals)")
print("=" * 80)

# NOTE: uses model_sample.parquet for consistent N (centralized data-prep)

# ============================================================================
# 1. PREPARE DATA (same as in 03_ols_price_analysis.py)
# ============================================================================
print("\n[STEP 1] Loading and preparing data... (centralized)")

# Prefer the saved model sample for reproducibility
model_sample_path = DATA_PROCESSED / "model_sample.parquet"
if model_sample_path.exists():
    model_df = pd.read_parquet(model_sample_path)
    print(f"  → Loaded model_sample.parquet with {len(model_df)} listings")
else:
    df_raw = load_listings_processed(DATA_PROCESSED / "listings_clean.parquet")
    model_df = build_model_df(df_raw, out_path=model_sample_path)
    print(f"  → Built and saved model_sample.parquet with {len(model_df)} listings")

df = model_df.reset_index(drop=True).copy()

# ============================================================================
# 2. FIT MODEL B (same as before)
# ============================================================================
print("\n[STEP 2] Fitting OLS Model B (Property + Host + Location + Neighbourhood)...")

# model_df already contains dummies and log_price
df_model = df.copy()

# Distance to CBD
PUERTA_DEL_SOL_LAT, PUERTA_DEL_SOL_LON = 40.4169, -3.7035

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df_model['dist_cbd_km'] = haversine_distance(
    df_model['latitude'], df_model['longitude'],
    PUERTA_DEL_SOL_LAT, PUERTA_DEL_SOL_LON
)

# Model B variables
property_vars = ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'minimum_nights']
host_vars = ['host_is_superhost', 'host_listings_count', 'number_of_reviews',
             'review_scores_rating', 'instant_bookable']
room_type_vars = [col for col in df_model.columns if col.startswith('room_type_')]
neighbourhood_vars = [col for col in df_model.columns if col.startswith('neigh_')]

model_b_vars = property_vars + host_vars + room_type_vars + ['dist_cbd_km'] + neighbourhood_vars

y = df_model['log_price'].astype(float)
X_b = df_model[model_b_vars].astype(float)
X_b_const = add_constant(X_b)

model_b = sm.OLS(y, X_b_const).fit(cov_type='HC1')
residuals = model_b.resid.values

print(f"  → Model B fitted (R² = {model_b.rsquared:.4f})")
print(f"  → Residuals: mean={residuals.mean():.6f}, std={residuals.std():.6f}")

# Add to dataframe
df_model['residuals_modelB'] = residuals
# Keep only necessary spatial columns (neighbourhood_cleansed may be absent in centralized sample)
cols_spatial = [c for c in ['listing_id', 'latitude', 'longitude', 'residuals_modelB'] if c in df_model.columns]
df_spatial = df_model[cols_spatial].copy()

# ============================================================================
# 3. MORAN'S I - NEIGHBOURHOOD WORKFLOW (only neighbourhood-level kept)
# ============================================================================
print("\n" + "=" * 80)
print("NEIGHBOURHOOD-LEVEL SPATIAL AUTOCORRELATION (Queen Contiguity)")
print("=" * 80)

# Create point geometries (EPSG:4326) for spatial join to neighbourhoods
gdf_listings = gpd.GeoDataFrame(
    df_spatial,
    geometry=gpd.points_from_xy(df_spatial['longitude'], df_spatial['latitude']),
    crs='EPSG:4326'
)

morans_results = []

# ============================================================================
# 4. MORAN'S I - APPROACH B: NEIGHBOURHOOD-LEVEL (QUEEN CONTIGUITY)
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH B: NEIGHBOURHOOD-LEVEL SPATIAL AUTOCORRELATION (Queen Contiguity)")
print("=" * 80)

# Load neighbourhoods GeoJSON
try:
    gdf_neigh = gpd.read_file(DATA_PROCESSED / "neighbourhoods_enriched.geojson")
    print(f"\nNeighbourhoods loaded: {len(gdf_neigh)} polygons")
    print(f"CRS: {gdf_neigh.crs}")
    print(f"Geometries valid: {gdf_neigh.is_valid.sum()}/{len(gdf_neigh)}")
    
    # Fix invalid geometries if needed
    if gdf_neigh.is_valid.sum() < len(gdf_neigh):
        print("  Fixing invalid geometries...")
        gdf_neigh['geometry'] = gdf_neigh.geometry.buffer(0)
    
    # Ensure EPSG:4326
    if gdf_neigh.crs != 'EPSG:4326':
        gdf_neigh = gdf_neigh.to_crs('EPSG:4326')
        print(f"  Reprojected to EPSG:4326")
    
    # Spatial join: robust workflow using polygon index (neigh_idx) only
    gdf_neigh_reset = gdf_neigh.reset_index().rename(columns={'index': 'neigh_idx'})
    gdf_listings_4326 = gdf_listings.to_crs('EPSG:4326')  # ensure same CRS

    joined = gpd.sjoin(gdf_listings_4326, gdf_neigh_reset[['neigh_idx', 'geometry']], predicate='within', how='left')

    n_null = joined['neigh_idx'].isna().sum()
    n_total = len(joined)
    coverage_pct = 100 * (n_total - n_null) / n_total
    print(f"\nSpatial join coverage (by neigh_idx): {n_total - n_null}/{n_total} ({coverage_pct:.1f}%)")

    # If many listings unmatched, warn and save examples for debugging
    if coverage_pct < 95:
        print(f"  WARNING: High unmatched listings after sjoin: {n_null} listings ({100 - coverage_pct:.1f}% unmatched)")
        examples = joined[joined['neigh_idx'].isna()][['listing_id', 'latitude', 'longitude']].head(200)
        examples_path = OUTPUT_TABLES / 'neighbourhoods_unmatched_examples.csv'
        examples.to_csv(examples_path, index=False)
        print(f"  ✓ Saved examples of unmatched listings to: {examples_path}")

    # Aggregate residuals by neigh_idx (drop unmatched)
    residuals_by_idx = (
        joined.dropna(subset=['neigh_idx']).groupby('neigh_idx')['residuals_modelB']
        .agg(['mean', 'count']).reset_index()
    )
    residuals_by_idx.columns = ['neigh_idx', 'residual_mean', 'count']
    print(f"\nAggregated residuals by neigh_idx: {len(residuals_by_idx)} neighbourhoods")

    # Merge aggregated residuals with neighbourhood polygons by neigh_idx (integer index)
    gdf_neigh_with_data = gdf_neigh_reset.merge(residuals_by_idx, on='neigh_idx', how='inner')
    print(f"  Neighbourhoods with residual data: {len(gdf_neigh_with_data)}")
    
    # Queen contiguity weights
    print(f"\n[Queen Contiguity]")
    
    try:
        w_queen = Queen.from_dataframe(gdf_neigh_with_data)
        w_queen.transform = 'r'  # Row-standardize
        
        n_weights = w_queen.n
        n_islands = sum(1 for ni in w_queen.neighbors.values() if len(ni) == 0)
        print(f"  Weights created: {n_weights} neighbourhoods, {n_islands} islands")
        
        if n_islands > 0:
            print(f"  WARNING: {n_islands} islands detected (neighbourhoods with no contiguous neighbours)")
        
        # Moran's I on aggregated residuals
        residuals_neigh = gdf_neigh_with_data['residual_mean'].values
        mi_queen = Moran(residuals_neigh, w_queen)
        
        print(f"  Moran's I: {mi_queen.I:.6f}")
        print(f"  p-value (simulation): {mi_queen.p_sim:.6f}")
        print(f"  z-score (simulation): {mi_queen.z_sim:.4f}")
        
        morans_results.append({
            'level': 'neighbourhood',
            'weights': 'queen',
            'n': n_weights,
            'morans_I': mi_queen.I,
            'p_sim': mi_queen.p_sim,
            'z_sim': mi_queen.z_sim,
            'islands': n_islands
        })
        
        # Moran scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        moran_scatterplot(mi_queen, ax=ax)
        ax.set_title('Moran Scatterplot: OLS Residuals (Neighbourhood-level, Queen)', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_FIGURES / "morans_scatter_neighbourhood_queen.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: morans_scatter_neighbourhood_queen.png")
        
        # LISA cluster map (optional but nice)
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            lisa_cluster(mi_queen, gdf_neigh_with_data, ax=ax, legend=True)
            ax.set_title('LISA Clusters: OLS Residuals (Neighbourhood-level, Queen)', 
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(OUTPUT_FIGURES / "lisa_clusters_neighbourhood_queen.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: lisa_clusters_neighbourhood_queen.png")
        except Exception as e:
            print(f"  LISA cluster map failed: {e}")
        
    except Exception as e:
        print(f"  ERROR in Queen weights: {e}")

except Exception as e:
    print(f"\nERROR loading neighbourhoods: {e}")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results_df = pd.DataFrame(morans_results)
results_df = results_df[['level', 'weights', 'n', 'morans_I', 'p_sim', 'z_sim', 'islands']]

results_df.to_csv(OUTPUT_TABLES / "morans_results.csv", index=False)
print(f"\n✓ Saved: {OUTPUT_TABLES / 'morans_results.csv'}")
print("\nResults summary:")
print(results_df.to_string(index=False))

# ============================================================================
# 6. INTERPRETATION
# ============================================================================
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

for idx, row in results_df.iterrows():
    level = row['level']
    weights = row['weights']
    p_val = row['p_sim']
    mi_val = row['morans_I']
    
    print(f"\n[{level.upper()} - {weights.upper()}]")
    print(f"  Moran's I = {mi_val:.6f}, p-value = {p_val:.6f}")
    
    if p_val < 0.05:
        direction = "positive" if mi_val > 0 else "negative"
        print(f"  ✓ SIGNIFICANT {direction.upper()} spatial autocorrelation (p < 0.05)")
        print(f"    → OLS residuals show spatial clustering")
    else:
        print(f"  ✗ No significant spatial autocorrelation detected (p ≥ 0.05)")
        print(f"    → Residuals appear spatially random")

print("\n" + "=" * 80)
print("✓ MORAN'S I ANALYSIS COMPLETE")
print("=" * 80)
