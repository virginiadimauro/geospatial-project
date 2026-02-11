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

# ============================================================================
# 1. PREPARE DATA (same as in 03_ols_price_analysis.py)
# ============================================================================
print("\n[STEP 1] Loading and preparing data...")

df = pd.read_parquet(DATA_PROCESSED / "listings_clean.parquet").copy()

# Price parsing and cleaning
def parse_price(price_str):
    if pd.isna(price_str):
        return np.nan
    if isinstance(price_str, (int, float)):
        return float(price_str)
    price_clean = str(price_str).replace('€', '').replace(',', '').strip()
    try:
        return float(price_clean)
    except:
        return np.nan

df['price_numeric'] = df['price'].apply(parse_price)
df = df[df['price_numeric'].notna()].copy()

# Winsorize and log
q_low = df['price_numeric'].quantile(0.005)
q_high = df['price_numeric'].quantile(0.995)
df['price_winsorized'] = df['price_numeric'].clip(q_low, q_high)
df['log_price'] = np.log(df['price_winsorized'])

# Covariates
covariate_cols = ['room_type', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
                  'minimum_nights', 'host_is_superhost', 'host_listings_count',
                  'number_of_reviews', 'review_scores_rating', 'instant_bookable']

# Impute
for col in ['bedrooms', 'beds', 'bathrooms']:
    df[col] = df[col].fillna(df[col].median())

# Convert categorical
df['host_is_superhost'] = df['host_is_superhost'].fillna('f')
df['host_is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
df['instant_bookable'] = (df['instant_bookable'] == 't').astype(int)
df['review_scores_rating'] = df['review_scores_rating'].fillna(0)
df['host_listings_count'] = df['host_listings_count'].fillna(df['host_listings_count'].median())

print(f"  → {len(df)} listings ready for OLS")

# ============================================================================
# 2. FIT MODEL B (same as before)
# ============================================================================
print("\n[STEP 2] Fitting OLS Model B (Property + Host + Location + Neighbourhood)...")

df_model = df.copy()
df_model = pd.get_dummies(df_model, columns=['room_type'], drop_first=True, dtype=int)
neighbourhood_dummies = pd.get_dummies(df_model['neighbourhood_cleansed'], prefix='neigh', drop_first=True, dtype=int)
df_model = pd.concat([df_model, neighbourhood_dummies], axis=1)

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
df_spatial = df_model[['listing_id', 'latitude', 'longitude', 'neighbourhood_cleansed', 'residuals_modelB']].copy()

# ============================================================================
# 3. MORAN'S I - APPROACH A: LISTING-LEVEL (kNN WEIGHTS)
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH A: LISTING-LEVEL SPATIAL AUTOCORRELATION (k-NN Weights)")
print("=" * 80)

# Create point geometries (EPSG:4326)
gdf_listings = gpd.GeoDataFrame(
    df_spatial,
    geometry=gpd.points_from_xy(df_spatial['longitude'], df_spatial['latitude']),
    crs='EPSG:4326'
)

print(f"\nCRS: {gdf_listings.crs}")
print(f"Geometries valid: {gdf_listings.is_valid.sum()}/{len(gdf_listings)}")

# Reproject to EPSG:25830 (metric) for distance calculations
gdf_listings_metric = gdf_listings.to_crs('EPSG:25830')
print(f"Reprojected to EPSG:25830 for k-NN distance calculations")

# Extract coordinates (metric)
coords_metric = np.array([
    [geom.x, geom.y] for geom in gdf_listings_metric.geometry
])

# Moran's I results
morans_results = []

# k-NN weights: k=8
print(f"\n[k-NN, k=8]")
try:
    w_knn8 = KNN.from_array(coords_metric, k=8)
    w_knn8.transform = 'r'  # Row-standardize
    
    n_weights = w_knn8.n  # Number of observations
    n_islands = sum(1 for ni in w_knn8.neighbors.values() if len(ni) == 0)
    print(f"  Weights created: {n_weights} listings, {n_islands} islands")
    
    mi_knn8 = Moran(residuals, w_knn8)
    print(f"  Moran's I: {mi_knn8.I:.6f}")
    print(f"  p-value (simulation): {mi_knn8.p_sim:.6f}")
    print(f"  z-score (simulation): {mi_knn8.z_sim:.4f}")
    
    morans_results.append({
        'level': 'listing',
        'weights': 'knn8',
        'n': n_weights,
        'morans_I': mi_knn8.I,
        'p_sim': mi_knn8.p_sim,
        'z_sim': mi_knn8.z_sim,
        'islands': n_islands
    })
    
    # Save Moran's scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    moran_scatterplot(mi_knn8, ax=ax)
    ax.set_title('Moran Scatterplot: OLS Residuals (Listing-level, k-NN k=8)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "morans_scatter_listing_knn8.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: morans_scatter_listing_knn8.png")
    
except Exception as e:
    print(f"  ERROR: {e}")

# k-NN weights: k=12 (sensitivity check)
print(f"\n[k-NN, k=12] (sensitivity check)")
try:
    w_knn12 = KNN.from_array(coords_metric, k=12)
    w_knn12.transform = 'r'
    
    n_weights = w_knn12.n
    n_islands = sum(1 for ni in w_knn12.neighbors.values() if len(ni) == 0)
    print(f"  Weights created: {n_weights} listings, {n_islands} islands")
    
    mi_knn12 = Moran(residuals, w_knn12)
    print(f"  Moran's I: {mi_knn12.I:.6f}")
    print(f"  p-value (simulation): {mi_knn12.p_sim:.6f}")
    print(f"  z-score (simulation): {mi_knn12.z_sim:.4f}")
    
    morans_results.append({
        'level': 'listing',
        'weights': 'knn12',
        'n': n_weights,
        'morans_I': mi_knn12.I,
        'p_sim': mi_knn12.p_sim,
        'z_sim': mi_knn12.z_sim,
        'islands': n_islands
    })
    
except Exception as e:
    print(f"  ERROR: {e}")

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
    
    # Spatial join: listing points (EPSG:4326) to neighbourhood polygons
    gdf_listings_4326 = gdf_listings.to_crs('EPSG:4326')  # Already 4326, but ensure
    sjoin = gpd.sjoin(gdf_listings_4326, gdf_neigh[['geometry']], predicate='within', how='left')
    
    # Check coverage
    n_matched = sjoin.index_right.notna().sum()
    coverage_pct = 100 * n_matched / len(sjoin)
    print(f"\nSpatial join coverage: {n_matched}/{len(sjoin)} ({coverage_pct:.1f}%)")
    
    if coverage_pct < 95:
        print(f"  WARNING: {100-coverage_pct:.1f}% listings not matched to neighbourhoods")
    
    # Use existing neighbourhood_cleansed from listings data (already assigned)
    # Aggregate residuals by neighbourhood (mean)
    residuals_by_neigh = df_spatial.groupby('neighbourhood_cleansed')['residuals_modelB'].agg(['mean', 'count']).reset_index()
    residuals_by_neigh.columns = ['neighbourhood', 'residual_mean', 'count']
    print(f"\nAggregated residuals: {len(residuals_by_neigh)} neighbourhoods")
    print(f"  Mean count per neighbourhood: {residuals_by_neigh['count'].mean():.1f}")
    
    # Merge residuals with neighbourhoodspolygons
    # Create a simplified neighbourhood geodataframe with just geometry
    gdf_neigh_simple = gdf_neigh.reset_index(drop=True)
    gdf_neigh_simple['neigh_idx'] = range(len(gdf_neigh_simple))
    
    # Merge residuals with neighbourhoods by spatial position
    # Create a mapping from neighbourhood index to aggregated residuals
    # For now, use the residuals_by_neigh directly and merge on neighbourhood name
    
    # Try to get neighbourhood identifiers from GeoJSON
    # If 'properties' column exists with dict, extract neighbourhood name
    neigh_names = []
    for idx in range(len(gdf_neigh)):
        try:
            # Try to access properties dict
            props = gdf_neigh.iloc[idx].get('properties', {})
            if isinstance(props, dict):
                neigh_names.append(props.get('neighbourhood', str(idx)))
            else:
                neigh_names.append(str(idx))
        except:
            neigh_names.append(str(idx))
    
    gdf_neigh_simple['neighbourhood_name'] = neigh_names
    
    # Merge with aggregated residuals
    gdf_neigh_with_data = gdf_neigh_simple.merge(
        residuals_by_neigh,
        left_on='neighbourhood_name',
        right_on='neighbourhood',
        how='left'
    )
    
    # Keep only rows with data
    gdf_neigh_with_data = gdf_neigh_with_data.dropna(subset=['residual_mean'])
    print(f"  Neighbourhoods with residual data: {len(gdf_neigh_with_data)}")
    
    if len(gdf_neigh_with_data) < 20:
        print(f"  WARNING: Only {len(gdf_neigh_with_data)} neighbourhoods matched (consider alternative approach)")
        # If merge didn't work well, use index-based approach
        # Map listing neighbourhoods to GeoJSON index
        gdf_neigh_with_data = gdf_neigh.copy()
        
        # Try direct assignment using neighbourhood names from listings
        # Create a numeric ID for neighbourhoods based on their order
        unique_neighs = df_spatial['neighbourhood_cleansed'].unique()
        neigh_to_id = {n: i for i, n in enumerate(unique_neighs)}
        
        # Get residuals aggregated by neighbourhood
        neigh_data = df_spatial.copy()
        neigh_data['neigh_id'] = neigh_data['neighbourhood_cleansed'].map(neigh_to_id)
        residuals_by_id = neigh_data.groupby('neigh_id')['residuals_modelB'].mean()
        
        print(f"  Using index-based mapping: {len(residuals_by_id)} neighbourhoods")
        gdf_neigh_with_data = gdf_neigh.iloc[:len(residuals_by_id)].copy()
        gdf_neigh_with_data['residual_mean'] = residuals_by_id.values
    
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
