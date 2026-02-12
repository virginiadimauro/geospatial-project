#!/usr/bin/env python
"""
Prepare map layers for Streamlit interactive visualization
===========================================================

Creates:
1. Layer A: Sample of 5000 listings points with price, residuals, attributes
2. Layer B: Grid cells with aggregated metrics (price, residuals, count)

Outputs:
- data/processed/map_points_sample.geojson (5k listings)
- data/processed/map_grid_cells.geojson (grid cells with aggregates)

NOTE: H3 support (hexagons) requires 'h3' package; falls back to regular grid if unavailable.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Auto-detect project root
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import PROCESSED_DIR, OUTPUT_TABLES, RANDOM_SEED
    DATA_PROCESSED = PROCESSED_DIR
except (ImportError, ModuleNotFoundError):
    # Fallback: use path inference
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
    RANDOM_SEED = 42

# Try importing H3
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    print("⚠ h3 not available - will create quadrat grid instead\n")

# Ensure output directory exists
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PREPARING MAP LAYERS")
print("=" * 80)
print(f"\nProject root: {PROJECT_ROOT}")
print(f"Data dir: {DATA_PROCESSED}")

# ============================================================================
# LAYER A: SAMPLE POINTS
# ============================================================================
print("\n[LAYER A] Creating sample points...")

try:
    # Load model sample
    model_sample_path = DATA_PROCESSED / "model_sample.parquet"
    if not model_sample_path.exists():
        raise FileNotFoundError(f"Missing {model_sample_path}. Run Phase A (05_final_pipeline.ipynb) first.")
    
    df_model = pd.read_parquet(model_sample_path)
    print(f"  ✓ Loaded model_sample.parquet: {len(df_model)} listings")
    
    # Load residuals (if available, optional for points layer)
    residuals_path = OUTPUT_TABLES / "residuals_for_map.csv"
    residuals_available = residuals_path.exists()
    
    if residuals_available:
        residuals_df = pd.read_csv(residuals_path)
        print(f"  ✓ Loaded residuals: {len(residuals_df)} records")
        # Merge residuals with model sample
        df_points = df_model.merge(residuals_df[['listing_id', 'residual_OLS', 'residual_SAR', 'residual_SEM']], 
                                    on='listing_id', how='left')
    else:
        print("  ⚠ residuals_for_map.csv not found (optional)")
        df_points = df_model.copy()
    
    # Select core columns
    core_cols = ['listing_id', 'latitude', 'longitude', 'price_numeric']
    optional_cols = ['log_price', 'accommodates', 'bedrooms', 'bathrooms', 
                     'review_scores_rating', 'host_is_superhost', 'instant_bookable',
                     'residual_OLS', 'residual_SAR', 'residual_SEM']
    
    available_cols = core_cols + [c for c in optional_cols if c in df_points.columns]
    df_points = df_points[available_cols].copy()
    
    print(f"  → Full points available: {len(df_points)}")
    
    # Sample for performance
    sample_size = min(5000, len(df_points))
    df_sample = df_points.sample(n=sample_size, random_state=RANDOM_SEED)
    
    # Create GeoDataFrame
    gdf_sample = gpd.GeoDataFrame(
        df_sample,
        geometry=gpd.points_from_xy(df_sample['longitude'], df_sample['latitude']),
        crs='EPSG:4326'
    )
    
    # Save as GeoJSON
    sample_path = DATA_PROCESSED / "map_points_sample.geojson"
    gdf_sample.to_file(sample_path, driver='GeoJSON')
    
    print(f"  ✓ Saved: {sample_path}")
    print(f"    Sample size: {len(gdf_sample)}")
    print(f"    Geometry type: {gdf_sample.geometry.geom_type.iloc[0]}")
    
except Exception as e:
    print(f"  ✗ Error creating points layer: {e}")
    sys.exit(1)

# ============================================================================
# LAYER B: GRID CELLS
# ============================================================================

# Only attempt aggregation if residuals are available
if residuals_available and 'residual_OLS' in df_points.columns:
    if H3_AVAILABLE:
        print("\n[LAYER B] Creating H3 hexagon grid (not yet fully supported)...")
        print("  ⚠ H3 grid creation requires additional validation")
        print("  → Falling back to quadrat grid...")
    
    print("\n[LAYER B] Creating quadrat grid...")
    
    try:
        # Use full dataset for aggregation (not sampled)
        df_agg = df_points.copy()
        
        # Create regular grid cells (0.05° ≈ 5-6 km at equator)
        grid_size = 0.05
        df_agg['grid_x'] = (df_agg['longitude'] / grid_size).astype(int)
        df_agg['grid_y'] = (df_agg['latitude'] / grid_size).astype(int)
        
        # Aggregate by grid cell
        agg_dict = {
            'price_numeric': ['mean', 'median', 'std', 'count'],
            'log_price': 'mean' if 'log_price' in df_agg.columns else lambda x: np.nan,
            'residual_OLS': 'mean',
            'residual_SAR': 'mean',
            'residual_SEM': 'mean'
        }
        
        # Only include columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df_agg.columns}
        
        grid_agg = df_agg.groupby(['grid_x', 'grid_y']).agg(agg_dict).reset_index()
        
        # Flatten column names
        grid_agg.columns = ['grid_x', 'grid_y', 'price_mean', 'price_median', 'price_std', 'count',
                            'log_price_mean', 'residual_OLS_mean', 'residual_SAR_mean', 'residual_SEM_mean']
        
        # Add absolute residuals for visualization
        if 'residual_SAR_mean' in grid_agg.columns:
            grid_agg['abs_residual_SAR_mean'] = np.abs(grid_agg['residual_SAR_mean'])
        
        # Create rectangular geometries
        from shapely.geometry import box
        grid_geometries = []
        for _, row in grid_agg.iterrows():
            x, y = row['grid_x'], row['grid_y']
            minx, miny = x * grid_size, y * grid_size
            maxx, maxy = (x + 1) * grid_size, (y + 1) * grid_size
            grid_geometries.append(box(minx, miny, maxx, maxy))
        
        grid_agg['geometry'] = grid_geometries
        
        gdf_grid = gpd.GeoDataFrame(grid_agg, crs='EPSG:4326')
        
        grid_path = DATA_PROCESSED / "map_grid_cells.geojson"
        gdf_grid.to_file(grid_path, driver='GeoJSON')
        
        print(f"  ✓ Saved: {grid_path}")
        print(f"    Grid cells: {len(gdf_grid)}")
        print(f"    Price range: [€{grid_agg['price_mean'].min():.0f}, €{grid_agg['price_mean'].max():.0f}]")
        if 'residual_SAR_mean' in grid_agg.columns:
            print(f"    SAR residual range: [{grid_agg['residual_SAR_mean'].min():.3f}, {grid_agg['residual_SAR_mean'].max():.3f}]")
    
    except Exception as e:
        print(f"  ✗ Error creating grid layer: {e}")
        sys.exit(1)

else:
    print("\n[LAYER B] Skipping aggregation layer (residuals not available)")
    print("  → Run 'python scripts/07b_extract_residuals.py' first to generate residuals")

print("\n" + "=" * 80)
print("✓ MAP LAYERS READY")
print("=" * 80)
print(f"\nFiles created:")
print(f"  - {sample_path.name}")
if residuals_available and 'residual_OLS' in df_points.columns:
    print(f"  - {grid_path.name}")
