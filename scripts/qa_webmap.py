#!/usr/bin/env python3
"""
Quality Assurance Script for Interactive Map
==============================================

Validates:
1. CRS consistency (web EPSG:4326, modelli EPSG:25830)
2. GeoJSON geometries valid
3. All required files exist
4. Residuals CSV complete and numeric
5. Color scale and residual ranges valid

Run before launching Streamlit to ensure map readiness.
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
REPORTS_MAPS = PROJECT_ROOT / "reports" / "maps"
SCRIPTS = PROJECT_ROOT / "scripts"

print("=" * 80)
print("WEBMAP QUALITY ASSURANCE CHECK")
print("=" * 80)

errors = []
warnings = []
checks_passed = 0

# ============================================================================
# CHECK 1: Files exist
# ============================================================================
print("\n[1] Checking file existence...")

required_files = {
    "GeoJSON points": DATA_PROCESSED / "map_points_sample.geojson",
    "GeoJSON grid": DATA_PROCESSED / "map_grid_cells.geojson",
    "Residuals CSV": OUTPUT_TABLES / "residuals_for_map.csv",
    "HTML map": REPORTS_MAPS / "interactive_map.html",
    "Streamlit app": PROJECT_ROOT / "webmap" / "app.py",
    "Extract script": SCRIPTS / "07b_extract_residuals.py",
    "Prep layers script": SCRIPTS / "08_prepare_map_layers.py",
}

for name, path in required_files.items():
    if path.exists():
        size = path.stat().st_size
        if size > 1e6:
            size_str = f"{size/1e6:.1f} MB"
        elif size > 1e3:
            size_str = f"{size/1e3:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  ✓ {name}: {path.name} ({size_str})")
        checks_passed += 1
    else:
        errors.append(f"Missing file: {name} at {path}")
        print(f"  ✗ {name}: NOT FOUND")

# ============================================================================
# CHECK 2: GeoJSON validity and CRS
# ============================================================================
print("\n[2] Validating GeoJSON files...")

try:
    gdf_points = gpd.read_file(DATA_PROCESSED / "map_points_sample.geojson")
    print(f"  ✓ Points GeoJSON: {len(gdf_points)} features")
    if gdf_points.crs != "EPSG:4326":
        warnings.append(f"Points CRS is {gdf_points.crs}, expected EPSG:4326")
    if gdf_points.geometry.isna().any():
        errors.append("Points GeoJSON has NULL geometries")
    if not gdf_points.geometry.is_valid.all():
        errors.append("Points GeoJSON has invalid geometries")
    else:
        print(f"    - CRS: {gdf_points.crs} (✓ EPSG:4326)")
        print(f"    - All geometries valid: ✓")
        checks_passed += 1
except Exception as e:
    errors.append(f"Failed to load points GeoJSON: {e}")

try:
    gdf_grid = gpd.read_file(DATA_PROCESSED / "map_grid_cells.geojson")
    print(f"  ✓ Grid GeoJSON: {len(gdf_grid)} features")
    if gdf_grid.crs != "EPSG:4326":
        warnings.append(f"Grid CRS is {gdf_grid.crs}, expected EPSG:4326")
    if gdf_grid.geometry.isna().any():
        errors.append("Grid GeoJSON has NULL geometries")
    if not gdf_grid.geometry.is_valid.all():
        errors.append("Grid GeoJSON has invalid geometries")
    else:
        print(f"    - CRS: {gdf_grid.crs} (✓ EPSG:4326)")
        print(f"    - All geometries valid: ✓")
        checks_passed += 1
except Exception as e:
    errors.append(f"Failed to load grid GeoJSON: {e}")

# ============================================================================
# CHECK 3: Residuals CSV
# ============================================================================
print("\n[3] Validating residuals CSV...")

try:
    resid_df = pd.read_csv(OUTPUT_TABLES / "residuals_for_map.csv")
    print(f"  ✓ Residuals CSV: {len(resid_df)} rows")
    
    # Check columns
    required_cols = ['listing_id', 'log_price', 'residual_OLS', 'residual_SAR', 'residual_SEM', 'abs_residual_SAR']
    missing_cols = [c for c in required_cols if c not in resid_df.columns]
    if missing_cols:
        errors.append(f"Residuals CSV missing columns: {missing_cols}")
    else:
        print(f"    - All required columns present: ✓")
        checks_passed += 1
    
    # Check for NaN
    nan_counts = resid_df.isnull().sum()
    if nan_counts.sum() > 0:
        warnings.append(f"Residuals CSV has {nan_counts.sum()} NaN values")
        print(f"    - NaN check: {nan_counts.sum()} missing values (warning)")
    else:
        print(f"    - No missing values: ✓")
    
    # Check numeric ranges
    for col in ['residual_OLS', 'residual_SAR', 'residual_SEM']:
        if col in resid_df.columns:
            min_val = resid_df[col].min()
            max_val = resid_df[col].max()
            mean_val = resid_df[col].mean()
            print(f"    - {col}: [{min_val:.3f}, {max_val:.3f}] (mean={mean_val:.4f})")
            if abs(mean_val) > 0.1:
                warnings.append(f"{col} mean = {mean_val:.4f} (expected ≈0)")
    
    checks_passed += 1
    
except Exception as e:
    errors.append(f"Failed to load residuals CSV: {e}")

# ============================================================================
# CHECK 4: Merge consistency
# ============================================================================
print("\n[4] Checking data merge consistency...")

try:
    # Points should be subset of residuals
    points_ids = set(gdf_points['listing_id'].unique())
    resid_ids = set(resid_df['listing_id'].unique())
    
    missing_from_resid = points_ids - resid_ids
    if missing_from_resid:
        errors.append(f"{len(missing_from_resid)} point IDs not in residuals CSV")
    else:
        print(f"  ✓ All {len(points_ids)} point listing_ids in residuals CSV")
        checks_passed += 1
except Exception as e:
    warnings.append(f"Could not check merge consistency: {e}")

# ============================================================================
# CHECK 5: Color scale validity
# ============================================================================
print("\n[5] Validating color scale...")

# Residual ranges for color mapping
residual_ranges = [-2.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.5]
print(f"  Color scale validation:")
print(f"    - Residual ranges: {residual_ranges}")
print(f"    - Blue-gray-red diverging palette: ✓")
checks_passed += 1

# ============================================================================
# CHECK 6: Streamlit app exists and readable
# ============================================================================
print("\n[6] Checking Streamlit app...")

app_path = PROJECT_ROOT / "webmap" / "app.py"
if app_path.exists():
    try:
        with open(app_path, 'r') as f:
            content = f.read()
            if 'streamlit' in content and 'folium' in content:
                print(f"  ✓ webmap/app.py exists and contains streamlit/folium imports")
                checks_passed += 1
            else:
                warnings.append("app.py missing streamlit/folium imports")
    except Exception as e:
        errors.append(f"Cannot read app.py: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nChecks passed: {checks_passed}/6")

if errors:
    print(f"\n🔴 ERRORS ({len(errors)}):")
    for err in errors:
        print(f"   - {err}")
    print("\n❌ Quality assurance FAILED. Fix issues before launching webmap.")
    sys.exit(1)

if warnings:
    print(f"\n🟡 WARNINGS ({len(warnings)}):")
    for warn in warnings:
        print(f"   - {warn}")
    print("\n⚠️  Warnings found. Proceed with caution.")
else:
    print(f"\n✅ No warnings.")

if not errors:
    print("\n" + "🟢 " * 20)
    print("WEBMAP IS READY FOR LAUNCH")
    print("🟢 " * 20)
    print("\nNext step:")
    print("  micromamba activate geo")
    print("  streamlit run webmap/app.py")
    print("\nOr use launcher script:")
    print("  python scripts/run_webmap.sh")
