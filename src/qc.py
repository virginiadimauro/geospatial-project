"""
Quality Control (QC) module: Assertions and data quality checks.
"""

import pandas as pd
import geopandas as gpd
import numpy as np

def check_unique_ids(df, id_col='listing_id'):
    """Assert IDs are unique (no duplicates)."""
    assert df[id_col].duplicated().sum() == 0, f"Duplicate {id_col} values found!"
    assert df[id_col].isnull().sum() == 0, f"Null {id_col} values found!"
    return f"✓ {id_col} is unique (n={len(df)})"

def check_no_negative_ids(df, id_col='listing_id'):
    """Assert no negative IDs."""
    assert (df[id_col] >= 0).all(), f"Found negative {id_col} values!"
    return f"✓ No negative {id_col} values"

def check_geometry_validity(gdf):
    """Assert all geometries are valid."""
    assert (~gdf.geometry.is_valid).sum() == 0, "Found invalid geometries!"
    assert gdf.geometry.is_empty.sum() == 0, "Found empty geometries!"
    return f"✓ All {len(gdf)} geometries are valid"

def check_crs(gdf, expected_crs='EPSG:4326'):
    """Assert CRS matches expected."""
    assert gdf.crs == expected_crs, f"CRS mismatch: {gdf.crs} != {expected_crs}"
    return f"✓ CRS is {expected_crs}"

def check_price_range(df, price_col='price', min_price=10, max_price=10000):
    """Assert prices are in expected range (logs outliers)."""
    if price_col not in df.columns:
        return f"⚠️  {price_col} column not found"
    
    outliers_low = (df[price_col] < min_price).sum()
    outliers_high = (df[price_col] > max_price).sum()
    
    if outliers_low > 0 or outliers_high > 0:
        return f"⚠️  Price outliers: <€{min_price}: {outliers_low}, >€{max_price}: {outliers_high}"
    
    return f"✓ Prices in range €{min_price}-€{max_price}"

def check_availability_values(df, avail_col='available'):
    """Assert availability is binary (0/1)."""
    if avail_col not in df.columns:
        return f"⚠️  {avail_col} column not found"
    
    assert df[avail_col].isin([0, 1]).all(), f"Availability not binary in {avail_col}"
    return f"✓ Availability is binary (0/1)"

def check_date_range(df, date_col='date'):
    """Check date range and missing values."""
    if date_col not in df.columns:
        return f"⚠️  {date_col} column not found"
    
    null_count = df[date_col].isnull().sum()
    if null_count > 0:
        return f"⚠️  {null_count} null dates in {date_col}"
    
    return f"✓ Date range: {df[date_col].min()} to {df[date_col].max()}"

def check_spatial_join_coverage(gdf_joined, min_coverage=0.95):
    """Assert spatial join coverage meets minimum threshold."""
    if 'neighbourhood_idx' in gdf_joined.columns:
        matched = gdf_joined['neighbourhood_idx'].notna().sum()
    else:
        matched = len(gdf_joined)
    
    total = len(gdf_joined)
    coverage = matched / total if total > 0 else 0
    
    assert coverage >= min_coverage, f"Spatial join coverage {coverage:.1%} < {min_coverage:.1%}"
    return f"✓ Spatial join coverage: {coverage:.1%}"

def check_neighbourhood_listing_counts(gdf_listings_with_neigh, gdf_neighbourhoods_enriched):
    """Verify that aggregated n_listings matches individual listings count."""
    total_listings_with_neigh = gdf_listings_with_neigh['neighbourhood_idx'].notna().sum()
    sum_n_listings = gdf_neighbourhoods_enriched['n_listings'].sum()
    
    assert total_listings_with_neigh == sum_n_listings, (
        f"Listing count mismatch: {total_listings_with_neigh} individual != {sum_n_listings} aggregated"
    )
    return f"✓ Listing counts match: {sum_n_listings:,} total"

def print_qc_report(data_dict, checks):
    """
    Print formatted QC report.
    
    Args:
        data_dict: Dict with dataset names and DataFrames
        checks: List of (name, check_func, kwargs) tuples
    """
    print("\n" + "=" * 80)
    print("QUALITY CONTROL REPORT")
    print("=" * 80)
    
    for name, check_func, kwargs in checks:
        try:
            result = check_func(**kwargs)
            print(f"\n{name}")
            print(f"  {result}")
        except AssertionError as e:
            print(f"\n❌ {name}")
            print(f"  ERROR: {e}")
        except Exception as e:
            print(f"\n⚠️  {name}")
            print(f"  WARNING: {e}")
    
    print("\n" + "=" * 80)
