"""
Spatial module: Create geometries, spatial joins, neighbourhood aggregation, area/density calculations.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from . import config

def listings_to_geodataframe(df_listings):
    """
    Convert listings DataFrame with lat/lon to GeoDataFrame with Point geometries.
    
    Args:
        df_listings: DataFrame with 'latitude' and 'longitude' columns
    
    Returns:
        GeoDataFrame with Point geometries in EPSG:4326 and log info
    """
    log = []
    
    # Check required columns
    if 'latitude' not in df_listings.columns or 'longitude' not in df_listings.columns:
        raise ValueError("'latitude' and 'longitude' columns required")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_listings.copy(),
        geometry=gpd.points_from_xy(df_listings['longitude'], df_listings['latitude']),
        crs='EPSG:4326'
    )
    
    # Validate geometries
    invalid = (~gdf.geometry.is_valid).sum()
    if invalid > 0:
        log.append(f"⚠️  Found {invalid} invalid geometries")
        gdf.geometry = gdf.geometry.buffer(0)
    
    log.append(f"✓ Created Point geometries for {len(gdf)} listings (CRS: EPSG:4326)")
    
    return gdf, log

def clean_neighbourhoods(gdf_neighbourhoods):
    """
    Validate and clean neighbourhood geometries.
    
    Args:
        gdf_neighbourhoods: Neighbourhood GeoDataFrame
    
    Returns:
        Cleaned GeoDataFrame and log info
    """
    log = []
    gdf_clean = gdf_neighbourhoods.copy()
    
    # 1. Check CRS
    if gdf_clean.crs is None:
        log.append(f"⚠️  CRS missing; assuming EPSG:4326")
        gdf_clean = gdf_clean.set_crs('EPSG:4326')
    else:
        log.append(f"✓ CRS: {gdf_clean.crs}")
    
    # 2. Validate geometries
    invalid_before = (~gdf_clean.geometry.is_valid).sum()
    if invalid_before > 0:
        log.append(f"⚠️  Found {invalid_before} invalid geometries; repairing...")
        gdf_clean.geometry = gdf_clean.geometry.buffer(0)
        invalid_after = (~gdf_clean.geometry.is_valid).sum()
        log.append(f"   → After repair: {invalid_after} invalid (target: 0)")
        assert invalid_after == 0, "Failed to repair geometries!"
    else:
        log.append(f"✓ All geometries are valid")
    
    log.append(f"✓ Neighbourhood cleaning complete")
    
    return gdf_clean, log

def spatial_join_listings_neighbourhoods(gdf_listings, gdf_neighbourhoods):
    """
    Assign listings to neighbourhoods via spatial join (point-in-polygon).
    
    Args:
        gdf_listings: Listings GeoDataFrame (Points, EPSG:4326)
        gdf_neighbourhoods: Neighbourhoods GeoDataFrame (Polygons)
    
    Returns:
        GeoDataFrame with 'neighbourhood_idx' column added and log info
    """
    log = []
    
    # Ensure same CRS
    if gdf_listings.crs != gdf_neighbourhoods.crs:
        log.append(f"⚠️  CRS mismatch; reprojecting listings to {gdf_neighbourhoods.crs}")
        gdf_listings = gdf_listings.to_crs(gdf_neighbourhoods.crs)
    
    # Create indexed neighbourhoods
    gdf_neigh_indexed = gdf_neighbourhoods.copy().reset_index()
    gdf_neigh_indexed = gdf_neigh_indexed.rename(columns={'index': 'neighbourhood_idx'})
    
    # Spatial join with 'within' predicate (safer than default)
    gdf_joined = gpd.sjoin(
        gdf_listings,
        gdf_neigh_indexed[['geometry', 'neighbourhood_idx']],
        how='left',
        predicate='within'
    )
    
    # If coverage is too low, try with 'intersects'
    if gdf_joined['neighbourhood_idx'].notna().sum() == 0:
        log.append(f"⚠️  No matches with 'within'; trying 'intersects'...")
        gdf_joined = gpd.sjoin(
            gdf_listings,
            gdf_neigh_indexed[['geometry', 'neighbourhood_idx']],
            how='left',
            predicate='intersects'
        )
    
    # Track matches
    total = len(gdf_joined)
    matched = gdf_joined['neighbourhood_idx'].notna().sum()
    coverage = (matched / total * 100) if total > 0 else 0
    
    log.append(f"✓ Spatial join complete:")
    log.append(f"  - Total listings: {total:,}")
    log.append(f"  - Matched to neighbourhood: {matched:,} ({coverage:.1f}%)")
    
    if coverage < config.MIN_SPATIAL_JOIN_COVERAGE * 100:
        log.append(f"⚠️  Coverage {coverage:.1f}% below threshold {config.MIN_SPATIAL_JOIN_COVERAGE*100:.1f}%")
    
    return gdf_joined, log

def aggregate_to_neighbourhoods(gdf_listings_with_neighbourhood, gdf_neighbourhoods):
    """
    Aggregate listing metrics (price, availability, reviews) to neighbourhood level.
    Includes area and density calculations in metric CRS.
    
    Args:
        gdf_listings_with_neighbourhood: Listings with neighbourhood assignment
        gdf_neighbourhoods: Neighbourhood polygons
    
    Returns:
        Enriched neighbourhood GeoDataFrame and log info
    """
    log = []
    
    # 1. Filter listings with neighbourhood assignment
    listings_with_neigh = gdf_listings_with_neighbourhood[
        gdf_listings_with_neighbourhood['neighbourhood_idx'].notna()
    ].copy()
    
    # Need neighbourhood identifier
    neigh_id_col = None
    for col in gdf_neighbourhoods.columns:
        if col != 'geometry' and (col.lower() in ['id', 'name', 'neighbourhood'] or 
                                   gdf_neighbourhoods[col].nunique() == len(gdf_neighbourhoods)):
            neigh_id_col = col
            break
    
    if neigh_id_col is None:
        neigh_id_col = 'neighbourhood_idx'
        gdf_neighbourhoods['neighbourhood_idx'] = range(len(gdf_neighbourhoods))
        log.append(f"⚠️  No neighbourhood ID column found; using index")
    else:
        listings_with_neigh['neighbourhood_idx'] = listings_with_neigh['neighbourhood_idx'].map(
            pd.Series(range(len(gdf_neighbourhoods)), index=gdf_neighbourhoods.index)
        )
        log.append(f"✓ Using neighbourhood ID column: '{neigh_id_col}'")
    
    # 2. Aggregate metrics
    agg_specs = {'listing_id': 'count'}  # n_listings
    
    if 'price' in listings_with_neigh.columns:
        agg_specs['price'] = ['median', 'mean']
    
    if 'availability_rate' in listings_with_neigh.columns:
        agg_specs['availability_rate'] = 'mean'
    
    if 'review_count_total' in listings_with_neigh.columns:
        agg_specs['review_count_total'] = 'mean'
    
    neigh_agg = listings_with_neigh.groupby('neighbourhood_idx', as_index=False).agg(agg_specs)
    
    # Flatten MultiIndex columns
    if isinstance(neigh_agg.columns, pd.MultiIndex):
        neigh_agg.columns = ['_'.join(col).strip('_') for col in neigh_agg.columns.values]
    
    # Rename for clarity
    neigh_agg.rename(columns={'listing_id_count': 'n_listings'}, inplace=True)
    log.append(f"✓ Aggregated to {len(neigh_agg)} neighbourhoods")
    
    # 3. Calculate area and density (using metric CRS)
    gdf_neighbourhoods_metric = gdf_neighbourhoods.to_crs(config.CRS_METRIC)
    gdf_neighbourhoods_metric['area_m2'] = gdf_neighbourhoods_metric.geometry.area
    gdf_neighbourhoods_metric['area_km2'] = gdf_neighbourhoods_metric['area_m2'] / 1e6
    
    # Add to aggregation
    area_data = gdf_neighbourhoods_metric[['area_km2']].reset_index(drop=True)
    neigh_agg['area_km2'] = area_data['area_km2']
    
    # Listing density (listings per km2)
    neigh_agg['listing_density'] = neigh_agg['n_listings'] / neigh_agg['area_km2'].clip(lower=0.01)
    
    log.append(f"✓ Area and density calculated (in {config.CRS_METRIC})")
    log.append(f"  - Area range: {neigh_agg['area_km2'].min():.2f} - {neigh_agg['area_km2'].max():.2f} km²")
    log.append(f"  - Density range: {neigh_agg['listing_density'].min():.1f} - {neigh_agg['listing_density'].max():.1f} listings/km²")
    
    # 4. Merge back with geometries (in web CRS)
    gdf_neighbourhoods_enriched = gdf_neighbourhoods.copy().reset_index(drop=True)
    gdf_neighbourhoods_enriched = gdf_neighbourhoods_enriched.merge(
        neigh_agg,
        left_index=True,
        right_on='neighbourhood_idx',
        how='left'
    )
    
    # Fill missing with zero for neighbourhoods without listings
    gdf_neighbourhoods_enriched['n_listings'] = gdf_neighbourhoods_enriched['n_listings'].fillna(0).astype('int64')
    
    # Ensure EPSG:4326 for final output
    if gdf_neighbourhoods_enriched.crs != 'EPSG:4326':
        gdf_neighbourhoods_enriched = gdf_neighbourhoods_enriched.to_crs('EPSG:4326')
    
    log.append(f"✓ Neighbourhoods enriched and exported to EPSG:4326")
    
    return gdf_neighbourhoods_enriched, log
