"""
I/O module: Load and save data in various formats (CSV, Parquet, GeoJSON).
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import warnings

def load_csv(filepath, **kwargs):
    """
    Load CSV file with error handling.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv()
    
    Returns:
        pd.DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    return pd.read_csv(filepath, **kwargs)

def load_geojson(filepath, **kwargs):
    """
    Load GeoJSON file with CRS validation.
    
    Args:
        filepath: Path to GeoJSON file
        **kwargs: Additional arguments for gpd.read_file()
    
    Returns:
        geopandas.GeoDataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {filepath}")
    
    gdf = gpd.read_file(filepath, **kwargs)
    
    if gdf.crs is None:
        warnings.warn(f"⚠️  CRS missing in {filepath.name}. Assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    
    return gdf

def load_parquet(filepath, **kwargs):
    """
    Load Parquet file.
    
    Args:
        filepath: Path to Parquet file
        **kwargs: Additional arguments for pd.read_parquet()
    
    Returns:
        pd.DataFrame or geopandas.GeoDataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")
    
    return pd.read_parquet(filepath, **kwargs)

def save_parquet(df, filepath, **kwargs):
    """
    Save DataFrame to Parquet with validation.
    
    Args:
        df: DataFrame to save
        filepath: Output path
        **kwargs: Additional arguments for df.to_parquet()
    
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Preserve geometry for GeoDataFrames
    if isinstance(df, gpd.GeoDataFrame):
        df.to_parquet(filepath, **kwargs)
    else:
        df.to_parquet(filepath, index=False, **kwargs)
    
    return filepath

def save_geojson(gdf, filepath, **kwargs):
    """
    Save GeoDataFrame to GeoJSON.
    
    Args:
        gdf: GeoDataFrame to save
        filepath: Output path
        **kwargs: Additional arguments for gdf.to_file()
    
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure EPSG:4326 for web compatibility
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    gdf.to_file(filepath, driver="GeoJSON", **kwargs)
    
    return filepath

def save_csv(df, filepath, **kwargs):
    """
    Save DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filepath: Output path
        **kwargs: Additional arguments for df.to_csv()
    
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=False, **kwargs)
    
    return filepath

def file_size_mb(filepath):
    """Get file size in MB."""
    return Path(filepath).stat().st_size / (1024 ** 2)
