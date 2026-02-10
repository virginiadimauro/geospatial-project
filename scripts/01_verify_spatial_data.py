"""
Quick verification of spatial data files
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path

data_path = Path("data/processed")

# Verifica file disponibili
print("Files in data/processed/:")
for f in sorted(data_path.glob("*")):
    size_mb = f.stat().st_size / (1024**2) if f.is_file() else 0
    print(f"  {f.name:50s} {size_mb:8.2f} MB")

# Carica listings
print("\n" + "="*80)
print("LISTINGS_CLEAN.PARQUET")
print("="*80)
df_listings = pd.read_parquet(data_path / "listings_clean.parquet")
print(f"Shape: {df_listings.shape}")
print(f"Columns: {df_listings.columns.tolist()}")
print(f"\nPrice stats:")
print(df_listings['price'].describe())
print(f"\nBeds column exists: {'beds' in df_listings.columns}")
if 'beds' in df_listings.columns:
    print(f"Beds non-null: {df_listings['beds'].notna().sum()} / {len(df_listings)} ({100*df_listings['beds'].notna().sum()/len(df_listings):.1f}%)")
    print(f"Beds stats:\n{df_listings['beds'].describe()}")

print(f"\nCoordinates check:")
print(f"  latitude non-null: {df_listings['latitude'].notna().sum()}")
print(f"  longitude non-null: {df_listings['longitude'].notna().sum()}")
print(f"  Lat range: {df_listings['latitude'].min():.4f} to {df_listings['latitude'].max():.4f}")
print(f"  Lon range: {df_listings['longitude'].min():.4f} to {df_listings['longitude'].max():.4f}")

# Carica neighbourhoods
print("\n" + "="*80)
print("NEIGHBOURHOODS_ENRICHED.GEOJSON")
print("="*80)
gdf_neigh = gpd.read_file(data_path / "neighbourhoods_enriched.geojson")
print(f"Shape: {gdf_neigh.shape}")
print(f"Columns: {gdf_neigh.columns.tolist()}")
print(f"CRS: {gdf_neigh.crs}")
print(f"Geometry type: {gdf_neigh.geometry.type.unique()}")
print(f"Valid geometries: {gdf_neigh.geometry.is_valid.sum()} / {len(gdf_neigh)}")

print("\n✓ Verification complete")
