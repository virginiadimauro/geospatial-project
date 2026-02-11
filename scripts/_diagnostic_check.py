#!/usr/bin/env python
"""Quick diagnostic check for W sparsity and data dimensions"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from libpysal.weights import KNN

# Load data
model_sample_path = Path('data/processed/model_sample.parquet')
df_model = pd.read_parquet(model_sample_path).copy()
df_model = df_model.reset_index(drop=True).copy()

# Prepare coordinates
coords_df = df_model[['longitude','latitude']].copy()

# Prepare covariates
exclude = {'listing_id','latitude','longitude','log_price','price_numeric','price_winsorized','geometry','_original_idx'}
all_cols_available = [c for c in df_model.columns if c not in exclude]
X_cols_parsimonious = [c for c in all_cols_available if not c.startswith('neigh_')]

# Prepare data
df_spatial = df_model[X_cols_parsimonious + ['log_price']].dropna()
y = df_spatial['log_price'].values
X_parsimonious = df_spatial[X_cols_parsimonious].astype(float).values

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    coords_df.reset_index(drop=True).copy(),
    geometry=gpd.points_from_xy(coords_df['longitude'], coords_df['latitude']),
    crs='EPSG:4326'
)
gdf = gdf.to_crs('EPSG:25830')

# Create weights
w_knn8 = KNN.from_dataframe(gdf, k=8)
w_knn8.transform = 'r'

print('=' * 80)
print('DIAGNOSTICS')
print('=' * 80)
print(f'\nData Arrays:')
print(f'  y.shape = {y.shape}')
print(f'  y.dtype = {y.dtype}')
print(f'  y memory = {y.nbytes/1e6:.2f} MB')
print(f'\n  X_parsimonious.shape = {X_parsimonious.shape}')
print(f'  X_parsimonious.dtype = {X_parsimonious.dtype}')
print(f'  X_parsimonious memory = {X_parsimonious.nbytes/1e6:.2f} MB')

print(f'\nSpatial Weights Matrix:')
print(f'  W.n = {w_knn8.n}')
print(f'  W.sparse type: {type(w_knn8.sparse)}')
print(f'  W.sparse.nnz = {w_knn8.sparse.nnz:,}')
print(f'  nnz / N = {w_knn8.sparse.nnz / w_knn8.n:.2f} (expected ~8.0)')
print(f'  W is sparse: {hasattr(w_knn8.sparse, "nnz")}')
print(f'  Sparsity = {w_knn8.sparse.nnz / (w_knn8.n ** 2) * 100:.4f}%')

print(f'\nExpected memory for dense W: {(w_knn8.n ** 2) * 8 / 1e9:.2f} GB')
print(f'Actual memory for sparse W: ~{w_knn8.sparse.nnz * 8 / 1e6:.2f} MB')
print('=' * 80)
