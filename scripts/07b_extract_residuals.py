#!/usr/bin/env python
"""
Extract residuals from OLS, SAR and SEM models for mapping
=========================================================

Loads model_sample.parquet, fits OLS/SAR/SEM, extracts residuals,
and saves residuals_for_map.csv with columns:
- listing_id, log_price, residual_OLS, residual_SAR, residual_SEM, abs_residual_SAR

This file is used by the Streamlit interactive map to show model fit quality.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import sys
import warnings

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

warnings.filterwarnings('ignore')

from libpysal.weights import KNN
import spreg
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

# ============================================================================
# SETUP
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXTRACTING RESIDUALS FOR INTERACTIVE MAP")
print("=" * 80)

# ============================================================================
# LOAD DATA (EXACT COPY FROM SCRIPT 07)
# ============================================================================
print("\n[STEP 1] Loading model sample...")

model_sample_path = DATA_PROCESSED / "model_sample.parquet"
df_model = pd.read_parquet(model_sample_path).copy()
print(f"  → Loaded {len(df_model)} listings")

df_model = df_model.reset_index(drop=True).copy()
df_model['_original_idx'] = df_model.index

# Define covariates
exclude = {'listing_id', 'latitude', 'longitude', 'log_price', 'price_numeric', 
           'price_winsorized', 'geometry', '_original_idx'}
all_cols_available = [c for c in df_model.columns if c not in exclude]
X_cols_parsimonious = [c for c in all_cols_available if not c.startswith('neigh_')]

# Prepare parsimonious specification
spatial_cols = X_cols_parsimonious + ['log_price', '_original_idx', 'latitude', 'longitude', 'listing_id']
df_spatial = df_model[spatial_cols].dropna()

print(f"  → After dropna: N={len(df_spatial)}")

model_indices = df_spatial['_original_idx'].values
y = df_spatial['log_price'].values
X_parsimonious = df_spatial[X_cols_parsimonious].astype(float).values
listing_ids = df_spatial['listing_id'].values

# ============================================================================
# CREATE SPATIAL WEIGHTS
# ============================================================================
print("\n[STEP 2] Creating spatial weights...")

gdf = gpd.GeoDataFrame(
    df_spatial.loc[:, ['_original_idx', 'longitude', 'latitude']].reset_index(drop=True).copy(),
    geometry=gpd.points_from_xy(df_spatial['longitude'], df_spatial['latitude']),
    crs='EPSG:4326'
)
gdf = gdf.to_crs('EPSG:25830')

w_knn8 = KNN.from_dataframe(gdf, k=8)
w_knn8.transform = 'r'

print(f"  → W: {w_knn8.n} obs, {w_knn8.sparse.nnz} non-zero, {len(w_knn8.islands)} islands")

# ============================================================================
# FIT OLS
# ============================================================================
print("\n[STEP 3] Fitting OLS (parsimonious)...")

X_pars_const = add_constant(X_parsimonious)
ols_model = sm.OLS(y, X_pars_const).fit(cov_type='HC1')
ols_residuals = np.asarray(ols_model.resid).flatten()

print(f"  → OLS R² = {ols_model.rsquared:.4f}")
print(f"  → Residuals shape: {ols_residuals.shape}")

# ============================================================================
# FIT SAR
# ============================================================================
print("\n[STEP 4] Fitting SAR (GMM)...")

try:
    sar_model = spreg.GM_Lag(
        y, X_parsimonious, w=w_knn8,
        name_y='log_price',
        name_x=X_cols_parsimonious,
        name_w='knn8',
        robust='white'
    )
    sar_residuals = np.asarray(sar_model.u).flatten()
    sar_rho = float(np.asarray(getattr(sar_model, 'rho', np.nan)).squeeze())
    print(f"  → SAR ρ = {sar_rho:.6f}")
    print(f"  → Residuals shape: {sar_residuals.shape}")
except Exception as e:
    print(f"  ⚠ SAR fit failed: {e}")
    sar_residuals = np.full_like(y, np.nan)

# ============================================================================
# FIT SEM
# ============================================================================
print("\n[STEP 5] Fitting SEM (GMM)...")

try:
    try:
        sem_model = spreg.GM_Error(
            y, X_parsimonious, w=w_knn8,
            name_y='log_price',
            name_x=X_cols_parsimonious,
            name_w='knn8',
            robust='white'
        )
    except TypeError:
        sem_model = spreg.GM_Error(
            y, X_parsimonious, w=w_knn8,
            name_y='log_price',
            name_x=X_cols_parsimonious,
            name_w='knn8'
        )
    sem_residuals = np.asarray(sem_model.u).flatten()
    sem_lambda = float(np.asarray(
        getattr(sem_model, 'lam', getattr(sem_model, 'lambda', getattr(sem_model, 'lamda', np.nan)))
    ).squeeze())
    print(f"  → SEM λ = {sem_lambda:.6f}")
    print(f"  → Residuals shape: {sem_residuals.shape}")
except Exception as e:
    print(f"  ⚠ SEM fit failed: {e}")
    sem_residuals = np.full_like(y, np.nan)

# ============================================================================
# SAVE RESIDUALS
# ============================================================================
print("\n[STEP 6] Saving residuals...")

residuals_df = pd.DataFrame({
    'listing_id': listing_ids,
    'log_price': y,
    'residual_OLS': ols_residuals,
    'residual_SAR': sar_residuals,
    'residual_SEM': sem_residuals,
    'abs_residual_SAR': np.abs(sar_residuals)
})

residuals_path = OUTPUT_TABLES / "residuals_for_map.csv"
residuals_df.to_csv(residuals_path, index=False)

print(f"\n✓ Saved: {residuals_path}")
print(f"   Shape: {residuals_df.shape}")
print(f"\n   Summary statistics:")
print(residuals_df[['residual_OLS', 'residual_SAR', 'residual_SEM']].describe())

print("\n" + "=" * 80)
print("RESIDUALS EXTRACTED")
print("=" * 80)
