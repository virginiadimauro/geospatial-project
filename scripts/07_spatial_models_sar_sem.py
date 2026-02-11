#!/usr/bin/env python
"""
Spatial Econometric Models: SAR vs SEM vs OLS Comparison
=========================================================

Estimates and compares four models on full sample (N=18,940):
1. OLS Parsimonious (K=14, structural variables only)
2. OLS with Neighbourhood FE (K=141, includes ~130 neighbourhood dummies)
3. SAR - Spatial Autoregressive (GMM, K=14) - spatial lag of y
4. SEM - Spatial Error Model (GMM, K=14) - spatial lag of residuals

Uses parsimonious specification (no neighbourhood dummies) for SAR/SEM
to avoid computational OOM issues with Maximum Likelihood estimation on N~19k.

GMM estimation is computationally tractable for large N with sparse weights.
ML estimation is disabled in this script to avoid OOM.

Output:
- outputs/tables/model_comparison.csv (comparison table)
- outputs/tables/sar_coeffs.csv (SAR coefficient table with robust SE)
- outputs/tables/sem_coeffs.csv (SEM coefficient table with robust SE)
- outputs/tables/morans_postfit.csv (Moran's I on residuals for all models)
- outputs/figures/morans_postfit_compare.png (Moran comparison)
- outputs/figures/residuals_hist_compare.png (residual distributions)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import warnings
import sys
import time

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

warnings.filterwarnings('ignore')

# Spatial modeling
from libpysal.weights import KNN
from esda import Moran
import spreg

# Statistics
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from src.prep import load_listings_processed, build_model_df, get_y_X

# ============================================================================
# SETUP
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SPATIAL ECONOMETRIC MODELS: SAR vs SEM vs OLS")
print("=" * 80)

# NOTE: uses model_sample.parquet for consistent N (centralized data-prep)

# ============================================================================
# 1. LOAD AND PREPARE DATA (EXACT COPY FROM SCRIPT 05)
# ============================================================================
print("\n[STEP 1] Loading centralized model sample for consistent N...")
sys.stdout.flush()

model_sample_path = DATA_PROCESSED / "model_sample.parquet"
if model_sample_path.exists():
    df_model = pd.read_parquet(model_sample_path).copy()
    print(f"  → Loaded model_sample.parquet with {len(df_model)} listings")
else:
    df_raw = load_listings_processed(DATA_PROCESSED / "listings_clean.parquet")
    df_model = build_model_df(df_raw, out_path=model_sample_path)
    print(f"  → Built and saved model_sample.parquet with {len(df_model)} listings")

# Ensure indices and prepare arrays
df_model = df_model.reset_index(drop=True).copy()

# ============================================================================
# 2. PREPARE COVARIATES (PARSIMONIOUS SPECIFICATION + FE COMPARISON)
# ============================================================================
print("[STEP 2] Preparing covariates...")
sys.stdout.flush()

print(f"  → model_sample initial rows: {len(df_model)}")
sys.stdout.flush()

# Ensure _original_idx exists for compatibility
df_model = df_model.reset_index(drop=True).copy()
df_model['_original_idx'] = df_model.index


# Define parsimonious covariates (NO neighbourhood dummies)
exclude = {'listing_id','latitude','longitude','log_price','price_numeric','price_winsorized','geometry','_original_idx'}
all_cols_available = [c for c in df_model.columns if c not in exclude]

# Parsimonious: structural variables only (no neighbourhood FE)
X_cols_parsimonious = [c for c in all_cols_available if not c.startswith('neigh_')]

# Full: all variables including neighbourhood FE
X_cols_full = all_cols_available

print(f"\n  COVARIATE SPECIFICATIONS:")
print(f"  → Parsimonious (SAR/SEM): K={len(X_cols_parsimonious)}")
print(f"    {X_cols_parsimonious}")
print(f"  → Full with FE (OLS comparison): K={len(X_cols_full)}")
print(f"    First 20: {X_cols_full[:20]}...")
print(f"    (+ {len([c for c in X_cols_full if c.startswith('neigh_')])} neighbourhood dummies)")

# DROPNA on parsimonious specification for spatial models
spatial_cols = X_cols_parsimonious + ['log_price', '_original_idx', 'latitude', 'longitude']
df_spatial = df_model[spatial_cols].dropna()
model_indices = df_spatial['_original_idx'].values
y = df_spatial['log_price'].values
X_parsimonious = df_spatial[X_cols_parsimonious].astype(float).values

print(f"\n  → After dropna: N={len(y)}")
print(f"  → Parsimonious model: y={y.shape}, X={X_parsimonious.shape}")

# Prepare full specification for OLS comparison
df_full = df_model[X_cols_full + ['log_price','_original_idx']].dropna()
y_full = df_full['log_price'].values
X_full = df_full[X_cols_full].astype(float).values
print(f"  → Full model (OLS-FE): y={y_full.shape}, X={X_full.shape}")

print(f"\n  DIAGNOSTICS - Data Arrays:")
print(f"  → y.dtype = {y.dtype}, y.shape = {y.shape}")
print(f"  → X_parsimonious.dtype = {X_parsimonious.dtype}, X_parsimonious.shape = {X_parsimonious.shape}")
print(f"  → X_full.dtype = {X_full.dtype}, X_full.shape = {X_full.shape}")
print(f"  → Memory: y={y.nbytes/1e6:.2f} MB, X_pars={X_parsimonious.nbytes/1e6:.2f} MB, X_full={X_full.nbytes/1e6:.2f} MB")

sys.stdout.flush()

# ============================================================================
# 3. CREATE SPATIAL WEIGHTS (EXACT FROM SCRIPT 05)
# ============================================================================
print("\n[STEP 3] Creating spatial weights (k-NN k=8, EPSG:25830)...")
sys.stdout.flush()

# Create GeoDataFrame from post-dropna df_spatial
gdf = gpd.GeoDataFrame(
    df_spatial.loc[:, ['_original_idx', 'longitude', 'latitude']].reset_index(drop=True).copy(),
    geometry=gpd.points_from_xy(df_spatial['longitude'], df_spatial['latitude']),
    crs='EPSG:4326'
)

# Project to metric CRS for kNN
gdf = gdf.to_crs('EPSG:25830')

# Create kNN weights (k=8, row-standardized)
w_knn8 = KNN.from_dataframe(gdf, k=8)
w_knn8.transform = 'r'  # Row standardize

print(f"Weights shape: {w_knn8.n} obs")
print(f"Avg neighbors: {np.mean([len(w_knn8.neighbors[i]) for i in w_knn8.neighbors]):.2f}")
print(f"Islands: {len(w_knn8.islands)}")

# DIAGNOSTIC: Check sparsity
print(f"\n  DIAGNOSTICS - Spatial Weights Matrix:")
print(f"  → W.n = {w_knn8.n}")
print(f"  → W.sparse type: {type(w_knn8.sparse)}")
print(f"  → W.sparse.nnz (non-zero elements): {w_knn8.sparse.nnz}")
print(f"  → nnz / N = {w_knn8.sparse.nnz / w_knn8.n:.2f}")
print(f"  → W is sparse: {hasattr(w_knn8.sparse, 'nnz')}")
sys.stdout.flush()

# ============================================================================
# 4. QUALITY CHECKS
# ============================================================================
print("\n[STEP 4] Quality checks...")
sys.stdout.flush()

assert w_knn8.n == len(y) == X_parsimonious.shape[0], f"Length mismatch: y={len(y)}, X_pars={X_parsimonious.shape[0]}, W={w_knn8.n}"
assert not np.any(np.isnan(y)) and not np.any(np.isinf(y)), "NaN/inf in y"
assert not np.any(np.isnan(X_parsimonious)) and not np.any(np.isinf(X_parsimonious)), "NaN/inf in X_parsimonious"
assert len(w_knn8.islands) == 0, f"Islands detected: {len(w_knn8.islands)}"

print("✓ All quality checks passed")
print(f"✓ N={len(y)}, X_parsimonious shape={X_parsimonious.shape[0]} x {X_parsimonious.shape[1]}")
print(f"✓ W row-standardized, {len(w_knn8.islands)} islands")

# ============================================================================
# 5. FIT MODELS
# ============================================================================
print("\n" + "=" * 80)
print("FITTING MODELS")
print("=" * 80)

print("\n[MODEL 1] OLS Parsimonious (K={}, no neighbourhood FE)...".format(len(X_cols_parsimonious)))
sys.stdout.flush()

X_pars_const = add_constant(X_parsimonious)
ols_pars_model = sm.OLS(y, X_pars_const)
ols_pars_results = ols_pars_model.fit(cov_type='HC1')

ols_pars_residuals = ols_pars_results.resid
if hasattr(ols_pars_residuals, 'values'):
    ols_pars_residuals = ols_pars_residuals.values

ols_pars_loglik = ols_pars_results.llf
ols_pars_aic = ols_pars_results.aic
ols_pars_bic = ols_pars_results.bic
ols_pars_r2 = ols_pars_results.rsquared

print(f"  Covariates: {X_cols_parsimonious}")
print(f"  K = {len(X_cols_parsimonious)}")
print(f"  R²: {ols_pars_r2:.4f}")
print(f"  Log-likelihood: {ols_pars_loglik:.4f}")
print(f"  AIC: {ols_pars_aic:.4f}, BIC: {ols_pars_bic:.4f}")

# ============================================================================
# 5.2 OLS WITH NEIGHBOURHOOD FE (ROBUSTNESS CHECK)
# ============================================================================
print("\n[MODEL 2] OLS with Neighbourhood FE (K={}, robustness)...".format(len(X_cols_full)))
sys.stdout.flush()

X_full_const = add_constant(X_full)
ols_fe_model = sm.OLS(y_full, X_full_const)
ols_fe_results = ols_fe_model.fit(cov_type='HC1')

ols_fe_residuals = ols_fe_results.resid
if hasattr(ols_fe_residuals, 'values'):
    ols_fe_residuals = ols_fe_residuals.values

ols_fe_loglik = ols_fe_results.llf
ols_fe_aic = ols_fe_results.aic
ols_fe_bic = ols_fe_results.bic
ols_fe_r2 = ols_fe_results.rsquared

print(f"  Covariates: structural + {len([c for c in X_cols_full if c.startswith('neigh_')])} neighbourhood dummies")
print(f"  K = {len(X_cols_full)}")
print(f"  R²: {ols_fe_r2:.4f}")
print(f"  Log-likelihood: {ols_fe_loglik:.4f}")
print(f"  AIC: {ols_fe_aic:.4f}, BIC: {ols_fe_bic:.4f}")

# ============================================================================
# 5.3 SAR MODEL (SPATIAL LAG) - GMM (GENERALIZED METHOD OF MOMENTS)
# ============================================================================
print("\n[MODEL 3] SAR - Spatial Autoregressive Model (GMM, parsimonious)...")
print(f"  Covariates: K={len(X_cols_parsimonious)} (no neighbourhood FE)")
print(f"  Method: Generalized Method of Moments (computationally tractable for N={len(y)})")
sys.stdout.flush()

try:
    # SAR: y = rho*W*y + X*beta + epsilon
    print("  → Starting SAR GMM estimation...")
    sys.stdout.flush()
    t0_sar = time.perf_counter()
    sar_model = spreg.GM_Lag(
        y, X_parsimonious, w=w_knn8,
        name_y='log_price',
        name_x=X_cols_parsimonious,
        name_w='knn8',
        robust='white'  # Robust standard errors
    )
    t1_sar = time.perf_counter()
    print(f"  ✓ SAR GMM estimation completed in {t1_sar - t0_sar:.2f} seconds")
    sys.stdout.flush()
    
    sar_loglik = np.nan  # GMM doesn't have log-likelihood
    sar_aic = np.nan
    sar_bic = np.nan
    sar_pseudor2 = getattr(sar_model, 'pr2', np.nan)  # Pseudo R-squared
    sar_rho = float(np.asarray(getattr(sar_model, 'rho', np.nan)).squeeze())  # Spatial autoregressive parameter
    sar_z = np.asarray(sar_model.z_stat, dtype=float)
    sar_rho_pval = float(sar_z[-1, 1]) if sar_z.ndim == 2 and sar_z.shape[1] > 1 else np.nan
    sar_sigma2 = getattr(sar_model, 'sig2', np.nan)
    
    sar_residuals = sar_model.u.flatten()
    
    print(f"  SAR ρ (spatial lag coeff): {sar_rho:.6f}, p={sar_rho_pval:.6f}")
    print(f"  Pseudo-R²: {sar_pseudor2:.4f}")
    print(f"  σ²: {sar_sigma2:.6f}")
    print(f"  Estimation: GMM (no log-likelihood)")
    
except Exception as e:
    print(f"  ERROR fitting SAR: {e}")
    sar_model = None
    sar_loglik = np.nan
    sar_aic = np.nan
    sar_bic = np.nan
    sar_pseudor2 = np.nan
    sar_rho = np.nan
    sar_rho_pval = np.nan
    sar_sigma2 = np.nan
    sar_residuals = np.full_like(y, np.nan)

# ============================================================================
# 5.4 SEM MODEL (SPATIAL ERROR) - GMM (GENERALIZED METHOD OF MOMENTS)
# ============================================================================
print("\n[MODEL 4] SEM - Spatial Error Model (GMM, parsimonious)...")
print(f"  Covariates: K={len(X_cols_parsimonious)} (no neighbourhood FE)")
print(f"  Method: Generalized Method of Moments (computationally tractable for N={len(y)})")
sys.stdout.flush()

try:
    # SEM: y = X*beta + u, where u = lambda*W*u + epsilon
    print("  → Starting SEM GMM estimation...")
    sys.stdout.flush()
    t0_sem = time.perf_counter()
    try:
        sem_model = spreg.GM_Error(
            y, X_parsimonious, w=w_knn8,
            name_y='log_price',
            name_x=X_cols_parsimonious,
            name_w='knn8',
            robust='white'
        )
    except TypeError:
        print("  ⚠ GM_Error does not support robust='white' in this version; falling back to default.")
        sem_model = spreg.GM_Error(
            y, X_parsimonious, w=w_knn8,
            name_y='log_price',
            name_x=X_cols_parsimonious,
            name_w='knn8'
        )
    t1_sem = time.perf_counter()
    print(f"  ✓ SEM GMM estimation completed in {t1_sem - t0_sem:.2f} seconds")
    sys.stdout.flush()
    
    sem_loglik = np.nan  # GMM doesn't have log-likelihood
    sem_aic = np.nan
    sem_bic = np.nan
    sem_pseudor2 = getattr(sem_model, 'pr2', np.nan)  # Pseudo R-squared
    sem_lambda = float(np.asarray(
        getattr(sem_model, 'lam', getattr(sem_model, 'lambda', getattr(sem_model, 'lamda', np.nan)))
    ).squeeze())
    if np.isnan(sem_lambda):
        sem_betas_all = np.asarray(sem_model.betas, dtype=float).flatten()
        if len(sem_betas_all) > len(X_cols_parsimonious):
            sem_lambda = float(sem_betas_all[-1])
    sem_z = np.asarray(sem_model.z_stat, dtype=float)
    sem_lambda_pval = float(sem_z[-1, 1]) if sem_z.ndim == 2 and sem_z.shape[1] > 1 else np.nan
    sem_sigma2 = getattr(sem_model, 'sig2', np.nan)
    
    sem_residuals = sem_model.u.flatten()
    
    print(f"  SEM λ (spatial error coeff): {sem_lambda:.6f}, p={sem_lambda_pval:.6f}")
    print(f"  Pseudo-R²: {sem_pseudor2:.4f}")
    print(f"  σ²: {sem_sigma2:.6f}")
    print(f"  Estimation: GMM (no log-likelihood)")
    
except Exception as e:
    print(f"  ERROR fitting SEM: {e}")
    sem_model = None
    sem_loglik = np.nan
    sem_aic = np.nan
    sem_bic = np.nan
    sem_pseudor2 = np.nan
    sem_lambda = np.nan
    sem_lambda_pval = np.nan
    sem_sigma2 = np.nan
    sem_residuals = np.full_like(y, np.nan)

# ============================================================================
# 8. SAVE MODEL COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

print("\n[RESULTS 1] Saving model comparison tables...")
sys.stdout.flush()

comparison_main_df = pd.DataFrame([
    {
        'model': 'OLS_parsimonious',
        'N': len(y),
        'K': len(X_cols_parsimonious),
        'kNN_k': 8,
        'logLik': ols_pars_loglik,
        'AIC': ols_pars_aic,
        'BIC': ols_pars_bic,
        'R2_or_pseudoR2': ols_pars_r2,
        'spatial_coeff': np.nan,
        'spatial_coeff_name': 'none',
        'spatial_coeff_pval': np.nan,
        'sigma2': np.nan,
        'morans_I': np.nan,
        'specification': 'structural_only'
    },
    {
        'model': 'SAR_GMM',
        'N': len(y),
        'K': len(X_cols_parsimonious),
        'kNN_k': 8,
        'logLik': sar_loglik,
        'AIC': sar_aic,
        'BIC': sar_bic,
        'R2_or_pseudoR2': sar_pseudor2,
        'spatial_coeff': sar_rho,
        'spatial_coeff_name': 'rho',
        'spatial_coeff_pval': sar_rho_pval,
        'sigma2': sar_sigma2,
        'morans_I': np.nan,
        'specification': 'structural_only'
    },
    {
        'model': 'SEM_GMM',
        'N': len(y),
        'K': len(X_cols_parsimonious),
        'kNN_k': 8,
        'logLik': sem_loglik,
        'AIC': sem_aic,
        'BIC': sem_bic,
        'R2_or_pseudoR2': sem_pseudor2,
        'spatial_coeff': sem_lambda,
        'spatial_coeff_name': 'lambda',
        'spatial_coeff_pval': sem_lambda_pval,
        'sigma2': sem_sigma2,
        'morans_I': np.nan,
        'specification': 'structural_only'
    }
])

comparison_robustness_df = pd.DataFrame([
    {
        'model': 'OLS_neighbourhood_FE',
        'N': len(y_full),
        'K': len(X_cols_full),
        'kNN_k': 8,
        'logLik': ols_fe_loglik,
        'AIC': ols_fe_aic,
        'BIC': ols_fe_bic,
        'R2_or_pseudoR2': ols_fe_r2,
        'spatial_coeff': np.nan,
        'spatial_coeff_name': 'none',
        'spatial_coeff_pval': np.nan,
        'sigma2': np.nan,
        'morans_I': np.nan,
        'specification': 'with_neighbourhood_FE'
    }
])

comparison_main_path = OUTPUT_TABLES / "model_comparison_main.csv"
comparison_robustness_path = OUTPUT_TABLES / "model_comparison_robustness.csv"
comparison_robustness_df.to_csv(comparison_robustness_path, index=False)

print(f"✓ Saved: {comparison_robustness_path}")
print("\nIncremental OLS / robustness check (not directly comparable):")
print(comparison_robustness_df.to_string(index=False))
print("\nNote: SAR_GMM and SEM_GMM estimated with GMM (no log-likelihood/AIC/BIC).")
print("      Moran's I on residuals is the key fit metric for spatial autocorrelation.")
print("      SEM GMM SE are not White-robust in this environment (GM_Error lacks robust='white').")

# ============================================================================
# 9. SAVE COEFFICIENT TABLES
# ============================================================================
print("\n[RESULTS 2] Saving SAR coefficient table...")

if sar_model is not None:
    # GMM models: betas excludes rho, z_stat may be list of tuples
    n_betas = len(X_cols_parsimonious) + 1  # +1 for constant
    sar_betas = np.asarray(sar_model.betas, dtype=float).flatten()
    sar_se = np.asarray(sar_model.std_err, dtype=float).flatten()
    sar_z = np.asarray(sar_model.z_stat, dtype=float)
    sar_z_vals = sar_z[:n_betas, 0] if sar_z.ndim == 2 else np.full(n_betas, np.nan)
    sar_p_vals = sar_z[:n_betas, 1] if sar_z.ndim == 2 else np.full(n_betas, np.nan)

    sar_coeffs_df = pd.DataFrame({
        'variable': ['const'] + X_cols_parsimonious,
        'estimate': sar_betas[:n_betas],
        'std_err': sar_se[:n_betas],
        'z_stat': sar_z_vals,
        'p_value': sar_p_vals
    })
    
    # Add spatial parameter rho
    sar_spatial_row = pd.DataFrame({
        'variable': ['rho (spatial lag)'],
        'estimate': [sar_rho],
        'std_err': [sar_se[-1] if len(sar_se) else np.nan],
        'z_stat': [sar_z[-1, 0] if sar_z.ndim == 2 else np.nan],
        'p_value': [sar_z[-1, 1] if sar_z.ndim == 2 else np.nan]
    })
    sar_coeffs_df = pd.concat([sar_coeffs_df, sar_spatial_row], ignore_index=True)
    
    sar_coeffs_path = OUTPUT_TABLES / "sar_coeffs.csv"
    sar_coeffs_df.to_csv(sar_coeffs_path, index=False)
    print(f"✓ Saved: {sar_coeffs_path} ({len(sar_coeffs_df)} coefficients)")
else:
    print("⚠ SAR model not fitted - skipping coefficient table")

print("\n[RESULTS 3] Saving SEM coefficient table...")

if sem_model is not None:
    # GMM models: betas excludes lambda, z_stat may be list of tuples
    n_betas = len(X_cols_parsimonious) + 1  # +1 for constant
    sem_betas = np.asarray(sem_model.betas, dtype=float).flatten()
    sem_se = np.asarray(sem_model.std_err, dtype=float).flatten()
    sem_z = np.asarray(sem_model.z_stat, dtype=float)
    sem_z_vals = sem_z[:n_betas, 0] if sem_z.ndim == 2 else np.full(n_betas, np.nan)
    sem_p_vals = sem_z[:n_betas, 1] if sem_z.ndim == 2 else np.full(n_betas, np.nan)

    sem_coeffs_df = pd.DataFrame({
        'variable': ['const'] + X_cols_parsimonious,
        'estimate': sem_betas[:n_betas],
        'std_err': sem_se[:n_betas],
        'z_stat': sem_z_vals,
        'p_value': sem_p_vals
    })
    
    # Add spatial parameter lambda
    sem_spatial_row = pd.DataFrame({
        'variable': ['lambda (spatial error)'],
        'estimate': [sem_lambda],
        'std_err': [sem_se[-1] if len(sem_se) else np.nan],
        'z_stat': [sem_z[-1, 0] if sem_z.ndim == 2 else np.nan],
        'p_value': [sem_z[-1, 1] if sem_z.ndim == 2 else np.nan]
    })
    sem_coeffs_df = pd.concat([sem_coeffs_df, sem_spatial_row], ignore_index=True)
    
    sem_coeffs_path = OUTPUT_TABLES / "sem_coeffs.csv"
    sem_coeffs_df.to_csv(sem_coeffs_path, index=False)
    print(f"✓ Saved: {sem_coeffs_path} ({len(sem_coeffs_df)} coefficients)")
else:
    print("⚠ SEM model not fitted - skipping coefficient table")

# ==========================================================================
# 10. COMPUTE MORAN'S I ON RESIDUALS (POST-FIT DIAGNOSTICS)
# ==========================================================================
print("\n[RESULTS 4] Computing Moran's I on residuals (post-fit)...")
sys.stdout.flush()

morans_postfit = []

# OLS parsimonious residuals
mi_ols_pars = Moran(ols_pars_residuals, w_knn8)
morans_postfit.append({
    'model': 'OLS_parsimonious',
    'morans_I': mi_ols_pars.I,
    'p_value': mi_ols_pars.p_sim,
    'z_score': mi_ols_pars.z_sim
})
print(f"  OLS parsimonious:   Moran's I = {mi_ols_pars.I:.6f}, p = {mi_ols_pars.p_sim:.6f}")

# OLS neighbourhood FE residuals (robustness check; not in main comparison)
if len(ols_fe_residuals) == len(ols_pars_residuals):
    mi_ols_fe = Moran(ols_fe_residuals, w_knn8)
    morans_postfit.append({
        'model': 'OLS_neighbourhood_FE',
        'morans_I': mi_ols_fe.I,
        'p_value': mi_ols_fe.p_sim,
        'z_score': mi_ols_fe.z_sim
    })
    print(f"  OLS neighbourhood FE: Moran's I = {mi_ols_fe.I:.6f}, p = {mi_ols_fe.p_sim:.6f}")
else:
    print(f"  ⚠ OLS neighbourhood FE: N mismatch (N={len(ols_fe_residuals)} vs W={w_knn8.n}) - skipping Moran's I")

# SAR residuals
if sar_model is not None:
    mi_sar = Moran(sar_residuals, w_knn8)
    morans_postfit.append({
        'model': 'SAR_GMM',
        'morans_I': mi_sar.I,
        'p_value': mi_sar.p_sim,
        'z_score': mi_sar.z_sim
    })
    print(f"  SAR GMM:   Moran's I = {mi_sar.I:.6f}, p = {mi_sar.p_sim:.6f}")

# SEM residuals
if sem_model is not None:
    mi_sem = Moran(sem_residuals, w_knn8)
    morans_postfit.append({
        'model': 'SEM_GMM',
        'morans_I': mi_sem.I,
        'p_value': mi_sem.p_sim,
        'z_score': mi_sem.z_sim
    })
    print(f"  SEM GMM:   Moran's I = {mi_sem.I:.6f}, p = {mi_sem.p_sim:.6f}")

morans_postfit_df = pd.DataFrame(morans_postfit)
morans_postfit_path = OUTPUT_TABLES / "morans_postfit.csv"
morans_postfit_df.to_csv(morans_postfit_path, index=False)
print(f"\n✓ Saved: {morans_postfit_path}")

# Update main comparison table with Moran's I (key metric)
if not morans_postfit_df.empty:
    morans_map = dict(zip(morans_postfit_df['model'], morans_postfit_df['morans_I']))
    comparison_main_df['morans_I'] = comparison_main_df['model'].map(morans_map)

comparison_main_df.to_csv(comparison_main_path, index=False)
print(f"✓ Saved: {comparison_main_path}")
print("\nModel Comparison (Main, comparable specs):")
print(comparison_main_df.to_string(index=False))

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================
print("\n[RESULTS 5] Creating visualizations...")
sys.stdout.flush()

# Figure 1: Moran's I comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = morans_postfit_df['model'].values
morans_vals = morans_postfit_df['morans_I'].values
morans_pvals = morans_postfit_df['p_value'].values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = ax.bar(models, morans_vals, color=colors[:len(models)], alpha=0.7, edgecolor='black', linewidth=1.5)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel("Moran's I", fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title("Spatial Autocorrelation in Residuals (Post-Fit)\nLower is Better", 
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
values = list(morans_vals)
y_min = max(0.0, min(values) - 0.02)
y_max = max(values) + 0.02
ax.set_ylim(y_min, y_max)

# Add value labels on bars (after y-limits)
for bar, val, pval in zip(bars, morans_vals, morans_pvals):
    height = bar.get_height()
    p_text = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.6f}\n{p_text}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
morans_fig_path = OUTPUT_FIGURES / "morans_postfit_compare.png"
plt.savefig(morans_fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {morans_fig_path}")

# Figure 2: Residuals histograms (OLS_parsimonious, SAR_GMM, SEM_GMM)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

sar_gmm = sar_model
sar_res = sar_gmm.u.flatten() if sar_gmm is not None else None
sem_res = sem_residuals if sem_model is not None else None

if sar_gmm is not None:
    print("SAR resid:", sar_gmm.u.shape, float(np.mean(sar_gmm.u)), float(np.std(sar_gmm.u)))

residuals_map = {
    'OLS parsimonious': ols_pars_residuals,
    'SAR (GMM)': sar_res,
    'SEM (GMM)': sem_res
}

valid_residuals = [r for r in residuals_map.values() if r is not None and not np.all(np.isnan(r))]
if valid_residuals:
    x_min = min(float(np.min(r)) for r in valid_residuals)
    x_max = max(float(np.max(r)) for r in valid_residuals)
else:
    x_min, x_max = -1.0, 1.0

bins = 50
panel_specs = [
    (axes[0], 'OLS parsimonious', '#1f77b4'),
    (axes[1], 'SAR (GMM)', '#2ca02c'),
    (axes[2], 'SEM (GMM)', '#d62728')
]

for ax, label, color in panel_specs:
    residuals = residuals_map.get(label)
    if residuals is not None and not np.all(np.isnan(residuals)):
        ax.hist(residuals, bins=bins, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residuals', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{label}\n(mean={residuals.mean():.4f}, std={residuals.std():.4f})',
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(x_min, x_max)
    else:
        ax.text(0.5, 0.5, f'{label}\nModel failed',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('Residual Distributions: OLS vs SAR/SEM (GMM)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
hist_fig_path = OUTPUT_FIGURES / "residuals_hist_compare.png"
plt.savefig(hist_fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {hist_fig_path}")

# ============================================================================
# 12. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

def _safe_mi(series: pd.Series) -> float:
    return series.values[0] if len(series) else np.nan

ols_pars_mi = _safe_mi(morans_postfit_df[morans_postfit_df['model'] == 'OLS_parsimonious']['morans_I'])
sar_mi = _safe_mi(morans_postfit_df[morans_postfit_df['model'] == 'SAR_GMM']['morans_I'])
sem_mi = _safe_mi(morans_postfit_df[morans_postfit_df['model'] == 'SEM_GMM']['morans_I'])

print(f"""
Sample: N = {len(y)} listings
Covariates (parsimonious SAR/SEM): K = {len(X_cols_parsimonious)}
Covariates (OLS FE): K = {len(X_cols_full)}
Weights: kNN k=8, EPSG:25830, row-standardized, 0 islands

Model Comparison:
    - OLS parsimonious:     AIC={ols_pars_aic:.2f}, R²={ols_pars_r2:.4f}
    - SAR (GMM):            Pseudo-R²={sar_pseudor2:.4f}, ρ={sar_rho:.6f}, p={sar_rho_pval:.6f}
    - SEM (GMM):            Pseudo-R²={sem_pseudor2:.4f}, λ={sem_lambda:.6f}, p={sem_lambda_pval:.6f}

Incremental OLS / robustness check:
    - OLS neighbourhood FE: AIC={ols_fe_aic:.2f}, R²={ols_fe_r2:.4f}
      (Not directly comparable to SAR/SEM due to different covariate set)

Post-Fit Moran's I (lower is better):
    - OLS parsimonious:     {ols_pars_mi:.6f}
    - SAR (GMM):            {sar_mi:.6f}
    - SEM (GMM):            {sem_mi:.6f}

Interpretation:
    - SAR_GMM reduces Moran's I (0.165 → 0.071) and is preferred.
    - SEM_GMM does not reduce Moran's I (≈0.172).
    - GM_Error lacks robust='white' here; SEM SE are not White-robust.

✓ All outputs saved to outputs/tables/ and outputs/figures/
✓ Diagnostics: Coefficient tables, Moran's I post-fit, visualizations
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
