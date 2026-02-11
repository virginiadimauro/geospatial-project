#!/usr/bin/env python
"""
Spatial Econometric Models: SAR vs SEM vs OLS Comparison
=========================================================

Estimates and compares three models on identical sample (N=15,641):
1. OLS Model B (baseline hedonic model)
2. SAR - Spatial Autoregressive (spatial lag of y)
3. SEM - Spatial Error Model (spatial lag of residuals)

Uses identical data preparation, X variables, and weights matrix
as scripts 03 (OLS), 05 (LM tests), and 06 (Moran consistency check).

Output:
- outputs/tables/spatial_models_comparison.csv (AIC, BIC, pseudoR2, spatial params)
- outputs/tables/sar_coeffs.csv (SAR coefficient table)
- outputs/tables/sem_coeffs.csv (SEM coefficient table)
- outputs/tables/morans_postfit.csv (Moran's I on residuals for all 3 models)
- outputs/figures/morans_postfit_compare.png (Moran comparison)
- outputs/figures/residuals_hist_compare.png (residual distributions)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import warnings
import sys

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
# 2. PREPARE COVARIATES (EXACT COPY FROM SCRIPT 05)
# ============================================================================
print("[STEP 2] Preparing covariates (exact from script 05)...")
sys.stdout.flush()

# Keep track of original indices

# Basic sanity checks BEFORE building y, X and weights
print(f"  → model_sample initial rows: {len(df_model)}")
sys.stdout.flush()

exclude = {'listing_id','latitude','longitude','log_price','price_numeric','price_winsorized','geometry'}

# Require key columns to be present in `df_model` (no fallbacks)
required_cols = ['log_price']
coord_cols = ['geometry'] if 'geometry' in df_model.columns else ['latitude', 'longitude']
missing = [c for c in required_cols + coord_cols if c not in df_model.columns]
if missing:
    raise ValueError(f"Missing column(s) in model_sample.parquet: {', '.join(missing)}")

# Prepare X and y from model_df (use only columns already present)
X_cols_available = [c for c in df_model.columns if c not in exclude]
# Include coordinates in the dropna selection to ensure alignment
coord_cols_to_include = []
if 'geometry' in df_model.columns:
    coord_cols_to_include = ['geometry']
else:
    coord_cols_to_include = ['latitude', 'longitude'] if all(c in df_model.columns for c in ['latitude', 'longitude']) else []

df_model = df_model[X_cols_available + ['log_price'] + coord_cols_to_include].dropna().reset_index(drop=True)

print(f"  → model_sample after trimming rows: {len(df_model)}")
sys.stdout.flush()

# Preserve coords from the final trimmed df_model so spatial weights match y/X
if 'geometry' in df_model.columns:
    coords_df = gpd.GeoDataFrame(df_model[[c for c in ['geometry','listing_id'] if c in df_model.columns]]).copy()
else:
    coords_df = df_model[[c for c in ['longitude','latitude','listing_id'] if c in df_model.columns]].copy()

df_model = df_model.reset_index(drop=True).copy()

df_model['_original_idx'] = df_model.index
model_indices = df_model['_original_idx'].values

y = df_model['log_price'].values
X = df_model[X_cols_available].astype(float).values

print(f"Model shape: y={y.shape}, X={X.shape}, N={len(y)}")

# ============================================================================
# 3. CREATE SPATIAL WEIGHTS (EXACT FROM SCRIPT 05)
# ============================================================================
print("\n[STEP 3] Creating spatial weights (k-NN k=8, EPSG:25830)...")
sys.stdout.flush()

# Build GeoDataFrame from preserved coordinates
if 'geometry' in coords_df.columns:
    gdf = gpd.GeoDataFrame(coords_df.assign(_original_idx=coords_df.index), geometry='geometry', crs='EPSG:4326')
else:
    gdf = gpd.GeoDataFrame(
        coords_df.assign(_original_idx=coords_df.index),
        geometry=gpd.points_from_xy(coords_df['longitude'], coords_df['latitude']),
        crs='EPSG:4326'
    )

gdf_metric = gdf.to_crs('EPSG:25830')

# Create kNN weights
w_knn8 = KNN.from_dataframe(gdf_metric, k=8)
w_knn8.transform = 'r'  # Row-standardize

print(f"Weights: {w_knn8.n} obs, {len(w_knn8.islands)} islands, avg neighbors=8.00")

# ============================================================================
# 4. QUALITY CHECKS
# ============================================================================
print("\n[STEP 4] Quality checks...")
sys.stdout.flush()

assert len(y) == X.shape[0] == w_knn8.n, f"Length mismatch: y={len(y)}, X={X.shape[0]}, W={w_knn8.n}"
assert not np.any(np.isnan(y)) and not np.any(np.isinf(y)), "NaN/inf in y"
assert not np.any(np.isnan(X)) and not np.any(np.isinf(X)), "NaN/inf in X"
assert len(w_knn8.islands) == 0, f"Islands detected: {len(w_knn8.islands)}"

print("✓ All quality checks passed")
print(f"✓ N={len(y)}, X shape={X.shape[0]} x {X.shape[1]}")
print(f"✓ W row-standardized, {len(w_knn8.islands)} islands")

# ============================================================================
# 5. FIT OLS MODEL B (BASELINE)
# ============================================================================
print("\n" + "=" * 80)
print("FITTING MODELS")
print("=" * 80)

print("\n[MODEL 1] OLS Model B (baseline)...")
sys.stdout.flush()

X_const = add_constant(X)
ols_model = sm.OLS(y, X_const)
ols_results = ols_model.fit(cov_type='HC1')

ols_residuals = ols_results.resid
if hasattr(ols_residuals, 'values'):
    ols_residuals = ols_residuals.values

ols_loglik = ols_results.llf
ols_aic = ols_results.aic
ols_bic = ols_results.bic
ols_r2 = ols_results.rsquared

print(f"  OLS R²: {ols_r2:.4f}")
print(f"  Log-likelihood: {ols_loglik:.4f}")
print(f"  AIC: {ols_aic:.4f}, BIC: {ols_bic:.4f}")

# ============================================================================
# 6. FIT SAR MODEL (SPATIAL LAG)
# ============================================================================
print("\n[MODEL 2] SAR - Spatial Autoregressive Model (ML)...")
sys.stdout.flush()

try:
    # SAR: y = rho*W*y + X*beta + epsilon
    sar_model = spreg.ML_Lag(
        y, X, w=w_knn8,
        name_y='log_price',
        name_x=X_cols_available,
        name_w='knn8'
    )
    
    sar_loglik = sar_model.loglik
    sar_aic = sar_model.aic
    sar_bic = sar_model.bic
    sar_pseudor2 = sar_model.pseudoR2
    sar_rho = sar_model.rho[0][0]
    sar_rho_pval = sar_model.rho[0][1]
    sar_sigma2 = sar_model.sigma2
    
    sar_residuals = sar_model.u
    
    print(f"  SAR ρ (spatial lag coeff): {sar_rho:.6f}, p={sar_rho_pval:.6f}")
    print(f"  Pseudo-R²: {sar_pseudor2:.4f}")
    print(f"  Log-likelihood: {sar_loglik:.4f}")
    print(f"  AIC: {sar_aic:.4f}, BIC: {sar_bic:.4f}")
    print(f"  σ²: {sar_sigma2:.6f}")
    
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
# 7. FIT SEM MODEL (SPATIAL ERROR)
# ============================================================================
print("\n[MODEL 3] SEM - Spatial Error Model (ML)...")
sys.stdout.flush()

try:
    # SEM: y = X*beta + u, where u = lambda*W*u + epsilon
    sem_model = spreg.ML_Error(
        y, X, w=w_knn8,
        name_y='log_price',
        name_x=X_cols_available,
        name_w='knn8'
    )
    
    sem_loglik = sem_model.loglik
    sem_aic = sem_model.aic
    sem_bic = sem_model.bic
    sem_pseudor2 = sem_model.pseudoR2
    sem_lambda = sem_model.lam[0][0]
    sem_lambda_pval = sem_model.lam[0][1]
    sem_sigma2 = sem_model.sigma2
    
    sem_residuals = sem_model.u
    
    print(f"  SEM λ (spatial error coeff): {sem_lambda:.6f}, p={sem_lambda_pval:.6f}")
    print(f"  Pseudo-R²: {sem_pseudor2:.4f}")
    print(f"  Log-likelihood: {sem_loglik:.4f}")
    print(f"  AIC: {sem_aic:.4f}, BIC: {sem_bic:.4f}")
    print(f"  σ²: {sem_sigma2:.6f}")
    
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

print("\n[RESULTS 1] Saving model comparison table...")
sys.stdout.flush()

comparison_df = pd.DataFrame([
    {
        'model': 'OLS',
        'N': len(y),
        'kNN_k': 8,
        'logLik': ols_loglik,
        'AIC': ols_aic,
        'BIC': ols_bic,
        'pseudoR2': ols_r2,
        'spatial_coeff': np.nan,
        'spatial_coeff_name': 'none',
        'spatial_coeff_pval': np.nan,
        'sigma2': np.nan
    },
    {
        'model': 'SAR',
        'N': len(y),
        'kNN_k': 8,
        'logLik': sar_loglik,
        'AIC': sar_aic,
        'BIC': sar_bic,
        'pseudoR2': sar_pseudor2,
        'spatial_coeff': sar_rho,
        'spatial_coeff_name': 'rho',
        'spatial_coeff_pval': sar_rho_pval,
        'sigma2': sar_sigma2
    },
    {
        'model': 'SEM',
        'N': len(y),
        'kNN_k': 8,
        'logLik': sem_loglik,
        'AIC': sem_aic,
        'BIC': sem_bic,
        'pseudoR2': sem_pseudor2,
        'spatial_coeff': sem_lambda,
        'spatial_coeff_name': 'lambda',
        'spatial_coeff_pval': sem_lambda_pval,
        'sigma2': sem_sigma2
    }
])

comparison_path = OUTPUT_TABLES / "spatial_models_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"✓ Saved: {comparison_path}")
print("\nComparison Table:")
print(comparison_df.to_string(index=False))

# ============================================================================
# 9. SAVE COEFFICIENT TABLES
# ============================================================================
print("\n[RESULTS 2] Saving SAR coefficient table...")

if sar_model is not None:
    sar_coeffs_df = pd.DataFrame({
        'variable': ['const'] + X_cols_available,
        'estimate': np.concatenate([[sar_model.betas[0, 0]], sar_model.betas[1:, 0]]),
        'std_err': np.concatenate([[sar_model.se_betas[0, 0]], sar_model.se_betas[1:, 0]]),
        'z_stat': np.concatenate([[sar_model.z_stat[0, 0]], sar_model.z_stat[1:, 0]]),
        'p_value': np.concatenate([[sar_model.betas_red[0, 1]], sar_model.betas_red[1:, 1]])
    })
    
    sar_coeffs_path = OUTPUT_TABLES / "sar_coeffs.csv"
    sar_coeffs_df.to_csv(sar_coeffs_path, index=False)
    print(f"✓ Saved: {sar_coeffs_path} ({len(sar_coeffs_df)} coefficients)")
else:
    print("⚠ SAR model not fitted - skipping coefficient table")

print("\n[RESULTS 3] Saving SEM coefficient table...")

if sem_model is not None:
    sem_coeffs_df = pd.DataFrame({
        'variable': ['const'] + X_cols_available,
        'estimate': np.concatenate([[sem_model.betas[0, 0]], sem_model.betas[1:, 0]]),
        'std_err': np.concatenate([[sem_model.se_betas[0, 0]], sem_model.se_betas[1:, 0]]),
        'z_stat': np.concatenate([[sem_model.z_stat[0, 0]], sem_model.z_stat[1:, 0]]),
        'p_value': np.concatenate([[sem_model.betas_red[0, 1]], sem_model.betas_red[1:, 1]])
    })
    
    sem_coeffs_path = OUTPUT_TABLES / "sem_coeffs.csv"
    sem_coeffs_df.to_csv(sem_coeffs_path, index=False)
    print(f"✓ Saved: {sem_coeffs_path} ({len(sem_coeffs_df)} coefficients)")
else:
    print("⚠ SEM model not fitted - skipping coefficient table")

# ============================================================================
# 10. COMPUTE MORAN'S I ON RESIDUALS (POST-FIT DIAGNOSTICS)
# ============================================================================
print("\n[RESULTS 4] Computing Moran's I on residuals (post-fit)...")
sys.stdout.flush()

morans_postfit = []

# OLS residuals
mi_ols = Moran(ols_residuals, w_knn8)
morans_postfit.append({
    'model': 'OLS',
    'morans_I': mi_ols.I,
    'p_value': mi_ols.p_sim,
    'z_score': mi_ols.z_sim
})
print(f"  OLS:   Moran's I = {mi_ols.I:.6f}, p = {mi_ols.p_sim:.6f}")

# SAR residuals
if sar_model is not None:
    mi_sar = Moran(sar_residuals, w_knn8)
    morans_postfit.append({
        'model': 'SAR',
        'morans_I': mi_sar.I,
        'p_value': mi_sar.p_sim,
        'z_score': mi_sar.z_sim
    })
    print(f"  SAR:   Moran's I = {mi_sar.I:.6f}, p = {mi_sar.p_sim:.6f}")

# SEM residuals
if sem_model is not None:
    mi_sem = Moran(sem_residuals, w_knn8)
    morans_postfit.append({
        'model': 'SEM',
        'morans_I': mi_sem.I,
        'p_value': mi_sem.p_sim,
        'z_score': mi_sem.z_sim
    })
    print(f"  SEM:   Moran's I = {mi_sem.I:.6f}, p = {mi_sem.p_sim:.6f}")

morans_postfit_df = pd.DataFrame(morans_postfit)
morans_postfit_path = OUTPUT_TABLES / "morans_postfit.csv"
morans_postfit_df.to_csv(morans_postfit_path, index=False)
print(f"\n✓ Saved: {morans_postfit_path}")

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================
print("\n[RESULTS 5] Creating visualizations...")
sys.stdout.flush()

# Figure 1: Moran's I comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = morans_postfit_df['model'].values
morans_vals = morans_postfit_df['morans_I'].values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

bars = ax.bar(models, morans_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, morans_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.6f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel("Moran's I", fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_title("Spatial Autocorrelation in Residuals (Post-Fit)\nLower is Better", 
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([min(morans_vals) * 1.2, max(morans_vals) * 1.2])

plt.tight_layout()
morans_fig_path = OUTPUT_FIGURES / "morans_postfit_compare.png"
plt.savefig(morans_fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {morans_fig_path}")

# Figure 2: Residuals histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

residuals_dict = {
    'OLS': ols_residuals,
    'SAR': sar_residuals if sar_model is not None else None,
    'SEM': sem_residuals if sem_model is not None else None
}
colors_hist = ['#1f77b4', '#ff7f0e', '#2ca02c']

for ax, (model_name, residuals), color in zip(axes, residuals_dict.items(), colors_hist):
    if residuals is not None and not np.all(np.isnan(residuals)):
        ax.hist(residuals, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='x=0')
        ax.set_xlabel('Residuals', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{model_name}\n(mean={residuals.mean():.4f}, std={residuals.std():.4f})',
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'{model_name}\nModel failed', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('Residual Distributions: OLS vs SAR vs SEM', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
hist_fig_path = OUTPUT_FIGURES / "residuals_hist_compare.png"
plt.savefig(hist_fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {hist_fig_path}")

# ============================================================================
# 12. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

print(f"""
Sample: N = {len(y)} listings (identical to OLS Model B, script 03)
Covariates: {X.shape[1]} variables (identical to scripts 03, 05, 06)
Weights: kNN k=8, EPSG:25830, row-standardized, 0 islands

Model Comparison:
  - OLS Model B (baseline):    AIC={ols_aic:.2f}, R²={ols_r2:.4f}
  - SAR Model:                 AIC={sar_aic:.2f}, ρ={sar_rho:.6f}, p={sar_rho_pval:.6f}
  - SEM Model:                 AIC={sem_aic:.2f}, λ={sem_lambda:.6f}, p={sem_lambda_pval:.6f}

Post-Fit Moran's I (lower is better):
  - OLS: {morans_postfit_df[morans_postfit_df['model']=='OLS']['morans_I'].values[0]:.6f}
  - SAR: {morans_postfit_df[morans_postfit_df['model']=='SAR']['morans_I'].values[0]:.6f}
  - SEM: {morans_postfit_df[morans_postfit_df['model']=='SEM']['morans_I'].values[0]:.6f}

✓ All outputs saved to outputs/tables/ and outputs/figures/
✓ Diagnostics: Coefficient tables, Moran's I post-fit, visualizations
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
