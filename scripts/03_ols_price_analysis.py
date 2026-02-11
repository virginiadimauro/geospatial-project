#!/usr/bin/env python
"""
OLS analysis: log_price estimation with hedonic model
Model A: Property + Host characteristics
Model B: Model A + Location (distance to CBD, neighbourhood dummies)
"""

# NOTE: uses model_sample.parquet for consistent N (centralized data-prep)

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys

# ensure repo root on path for `src` package imports when running as script
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

# Statsmodels
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from src.prep import load_listings_processed, build_model_df, get_y_X

# ============================================================================
# 0. SETUP PATHS
# ============================================================================
PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("=" * 80)
print("OLS PRICE ANALYSIS: HEDONIC MODEL")
print("=" * 80)

# Use centralized preprocessing
df_raw = load_listings_processed(DATA_PROCESSED / "listings_clean.parquet")
# build_model_df will also save model_sample.parquet when instructed by other scripts
model_sample_path = DATA_PROCESSED / "model_sample.parquet"
model_df = build_model_df(df_raw, out_path=model_sample_path)

print(f"\n[DATA] Loaded {len(model_df)} listings after centralized prep (model_sample)")

# model_df already contains covariates, dummies and log_price
df = model_df.reset_index(drop=True).copy()

# ============================================================================
# 2. CREATE LOCATION FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("LOCATION FEATURES")
print("=" * 80)

# Puerta del Sol coordinates
PUERTA_DEL_SOL_LAT, PUERTA_DEL_SOL_LON = 40.4169, -3.7035

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km using Haversine formula."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['dist_cbd_km'] = haversine_distance(
    df['latitude'], df['longitude'],
    PUERTA_DEL_SOL_LAT, PUERTA_DEL_SOL_LON
)

print(f"\nDistance to CBD (Puerta del Sol):")
print(f"  Min:    {df['dist_cbd_km'].min():.2f} km")
print(f"  Median: {df['dist_cbd_km'].median():.2f} km")
print(f"  Max:    {df['dist_cbd_km'].max():.2f} km")

# ============================================================================
# 3. PREPARE OLS DATA
# ============================================================================
print("\n" + "=" * 80)
print("MODEL PREPARATION")
print("=" * 80)

# Create dummy variables for categorical variables if not already present
df_model = df.copy()
if 'room_type' in df_model.columns:
    df_model = pd.get_dummies(df_model, columns=['room_type'], drop_first=True, dtype=int)

if 'neighbourhood_cleansed' in df_model.columns and not any(col.startswith('neigh_') for col in df_model.columns):
    neighbourhood_dummies = pd.get_dummies(df_model['neighbourhood_cleansed'], prefix='neigh', drop_first=True, dtype=int)
    df_model = pd.concat([df_model, neighbourhood_dummies], axis=1)

# Define variable sets
property_vars = ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'minimum_nights']
host_vars = ['host_is_superhost', 'host_listings_count', 'number_of_reviews', 
             'review_scores_rating', 'instant_bookable']
room_type_vars = [col for col in df_model.columns if col.startswith('room_type_')]
neighbourhood_vars = [col for col in df_model.columns if col.startswith('neigh_')]

# Model A: Property + Host + Room type
model_a_vars = property_vars + host_vars + room_type_vars
print(f"\nModel A variables ({len(model_a_vars)}):")
print(f"  Property: {property_vars}")
print(f"  Host:     {host_vars}")
print(f"  Room type: {room_type_vars}")

# Model B: Model A + Location
model_b_vars = model_a_vars + ['dist_cbd_km'] + neighbourhood_vars
print(f"\nModel B variables ({len(model_b_vars)}):")
print(f"  Model A vars: {len(model_a_vars)}")
print(f"  + Distance to CBD: 1")
print(f"  + Neighbourhoods: {len(neighbourhood_vars)}")

# ============================================================================
# 4. FIT MODELS
# ============================================================================
print("\n" + "=" * 80)
print("OLS ESTIMATION")
print("=" * 80)

# Target
y = df_model['log_price'].astype(float)

# Ensure all X variables are numeric
X_a = df_model[model_a_vars].astype(float)
X_b = df_model[model_b_vars].astype(float)

# Model A
X_a_const = add_constant(X_a)
model_a = sm.OLS(y, X_a_const).fit(cov_type='HC1')  # HC1 robust errors

# Model B
X_b_const = add_constant(X_b)
model_b = sm.OLS(y, X_b_const).fit(cov_type='HC1')  # HC1 robust errors

print("\n" + model_a.summary().as_text())
print("\n" + model_b.summary().as_text())

# ============================================================================
# 5. MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison_data = {
    'Metric': ['N', 'R-squared', 'Adj. R-squared', 'AIC', 'BIC', 'Residual Std Err'],
    'Model A': [
        len(y),
        f"{model_a.rsquared:.4f}",
        f"{model_a.rsquared_adj:.4f}",
        f"{model_a.aic:.2f}",
        f"{model_a.bic:.2f}",
        f"{np.sqrt(model_a.mse_resid):.4f}"
    ],
    'Model B': [
        len(y),
        f"{model_b.rsquared:.4f}",
        f"{model_b.rsquared_adj:.4f}",
        f"{model_b.aic:.2f}",
        f"{model_b.bic:.2f}",
        f"{np.sqrt(model_b.mse_resid):.4f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(OUTPUT_TABLES / "ols_comparison.csv", index=False)
print(f"\n✓ Saved: {OUTPUT_TABLES / 'ols_comparison.csv'}")

# ============================================================================
# 6. EXTRACT AND SAVE COEFFICIENTS
# ============================================================================
# Model A coefficients
coeff_a = pd.DataFrame({
    'variable': model_a.params.index,
    'coefficient': model_a.params.values,
    'std_err': model_a.bse.values,
    't_stat': model_a.tvalues.values,
    'p_value': model_a.pvalues.values
}).reset_index(drop=True)
coeff_a.to_csv(OUTPUT_TABLES / "ols_coeffs_modelA.csv", index=False)
print(f"✓ Saved: {OUTPUT_TABLES / 'ols_coeffs_modelA.csv'}")

# Model B coefficients
coeff_b = pd.DataFrame({
    'variable': model_b.params.index,
    'coefficient': model_b.params.values,
    'std_err': model_b.bse.values,
    't_stat': model_b.tvalues.values,
    'p_value': model_b.pvalues.values
}).reset_index(drop=True)
coeff_b.to_csv(OUTPUT_TABLES / "ols_coeffs_modelB.csv", index=False)
print(f"✓ Saved: {OUTPUT_TABLES / 'ols_coeffs_modelB.csv'}")

# Interpret distance coefficient (CBD)
dist_cbd_coeff = model_b.params['dist_cbd_km']
dist_cbd_pct_change = 100 * (np.exp(dist_cbd_coeff) - 1)
print(f"\n[INTERPRETATION] Distance to CBD coefficient (Model B):")
print(f"  Coefficient: {dist_cbd_coeff:.6f}")
print(f"  % price change per km from CBD: {dist_cbd_pct_change:.4f}%")

# ============================================================================
# 7. QUALITY CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("QUALITY CHECKS (Model B)")
print("=" * 80)

# Check for inf/nan in log_price
n_inf = np.isinf(y).sum()
n_nan = y.isna().sum()
print(f"\n[Target variable] log_price:")
print(f"  inf values: {n_inf}")
print(f"  nan values: {n_nan}")

# Residuals
residuals_b = model_b.resid

print(f"\nResiduals (Model B):")
print(f"  Mean: {residuals_b.mean():.6f}")
print(f"  Std: {residuals_b.std():.6f}")
print(f"  Min: {residuals_b.min():.4f}")
print(f"  Max: {residuals_b.max():.4f}")

# VIF for numeric covariates
print(f"\n[VIF] Top 10 collinearity issues (numeric covariates):")
numeric_vars_idx = [i for i, var in enumerate(X_b.columns) if X_b[var].dtype in [int, float]]
vif_values = []
for i in numeric_vars_idx:
    try:
        vif_val = vif(X_b.values, i)
        vif_values.append((X_b.columns[i], vif_val))
    except:
        pass

vif_values.sort(key=lambda x: x[1], reverse=True)
for var, vif_val in vif_values[:10]:
    print(f"  {var:30s}: {vif_val:8.2f}")

# ============================================================================
# 8. VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("CREATING FIGURES")
print("=" * 80)

# Figure 1: Residuals histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(residuals_b, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Residuals', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of Residuals (Model B)', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# Q-Q plot
stats.probplot(residuals_b, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "residuals_hist_modelB.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_FIGURES / 'residuals_hist_modelB.png'}")
plt.close()

# Figure 2: Fitted vs Residuals
fig, ax = plt.subplots(figsize=(12, 6))
fitted_values = model_b.fittedvalues
ax.scatter(fitted_values, residuals_b, alpha=0.5, s=30)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero line')
ax.set_xlabel('Fitted Values', fontsize=11)
ax.set_ylabel('Residuals', fontsize=11)
ax.set_title('Fitted Values vs Residuals (Model B)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURES / "residuals_fitted_modelB.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_FIGURES / 'residuals_fitted_modelB.png'}")
plt.close()

print("\n" + "=" * 80)
print("✓ ANALYSIS COMPLETE")
print("=" * 80)
