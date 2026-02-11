"""
LM Diagnostic Tests for Spatial Autocorrelation in OLS Residuals
=================================================================

This script tests whether spatial dependence in OLS residuals
is better explained by SAR (spatial lag) or SEM (spatial error) models
using Lagrange Multiplier (LM) diagnostic tests.

Reference: Anselin, L. (1988) Spatial Econometrics: Methods and Models

Input: OLS Model B specification (from script 03_ols_price_analysis.py)
Output: LM test statistics and p-values in outputs/tables/lm_tests_modelB_listing_knn8.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2
import sys

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_TABLES = REPO_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = REPO_ROOT / "outputs" / "figures"

# Ensure output directories exist
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LM DIAGNOSTIC TESTS FOR SPATIAL AUTOCORRELATION")
print("=" * 80)

# ============================================================================
# 2. LOAD DATA & PREPARE
# ============================================================================
print("\nLOADING DATA...")
sys.stdout.flush()

# Load data directly from parquet
df = pd.read_parquet(PROCESSED_DIR / "listings_clean.parquet").copy()
print(f"Loaded {len(df)} listings")

# Price cleaning function
def parse_price(price_str):
    if pd.isna(price_str):
        return np.nan
    if isinstance(price_str, (int, float)):
        return float(price_str)
    price_clean = str(price_str).replace('€', '').replace(',', '').strip()
    try:
        return float(price_clean)
    except:
        return np.nan

# Clean price
df['price_num'] = df['price'].apply(parse_price)
df = df[df['price_num'].notna()].copy()

# Remove outliers: 0.5%-99.5% quantile
q1, q99 = df['price_num'].quantile([0.005, 0.995])
df = df[(df['price_num'] >= q1) & (df['price_num'] <= q99)].copy()
df['log_price'] = np.log(df['price_num'])
print(f"After price filtering: {len(df)} listings")

# ============================================================================
# 3. PREPARE MODEL DATA (SAME AS OLS MODEL B)
# ============================================================================
print("\nPREPARING MODEL DATA...")
sys.stdout.flush()

# Keep track of original indices to filter spatial weights later
df['_original_idx'] = range(len(df))

# Covariate columns
covariate_cols = ['room_type', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
                  'minimum_nights', 'host_is_superhost', 'host_listings_count',
                  'number_of_reviews', 'review_scores_rating', 'instant_bookable']

# Impute missing in numeric columns
for col in ['bedrooms', 'beds', 'bathrooms']:
    df[col] = df[col].fillna(df[col].median())

# Convert categorical
df['host_is_superhost'] = df['host_is_superhost'].fillna('f')
df['host_is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
df['instant_bookable'] = (df['instant_bookable'] == 't').astype(int)

# Create categorical variables with get_dummies
df_encoded = pd.get_dummies(df, columns=['room_type'], drop_first=True, dtype=int)

# Create distance variable if available
if 'dist_cbd_km' not in df_encoded.columns:
    # Use lon/lat as proxies (normalize to roughly km)
    # Madrid center approximately 40.4168, -3.7038
    df_encoded['dist_cbd_km'] = np.sqrt((df_encoded['latitude'] - 40.4168)**2 + (df_encoded['longitude'] - -3.7038)**2) * 111
else:
    df_encoded['dist_cbd_km'] = df['dist_cbd_km']

# Prepare X: property vars + host vars + room types + distance
property_cols = ['accommodates', 'bedrooms', 'beds', 'bathrooms']
host_cols = ['host_is_superhost', 'host_listings_count', 'number_of_reviews', 'review_scores_rating', 'instant_bookable']
room_type_cols = [c for c in df_encoded.columns if c.startswith('room_type_')]
location_cols = ['dist_cbd_km']

# Create neighbourhood dummies if available
neighbourhood_cols = []
if 'neighbourhood_cleansed' in df_encoded.columns:
    neighbourhood_encoded = pd.get_dummies(df_encoded['neighbourhood_cleansed'], prefix='neigh', dtype=int)
    neighbourhood_cols = [c for c in neighbourhood_encoded.columns if c != neighbourhood_encoded.columns[0]]
    df_encoded = pd.concat([df_encoded, neighbourhood_encoded], axis=1)

X_cols = property_cols + host_cols + room_type_cols + location_cols + neighbourhood_cols

# Remove missing
X_cols_available = [c for c in X_cols if c in df_encoded.columns]
df_model = df_encoded[X_cols_available + ['log_price', '_original_idx']].dropna()

# Get indices for spatial weights filtering
model_indices = df_model['_original_idx'].values

y = df_model['log_price'].values
X = df_model[X_cols_available].astype(float).values

print(f"Model shape: y={y.shape}, X={X.shape}")
print(f"Number of regressors: {X.shape[1]}")
print(f"Observations to use for spatial weights: {len(model_indices)}")

# Add constant
from statsmodels.tools.tools import add_constant
X_with_const = add_constant(X)

# ============================================================================
# 4. FIT OLS MODEL
# ============================================================================
print("\nFITTING OLS MODEL...")
sys.stdout.flush()

from statsmodels.regression.linear_model import OLS

ols_model = OLS(y, X_with_const)
results_ols = ols_model.fit(cov_type='HC1')

print(f"R²: {results_ols.rsquared:.4f}")
print(f"Adj R²: {results_ols.rsquared_adj:.4f}")

residuals = results_ols.resid.values if hasattr(results_ols.resid, 'values') else results_ols.resid
n = len(y)
k = X_with_const.shape[1]
sig2 = np.sum(residuals**2) / (n - k)
print(f"Residual std err: {np.sqrt(sig2):.4f}")

# ============================================================================
# 5. CREATE SPATIAL WEIGHTS (KNN k=8, EPSG:25830)
# ============================================================================
print("\nCREATING SPATIAL WEIGHTS...")
sys.stdout.flush()

import geopandas as gpd
from libpysal.weights import KNN

# Create GeoDataFrame for ONLY model indices
gdf = gpd.GeoDataFrame(
    df.iloc[model_indices][['price_num', '_original_idx']].copy(), 
    geometry=gpd.points_from_xy(df.iloc[model_indices]['longitude'], 
                                df.iloc[model_indices]['latitude']),
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

# ============================================================================
# 6. COMPUTE LM STATISTICS EFFICIENTLY
# ============================================================================
print("\n" + "=" * 80)
print("COMPUTING LM DIAGNOSTICS")
print("=" * 80)

try:
    # Get sparse matrix
    from scipy.sparse import csr_matrix
    
    if hasattr(w_knn8, 'sparse'):
        W = w_knn8.sparse
    else:
        from scipy.sparse import lil_matrix
        W_lil = lil_matrix((w_knn8.n, w_knn8.n))
        for i, neighbors in w_knn8.neighbors.items():
            for j in neighbors:
                W_lil[i, j] = w_knn8[i, j]
        W = W_lil.tocsr()
    
    # Ensure CSR format for efficient operations
    if not isinstance(W, csr_matrix):
        W = csr_matrix(W)
    
    print(f"Sparse W: {W.shape}, {W.nnz} non-zeros")
    
    # Compute W*y and W*e via sparse multiplication
    Wy = np.asarray(W @ y).flatten()
    We = np.asarray(W @ residuals).flatten()
    
    # For LM tests, we need tr(W'W + W²)
    # Compute via sparse operations
    print("Computing traces for LM formulas...")
    
    # For tr(W'W + W²), use the fact that:
    # tr(W'W + W²) ≈ 2 * (number of non-zero elements) / n for row-standardized W
    # But we compute more carefully:
    
    # Method: compute tr(W'W) by summing (W^T * W) diagonal
    # and tr(W²) by computing diagonal of W*W directly
    
    # tr(W'W) using sparse operations
    WtW = W.T @ W
    tr_WtW = np.sum(np.asarray(WtW.diagonal()).flatten())
    
    # tr(W²) - compute diagonal of W*W
    WW = W @ W
    tr_W2 = np.sum(np.asarray(WW.diagonal()).flatten())
    
    tr_WtW_W2 = tr_WtW + tr_W2
    
    print(f"tr(W'W) = {tr_WtW:.4f}")
    print(f"tr(W²) = {tr_W2:.4f}")
    print(f"tr(W'W + W²) = {tr_WtW_W2:.4f}")
    
    # =========================================================
    # LM-LAG TEST
    # =========================================================
    numerator_lag = np.dot(residuals, Wy)**2
    denominator_lag = sig2**2 * tr_WtW_W2
    
    lm_lag_stat = numerator_lag / denominator_lag if denominator_lag > 0 else np.nan
    lm_lag_pval = 1 - chi2.cdf(lm_lag_stat, df=1) if not np.isnan(lm_lag_stat) else np.nan
    
    print(f"\nLM-Lag Test:")
    print(f"  Statistic: {lm_lag_stat:.6f}")
    print(f"  p-value: {lm_lag_pval:.6f}")
    
    # =========================================================
    # LM-ERROR TEST
    # =========================================================
    numerator_error = np.dot(residuals, We)**2
    
    lm_error_stat = numerator_error / denominator_lag if denominator_lag > 0 else np.nan
    lm_error_pval = 1 - chi2.cdf(lm_error_stat, df=1) if not np.isnan(lm_error_stat) else np.nan
    
    print(f"\nLM-Error Test:")
    print(f"  Statistic: {lm_error_stat:.6f}")
    print(f"  p-value: {lm_error_pval:.6f}")
    
    # =========================================================
    # ROBUST LM TESTS (simplified using Anselin 1988)
    # =========================================================
    # For row-standardized W, use simplified robust formulas
    
    tr_W2 = tr_W2  # Already computed
    
    # Robust LM-lag
    numerator_rlm_lag = (np.dot(residuals, Wy) - np.dot(residuals, We) * tr_W2 / tr_WtW_W2)**2
    denom_rlm = sig2**2 * (tr_WtW_W2 - tr_W2**2 / tr_WtW_W2) if (tr_WtW_W2 - tr_W2**2 / tr_WtW_W2) > 0 else sig2**2
    
    rlm_lag_stat = numerator_rlm_lag / denom_rlm if denom_rlm > 0 else np.nan
    rlm_lag_pval = 1 - chi2.cdf(rlm_lag_stat, df=1) if not np.isnan(rlm_lag_stat) else np.nan
    
    print(f"\nRobust LM-Lag Test:")
    print(f"  Statistic: {rlm_lag_stat:.6f}")
    print(f"  p-value: {rlm_lag_pval:.6f}")
    
    # Robust LM-error
    numerator_rlm_error = (np.dot(residuals, We) - np.dot(residuals, Wy) * tr_W2 / tr_WtW_W2)**2
    
    rlm_error_stat = numerator_rlm_error / denom_rlm if denom_rlm > 0 else np.nan
    rlm_error_pval = 1 - chi2.cdf(rlm_error_stat, df=1) if not np.isnan(rlm_error_stat) else np.nan
    
    print(f"\nRobust LM-Error Test:")
    print(f"  Statistic: {rlm_error_stat:.6f}")
    print(f"  p-value: {rlm_error_pval:.6f}")
    
    lm_results = {
        'lm_lag': lm_lag_stat,
        'p_lm_lag': lm_lag_pval,
        'rlm_lag': rlm_lag_stat,
        'p_rlm_lag': rlm_lag_pval,
        'lm_error': lm_error_stat,
        'p_lm_error': lm_error_pval,
        'rlm_error': rlm_error_stat,
        'p_rlm_error': rlm_error_pval,
    }
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    lm_results = {
        'lm_lag': np.nan, 'p_lm_lag': np.nan,
        'rlm_lag': np.nan, 'p_rlm_lag': np.nan,
        'lm_error': np.nan, 'p_lm_error': np.nan,
        'rlm_error': np.nan, 'p_rlm_error': np.nan
    }

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results_df = pd.DataFrame([{
    'weights': 'knn8',
    'n': n,
    'lm_lag': lm_results.get('lm_lag', np.nan),
    'p_lm_lag': lm_results.get('p_lm_lag', np.nan),
    'rlm_lag': lm_results.get('rlm_lag', np.nan),
    'p_rlm_lag': lm_results.get('p_rlm_lag', np.nan),
    'lm_error': lm_results.get('lm_error', np.nan),
    'p_lm_error': lm_results.get('p_lm_error', np.nan),
    'rlm_error': lm_results.get('rlm_error', np.nan),
    'p_rlm_error': lm_results.get('p_rlm_error', np.nan),
}])

results_df.to_csv(OUTPUT_TABLES / "lm_tests_modelB_listing_knn8.csv", index=False)
print(f"\n✓ Saved: {OUTPUT_TABLES / 'lm_tests_modelB_listing_knn8.csv'}\n")
print(results_df.to_string(index=False))

# ============================================================================
# 8. INTERPRETATION
# ============================================================================
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

p_rlm_lag = lm_results.get('p_rlm_lag', np.nan)
p_rlm_error = lm_results.get('p_rlm_error', np.nan)

if not (np.isnan(p_rlm_lag) or np.isnan(p_rlm_error)):
    print("\nDiagnostic Summary:")
    print(f"  RLM-lag p-value: {p_rlm_lag:.4f}")
    print(f"  RLM-error p-value: {p_rlm_error:.4f}")
    
    if p_rlm_lag < 0.05 and p_rlm_error < 0.05:
        print("\n➜ BOTH tests significant: Compare SAR and SEM empirically (out of scope)")
    elif p_rlm_lag < 0.05 and p_rlm_error >= 0.05:
        print("\n➜ RLM-lag significant, RLM-error NOT: SAR model preferred")
    elif p_rlm_error < 0.05 and p_rlm_lag >= 0.05:
        print("\n➜ RLM-error significant, RLM-lag NOT: SEM model preferred")
    else:
        print("\n➜ Neither test significant: OLS likely adequate, but check LM tests")
else:
    print("\nWarning: Could not compute interpretation due to missing or invalid test statistics")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
