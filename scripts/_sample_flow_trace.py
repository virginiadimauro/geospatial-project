#!/usr/bin/env python
"""
Trace Sample Flow: From listings_clean.parquet to Final Model (N=15,641)

Tracks:
1. N from listings_clean.parquet
2. N after price parsing
3. N after winsorization (0.5% - 99.5%)
4. N after dropna on model features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TRACE SAMPLE FLOW
# ============================================================================

flow_log = []

# Step 0: Initial load
print("=" * 80)
print("SAMPLE FLOW TRACE: N=18,940 → N=15,641")
print("=" * 80)

df = pd.read_parquet(DATA_PROCESSED / "listings_clean.parquet").copy()
n_initial = len(df)
flow_log.append({
    'step': 0,
    'description': 'Initial load from listings_clean.parquet',
    'N_remaining': n_initial,
    'N_dropped': 0,
    'reason': 'Starting dataset'
})
print(f"\n[0] Initial: {n_initial} listings")

# Step 1: Price parsing
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

df['price_num'] = df['price'].apply(parse_price)
df_after_price = df[df['price_num'].notna()].copy()
n_after_price = len(df_after_price)
n_dropped_price = n_initial - n_after_price

flow_log.append({
    'step': 1,
    'description': 'Price parsing (exclude NaN)',
    'N_remaining': n_after_price,
    'N_dropped': n_dropped_price,
    'reason': f'Exclude unparseable/NaN price values'
})
print(f"[1] After price parsing: {n_after_price} ({n_dropped_price} dropped)")

# Step 2: Winsorization
q1, q99 = df_after_price['price_num'].quantile([0.005, 0.995])
df_after_winsor = df_after_price[(df_after_price['price_num'] >= q1) & (df_after_price['price_num'] <= q99)].copy()
n_after_winsor = len(df_after_winsor)
n_dropped_winsor = n_after_price - n_after_winsor

flow_log.append({
    'step': 2,
    'description': f'Winsorization (p=0.5%, p=99.5%)',
    'N_remaining': n_after_winsor,
    'N_dropped': n_dropped_winsor,
    'reason': f'Exclude extreme outliers (price < €{q1:.0f} or > €{q99:.0f})'
})
print(f"[2] After winsorization (0.5%-99.5%): {n_after_winsor} ({n_dropped_winsor} dropped)")
print(f"    Price range: €{q1:.0f} - €{q99:.0f}")

# Step 3: Log transformation and feature engineering
df_after_winsor['log_price'] = np.log(df_after_winsor['price_num'])
df_after_winsor['_original_idx'] = range(len(df_after_winsor))

# Impute missing in numeric columns
for col in ['bedrooms', 'beds', 'bathrooms']:
    df_after_winsor[col] = df_after_winsor[col].fillna(df_after_winsor[col].median())

# Convert categorical
df_after_winsor['host_is_superhost'] = df_after_winsor['host_is_superhost'].fillna('f')
df_after_winsor['host_is_superhost'] = (df_after_winsor['host_is_superhost'] == 't').astype(int)
df_after_winsor['instant_bookable'] = (df_after_winsor['instant_bookable'] == 't').astype(int)

# Encode room_type
df_encoded = pd.get_dummies(df_after_winsor, columns=['room_type'], drop_first=True, dtype=int)

# Distance variable
if 'dist_cbd_km' not in df_encoded.columns:
    df_encoded['dist_cbd_km'] = np.sqrt((df_encoded['latitude'] - 40.4168)**2 + (df_encoded['longitude'] - -3.7038)**2) * 111
else:
    df_encoded['dist_cbd_km'] = df_encoded['dist_cbd_km']

# Build feature list
property_cols = ['accommodates', 'bedrooms', 'beds', 'bathrooms']
host_cols = ['host_is_superhost', 'host_listings_count', 'number_of_reviews', 'review_scores_rating', 'instant_bookable']
room_type_cols = [c for c in df_encoded.columns if c.startswith('room_type_')]
location_cols = ['dist_cbd_km']

neighbourhood_cols = []
if 'neighbourhood_cleansed' in df_encoded.columns:
    neighbourhood_encoded = pd.get_dummies(df_encoded['neighbourhood_cleansed'], prefix='neigh', dtype=int)
    neighbourhood_cols = [c for c in neighbourhood_encoded.columns if c != neighbourhood_encoded.columns[0]]
    df_encoded = pd.concat([df_encoded, neighbourhood_encoded], axis=1)

X_cols = property_cols + host_cols + room_type_cols + location_cols + neighbourhood_cols
X_cols_available = [c for c in X_cols if c in df_encoded.columns]

# Step 4: Remove rows with missing values in model features
df_model = df_encoded[X_cols_available + ['log_price', '_original_idx']].dropna()
n_after_dropna = len(df_model)
n_dropped_missing = len(df_encoded) - n_after_dropna

flow_log.append({
    'step': 3,
    'description': f'Feature engineering + imputation (missing in {len(X_cols_available)} features)',
    'N_remaining': n_after_dropna,
    'N_dropped': n_dropped_missing,
    'reason': f'Exclude rows with missing values in log_price or {len(X_cols_available)} model features'
})
print(f"[3] After feature engineering & dropna: {n_after_dropna} ({n_dropped_missing} dropped)")
print(f"    Features checked: {len(X_cols_available)} (property, host, room_type, location, neighbourhood)")

# ============================================================================
# SAVE SAMPLE FLOW
# ============================================================================

flow_df = pd.DataFrame(flow_log)
output_path = OUTPUT_TABLES / "sample_flow.csv"
flow_df.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")
print("\n" + "=" * 80)
print("SAMPLE FLOW SUMMARY")
print("=" * 80)
print(flow_df.to_string(index=False))

print(f"\n✓ FINAL SAMPLE: N = {n_after_dropna}")
print(f"  ├─ Lost to price parsing: {n_dropped_price}")
print(f"  ├─ Lost to winsorization: {n_dropped_winsor}")
print(f"  └─ Lost to missing features: {n_dropped_missing}")
print(f"  Total lost: {n_initial - n_after_dropna}")
print("=" * 80)
