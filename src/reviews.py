"""
Reviews module: Clean and aggregate reviews data.
"""

import pandas as pd
import numpy as np

def clean_and_aggregate_reviews(df):
    """
    Clean reviews and aggregate to listing level.
    
    Args:
        df: Raw reviews DataFrame
    
    Returns:
        Aggregated DataFrame (1 row per listing) with log info
    """
    log = []
    df_clean = df.copy()
    
    # 1. Normalize columns
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    log.append(f"✓ Column names normalized")
    
    # 2. Ensure listing_id is int64
    if 'listing_id' in df_clean.columns:
        df_clean['listing_id'] = df_clean['listing_id'].astype('int64')
        assert (df_clean['listing_id'] >= 0).all(), "Found negative listing_id!"
        log.append(f"✓ listing_id converted to int64")
    else:
        raise ValueError("'listing_id' column not found in reviews")
    
    # 3. Parse date column
    date_col = None
    for col in df_clean.columns:
        if 'date' in col.lower():
            date_col = col
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            null_dates = df_clean[col].isnull().sum()
            if null_dates > 0:
                log.append(f"⚠️  {null_dates} null dates in '{col}'")
            break
    
    if date_col is None:
        log.append(f"⚠️  No date column found; skipping date-based aggregations")
    else:
        log.append(f"✓ Date column '{date_col}': {df_clean[date_col].min()} to {df_clean[date_col].max()}")
    
    # 4. Remove duplicates
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        log.append(f"⚠️  Removed {dup_count} duplicate review rows")
        df_clean = df_clean.drop_duplicates()
    
    # 5. Aggregate to listing level
    # Create aggregation dictionary - count reviews per listing
    agg_specs = {}
    
    if date_col:
        agg_specs[date_col] = ['min', 'max']  # First and last review dates
    else:
        # If no date column, just count rows
        agg_specs['listing_id'] = 'count'
    
    # Group by listing_id - this keeps listing_id as a regular column (not index)
    reviews_agg = df_clean.groupby('listing_id', as_index=False).agg(agg_specs) if agg_specs else df_clean.groupby('listing_id', as_index=False).size().reset_index(name='review_count_total')
    
    # If we only have date specs, we need to add review count manually
    if date_col and agg_specs:
        reviews_agg['review_count_total'] = df_clean.groupby('listing_id').size().values
    
    # Flatten column names from MultiIndex
    if isinstance(reviews_agg.columns, pd.MultiIndex):
        reviews_agg.columns = ['_'.join(col).strip('_') for col in reviews_agg.columns.values]
    
    # Ensure listing_id is a column (not index)
    if 'listing_id' not in reviews_agg.columns:
        reviews_agg = reviews_agg.reset_index()
    
    # Rename date columns for clarity
    if date_col:
        if f'{date_col}_min' in reviews_agg.columns:
            reviews_agg.rename(columns={
                f'{date_col}_min': 'first_review_date',
                f'{date_col}_max': 'last_review_date'
            }, inplace=True)
    
    log.append(f"✓ Aggregated to {len(reviews_agg)} unique listings")
    
    # 6. Calculate metrics if dates are available
    if 'last_review_date' in reviews_agg.columns:
        today = pd.Timestamp.now()
        reviews_with_dates = reviews_agg[reviews_agg['last_review_date'].notna()].copy()
        
        reviews_agg['days_since_last_review'] = (
            today - reviews_agg['last_review_date']
        ).dt.days
        
        reviews_agg['months_active'] = (
            today - reviews_agg['first_review_date']
        ).dt.days / 30.0
        
        reviews_agg['reviews_per_month'] = (
            reviews_agg['review_count_total'] / 
            reviews_agg['months_active'].clip(lower=1)
        )
        
        log.append(f"✓ Temporal metrics calculated")
        log.append(f"  - Days since last review: {reviews_agg['days_since_last_review'].min():.0f} to {reviews_agg['days_since_last_review'].max():.0f}")
        log.append(f"  - Mean reviews/month: {reviews_agg['reviews_per_month'].mean():.2f}")
    
    return reviews_agg, log
