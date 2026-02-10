"""
Cleaning module: Data cleaning and normalization for calendar, listings, and reviews.
"""

import pandas as pd
import numpy as np
import re
from . import config


def parse_price_series(s: pd.Series) -> pd.Series:
    """
    Robustly parse price series, handling $ € £ symbols, spaces, NBSP, and decimal formats.

    Handles both 1,234.56 (comma thousands sep) and 1.234,56 (comma decimal sep).
    Decision: use the LAST separator (comma or dot) as decimal point.

    Args:
        s: pd.Series of price strings (e.g., '$157.00', '€100,50', etc.)

    Returns:
        pd.Series of float values (NaN for unparseable)
    """
    def parse_single(val):
        if pd.isna(val) or val == '' or val is None:
            return np.nan

        # Convert to string and strip
        val_str = str(val).strip()

        # Remove currency symbols, spaces, NBSP
        val_str = re.sub(r'[$€£\s\xa0]', '', val_str)

        # Keep only digits, dots, commas, minus sign
        val_str = re.sub(r'[^\d.,\-]', '', val_str)

        if not val_str or val_str == '-':
            return np.nan

        # Find last occurrence of comma or dot (to decide decimal separator)
        last_comma_idx = val_str.rfind(',')
        last_dot_idx = val_str.rfind('.')

        # Determine decimal separator based on last occurrence
        if last_comma_idx > last_dot_idx:
            # Last separator is comma → comma is decimal sep
            # Remove all dots, replace comma with dot
            val_str = val_str.replace('.', '').replace(',', '.')
        elif last_dot_idx > last_comma_idx:
            # Last separator is dot → dot is decimal sep
            # Remove all commas (they're thousands separators)
            val_str = val_str.replace(',', '')
        # else: no separator found, keep as-is

        try:
            return float(val_str)
        except ValueError:
            return np.nan

    return s.apply(parse_single)


def clean_calendar(df_calendar):
    """
    Clean calendar data: normalize columns, fix data types, handle outliers.
    
    Args:
        df_calendar: Raw calendar DataFrame
    
    Returns:
        Cleaned DataFrame and log info
    """
    log = []
    df_clean = df_calendar.copy()
    
    # 1. Normalize column names
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    log.append(f"✓ Column names normalized")
    
    # 2. Ensure listing_id is int64
    if 'listing_id' in df_clean.columns:
        df_clean['listing_id'] = df_clean['listing_id'].astype('int64')
        assert (df_clean['listing_id'] >= 0).all(), "Found negative listing_id!"
        log.append(f"✓ listing_id converted to int64 ({len(df_clean['listing_id'].unique())} unique listings)")
    else:
        raise ValueError("'listing_id' column not found in calendar")
    
    # 3. Parse date column
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        null_dates = df_clean['date'].isnull().sum()
        if null_dates > 0:
            log.append(f"⚠️  {null_dates} invalid dates (set to NaT)")
            df_clean = df_clean[df_clean['date'].notna()]
        log.append(f"✓ Date column parsed ({df_clean['date'].min()} to {df_clean['date'].max()})")
    
    # 4. Clean availability column
    avail_col = None
    for col in df_clean.columns:
        if 'available' in col.lower():
            avail_col = col
            break
    
    if avail_col:
        # Standardize to 0/1 binary
        df_clean[avail_col] = df_clean[avail_col].astype(str).str.lower()
        bool_map = {'t': 1, 'f': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0}
        df_clean[avail_col] = df_clean[avail_col].map(bool_map).astype('int8')
        
        # Check for remaining NaN
        null_avail = df_clean[avail_col].isnull().sum()
        if null_avail > 0:
            df_clean[avail_col] = df_clean[avail_col].fillna(0).astype('int8')
            log.append(f"⚠️  Filled {null_avail} missing availability values with 0")
        
        log.append(f"✓ {avail_col} converted to binary (0/1)")

        # Rename for consistency
        if avail_col != 'available':
            df_clean.rename(columns={avail_col: 'available'}, inplace=True)
    
    # 5. Clean price columns
    # Note: calendar.price and adjusted_price are typically empty in this dataset
    # We skip parsing them entirely - they will remain NaN
    # Price statistics should come from listings only
    price_cols = ['price', 'adjusted_price']
    for price_col in price_cols:
        if price_col in df_clean.columns:
            # Just check if data exists, don't try to parse empty columns
            price_not_null = df_clean[price_col].notna().sum()
            if price_not_null == 0:
                log.append(f"⚠️  {price_col}: All values are NULL (skipping - use listings.price for statistics)")
            else:
                log.append(f"⚠️  {price_col}: Found {price_not_null} values (not parsing - check manually if needed)")
    
    # 6. Clean minimum/maximum nights
    for nights_col in ['minimum_nights', 'maximum_nights']:
        if nights_col in df_clean.columns:
            df_clean[nights_col] = pd.to_numeric(df_clean[nights_col], errors='coerce').astype('int64')
    
    # 7. Remove complete duplicates
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        log.append(f"⚠️  Removed {dup_count} duplicate calendar rows")
        df_clean = df_clean.drop_duplicates()
    
    # 8. Final shape
    log.append(f"✓ Calendar cleaning complete: {df_calendar.shape} → {df_clean.shape}")
    
    return df_clean, log


def clean_listings(df_listings):
    """
    Clean listings data: normalize columns, fix geo columns, handle prices.
    
    Args:
        df_listings: Raw listings DataFrame
    
    Returns:
        Cleaned DataFrame and log info
    """
    log = []
    df_clean = df_listings.copy()
    
    # 1. Normalize column names
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    log.append(f"✓ Column names normalized")
    
    # 2. Find and rename ID column
    id_col = None
    for col in df_clean.columns:
        if col == 'id' or (col.endswith('_id') and 'listing' in col):
            id_col = col
            break
    
    if id_col and id_col != 'listing_id':
        df_clean.rename(columns={id_col: 'listing_id'}, inplace=True)
        log.append(f"✓ Renamed '{id_col}' to 'listing_id'")
    
    # 3. Ensure listing_id is int64
    if 'listing_id' in df_clean.columns:
        df_clean['listing_id'] = df_clean['listing_id'].astype('int64')
        assert (df_clean['listing_id'] >= 0).all(), "Found negative listing_id!"
        log.append(f"✓ listing_id converted to int64 ({len(df_clean)} unique listings)")
    
    # 4. Clean price column using robust parser
    price_col = None
    for col in df_clean.columns:
        if col.lower() == 'price':
            price_col = col
            break

    if price_col:
        # Use robust price parser
        df_clean[price_col] = parse_price_series(df_clean[price_col])

        # Check if price data was successfully parsed (MANDATORY CHECK)
        price_not_null = df_clean[price_col].notna().sum()
        if price_not_null == 0:
            raw_examples = df_listings[price_col].dropna().astype(str).head(10).tolist()
            raise ValueError(f"PRICE PARSING FAILED: 0 valid prices. Raw examples: {raw_examples}")

        log.append(f"✓ {price_col}: {price_not_null:,} prices parsed successfully")

        # Remove price outliers (only from non-null values)
        before = len(df_clean)
        mask = (df_clean[price_col].isna()) | (
            (df_clean[price_col] >= config.PRICE_OUTLIER_THRESHOLD_LOW) &
            (df_clean[price_col] <= config.PRICE_OUTLIER_THRESHOLD_HIGH)
        )
        df_clean = df_clean[mask]
        removed = before - len(df_clean)

        if removed > 0:
            log.append(f"⚠ Removed {removed} listings with price outliers " +
                      f"(<€{config.PRICE_OUTLIER_THRESHOLD_LOW} or >€{config.PRICE_OUTLIER_THRESHOLD_HIGH})")
        else:
            log.append(f"✓ {price_col}: {price_not_null:,} non-null values in range " +
                      f"€{config.PRICE_OUTLIER_THRESHOLD_LOW}-€{config.PRICE_OUTLIER_THRESHOLD_HIGH}")
    
    # 5. Check for latitude/longitude columns
    lat_col = None
    lon_col = None
    
    for col in df_clean.columns:
        if 'latitude' in col.lower():
            lat_col = col
        if 'longitude' in col.lower():
            lon_col = col
    
    if lat_col and lon_col:
        # Ensure numeric
        df_clean[lat_col] = pd.to_numeric(df_clean[lat_col], errors='coerce')
        df_clean[lon_col] = pd.to_numeric(df_clean[lon_col], errors='coerce')
        
        # Remove rows with missing geometry
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=[lat_col, lon_col])
        removed = before - len(df_clean)
        
        if removed > 0:
            log.append(f"⚠️  Removed {removed} listings with missing coordinates")
        
        # Rename to standard names if different
        if lat_col != 'latitude':
            df_clean.rename(columns={lat_col: 'latitude'}, inplace=True)
        if lon_col != 'longitude':
            df_clean.rename(columns={lon_col: 'longitude'}, inplace=True)
        
        log.append(f"✓ Geometry columns found (latitude, longitude)")
    else:
        log.append(f"⚠️  No latitude/longitude columns found")
    
    # 6. Clean room_type
    room_type_col = None
    for col in df_clean.columns:
        if 'room_type' in col.lower() or 'type' in col.lower():
            room_type_col = col
            break
    
    if room_type_col:
        df_clean[room_type_col] = df_clean[room_type_col].astype(str).str.lower().str.strip()
        log.append(f"✓ room_type standardized")
    
    # 7. Remove complete duplicates
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        log.append(f"⚠️  Removed {dup_count} duplicate listing rows")
        df_clean = df_clean.drop_duplicates()
    
    # 8. Remove duplicate listing_ids (keep first)
    duplicate_ids = df_clean['listing_id'].duplicated(keep='first').sum()
    if duplicate_ids > 0:
        log.append(f"⚠️  Removed {duplicate_ids} duplicate listing_ids (kept first occurrence)")
        df_clean = df_clean.drop_duplicates(subset=['listing_id'], keep='first')
    
    # 9. Final shape
    log.append(f"✓ Listings cleaning complete: {df_listings.shape} → {df_clean.shape}")
    
    # 10. Calculate availability_rate from reviews if available
    if 'review_count' in df_clean.columns:
        # Rough estimate: active listings tend to have more reviews
        # This will be properly calculated later
        log.append(f"⚠️  review_count found; availability will be calculated later with full review data")
    
    return df_clean, log
