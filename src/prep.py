import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path


def _parse_price_series(s: pd.Series) -> pd.Series:
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
    return s.apply(parse_price)


def load_listings_processed(path: str or Path) -> pd.DataFrame:
    """Load processed listings parquet and apply deterministic normalizations.

    - ensures `listing_id` exists
    - parses price into `price_numeric`
    - keeps geometry/coords if present
    """
    path = Path(path)
    df = pd.read_parquet(path).copy()

    # ensure stable listing_id
    if 'listing_id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'listing_id'})

    # normalize column names to snake_case minimally
    df.columns = [c.strip() for c in df.columns]

    # parse prices
    df['price_numeric'] = _parse_price_series(df.get('price', df.get('price_num', pd.Series([np.nan]*len(df)))))

    return df


def build_model_df(df: pd.DataFrame, out_path: str or Path = None, winsorize_q=(0.005, 0.995)) -> pd.DataFrame:
    """Build a deterministic model dataframe used across scripts.

    - imputes basic missing values
    - normalizes host booleans
    - winsorizes price and creates `log_price`
    - adds `dist_cbd_km`, room_type dummies, neighbourhood dummies
    - saves sample to `out_path` if provided
    Returns: model_df with stable `listing_id` and only necessary columns + coords if present
    """
    df = df.copy()

    # stable listing id
    if 'listing_id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'listing_id'})

    # price
    if 'price_numeric' not in df.columns:
        df['price_numeric'] = _parse_price_series(df.get('price', pd.Series([np.nan]*len(df))))
    df = df[df['price_numeric'].notna()].copy()

    # winsorize
    q_low, q_high = df['price_numeric'].quantile(list(winsorize_q))
    df['price_winsorized'] = df['price_numeric'].clip(q_low, q_high)
    df['log_price'] = np.log(df['price_winsorized'])

    # basic imputations
    for col in ['bedrooms', 'beds', 'bathrooms']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # host booleans
    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].fillna('f')
        df['host_is_superhost'] = (df['host_is_superhost'] == 't').astype(int)

    if 'instant_bookable' in df.columns:
        df['instant_bookable'] = (df['instant_bookable'] == 't').astype(int)

    if 'review_scores_rating' in df.columns:
        df['review_scores_rating'] = df['review_scores_rating'].fillna(0)

    if 'host_listings_count' in df.columns:
        df['host_listings_count'] = df['host_listings_count'].fillna(df['host_listings_count'].median())

    # location: distance to CBD (Puerta del Sol)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        PUERTA_DEL_SOL_LAT, PUERTA_DEL_SOL_LON = 40.4169, -3.7035

        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        df['dist_cbd_km'] = haversine_distance(df['latitude'], df['longitude'], PUERTA_DEL_SOL_LAT, PUERTA_DEL_SOL_LON)

    # dummies for room_type and neighbourhood
    model_df = df.copy()
    if 'room_type' in model_df.columns:
        room_dummies = pd.get_dummies(model_df['room_type'], prefix='room_type', drop_first=True, dtype=int)
        model_df = pd.concat([model_df, room_dummies], axis=1)

    if 'neighbourhood_cleansed' in model_df.columns:
        neigh_dummies = pd.get_dummies(model_df['neighbourhood_cleansed'], prefix='neigh', drop_first=True, dtype=int)
        model_df = pd.concat([model_df, neigh_dummies], axis=1)

    # ensure listing_id and coordinates included
    keep_cols = ['listing_id', 'latitude', 'longitude', 'log_price', 'price_numeric']
    keep_cols = [c for c in keep_cols if c in model_df.columns]

    # include all model covariates
    covariate_candidates = ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'minimum_nights',
                            'host_is_superhost', 'host_listings_count', 'number_of_reviews',
                            'review_scores_rating', 'instant_bookable', 'dist_cbd_km']
    for c in covariate_candidates:
        if c in model_df.columns and c not in keep_cols:
            keep_cols.append(c)

    # include dummies
    dummy_cols = [c for c in model_df.columns if c.startswith('room_type_') or c.startswith('neigh_')]
    keep_cols += dummy_cols

    model_sample = model_df[keep_cols].copy()

    # if geometry present, preserve it
    if 'geometry' in df.columns:
        gdf = gpd.GeoDataFrame(model_sample, geometry=df.loc[model_sample.index].geometry, crs=df.crs if hasattr(df, 'crs') else None)
        model_sample = gdf

    # save if requested
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model_sample.to_parquet(out_path)

    return model_sample


def get_y_X(model_df: pd.DataFrame):
    """Return (y, X, cols) aligned. X is a DataFrame without constant.
    y is a pandas Series indexed as model_df.
    """
    df = model_df.copy()
    if 'log_price' not in df.columns:
        raise ValueError('log_price not found in model_df')

    y = df['log_price']

    # select covariates: all columns except listing_id, coords, price fields, geometry
    exclude = {'listing_id', 'latitude', 'longitude', 'log_price', 'price_numeric', 'price_winsorized', 'geometry'}
    X_cols = [c for c in df.columns if c not in exclude]
    X = df[X_cols].copy()

    return y, X, X_cols
