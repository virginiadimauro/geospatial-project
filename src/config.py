"""
Configuration module: paths, CRS constants, and global settings.
"""

from pathlib import Path
import os

# ============================================================================
# PROJECT PATHS (zero hardcoding - all relative to PROJECT_ROOT)
# ============================================================================

# Detect PROJECT_ROOT: either cwd or parent if in notebooks/scripts
def get_project_root():
    """Auto-detect project root by checking for data/ and src/ folders."""
    cwd = Path.cwd()
    
    # If already in project root
    if (cwd / "data").exists() and (cwd / "src").exists():
        return cwd
    
    # If in notebooks/ or scripts/
    if cwd.name in ["notebooks", "scripts"] and (cwd.parent / "data").exists():
        return cwd.parent
    
    # Fallback: use parent if src exists there
    if (cwd.parent / "src").exists() and (cwd.parent / "data").exists():
        return cwd.parent
    
    # Last resort: assume project root is one level up
    return cwd.parent

PROJECT_ROOT = get_project_root()

# Core data paths
DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL_DIR = DATA_DIR / "original"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Create directories if missing
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Input files (raw data)
INPUT_FILES = {
    "calendar": ORIGINAL_DIR / "calendar.csv",
    "listings": ORIGINAL_DIR / "listings.csv",
    "reviews": ORIGINAL_DIR / "reviews.csv",
    "neighbourhoods": ORIGINAL_DIR / "neighbourhoods.geojson",
    "listings_summary": ORIGINAL_DIR / "listings_summary.csv",  # Optional
    "reviews_summary": ORIGINAL_DIR / "reviews_summary.csv",    # Optional
}

# Output files (processed)
OUTPUT_FILES = {
    "calendar_clean": PROCESSED_DIR / "calendar_clean.parquet",
    "listings_clean": PROCESSED_DIR / "listings_clean.parquet",
    "reviews_clean": PROCESSED_DIR / "reviews_clean.parquet",
    "reviews_listing_features": PROCESSED_DIR / "reviews_listing_features.parquet",
    "calendar_enriched": PROCESSED_DIR / "calendar_enriched.parquet",
    "calendar_enriched_neighbourhoods": PROCESSED_DIR / "calendar_enriched_with_neighbourhoods.parquet",
    "neighbourhoods_clean": PROCESSED_DIR / "neighbourhoods_clean.parquet",
    "neighbourhoods_enriched": PROCESSED_DIR / "neighbourhoods_enriched.parquet",
    "neighbourhoods_enriched_geojson": PROCESSED_DIR / "neighbourhoods_enriched.geojson",
    "listings_points_sample": PROCESSED_DIR / "listings_points_enriched_sample.geojson",
}

# ============================================================================
# GEOSPATIAL & CRS CONSTANTS
# ============================================================================

# Web mapping CRS (WGS84 - standard for all web outputs)
CRS_WEB = "EPSG:4326"

# Metric CRS for Madrid (UTM Zone 30N - for area/distance calculations)
CRS_METRIC = "EPSG:25830"  # ETRS89 UTM Zone 30N (metric, accurate for Spain)

# Alternative: EPSG:32630 (WGS84 UTM Zone 30N) - also valid but slightly less precise for Spain
# CRS_METRIC = "EPSG:32630"

# ============================================================================
# DATA QUALITY CONSTANTS
# ============================================================================

# Price cleaning
PRICE_OUTLIER_THRESHOLD_HIGH = 10000  # Remove prices > €10k/night
PRICE_OUTLIER_THRESHOLD_LOW = 10      # Remove prices < €10/night
PRICE_NULL_STRATEGY = "median"         # Strategy for missing prices: 'median', 'drop', 'flag'

# Availability
AVAILABILITY_VALID_VALUES = [0, 1, True, False, 't', 'f', 'true', 'false']

# Spatial join
SPATIAL_JOIN_PREDICATE = "within"      # Use 'within' or 'contains' for point-in-polygon
MIN_SPATIAL_JOIN_COVERAGE = 0.95       # Minimum % of listings matched to neighbourhoods

# ============================================================================
# SAMPLE SETTINGS
# ============================================================================

LISTINGS_SAMPLE_SIZE = 500  # For lightweight webmap point layer
RANDOM_SEED = 42

# ============================================================================
# LOGGING & VERBOSITY
# ============================================================================

VERBOSE = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

def print_config():
    """Print all configuration settings."""
    print("\n" + "=" * 80)
    print("PIPELINE CONFIGURATION")
    print("=" * 80)
    print(f"\n📁 PROJECT ROOT: {PROJECT_ROOT}")
    print(f"📂 DATA DIR: {DATA_DIR}")
    print(f"📂 PROCESSED DIR: {PROCESSED_DIR}")
    print(f"📂 OUTPUTS DIR: {OUTPUTS_DIR}")
    print(f"\n🗺️  CRS Settings:")
    print(f"   Web (output): {CRS_WEB}")
    print(f"   Metric (calculations): {CRS_METRIC}")
    print(f"\n✓ Configuration loaded successfully")
    print("=" * 80 + "\n")
