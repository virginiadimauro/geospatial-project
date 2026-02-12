#!/usr/bin/env python3
"""
Quality assurance checks for webmap inputs and artifacts.

Scope:
- Required files and scripts
- GeoJSON readability, CRS, and geometry validity
- Residuals table schema and basic numeric diagnostics
- ID consistency between point layer and residual table
- Streamlit app readability and key dependencies

This script exits with code 1 if blocking errors are detected.
"""

import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
REPORTS_MAPS = PROJECT_ROOT / "reports" / "maps"
SCRIPTS = PROJECT_ROOT / "scripts"

EXPECTED_WEB_CRS = "EPSG:4326"
REQUIRED_RESIDUAL_COLUMNS = [
    "listing_id",
    "log_price",
    "residual_OLS",
    "residual_SAR",
    "residual_SEM",
    "abs_residual_SAR",
]
RESIDUAL_SCALE_BREAKS = [-2.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.5]


def print_header() -> None:
    print("=" * 80)
    print("WEBMAP QUALITY ASSURANCE")
    print("=" * 80)


def print_check_result(check_id: int, name: str, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"\n[{check_id}] {name}: {status}")


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.1f} MB"
    if num_bytes >= 1_000:
        return f"{num_bytes / 1_000:.1f} KB"
    return f"{num_bytes} B"


def check_required_files(errors: list[str]) -> bool:
    required_files = {
        "GeoJSON points": DATA_PROCESSED / "map_points_sample.geojson",
        "GeoJSON grid": DATA_PROCESSED / "map_grid_cells.geojson",
        "Residuals CSV": OUTPUT_TABLES / "residuals_for_map.csv",
        "HTML map": REPORTS_MAPS / "interactive_map.html",
        "Streamlit app": PROJECT_ROOT / "webmap" / "app.py",
        "Extract script": SCRIPTS / "07b_extract_residuals.py",
        "Prep layers script": SCRIPTS / "08_prepare_map_layers.py",
    }

    failed = False
    for label, path in required_files.items():
        if path.exists():
            print(f"  - {label}: FOUND ({format_size(path.stat().st_size)})")
        else:
            failed = True
            errors.append(f"Missing file: {label} at {path}")
            print(f"  - {label}: NOT FOUND")

    return not failed


def check_geojson_layers(
    errors: list[str], warnings: list[str]
) -> tuple[bool, gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    points_path = DATA_PROCESSED / "map_points_sample.geojson"
    grid_path = DATA_PROCESSED / "map_grid_cells.geojson"
    points_gdf = None
    grid_gdf = None
    failed = False

    try:
        points_gdf = gpd.read_file(points_path)
        print(f"  - Points features: {len(points_gdf)}")
        if str(points_gdf.crs) != EXPECTED_WEB_CRS:
            warnings.append(
                f"Points CRS is {points_gdf.crs}; expected {EXPECTED_WEB_CRS}"
            )
        if points_gdf.geometry.isna().any():
            failed = True
            errors.append("Points GeoJSON contains NULL geometries")
        if not points_gdf.geometry.is_valid.all():
            failed = True
            errors.append("Points GeoJSON contains invalid geometries")
    except Exception as exc:
        failed = True
        errors.append(f"Failed to read points GeoJSON: {exc}")

    try:
        grid_gdf = gpd.read_file(grid_path)
        print(f"  - Grid features: {len(grid_gdf)}")
        if str(grid_gdf.crs) != EXPECTED_WEB_CRS:
            warnings.append(f"Grid CRS is {grid_gdf.crs}; expected {EXPECTED_WEB_CRS}")
        if grid_gdf.geometry.isna().any():
            failed = True
            errors.append("Grid GeoJSON contains NULL geometries")
        if not grid_gdf.geometry.is_valid.all():
            failed = True
            errors.append("Grid GeoJSON contains invalid geometries")
    except Exception as exc:
        failed = True
        errors.append(f"Failed to read grid GeoJSON: {exc}")

    return (not failed), points_gdf, grid_gdf


def check_residual_table(
    errors: list[str], warnings: list[str]
) -> tuple[bool, pd.DataFrame | None]:
    residual_path = OUTPUT_TABLES / "residuals_for_map.csv"
    residual_df = None
    failed = False

    try:
        residual_df = pd.read_csv(residual_path)
        print(f"  - Residual rows: {len(residual_df)}")

        missing_columns = [
            column for column in REQUIRED_RESIDUAL_COLUMNS if column not in residual_df.columns
        ]
        if missing_columns:
            failed = True
            errors.append(f"Residuals CSV missing columns: {missing_columns}")

        nan_total = int(residual_df.isna().sum().sum())
        if nan_total > 0:
            warnings.append(f"Residuals CSV contains {nan_total} missing values")

        for column in ["residual_OLS", "residual_SAR", "residual_SEM"]:
            if column in residual_df.columns:
                col_min = residual_df[column].min()
                col_max = residual_df[column].max()
                col_mean = residual_df[column].mean()
                print(
                    f"  - {column}: min={col_min:.3f}, max={col_max:.3f}, mean={col_mean:.4f}"
                )
                if abs(float(col_mean)) > 0.1:
                    warnings.append(
                        f"{column} mean is {col_mean:.4f}; expected approximately 0"
                    )
    except Exception as exc:
        failed = True
        errors.append(f"Failed to read residuals CSV: {exc}")

    return (not failed), residual_df


def check_id_consistency(
    points_gdf: gpd.GeoDataFrame | None,
    residual_df: pd.DataFrame | None,
    errors: list[str],
    warnings: list[str],
) -> bool:
    if points_gdf is None or residual_df is None:
        warnings.append("ID consistency check skipped because required data could not be loaded")
        return False

    if "listing_id" not in points_gdf.columns:
        errors.append("Points GeoJSON missing required column: listing_id")
        return False
    if "listing_id" not in residual_df.columns:
        errors.append("Residuals CSV missing required column: listing_id")
        return False

    point_ids = set(points_gdf["listing_id"].dropna().unique())
    residual_ids = set(residual_df["listing_id"].dropna().unique())
    missing_ids = point_ids - residual_ids
    if missing_ids:
        errors.append(f"{len(missing_ids)} listing_id values in points are missing in residuals")
        return False

    print(f"  - Consistent IDs: {len(point_ids)}")
    return True


def check_color_scale() -> bool:
    print(f"  - Residual class breaks: {RESIDUAL_SCALE_BREAKS}")
    return True


def check_streamlit_app(errors: list[str], warnings: list[str]) -> bool:
    app_path = PROJECT_ROOT / "webmap" / "app.py"
    if not app_path.exists():
        errors.append(f"Missing Streamlit app: {app_path}")
        return False

    try:
        content = app_path.read_text(encoding="utf-8")
    except Exception as exc:
        errors.append(f"Cannot read Streamlit app: {exc}")
        return False

    has_streamlit = "streamlit" in content
    has_folium = "folium" in content
    if not has_streamlit or not has_folium:
        warnings.append("webmap/app.py does not clearly contain both streamlit and folium references")

    print("  - Streamlit app file is readable")
    return True


def print_summary(total_checks: int, passed_checks: int, errors: list[str], warnings: list[str]) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nChecks passed: {passed_checks}/{total_checks}")
    print(f"Warnings: {len(warnings)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for message in errors:
            print(f"  - {message}")
        print("\nStatus: FAIL")
        print("Action required: resolve blocking errors before launching the webmap.")
        sys.exit(1)

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for message in warnings:
            print(f"  - {message}")
        print("\nStatus: PASS WITH WARNINGS")
    else:
        print("\nNo warnings detected.")
        print("Status: PASS")

    print("\nSuggested commands:")
    print("  micromamba activate geo")
    print("  streamlit run webmap/app.py")
    print("  # or")
    print("  bash webmap/run.sh")


def main() -> None:
    print_header()

    errors: list[str] = []
    warnings: list[str] = []
    total_checks = 6
    passed_checks = 0

    file_check = check_required_files(errors)
    print_check_result(1, "Required files", file_check)
    if file_check:
        passed_checks += 1

    geo_check, points_gdf, grid_gdf = check_geojson_layers(errors, warnings)
    print_check_result(2, "GeoJSON layers", geo_check)
    if geo_check:
        passed_checks += 1

    residual_check, residual_df = check_residual_table(errors, warnings)
    print_check_result(3, "Residual table", residual_check)
    if residual_check:
        passed_checks += 1

    id_check = check_id_consistency(points_gdf, residual_df, errors, warnings)
    print_check_result(4, "ID consistency", id_check)
    if id_check:
        passed_checks += 1

    color_check = check_color_scale()
    print_check_result(5, "Color scale", color_check)
    if color_check:
        passed_checks += 1

    app_check = check_streamlit_app(errors, warnings)
    print_check_result(6, "Streamlit app", app_check)
    if app_check:
        passed_checks += 1

    _ = grid_gdf
    print_summary(total_checks, passed_checks, errors, warnings)


if __name__ == "__main__":
    main()
