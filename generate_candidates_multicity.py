# =============================================================
# generate_candidates.py  (UPDATED: city-agnostic UTM handling)
#
# Generates candidate locations from OSM commercial zones.
# Works for ANY city — not hardcoded to Charlotte.
#
# WHAT IT DOES:
#   1. Downloads OSM features tagged as commercial/retail land use
#   2. Filters to meaningful retail-viable polygons (min/max area)
#   3. Takes centroid of each polygon as candidate point
#   4. Clips to city boundary (optional)
#   5. Saves as GeoJSON (EPSG:4326 points)
#
# OUTPUT:
#   GeoJSON with columns: candidate_id, osm_tag, area_sqm, geometry(Point)
#   CRS: EPSG:4326
# =============================================================

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# CONFIG — change these or pass as CLI args
# ---------------------------------------------------------------
DEFAULT_CITY      = "Charlotte, North Carolina, USA"
DEFAULT_OUT       = "data/candidates_osm.geojson"
DEFAULT_BOUNDARY  = "data/charlotte_boundary.geojson"

MIN_AREA_SQM      = 500        # 500 sqm ~ 5,400 sqft
MAX_AREA_SQM      = 500_000    # filters huge polygons (airports, etc.)
BUFFER_M          = 200        # buffer around boundary when clipping

# OSM tags to query — commercially viable locations
OSM_TAGS = {
    "landuse": ["retail", "commercial"],
    "building": ["retail", "commercial", "supermarket", "mall",
                 "shop", "store"],
    "shop":    ["supermarket", "mall", "department_store",
                "convenience", "wholesale"],
}
# ---------------------------------------------------------------


def _import_osmnx():
    try:
        import osmnx as ox
        return ox
    except ImportError:
        print("ERROR: osmnx not installed.")
        print("Install with: pip install osmnx")
        sys.exit(1)


def _ensure_crs_ll(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame is in EPSG:4326."""
    if gdf is None or gdf.empty:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf.to_crs("EPSG:4326")


def _utm_for_gdf_ll(gdf_ll: gpd.GeoDataFrame):
    """
    Pick a local UTM CRS appropriate for the geometry extent.
    Requires gdf in EPSG:4326.
    """
    if gdf_ll is None or gdf_ll.empty:
        # fallback (won't be used if empty)
        return "EPSG:3857"
    return gdf_ll.estimate_utm_crs()


def download_commercial_features(city_name: str, tags: dict) -> gpd.GeoDataFrame:
    """
    Download OSM polygon features for a city using specified tags.
    Returns GeoDataFrame in EPSG:4326.
    """
    ox = _import_osmnx()

    print(f"Downloading OSM commercial features for: {city_name}")
    print(f"Tags: {list(tags.keys())}")

    all_gdfs = []

    for tag_key, tag_values in tags.items():
        for tag_val in tag_values:
            try:
                gdf = ox.features_from_place(city_name, tags={tag_key: tag_val})

                # keep only polygon features (not points/lines)
                gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

                if len(gdf) > 0:
                    gdf["osm_tag"] = f"{tag_key}={tag_val}"
                    all_gdfs.append(gdf[["osm_tag", "geometry"]])
                    print(f"  {tag_key}={tag_val}: {len(gdf)} polygons")
                else:
                    print(f"  {tag_key}={tag_val}: 0 polygons")

            except Exception as e:
                print(f"  {tag_key}={tag_val}: skipped ({type(e).__name__})")

    if not all_gdfs:
        print("WARNING: No OSM features found. Check city name or tags.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    combined = pd.concat(all_gdfs, ignore_index=True)
    combined = combined.drop_duplicates(subset="geometry").reset_index(drop=True)

    combined = _ensure_crs_ll(combined)

    print(f"\nTotal unique commercial polygons: {len(combined)}")
    return combined


def compute_centroids_and_filter(
    gdf_polygons_ll: gpd.GeoDataFrame,
    min_area_sqm: float,
    max_area_sqm: float
) -> gpd.GeoDataFrame:
    """
    Project to local UTM (metres), compute area, filter by size, take centroids.
    Returns GeoDataFrame of Points in EPSG:4326.
    """
    if gdf_polygons_ll is None or gdf_polygons_ll.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf_polygons_ll = _ensure_crs_ll(gdf_polygons_ll)
    utm = _utm_for_gdf_ll(gdf_polygons_ll)

    gdf_m = gdf_polygons_ll.to_crs(utm)
    gdf_m["area_sqm"] = gdf_m.geometry.area

    before = len(gdf_m)
    gdf_m = gdf_m[(gdf_m["area_sqm"] >= min_area_sqm) & (gdf_m["area_sqm"] <= max_area_sqm)].copy()
    after = len(gdf_m)
    print(f"\nArea filter ({min_area_sqm}–{max_area_sqm} sqm): {before} → {after} polygons")

    # centroids
    gdf_m["geometry"] = gdf_m.geometry.centroid

    # back to WGS84
    result = gdf_m.to_crs("EPSG:4326").reset_index(drop=True)
    result["candidate_id"] = [f"osm_{i:06d}" for i in range(len(result))]

    return result[["candidate_id", "osm_tag", "area_sqm", "geometry"]]


def clip_to_boundary(
    candidates_ll: gpd.GeoDataFrame,
    boundary_file: str,
    buffer_m: float = BUFFER_M
) -> gpd.GeoDataFrame:
    """
    Clip candidate points to boundary + buffer.
    If boundary file missing/empty string, returns candidates unchanged.
    """
    if candidates_ll is None or candidates_ll.empty:
        return candidates_ll

    boundary_file = (boundary_file or "").strip()
    if not boundary_file:
        print("No boundary provided — skipping clip")
        return candidates_ll
    if not os.path.exists(boundary_file):
        print(f"Boundary file not found: {boundary_file} — skipping clip")
        return candidates_ll

    # Read boundary in lat/lon first
    boundary_ll = gpd.read_file(boundary_file)
    boundary_ll = _ensure_crs_ll(boundary_ll)

    utm = _utm_for_gdf_ll(boundary_ll)
    boundary_m = boundary_ll.to_crs(utm)
    boundary_buf = boundary_m.geometry.buffer(buffer_m).unary_union

    cands_m = candidates_ll.to_crs(utm)
    mask = cands_m.geometry.within(boundary_buf)

    before = len(candidates_ll)
    clipped = candidates_ll[mask].reset_index(drop=True)
    after = len(clipped)
    print(f"Boundary clip: {before} → {after} candidates")

    return clipped


def print_summary(candidates_ll: gpd.GeoDataFrame):
    print("\n" + "=" * 55)
    print("  CANDIDATE SUMMARY")
    print("=" * 55)
    print(f"  Total candidates : {len(candidates_ll)}")

    if not candidates_ll.empty and "osm_tag" in candidates_ll.columns:
        print("\n  By OSM tag:")
        for tag, count in candidates_ll["osm_tag"].value_counts().items():
            print(f"    {tag:<35} {count:>6}")

    if not candidates_ll.empty and "area_sqm" in candidates_ll.columns:
        areas = candidates_ll["area_sqm"]
        print("\n  Area distribution (sqm):")
        print(f"    min    = {areas.min():>12,.0f}")
        print(f"    median = {areas.median():>12,.0f}")
        print(f"    mean   = {areas.mean():>12,.0f}")
        print(f"    max    = {areas.max():>12,.0f}")

    if not candidates_ll.empty:
        bounds = candidates_ll.total_bounds
        print("\n  Bounding box (EPSG:4326):")
        print(f"    lon: {bounds[0]:.4f} to {bounds[2]:.4f}")
        print(f"    lat: {bounds[1]:.4f} to {bounds[3]:.4f}")

    print("=" * 55)


def generate(
    city_name: str = DEFAULT_CITY,
    out_path: str = DEFAULT_OUT,
    boundary_file: str = DEFAULT_BOUNDARY,
    min_area: float = MIN_AREA_SQM,
    max_area: float = MAX_AREA_SQM,
    tags: dict = OSM_TAGS,
):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    # 1) Download polygons
    polygons_ll = download_commercial_features(city_name, tags)
    if polygons_ll.empty:
        print("No polygons downloaded. Exiting.")
        return None

    # 2) Filter + centroid
    candidates_ll = compute_centroids_and_filter(polygons_ll, min_area, max_area)

    # 3) Optional boundary clip
    candidates_ll = clip_to_boundary(candidates_ll, boundary_file, buffer_m=BUFFER_M)

    if candidates_ll is None or candidates_ll.empty:
        print("No candidates after filtering/clipping. Check city/boundary/tags.")
        return None

    # 4) Summary
    print_summary(candidates_ll)

    # 5) Save GeoJSON (EPSG:4326)
    candidates_ll = _ensure_crs_ll(candidates_ll)
    candidates_ll.to_file(out_path, driver="GeoJSON")
    print(f"\nSaved: {out_path}")
    print(f'Use this file as CANDIDATES_FILE, e.g.: CANDIDATES_FILE = "{out_path}"')

    return candidates_ll


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OSM commercial candidate locations.")
    parser.add_argument("--city", default=DEFAULT_CITY, help=f"City name (default: {DEFAULT_CITY})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output GeoJSON (default: {DEFAULT_OUT})")
    parser.add_argument("--boundary", default=DEFAULT_BOUNDARY, help="Boundary GeoJSON for clipping (optional)")
    parser.add_argument("--min-area", type=float, default=MIN_AREA_SQM, help=f"Min polygon area sqm (default: {MIN_AREA_SQM})")
    parser.add_argument("--max-area", type=float, default=MAX_AREA_SQM, help=f"Max polygon area sqm (default: {MAX_AREA_SQM})")

    args = parser.parse_args()

    generate(
        city_name=args.city,
        out_path=args.out,
        boundary_file=args.boundary,
        min_area=args.min_area,
        max_area=args.max_area,
    )