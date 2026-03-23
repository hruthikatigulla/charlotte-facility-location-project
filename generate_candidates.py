# =============================================================
# generate_candidates.py
#
# Generates candidate locations from OSM commercial zones.
# Works for ANY city — not hardcoded to Charlotte.
#
# WHAT IT DOES:
#   1. Downloads OSM features tagged as commercial/retail land use
#   2. Filters to meaningful retail-viable polygons (min area)
#   3. Takes centroid of each polygon as candidate point
#   4. Clips to city boundary
#   5. Saves as candidates_osm.geojson
#
# OSM TAGS USED:
#   landuse   = retail, commercial
#   building  = retail, commercial, supermarket, mall
#   shop      = supermarket, mall, convenience (existing — for context)
#
# WHY OSM NOT SYNTHETIC GRID:
#   - Points represent real commercially zoned locations
#   - Reproducible — anyone can re-download the same data
#   - Generalises to any city with OSM coverage
#   - No proprietary data required
#
# USAGE:
#   python generate_candidates.py
#   python generate_candidates.py --city "Nashville, TN" --out data/candidates_nashville.geojson
#
# OUTPUT:
#   data/candidates_osm.geojson
#   Columns: osm_id, osm_type, area_sqm, geometry (Point, EPSG:4326)
#
# PAPER CITATION:
#   OpenStreetMap contributors (2024). OpenStreetMap.
#   Retrieved via Overpass API through osmnx.
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
BOUNDARY_FILE     = "data/charlotte_boundary.geojson"
MIN_AREA_SQM      = 500       # minimum parcel area — filters slivers
                               # 500 sqm ~ 5,400 sqft — reasonable min retail
MAX_AREA_SQM      = 500_000   # maximum — filters airports, industrial parks
BUFFER_M          = 200       # buffer around boundary when clipping

# OSM tags to query — these represent commercially viable locations
# Reference: OpenStreetMap Wiki — Map Features
OSM_TAGS = {
    "landuse": ["retail", "commercial"],
    "building": ["retail", "commercial", "supermarket", "mall",
                 "shop", "store"],
    "shop":    ["supermarket", "mall", "department_store",
                "convenience", "wholesale"],
}

# ---------------------------------------------------------------


def _import_osmnx():
    """Import osmnx with helpful error if missing."""
    try:
        import osmnx as ox
        return ox
    except ImportError:
        print("ERROR: osmnx not installed.")
        print("Install with:  pip install osmnx")
        sys.exit(1)


def download_commercial_features(city_name: str, tags: dict):
    """
    Download OSM polygon features for a city using specified tags.
    Returns a GeoDataFrame in EPSG:4326.
    """
    ox = _import_osmnx()

    print(f"Downloading OSM commercial features for: {city_name}")
    print(f"Tags: {list(tags.keys())}")

    all_gdfs = []

    for tag_key, tag_values in tags.items():
        for tag_val in tag_values:
            try:
                gdf = ox.features_from_place(
                    city_name,
                    tags={tag_key: tag_val}
                )
                # Keep only polygon features (not points/lines)
                gdf = gdf[gdf.geometry.geom_type.isin(
                    ["Polygon", "MultiPolygon"]
                )].copy()

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
    combined = combined.drop_duplicates(
        subset="geometry"
    ).reset_index(drop=True)

    # Ensure CRS
    if combined.crs is None:
        combined = combined.set_crs("EPSG:4326")
    else:
        combined = combined.to_crs("EPSG:4326")

    print(f"\nTotal unique commercial polygons: {len(combined)}")
    return combined


def compute_centroids_and_filter(gdf_polygons, min_area_sqm, max_area_sqm):
    """
    Project to metres, compute area, filter by size, take centroids.
    Returns GeoDataFrame of Points in EPSG:4326.
    """
    if gdf_polygons.empty:
        return gdf_polygons

    # Project to metres for area calculation
    # EPSG:32617 = UTM Zone 17N covers Charlotte/eastern US
    gdf_m = gdf_polygons.to_crs("EPSG:32617")
    gdf_m["area_sqm"] = gdf_m.geometry.area

    # Filter by area
    before = len(gdf_m)
    gdf_m = gdf_m[
        (gdf_m["area_sqm"] >= min_area_sqm) &
        (gdf_m["area_sqm"] <= max_area_sqm)
    ].copy()
    after = len(gdf_m)
    print(f"\nArea filter ({min_area_sqm}–{max_area_sqm} sqm): "
          f"{before} → {after} polygons")

    # Centroids
    gdf_m["geometry"] = gdf_m.geometry.centroid

    # Back to WGS84
    result = gdf_m.to_crs("EPSG:4326")
    result = result.reset_index(drop=True)
    result["candidate_id"] = [f"osm_{i:04d}" for i in range(len(result))]

    return result[["candidate_id", "osm_tag", "area_sqm", "geometry"]]


def clip_to_boundary(candidates, boundary_file, buffer_m=200):
    """
    Clip candidates to city boundary + buffer.
    If boundary file not found, returns candidates unchanged.
    """
    if not os.path.exists(boundary_file):
        print(f"Boundary file not found: {boundary_file} — skipping clip")
        return candidates

    boundary = gpd.read_file(boundary_file).to_crs("EPSG:32617")
    boundary_buf = boundary.geometry.buffer(buffer_m).unary_union

    cands_m = candidates.to_crs("EPSG:32617")
    mask = cands_m.geometry.within(boundary_buf)

    before = len(candidates)
    clipped = candidates[mask].reset_index(drop=True)
    after = len(clipped)
    print(f"Boundary clip: {before} → {after} candidates")

    return clipped


def print_summary(candidates):
    """Print summary statistics of generated candidates."""
    print("\n" + "="*55)
    print("  CANDIDATE SUMMARY")
    print("="*55)
    print(f"  Total candidates : {len(candidates)}")

    if "osm_tag" in candidates.columns:
        print(f"\n  By OSM tag:")
        for tag, count in candidates["osm_tag"].value_counts().items():
            print(f"    {tag:<35} {count:>5}")

    if "area_sqm" in candidates.columns:
        areas = candidates["area_sqm"]
        print(f"\n  Area distribution (sqm):")
        print(f"    min    = {areas.min():>10,.0f}")
        print(f"    median = {areas.median():>10,.0f}")
        print(f"    mean   = {areas.mean():>10,.0f}")
        print(f"    max    = {areas.max():>10,.0f}")

    if not candidates.empty:
        bounds = candidates.total_bounds
        print(f"\n  Bounding box:")
        print(f"    lon: {bounds[0]:.4f} to {bounds[2]:.4f}")
        print(f"    lat: {bounds[1]:.4f} to {bounds[3]:.4f}")

    print("="*55)


def generate(city_name=DEFAULT_CITY,
             out_path=DEFAULT_OUT,
             boundary_file=BOUNDARY_FILE,
             min_area=MIN_AREA_SQM,
             max_area=MAX_AREA_SQM,
             tags=OSM_TAGS):
    """
    Main function — download, filter, and save OSM commercial candidates.
    Can be called from other scripts or run directly.
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path)
                else ".", exist_ok=True)

    # 1. Download
    polygons = download_commercial_features(city_name, tags)

    if polygons.empty:
        print("No polygons downloaded. Exiting.")
        return None

    # 2. Filter and centroid
    candidates = compute_centroids_and_filter(polygons, min_area, max_area)

    # 3. Clip to boundary
    candidates = clip_to_boundary(candidates, boundary_file, BUFFER_M)

    if candidates.empty:
        print("No candidates after filtering. Check city/boundary.")
        return None

    # 4. Summary
    print_summary(candidates)

    # 5. Save
    candidates.to_file(out_path, driver="GeoJSON")
    print(f"\nSaved: {out_path}")
    print(f"Use this file as CANDIDATES_FILE in model_approach2.py")
    print(f"  CANDIDATES_FILE = \"{out_path}\"")

    return candidates


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate OSM commercial candidate locations."
    )
    parser.add_argument(
        "--city",
        default=DEFAULT_CITY,
        help=f"City name for OSM query (default: {DEFAULT_CITY})"
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output GeoJSON path (default: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--boundary",
        default=BOUNDARY_FILE,
        help=f"Boundary GeoJSON for clipping (default: {BOUNDARY_FILE})"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=MIN_AREA_SQM,
        help=f"Min polygon area sqm (default: {MIN_AREA_SQM})"
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=MAX_AREA_SQM,
        help=f"Max polygon area sqm (default: {MAX_AREA_SQM})"
    )
    args = parser.parse_args()

    generate(
        city_name=args.city,
        out_path=args.out,
        boundary_file=args.boundary,
        min_area=args.min_area,
        max_area=args.max_area,
    )