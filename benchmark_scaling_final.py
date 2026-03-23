"""
==========================================================================
benchmark_scaling_final.py — Self-Contained Scaling Benchmark
==========================================================================

HOW charlotte_roads_drive.geojson PRODUCED 282,751 NODES:
  1. osmnx graph_from_place → 89k intersection nodes
  2. ox.graph_to_gdfs(edges=True) → edges as LineStrings with ALL vertices
  3. Saved as GeoJSON → charlotte_roads_drive.geojson
  4. _build_graph_from_roads() reads GeoJSON → node per vertex → 282k nodes

This script reproduces that EXACT pipeline for any city.

DATA NEEDED (put in data/ folder):
    data/candidates_osm.geojson        (Charlotte, 2,450)  — already have
    data/candidates_atlanta.geojson    (Atlanta,   1,148)  — already have
    data/candidates_la.geojson         (LA,       14,208)  — already have

ROADS DOWNLOAD AUTOMATICALLY via osmnx.

REQUIREMENTS:
    pip install osmnx scipy numpy geopandas shapely networkx

USAGE:
    python benchmark_scaling_final.py --city Charlotte
    python benchmark_scaling_final.py --city Atlanta
    python benchmark_scaling_final.py --city "Los Angeles"
    python benchmark_scaling_final.py                     # all cities
==========================================================================
"""

import os
import sys
import time
import json
import gc
import argparse
import warnings
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csr_dijkstra

warnings.filterwarnings("ignore")

MILE_M = 1609.344
SNAP_TOL_M = 1.0
STD_DIJKSTRA_SAMPLE = 50
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================
# CITY CONFIGS
# =============================================================
CITIES = {
    "Charlotte": {
        # USE THE ORIGINAL FILE — this is the exact file that produced 282k nodes
        "local_roads_file": "data/charlotte_roads_drive.geojson",
        "osmnx_query": "Mecklenburg County, North Carolina, USA",
        "candidates_file": "data/candidates_osm.geojson",
        "n_block_groups": 554,
        "crs_m": 32119,
    },
    "Atlanta": {
        "local_roads_file": "data/atlanta_roads_drive.geojson",
        "osmnx_query": [
            "Fulton County, Georgia, USA",
            "DeKalb County, Georgia, USA",
            "Cobb County, Georgia, USA",
            "Gwinnett County, Georgia, USA",
            "Clayton County, Georgia, USA",
        ],
        "candidates_file": "data/candidates_atlanta.geojson",
        "n_block_groups": 1800,
        "crs_m": 32117,
    },
    "Los Angeles": {
        "local_roads_file": "data/la_roads_drive.geojson",
        "osmnx_query": "Los Angeles County, California, USA",
        "candidates_file": "data/candidates_la.geojson",
        "n_block_groups": 6400,
        "crs_m": 32111,
    },
}


# =============================================================
# STEP 1: Download osmnx graph → export EDGES as GeoJSON
#
# osmnx graph has ~89k intersection nodes for Charlotte.
# But EDGE geometries contain ~282k total vertices (curves, bends).
# Saving edges as GeoJSON and rebuilding with _build_graph_from_roads()
# creates a node at every vertex → 282k nodes. This is exactly how
# charlotte_roads_drive.geojson was created and used.
# =============================================================
def _count_vertices(geometry_series):
    """Count total vertices, handling both LineString and MultiLineString."""
    total = 0
    for g in geometry_series:
        if g is None or g.is_empty:
            continue
        if g.geom_type == "MultiLineString":
            for part in g.geoms:
                total += len(list(part.coords))
        elif g.geom_type == "LineString":
            total += len(list(g.coords))
    return total


def download_roads_geojson(city_name, config):
    """
    Get road edges as GeoJSON LineStrings.
    If local_roads_file exists (e.g. charlotte_roads_drive.geojson), use it directly.
    Otherwise, download via osmnx and export edges.
    """
    import osmnx as ox
    import geopandas as gpd

    # Check for local file first (e.g. Charlotte's original file)
    local_file = config.get("local_roads_file")
    if local_file and os.path.exists(local_file):
        print(f"  Using LOCAL road file: {local_file}")
        edges_gdf = gpd.read_file(local_file)
        if edges_gdf.crs is None:
            edges_gdf.set_crs("EPSG:4326", inplace=True)
        edges_gdf = edges_gdf.to_crs("EPSG:4326")
        total_verts = _count_vertices(edges_gdf.geometry)
        print(f"  {len(edges_gdf):,} segments, {total_verts:,} vertices")
        return edges_gdf

    # Check for cached osmnx export
    cache_file = os.path.join(OUTPUT_DIR, f"{city_name}_roads_drive.geojson")

    if os.path.exists(cache_file):
        print(f"  [cached] Loading {cache_file}")
        edges_gdf = gpd.read_file(cache_file)
        total_verts = _count_vertices(edges_gdf.geometry)
        print(f"  {len(edges_gdf):,} segments, {total_verts:,} vertices")
        return edges_gdf

    query = config["osmnx_query"]

    if isinstance(query, list):
        # Multi-county: download each, compose
        print(f"  Downloading {len(query)} counties...")
        graphs = []
        for i, q in enumerate(query):
            print(f"    [{i+1}/{len(query)}] {q}...", end=" ", flush=True)
            t0 = time.perf_counter()
            g = ox.graph_from_place(q, network_type="drive")
            print(f"{g.number_of_nodes():,} nodes ({time.perf_counter()-t0:.0f}s)")
            graphs.append(g)
        print(f"  Composing {len(graphs)} county graphs...")
        G = nx.compose_all(graphs)
    else:
        print(f"  Downloading: {query}")
        print(f"  (First run downloads from OSM — cached after this)")
        t0 = time.perf_counter()
        G = ox.graph_from_place(query, network_type="drive")
        print(f"  Downloaded in {time.perf_counter()-t0:.0f}s")

    osmnx_nodes = G.number_of_nodes()
    osmnx_edges = G.number_of_edges()
    print(f"  osmnx intersection graph: {osmnx_nodes:,} nodes, {osmnx_edges:,} edges")

    # KEY STEP: Export edges as GeoDataFrame with full LineString geometry.
    # This preserves ALL intermediate vertices (road curves, bends, etc).
    # When _build_graph_from_roads() processes these LineStrings, each vertex
    # becomes a graph node → 282k nodes for Charlotte (vs 89k osmnx nodes).
    print(f"  Exporting edges as GeoJSON (preserving all vertices)...")
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges_gdf = edges_gdf[["geometry"]].copy()
    edges_gdf = edges_gdf.to_crs(4326)

    total_verts = _count_vertices(edges_gdf.geometry)
    print(f"  {len(edges_gdf):,} edge segments")
    print(f"  {total_verts:,} total vertices → ~{total_verts:,} graph nodes")

    # Save as GeoJSON (same format as charlotte_roads_drive.geojson)
    print(f"  Saving GeoJSON...", end=" ", flush=True)
    edges_gdf.to_file(cache_file, driver="GeoJSON")
    fsize_mb = os.path.getsize(cache_file) / (1024 * 1024)
    print(f"{fsize_mb:.1f} MB")
    print(f"  Saved: {cache_file}")

    del G
    gc.collect()
    return edges_gdf


# =============================================================
# STEP 2: Build vertex graph
# EXACT COPY of model_approach2._build_graph_from_roads()
# Every vertex in every LineString becomes a graph node.
# =============================================================
def _build_graph_from_roads(roads_m, snap_tol_m=SNAP_TOL_M):
    """
    Exact same logic as model_approach2.py _build_graph_from_roads().
    """
    if roads_m is None or roads_m.empty:
        return None, None

    def _snap(x, y):
        if snap_tol_m and snap_tol_m > 0:
            return (round(x / snap_tol_m) * snap_tol_m,
                    round(y / snap_tol_m) * snap_tol_m)
        return (float(x), float(y))

    G = nx.Graph()
    node_coords = {}

    for geom in roads_m.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = (list(geom.geoms)
                 if geom.geom_type == "MultiLineString" else [geom])
        for line in lines:
            if line is None or line.is_empty:
                continue
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            x_prev, y_prev = coords[0]
            u = _snap(x_prev, y_prev)
            node_coords.setdefault(u, u)

            for (x, y) in coords[1:]:
                v = _snap(x, y)
                if v != u:
                    dist = float(np.hypot(v[0] - u[0], v[1] - u[1]))
                    if np.isfinite(dist) and dist > 0:
                        if G.has_edge(u, v):
                            if dist < G[u][v]["weight"]:
                                G[u][v]["weight"] = dist
                        else:
                            G.add_edge(u, v, weight=dist)
                    node_coords.setdefault(v, v)
                    u = v

    if G.number_of_nodes() == 0:
        return None, None

    return G, node_coords


def graph_to_csr(G, node_coords):
    """Convert NetworkX vertex graph to CSR sparse matrix."""
    node_ids = list(node_coords.keys())
    node_xy = np.array([node_coords[n] for n in node_ids], dtype=np.float64)
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)

    u_list, v_list, w_list = [], [], []
    for u, v, data in G.edges(data=True):
        ui, vi = node_to_idx[u], node_to_idx[v]
        w = float(data.get("weight", 0.0))
        if w <= 0:
            continue
        u_list.extend([ui, vi])
        v_list.extend([vi, ui])
        w_list.extend([w, w])

    csr = csr_matrix(
        (np.array(w_list, dtype=np.float64),
         (np.array(u_list, dtype=np.int32),
          np.array(v_list, dtype=np.int32))),
        shape=(n, n)
    )
    return csr, n


# =============================================================
# STEP 3: RAM-aware chunked Dijkstra benchmark
# =============================================================
def get_available_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemAvailable" in line:
                    return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return 6.0



def benchmark_dijkstra(csr, n_nodes, n_sources, n_candidates):
    """Smart direction: run from min(candidates, BGs). Always optimal."""
    rng = np.random.default_rng(seed=42)
    n_smart = min(n_sources, n_candidates)
    n_naive = max(n_sources, n_candidates)

    if n_candidates > n_sources:
        smart_label = 'BG sources (%d)' % n_sources
        naive_label = 'candidates (%d)' % n_candidates
    else:
        smart_label = 'candidates (%d)' % n_candidates
        naive_label = 'BG sources (%d)' % n_sources

    print('')
    print('  SMART DIRECTION: run from smaller set')
    print('    Ours (smart):     %s runs <- %s' % ('{:,}'.format(n_smart), smart_label))
    print('    Standard (naive): %s runs <- %s' % ('{:,}'.format(n_naive), naive_label))
    print('    Theoretical speedup: %.2fx' % (n_naive / n_smart))

    sources = rng.choice(n_nodes, size=min(n_smart, n_nodes), replace=False).astype(np.int32)
    available_ram = get_available_ram_gb()
    full_matrix_gb = (len(sources) * n_nodes * 8) / (1024**3)
    usable_bytes = available_ram * 0.5 * (1024**3)
    bytes_per_source = n_nodes * 8
    chunk_size = max(1, int(usable_bytes / bytes_per_source))
    n_chunks = max(1, int(np.ceil(len(sources) / chunk_size)))

    print('')
    print('  --- Multi-Source Dijkstra (ours - smart direction) ---')
    print('  Sources: %d  |  Graph: %s nodes' % (len(sources), '{:,}'.format(n_nodes)))
    print('  Available RAM: ~%.1f GB  |  Full matrix: %.1f GB' % (available_ram, full_matrix_gb))

    if n_chunks == 1:
        print('  Fits in RAM - running all %d sources at once' % len(sources))
    else:
        per_chunk_gb = chunk_size * n_nodes * 8 / (1024**3)
        print('  Chunking: %d batches of ~%d sources (~%.1f GB/chunk)' % (n_chunks, chunk_size, per_chunk_gb))

    total_time = 0.0
    source_chunks = np.array_split(sources, n_chunks)
    for i, chunk in enumerate(source_chunks):
        if n_chunks > 1:
            pct = 100 * (i + 1) / n_chunks
            print('  Chunk %d/%d (%d sources, %d%%)...' % (i+1, n_chunks, len(chunk), int(pct)), end=' ', flush=True)
        t0 = time.perf_counter()
        dist_matrix = csr_dijkstra(csr, directed=False, indices=chunk, return_predecessors=False)
        chunk_time = time.perf_counter() - t0
        total_time += chunk_time
        if n_chunks > 1:
            print('%.1fs' % chunk_time)
        del dist_matrix
        gc.collect()

    multisource_time = total_time
    print('  Multi-source TOTAL: %.1fs' % multisource_time)

    print('')
    print('  --- Standard Dijkstra (naive - from larger set) ---')
    sample_n = min(STD_DIJKSTRA_SAMPLE, n_naive)
    sample_sources = rng.choice(n_nodes, size=sample_n, replace=False)
    t0 = time.perf_counter()
    for src in sample_sources:
        d = csr_dijkstra(csr, directed=False, indices=int(src), return_predecessors=False)
        del d
    sample_time = time.perf_counter() - t0
    gc.collect()

    per_run = sample_time / sample_n
    std_estimated = per_run * n_naive
    print('  %d runs in %.1fs  (%.3fs/run)' % (sample_n, sample_time, per_run))
    print('  Estimated for %s (%s): %.1fs' % ('{:,}'.format(n_naive), naive_label, std_estimated))

    speedup = std_estimated / multisource_time if multisource_time > 0 else 0
    return {
        'multisource_time_s': round(multisource_time, 2),
        'std_estimated_s': round(std_estimated, 2),
        'speedup': round(speedup, 2),
        'smart_sources': int(n_smart),
        'naive_sources': int(n_naive),
        'n_sources': int(len(sources)),
        'n_candidates': int(n_candidates),
        'n_chunks': int(n_chunks),
        'per_dijkstra_run_s': round(per_run, 4),
        'available_ram_gb': round(available_ram, 1),
        'full_matrix_gb': round(full_matrix_gb, 1),
    }


# =============================================================
# Run one city
# =============================================================
def run_city(city_name, config):
    import geopandas as gpd

    crs_m = config["crs_m"]

    print(f"\n{'='*70}")
    print(f"  {city_name.upper()}")
    print(f"  osmnx query: {config['osmnx_query']}")
    print(f"  BG sources: {config['n_block_groups']}")
    print(f"{'='*70}")

    # ---- Step 1: Download roads ----
    print(f"\n[Step 1] Download osmnx graph → export edges as GeoJSON")
    edges_ll = download_roads_geojson(city_name, config)

    # ---- Step 2: Project and build vertex graph ----
    print(f"\n[Step 2] Project to EPSG:{crs_m} → build vertex graph")
    edges_m = edges_ll.to_crs(f"EPSG:{crs_m}")

    # Free original
    del edges_ll
    gc.collect()

    t0 = time.perf_counter()
    G, node_coords = _build_graph_from_roads(edges_m)
    build_time = time.perf_counter() - t0

    del edges_m
    gc.collect()

    if G is None:
        print("  FATAL: No graph built!")
        return None

    n_graph_nodes = G.number_of_nodes()
    n_graph_edges = G.number_of_edges()
    print(f"  Vertex graph: {n_graph_nodes:,} nodes, {n_graph_edges:,} edges "
          f"({build_time:.1f}s)")

    # Convert to CSR
    print(f"  Building CSR matrix...")
    csr, n_nodes = graph_to_csr(G, node_coords)
    print(f"  CSR: {n_nodes:,} nodes, {csr.nnz:,} non-zeros")

    del G, node_coords
    gc.collect()

    # ---- Count candidates ----
    cands_file = config["candidates_file"]
    if os.path.exists(cands_file):
        cands = gpd.read_file(cands_file)
        n_candidates = len(cands)
        print(f"\n  Candidates: {n_candidates:,} (from {cands_file})")
        del cands
    else:
        n_candidates = 2450
        print(f"\n  WARNING: {cands_file} not found! Using default {n_candidates}")

    # ---- Benchmark ----
    print(f"\n[Step 3] Dijkstra Benchmark")
    results = benchmark_dijkstra(
        csr, n_nodes,
        n_sources=config["n_block_groups"],
        n_candidates=n_candidates,
    )

    timing = {
        "city":           city_name,
        "n_road_nodes":   n_nodes,
        "n_road_edges":   int(csr.nnz),
        "n_candidates":   n_candidates,
        "n_bg_sources":   config["n_block_groups"],
        **results,
    }

    timing_file = os.path.join(OUTPUT_DIR, f"{city_name}_timing_final.json")
    with open(timing_file, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"\n  Saved: {timing_file}")

    return timing


# =============================================================
# Paper table
# =============================================================
def print_scaling_table(all_timings):
    def fmt(s):
        if s < 60:     return f"{s:.1f}s"
        elif s < 3600: return f"{s/60:.1f}min"
        else:          return f"{s/3600:.1f}hr"

    print(f"\n{'='*100}")
    print(f"  PAPER TABLE — Multi-Source Dijkstra Scaling (ALL MEASURED)")
    print(f"  Graph method: osmnx edges → _build_graph_from_roads() vertex graph")
    print(f"  (Same pipeline as charlotte_roads_drive.geojson → 282k nodes)")
    print(f"{'='*100}\n")

    header = (f"  {'City':<18} {'Road Nodes':>12} {'Edges':>12} {'Cands':>8} "
              f"{'BGs':>6} {'Std Dijkstra':>14} {'Multi-src':>12} "
              f"{'Speedup':>9}")
    print(header)
    print(f"  {'-'*93}")

    rows = []
    for t in all_timings:
        nodes    = f"{t['n_road_nodes']:,}"
        edges    = f"{t['n_road_edges']:,}"
        cands    = f"{t['n_candidates']:,}"
        sources  = f"{t['n_bg_sources']:,}"
        std_time = fmt(t["std_estimated_s"])
        ms_time  = fmt(t["multisource_time_s"])
        speedup  = f"{t['speedup']:.2f}x"

        print(f"  {t['city']:<18} {nodes:>12} {edges:>12} {cands:>8} "
              f"{sources:>6} {std_time:>14} {ms_time:>12} {speedup:>9}")

        rows.append({
            "City":                t["city"],
            "Road Nodes":          t["n_road_nodes"],
            "Road Edges (nnz)":    t["n_road_edges"],
            "Candidates":          t["n_candidates"],
            "BG Sources":          t["n_bg_sources"],
            "Std Dijkstra (est)":  std_time,
            "Multi-source (ours)": ms_time,
            "Speedup":             speedup,
        })

    print(f"  {'-'*93}")

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_file = os.path.join(OUTPUT_DIR, "scaling_table_final.csv")
    df.to_csv(csv_file, index=False)
    print(f"\n  Saved: {csv_file}")


# =============================================================
# MAIN
# =============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-source Dijkstra scaling benchmark")
    parser.add_argument("--city", type=str,
                        help="Run single city (Charlotte, Atlanta, 'Los Angeles')")
    parser.add_argument("--list", action="store_true",
                        help="List available cities")
    args = parser.parse_args()

    if args.list:
        print("Available cities:")
        for name, cfg in CITIES.items():
            print(f"  {name:<20} BGs={cfg['n_block_groups']:<6} "
                  f"query={cfg['osmnx_query']}")
        return

    print("=" * 70)
    print("  MULTI-SOURCE DIJKSTRA SCALING BENCHMARK")
    print("  Method: osmnx edges → vertex graph (matches Charlotte 282k)")
    print("=" * 70)

    all_timings = []

    if args.city:
        if args.city not in CITIES:
            print(f"Unknown city: {args.city}")
            print(f"Available: {list(CITIES.keys())}")
            sys.exit(1)
        t = run_city(args.city, CITIES[args.city])
        if t:
            all_timings.append(t)
    else:
        for city_name, config in CITIES.items():
            try:
                t = run_city(city_name, config)
                if t:
                    all_timings.append(t)
            except Exception as e:
                print(f"\n  ERROR on {city_name}: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Skipping {city_name}...\n")

    if all_timings:
        print_scaling_table(all_timings)

    print(f"\n  Done! Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()