"""
download_roads.py — Download road GeoJSON files for Atlanta and LA

Creates files identical in format to charlotte_roads_drive.geojson:
  - osmnx graph → export edges as GeoJSON with full LineString geometry
  - Every vertex in each LineString will become a graph node when
    processed by _build_graph_from_roads() in the benchmark script

Charlotte: already have data/charlotte_roads_drive.geojson (282k nodes)
Atlanta:   generates data/atlanta_roads_drive.geojson
LA:        generates data/la_roads_drive.geojson

USAGE:
    pip install osmnx networkx geopandas
    python download_roads.py
"""

import os
import time
import osmnx as ox
import networkx as nx

os.makedirs("data", exist_ok=True)


# ---- ATLANTA (5-county metro) ----
atlanta_file = "data/atlanta_roads_drive.geojson"
if os.path.exists(atlanta_file):
    print(f"[Atlanta] Already exists: {atlanta_file} — skipping")
else:
    print("[Atlanta] Downloading 5-county road network...")
    counties = [
        "Fulton County, Georgia, USA",
        "DeKalb County, Georgia, USA",
        "Cobb County, Georgia, USA",
        "Gwinnett County, Georgia, USA",
        "Clayton County, Georgia, USA",
    ]
    graphs = []
    for i, county in enumerate(counties):
        print(f"  [{i+1}/5] {county}...", end=" ", flush=True)
        t0 = time.perf_counter()
        g = ox.graph_from_place(county, network_type="drive")
        print(f"{g.number_of_nodes():,} nodes ({time.perf_counter()-t0:.0f}s)")
        graphs.append(g)

    print("  Composing graphs...")
    G = nx.compose_all(graphs)
    print(f"  Combined: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print("  Exporting edges as GeoJSON...")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)[["geometry"]]
    edges.to_file(atlanta_file, driver="GeoJSON")
    fsize = os.path.getsize(atlanta_file) / (1024 * 1024)
    total_verts = sum(len(list(g.coords)) for g in edges.geometry if g)
    print(f"  Saved: {atlanta_file} ({fsize:.1f} MB)")
    print(f"  {len(edges):,} edges, {total_verts:,} vertices")
    print(f"  (Will produce ~{total_verts:,} graph nodes)")
    del G, graphs, edges


# ---- LOS ANGELES (LA County) ----
la_file = "data/la_roads_drive.geojson"
if os.path.exists(la_file):
    print(f"\n[LA] Already exists: {la_file} — skipping")
else:
    print("\n[LA] Downloading LA County road network...")
    print("  (This is a large county — may take 15-30 minutes)")
    t0 = time.perf_counter()
    G = ox.graph_from_place("Los Angeles County, California, USA", network_type="drive")
    print(f"  Downloaded in {time.perf_counter()-t0:.0f}s")
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print("  Exporting edges as GeoJSON...")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)[["geometry"]]
    edges.to_file(la_file, driver="GeoJSON")
    fsize = os.path.getsize(la_file) / (1024 * 1024)
    total_verts = sum(len(list(g.coords)) for g in edges.geometry if g)
    print(f"  Saved: {la_file} ({fsize:.1f} MB)")
    print(f"  {len(edges):,} edges, {total_verts:,} vertices")
    print(f"  (Will produce ~{total_verts:,} graph nodes)")
    del G, edges


print("\n" + "=" * 50)
print("  DONE! Road files ready:")
print(f"  Charlotte: data/charlotte_roads_drive.geojson (already had)")
print(f"  Atlanta:   {atlanta_file}")
print(f"  LA:        {la_file}")
print("=" * 50)
print("\nNow run:")
print("  python benchmark_scaling_final.py --city Charlotte")
print("  python benchmark_scaling_final.py --city Atlanta")
print('  python benchmark_scaling_final.py --city "Los Angeles"')