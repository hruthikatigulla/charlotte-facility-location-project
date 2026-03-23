# =========================
# model_nashville.py
# Nashville BJJ / CrossFit site selection
#
# FINAL + FAST LOAD (Parquet + joblib graph cache):
#   - Keep BG polygons (bg_m) for /blocks map rendering
#   - Scoring uses BG centroids + cKDTree.query_ball_point() (NO overlay/intersection in loop)
#   - Dijkstra-based accessibility (access_score_dj) supported
#   - Disk cache for processed state:
#        * Parquet: bg_m, cands_m, cands_ll, bjj_m, cf_m, others_m
#        * joblib : road graph arrays + node_xy + bg_node_ids (node indices)
#   - Later startups load in seconds (no GeoJSON read, no graph rebuild)
#
# ENV:
#   MODEL_CACHE_DIR=<path>    (default: cache/)
#   FORCE_REBUILD_CACHE=1     (rebuild caches)
#   LOAD_ONLY_CACHE=1         (fail if cache missing; don't rebuild)
# =========================

import os
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import networkx as nx
from joblib import dump, load as joblib_load

# ----------- CONSTANTS -----------
CRS_LL = 4326      # WGS84 lat/lon
CRS_M  = 3857      # Web Mercator (metric-ish)
MILE_M = 1609.344

# ----------- FILES -----------
BG_FILE       = "nashville_data/nashville_bg_with_income.geojson"
ROADS_FILE    = "nashville_data/nashville_roads_drive.geojson"
BOUNDARY_FILE = "nashville_data/nashville_boundary.geojson"
GYMS_FILE     = "nashville_data/gyms_nashville_clean.geojson"

# ----------- CACHE -----------
CACHE_DIR = "cache"
CACHE_VERSION = "nashville_v6_parquet_graph_arrays"
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------ utility helpers ------------
def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)

def _file_sig(path: str) -> str:
    """Signature based on path + mtime + size (changes when data changes)."""
    if not _exists(path):
        return f"{path}:MISSING"
    st = os.stat(path)
    return f"{path}:{int(st.st_mtime)}:{int(st.st_size)}"

def _cache_key_for_inputs() -> str:
    sig = "|".join([
        CACHE_VERSION,
        _file_sig(BG_FILE),
        _file_sig(ROADS_FILE),
        _file_sig(BOUNDARY_FILE),
        _file_sig(GYMS_FILE),
    ])
    return hashlib.md5(sig.encode("utf-8")).hexdigest()

def _cache_base(prefix: str, cache_dir: str, cache_key: str) -> str:
    d = os.path.join(cache_dir, f"{prefix}_{cache_key}")
    os.makedirs(d, exist_ok=True)
    return d

def _read_to_ll(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(CRS_LL, inplace=True, allow_override=True)
    return gdf.to_crs(CRS_LL)

def _first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

def _norm_weight(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, float).copy()
    a[~np.isfinite(a)] = np.nan
    valid = a > 0
    if not np.any(valid):
        return np.ones_like(a)
    vals = a[valid]
    lo, hi = np.nanpercentile(vals, [1, 99])
    if hi <= lo:
        w = np.ones_like(a)
    else:
        clipped = np.clip(a, lo, hi)
        w = (clipped - lo) / (hi - lo)
    w[~np.isfinite(w)] = 0.0
    return 0.01 + 0.99 * np.clip(w, 0.0, 1.0)

# ------------------- HUFF -------------------
def huff_share_vs_competitors(t_new: np.ndarray, t_comps: np.ndarray, beta: float) -> np.ndarray:
    """
    Huff-style share vs competitors based on travel times (minutes).
    """
    t_new = np.asarray(t_new, float)
    t_comps = np.asarray(t_comps, float)

    eps = 1e-6
    t_new_safe = np.where(t_new <= 0, eps, t_new)
    t_comps_safe = np.where(t_comps <= 0, np.inf, t_comps)

    A_new = 1.0 / (t_new_safe ** beta)
    A_comp = np.sum(1.0 / (t_comps_safe ** beta), axis=1)

    return A_new / (A_new + A_comp + 1e-9)

# ------------------- ROAD GRAPH HELPERS -------------------
def _build_graph_from_roads(roads_m: gpd.GeoDataFrame):
    if roads_m is None or roads_m.empty:
        return None, None

    G = nx.Graph()
    node_coords = {}

    for geom in roads_m.geometry:
        if geom is None:
            continue

        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            x1, y1 = coords[0]
            x2, y2 = coords[-1]

            dist = float(np.hypot(x2 - x1, y2 - y1))
            if not np.isfinite(dist) or dist <= 0:
                continue

            u = (x1, y1)
            v = (x2, y2)

            # keep shortest if duplicates
            if G.has_edge(u, v):
                if dist < G[u][v]["weight"]:
                    G[u][v]["weight"] = dist
            else:
                G.add_edge(u, v, weight=dist)

            node_coords.setdefault(u, (x1, y1))
            node_coords.setdefault(v, (x2, y2))

    if G.number_of_nodes() == 0:
        return None, None

    print(f"[roads] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, node_coords

def _build_road_kdtree(node_coords: dict):
    if not node_coords:
        return None, None, None
    node_ids = list(node_coords.keys())
    node_xy = np.array([node_coords[n] for n in node_ids], dtype=np.float32)
    tree = cKDTree(node_xy)
    return tree, node_ids, node_xy

def _graph_to_arrays(G: nx.Graph, road_node_ids: list, node_coords: dict):
    """
    Convert graph with tuple-nodes into compact arrays.
    Returned graph nodes will be ints 0..N-1.
    """
    node_xy = np.array([node_coords[n] for n in road_node_ids], dtype=np.float32)
    node_to_idx = {n: i for i, n in enumerate(road_node_ids)}

    u_idx = []
    v_idx = []
    w = []
    for u, v, data in G.edges(data=True):
        u_idx.append(node_to_idx[u])
        v_idx.append(node_to_idx[v])
        w.append(float(data.get("weight", 0.0)))

    return (
        np.asarray(u_idx, dtype=np.int32),
        np.asarray(v_idx, dtype=np.int32),
        np.asarray(w, dtype=np.float32),
        node_xy,
    )

def _arrays_to_graph(u_idx, v_idx, w):
    G = nx.Graph()
    G.add_weighted_edges_from(zip(u_idx.tolist(), v_idx.tolist(), w.tolist()))
    return G

def _nearest_node_idx(tree: cKDTree, x, y) -> int:
    _, idx = tree.query([x, y], k=1)
    return int(idx)

# ------------------- FAST LOCAL METRICS (CENTROID + KDTREE) -------------------
def _bg_pop_income_centroid(bg_tree: cKDTree, cx: float, cy: float, rad_m: float,
                            bg_pop: np.ndarray, bg_inc: np.ndarray):
    """
    Approx pop in radius + population-weighted income, using centroid inclusion.
    Returns (pop_buf, income_wavg).
    """
    idxs = bg_tree.query_ball_point([cx, cy], rad_m)
    if not idxs:
        return 0.0, 0.0

    pop = np.asarray(bg_pop[idxs], float)
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, 0.0)
    pop_buf = float(pop.sum())
    if pop_buf <= 0:
        return 0.0, 0.0

    inc = np.asarray(bg_inc[idxs], float)
    mask = np.isfinite(inc) & (inc > 0) & (pop > 0)
    if not np.any(mask):
        return pop_buf, 0.0

    inc_wavg = float(np.average(inc[mask], weights=pop[mask]))
    return pop_buf, float(np.clip(inc_wavg, 0, 300000))

def _stores_per_10k_centroid(stores_tree: cKDTree, cx: float, cy: float, rad_m: float, pop_buf: float):
    if pop_buf <= 0 or stores_tree is None:
        return 0.0
    n_stores = len(stores_tree.query_ball_point([cx, cy], rad_m))
    return float(n_stores) / (pop_buf / 10000.0)

# ------------------- DIVERSE SELECTION -------------------
def select_top_diverse(out_m: gpd.GeoDataFrame,
                       scores_col: str = "pair_score",
                       N: int = 10,
                       min_sep_m: float = 3.0 * MILE_M) -> gpd.GeoDataFrame:
    if out_m.empty:
        return out_m

    cand = out_m.sort_values(scores_col, ascending=False).reset_index(drop=True)
    kept_idx = []
    xs = cand.geometry.x.values
    ys = cand.geometry.y.values

    for i in range(len(cand)):
        if len(kept_idx) >= N:
            break
        if not kept_idx:
            kept_idx.append(i)
            continue
        dx = xs[i] - xs[kept_idx]
        dy = ys[i] - ys[kept_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.all(dist >= min_sep_m):
            kept_idx.append(i)

    if not kept_idx:
        kept_idx = [0]
    return cand.iloc[kept_idx].reset_index(drop=True)

# ------------------- LOAD & PREP -------------------
def load_nashville():
    """
    FAST LOAD:
      - If parquet+joblib cache exists, load it in seconds.
      - Else build once (slow), save cache, then next runs are fast.
    """
    cache_root = os.environ.get("MODEL_CACHE_DIR", CACHE_DIR)
    force = os.environ.get("FORCE_REBUILD_CACHE", "0").strip().lower() in ("1", "true", "yes")
    load_only = os.environ.get("LOAD_ONLY_CACHE", "0").strip().lower() in ("1", "true", "yes")

    cache_key = _cache_key_for_inputs()
    base = _cache_base("nashville", cache_root, cache_key)

    p_bg       = os.path.join(base, "bg_m.parquet")
    p_cands_m  = os.path.join(base, "cands_m.parquet")
    p_cands_ll = os.path.join(base, "cands_ll.parquet")
    p_bjj      = os.path.join(base, "bjj_m.parquet")
    p_cf       = os.path.join(base, "cf_m.parquet")
    p_others   = os.path.join(base, "others_m.parquet")
    p_graph    = os.path.join(base, "road_graph.joblib")

    expected = [p_bg, p_cands_m, p_cands_ll, p_bjj, p_cf, p_others, p_graph]

    # ---------- FAST PATH ----------
    if (not force) and all(os.path.exists(p) for p in expected):
        bg_m = gpd.read_parquet(p_bg)
        cands_m = gpd.read_parquet(p_cands_m)
        cands_ll = gpd.read_parquet(p_cands_ll)
        bjj_m = gpd.read_parquet(p_bjj)
        cf_m = gpd.read_parquet(p_cf)
        others_m = gpd.read_parquet(p_others)

        g = joblib_load(p_graph)
        node_xy = g["node_xy"]
        road_kdtree = cKDTree(node_xy)
        road_graph = _arrays_to_graph(g["u_idx"], g["v_idx"], g["w"])
        bg_node_ids = g["bg_node_ids"].astype(np.int32)

        print(f"[cache] Nashville FAST load from {base}")
        return {
            "bg_m": bg_m,
            "cands_ll": cands_ll,
            "cands_m": cands_m,
            "roads_m": None,  # intentionally not loaded
            "road_graph": road_graph,
            "road_kdtree": road_kdtree,
            "bg_node_ids": bg_node_ids,
            "gyms_m": None,
            "bjj_m": bjj_m,
            "cf_m": cf_m,
            "others_m": others_m,
        }

    if load_only:
        raise RuntimeError(f"LOAD_ONLY_CACHE=1 but cache missing/incomplete at: {base}")

    # ---------- BUILD PATH ----------
    if not _exists(BG_FILE):
        raise FileNotFoundError(f"Missing Nashville BG file: {BG_FILE}")
    if not _exists(GYMS_FILE):
        raise FileNotFoundError(f"Missing gyms file: {GYMS_FILE}")
    if not _exists(ROADS_FILE):
        raise FileNotFoundError(f"Missing Nashville roads file: {ROADS_FILE}")

    # ---- BG polygons ----
    bg_ll = _read_to_ll(BG_FILE)

    pop_col = _first_existing(bg_ll.columns, ["population", "pop_total", "pop", "POP", "B01001_001E"])
    if pop_col is None:
        raise ValueError("Nashville BG file must have a population-like column.")
    if pop_col != "population":
        bg_ll = bg_ll.rename(columns={pop_col: "population"})

    inc_col = _first_existing(bg_ll.columns, ["income", "median_income", "med_income", "B19013_001E"])
    if inc_col is None:
        bg_ll["income"] = np.nan
    elif inc_col != "income":
        bg_ll = bg_ll.rename(columns={inc_col: "income"})

    bg_ll["population"] = pd.to_numeric(bg_ll["population"], errors="coerce").clip(lower=0)
    bg_ll["income"] = pd.to_numeric(bg_ll["income"], errors="coerce").clip(lower=0, upper=300000)
    med_inc = bg_ll["income"].median(skipna=True)
    bg_ll["income"] = bg_ll["income"].fillna(med_inc if np.isfinite(med_inc) else 0)

    # ---- roads ----
    roads_ll = _read_to_ll(ROADS_FILE)

    # ---- gyms ----
    gyms_ll = _read_to_ll(GYMS_FILE)
    name_series = gyms_ll.get("name", gyms_ll.get("Name", "")).astype(str).str.lower()

    is_bjj = (
        name_series.str.contains("bjj")
        | name_series.str.contains("jiu jitsu")
        | name_series.str.contains("jiu-jitsu")
        | name_series.str.contains("gracie")
    )
    is_cf = name_series.str.contains("crossfit") | name_series.str.contains("cross fit")

    gyms_ll["is_bjj"] = is_bjj
    gyms_ll["is_cf"] = is_cf

    print(f"[nashville] gyms total={len(gyms_ll)}, bjj={int(is_bjj.sum())}, cf={int(is_cf.sum())}, others={int((~(is_bjj | is_cf)).sum())}")

    # ---- boundary clip (load-time only; OK) ----
    try:
        if _exists(BOUNDARY_FILE):
            bound_ll = _read_to_ll(BOUNDARY_FILE)[["geometry"]]
            bound_m = bound_ll.to_crs(CRS_M)
            bound_buf = bound_m.buffer(200).unary_union

            try:
                bg_ll = gpd.overlay(bg_ll, bound_ll, how="intersection")
            except Exception as e:
                print("[nashville] BG overlay fallback:", e)
                bg_ll = bg_ll[bg_ll.to_crs(CRS_M).intersects(bound_buf)].to_crs(CRS_LL)

            gyms_ll = gyms_ll[gyms_ll.to_crs(CRS_M).within(bound_buf)].to_crs(CRS_LL)

            try:
                roads_ll = gpd.overlay(roads_ll, bound_ll, how="intersection")
            except Exception as e:
                print("[nashville] roads overlay fallback:", e)
                roads_ll = roads_ll[roads_ll.to_crs(CRS_M).intersects(bound_buf)].to_crs(CRS_LL)
    except Exception as e:
        print("[nashville] boundary clip skipped:", e)

    # ---- project to metric ----
    bg_m = bg_ll.to_crs(CRS_M)          # polygons kept for /blocks
    roads_m = roads_ll.to_crs(CRS_M)[["geometry"]]

    # tooltip fields
    bg_m["area_sqmi"] = bg_m.geometry.area / (MILE_M ** 2)
    bg_m["pop_per_sqmi"] = bg_m["population"] / bg_m["area_sqmi"].replace(0, np.nan)
    bg_m["median_income"] = bg_m["income"]

    gyms_m = gyms_ll.to_crs(CRS_M)
    gyms_m["is_bjj"] = gyms_ll["is_bjj"].values
    gyms_m["is_cf"] = gyms_ll["is_cf"].values

    # candidates = BG centroids
    bg_m_cent = bg_m.copy()
    bg_m_cent["cent"] = bg_m_cent.geometry.centroid

    cands_m = gpd.GeoDataFrame(geometry=bg_m_cent["cent"].rename("geometry"), crs=CRS_M)
    cands_ll = cands_m.to_crs(CRS_LL)

    # build road graph + KDTree + BG node mapping
    road_graph_raw, node_coords = _build_graph_from_roads(roads_m)
    road_kdtree_raw, road_node_ids, node_xy = _build_road_kdtree(node_coords)

    bg_xy = np.c_[bg_m_cent["cent"].x.values, bg_m_cent["cent"].y.values]
    _, idxs = road_kdtree_raw.query(bg_xy, k=1)
    bg_node_ids = idxs.astype(np.int32)

    u_idx, v_idx, w, node_xy = _graph_to_arrays(road_graph_raw, road_node_ids, node_coords)

    # subsets
    bjj_m = gyms_m[gyms_m["is_bjj"]].copy()
    cf_m = gyms_m[gyms_m["is_cf"]].copy()
    others_m = gyms_m[~(gyms_m["is_bjj"] | gyms_m["is_cf"])].copy()
    for gdf in (bjj_m, cf_m, others_m):
        gdf.set_crs(CRS_M, inplace=True, allow_override=True)

    # ---- SAVE CACHE ----
    try:
        bg_m.to_parquet(p_bg, index=False)
        cands_m.to_parquet(p_cands_m, index=False)
        cands_ll.to_parquet(p_cands_ll, index=False)
        bjj_m.to_parquet(p_bjj, index=False)
        cf_m.to_parquet(p_cf, index=False)
        others_m.to_parquet(p_others, index=False)

        dump(
            {"u_idx": u_idx, "v_idx": v_idx, "w": w, "node_xy": node_xy, "bg_node_ids": bg_node_ids},
            p_graph,
            compress=3,
        )
        print(f"[cache] Nashville saved parquet+graph to {base}")
    except Exception as e:
        print(f"[cache] Failed to save Nashville cache: {e}")

    # return warm-style state (graph uses int node ids)
    road_kdtree = cKDTree(node_xy)
    road_graph = _arrays_to_graph(u_idx, v_idx, w)

    return {
        "bg_m": bg_m,
        "cands_ll": cands_ll,
        "cands_m": cands_m,
        "roads_m": None,  # do not carry heavy roads in memory post-build
        "road_graph": road_graph,
        "road_kdtree": road_kdtree,
        "bg_node_ids": bg_node_ids,
        "gyms_m": gyms_m,
        "bjj_m": bjj_m,
        "cf_m": cf_m,
        "others_m": others_m,
    }

# -------------- core scoring --------------
def _core_score(state_core,
                radius_miles=5.0,
                beta=2.5,
                penalty_lambda=0.25,
                K=3,
                heat_sample=500,
                max_candidates=None,
                topN=10,
                min_sep_miles=3.0,
                W1=1.0, W2=1.0, W3=1.0):

    ht_m = state_core["ht_m"]
    comp_m = state_core["comp_m"]
    cands_ll = state_core["cands_ll"].copy()
    cands_m = state_core["cands_m"].copy()
    bg_m = state_core["bg_m"]

    road_graph = state_core.get("road_graph", None)
    road_kdtree = state_core.get("road_kdtree", None)
    bg_node_ids = state_core.get("bg_node_ids", None)

    rad_m = radius_miles * MILE_M
    SPEED_MPS = 35 * MILE_M / 3600.0

    if max_candidates is not None and len(cands_ll) > max_candidates:
        cands_ll = cands_ll.iloc[:max_candidates].reset_index(drop=True)
        cands_m = cands_m.iloc[:max_candidates].reset_index(drop=True)

    cands_ll = cands_ll.reset_index(drop=True)
    cands_m = cands_m.reset_index(drop=True)

    # BG centroid arrays (used for scoring, BG polygons still kept in bg_m)
    bg_cent = bg_m.copy()
    bg_cent["cent"] = bg_cent.geometry.centroid
    bg_xy = np.c_[bg_cent["cent"].x.values, bg_cent["cent"].y.values]

    pop_col = _first_existing(bg_cent.columns, ["population", "pop_total", "pop", "POP", "B01001_001E"])
    inc_col = _first_existing(bg_cent.columns, ["income", "median_income", "med_income", "B19013_001E"])

    bg_pop = pd.to_numeric(bg_cent[pop_col], errors="coerce").to_numpy()
    bg_pop = np.where(np.isfinite(bg_pop) & (bg_pop > 0), bg_pop, 0.0)

    if inc_col is not None:
        bg_inc = pd.to_numeric(bg_cent[inc_col], errors="coerce").to_numpy()
        bg_inc = np.where(np.isfinite(bg_inc) & (bg_inc > 0), bg_inc, 0.0)
    else:
        bg_inc = np.zeros_like(bg_pop)

    pop_w = _norm_weight(bg_pop)
    inc_w = _norm_weight(bg_inc) if np.any(bg_inc > 0) else np.ones_like(bg_pop)

    dens_col = _first_existing(bg_cent.columns, ["pop_per_sqmi", "density", "DENSITY"])
    if dens_col is not None:
        dens = pd.to_numeric(bg_cent[dens_col], errors="coerce").to_numpy()
        dens_w = _norm_weight(dens)
    else:
        dens_w = np.ones_like(bg_pop)

    block_weight = 0.4 * pop_w + 0.3 * inc_w + 0.3 * dens_w

    # competitor times (Euclidean)
    if not comp_m.empty:
        comp_xy = np.c_[comp_m.geometry.x.values, comp_m.geometry.y.values]
        tree = cKDTree(comp_xy)
        k_eff = min(max(1, K), len(comp_xy))
        dists_bg_to_comp, _ = tree.query(bg_xy, k=k_eff)
        if dists_bg_to_comp.ndim == 1:
            dists_bg_to_comp = dists_bg_to_comp[:, None]
        bg_tcomp = (dists_bg_to_comp / SPEED_MPS) / 60.0
    else:
        bg_tcomp = np.full((len(bg_xy), 1), 60.0)

    # KDTree for BG centroids
    bg_tree = cKDTree(bg_xy)

    # KDTree for store count metric
    stores_all_xy = np.vstack([
        np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values] if (ht_m is not None and len(ht_m) > 0) else np.empty((0, 2)),
        np.c_[comp_m.geometry.x.values, comp_m.geometry.y.values] if (comp_m is not None and len(comp_m) > 0) else np.empty((0, 2)),
    ])
    stores_tree = cKDTree(stores_all_xy) if stores_all_xy.shape[0] > 0 else None

    metr_s10k, metr_inc, metr_access_score, metr_potential = [], [], [], []

    cand_xy = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]

    for cx, cy in cand_xy:
        pop_buf, incM = _bg_pop_income_centroid(bg_tree, cx, cy, rad_m, bg_pop, bg_inc)
        s10k = _stores_per_10k_centroid(stores_tree, cx, cy, rad_m, pop_buf)

        metr_s10k.append(float(s10k))
        metr_inc.append(float(incM))

        # Huff capture (Euclidean time)
        d_new = np.hypot(bg_xy[:, 0] - cx, bg_xy[:, 1] - cy)
        t_new = (d_new / SPEED_MPS) / 60.0
        share = huff_share_vs_competitors(t_new, bg_tcomp, beta)
        metr_potential.append(float(np.nansum(block_weight * share)))

        # Dijkstra accessibility (graph nodes are int indices; bg_node_ids is int array)
        access_score = 0.0
        if (road_graph is not None) and (road_kdtree is not None) and (bg_node_ids is not None):
            cand_node = _nearest_node_idx(road_kdtree, cx, cy)
            lengths = nx.single_source_dijkstra_path_length(road_graph, cand_node, weight="weight")

            dist_m = np.fromiter(
                (lengths.get(int(n), np.inf) for n in bg_node_ids),
                dtype=np.float64,
                count=len(bg_node_ids),
            )

            t_road = (dist_m / SPEED_MPS) / 60.0
            valid = np.isfinite(t_road) & (t_road > 0) & np.isfinite(block_weight) & (block_weight > 0)
            if valid.any():
                wv = block_weight[valid]
                weighted_time_min = float(np.nansum(wv * t_road[valid]) / np.nansum(wv))
                t_hours = max(weighted_time_min, 0.0) / 60.0
                access_score = float(np.clip(1.0 / (1.0 + t_hours), 0.01, 1.0))

        metr_access_score.append(access_score)

    s10k_arr_raw = np.asarray(metr_s10k, float)
    inc_arr = np.asarray(metr_inc, float)
    access_dj_arr = np.asarray(metr_access_score, float)
    potential_arr_raw = np.asarray(metr_potential, float)

    potential_norm = _norm_weight(potential_arr_raw)

    final_scores = (W1 * potential_norm) + (W2 * access_dj_arr) - (W3 * penalty_lambda * s10k_arr_raw)
    ps = np.asarray(final_scores, float)
    inten = (ps - np.nanmin(ps)) / (np.nanmax(ps) - np.nanmin(ps) + 1e-9)

    out_ll = cands_ll.copy()
    out_ll["pair_score"] = np.round(ps, 2)
    out_ll["intensity"] = inten
    out_ll["stores_per_10k"] = np.round(s10k_arr_raw, 2)
    out_ll["income_med"] = np.round(inc_arr, 2)
    out_ll["access_score_dj"] = np.round(access_dj_arr, 2)

    ht_gdf = None

    heat_points = []

    out_m = out_ll.to_crs(CRS_M)
    diverse_top = select_top_diverse(
        out_m[["geometry", "pair_score", "stores_per_10k", "income_med", "access_score_dj"]].copy(),
        scores_col="pair_score",
        N=topN,
        min_sep_m=float(min_sep_miles) * MILE_M,
    )
    top10 = diverse_top.to_crs(CRS_LL).reset_index(drop=True)

    return top10, heat_points, ht_gdf

# -------------- public scoring wrappers --------------
def score_nashville_bjj(state, **kwargs):
    """
    New BJJ gym: BJJ = brand, (CF + others) = competitors.
    """
    comp_m = gpd.GeoDataFrame(
        geometry=pd.concat([state["cf_m"].geometry, state["others_m"].geometry], ignore_index=True),
        crs=CRS_M,
    )
    core_state = {
        "ht_m": state["bjj_m"],
        "comp_m": comp_m,
        "bg_m": state["bg_m"],
        "cands_ll": state["cands_ll"],
        "cands_m": state["cands_m"],
        "road_graph": state.get("road_graph", None),
        "road_kdtree": state.get("road_kdtree", None),
        "bg_node_ids": state.get("bg_node_ids", None),
    }
    return _core_score(core_state, **kwargs)

def score_nashville_cf(state, **kwargs):
    """
    New CrossFit gym: CF = brand, (BJJ + others) = competitors.
    """
    comp_m = gpd.GeoDataFrame(
        geometry=pd.concat([state["bjj_m"].geometry, state["others_m"].geometry], ignore_index=True),
        crs=CRS_M,
    )
    core_state = {
        "ht_m": state["cf_m"],
        "comp_m": comp_m,
        "bg_m": state["bg_m"],
        "cands_ll": state["cands_ll"],
        "cands_m": state["cands_m"],
        "road_graph": state.get("road_graph", None),
        "road_kdtree": state.get("road_kdtree", None),
        "bg_node_ids": state.get("bg_node_ids", None),
    }
    return _core_score(core_state, **kwargs)
