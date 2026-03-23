# Facility Location with Composite Spatial Accessibility Index

A framework for scoring candidate facility locations by combining demographic demand, road-network accessibility, and competitive saturation into a single bounded, weight-controllable score.

**Paper:** *A Composite Spatial Accessibility Index for Facility Location: Algorithm Benchmark and Application to Grocery Retail* (IEEE conference submission)

## Overview

The framework integrates three traditionally separate spatial analyses:

| Component | Symbol | What it measures | Source |
|-----------|--------|-----------------|--------|
| Market Potential | P̃_k | Huff gravity demand capture vs K nearest competitors | Census ACS + store locations |
| Accessibility | Ã_k | Hansen index via Dijkstra shortest paths on road network | OpenStreetMap road graph |
| Competition | S̃_k | Stores per 10,000 people (all brands) | Store locations + Census pop |

**Composite Score:**
```
Score_k = W1·P̃_k + W2·Ã_k - W3·S̃_k
```
where W1 + W2 + W3 = 1 and all components are normalised to [0.01, 1.0].

## Key Results (Charlotte, NC)

| Metric | Value |
|--------|-------|
| Census block groups | 554 |
| Candidate sites (OSM commercial) | 2,450 |
| Road network nodes | 282,751 |
| Road network edges | 585,666 |
| Harris Teeter ground-truth stores | 28 |
| Competitor grocery stores | 59 |

### Algorithm Benchmark

| Algorithm | Strategy | Runs | Time (s) | Speedup |
|-----------|----------|------|----------|---------|
| Standard Dijkstra | Forward, per candidate | 2,450 | 209.7 | 1.00× |
| Reverse Dijkstra | Backward, per BG | 554 | 50.6 | 4.14× |
| **Multi-source Dijkstra** | **All BGs, 1 scipy call (C)** | **1** | **43.4** | **4.83×** |
| A* (haversine) | Point-to-point (sample) | timeout | >300 | N/A |

All SSSP methods produce identical rankings: Spearman ρ = 1.000, top-10 overlap 10/10.

**The production code (`model_approach2.py`) uses Multi-source Dijkstra** — the BG distance matrix is pre-computed in `load_all()` via a single `scipy.sparse.csgraph.dijkstra` call with all 554 BG source nodes.

### Weight Calibration (Harris Teeter)

| Parameter | Value | Interpretation |
|-----------|-------|---------------|
| W1 (demand) | 0.97 | Dominant — siting is demand-driven |
| W2 (accessibility) | 0.03 | Negligible — Charlotte network is uniform |
| W3 (competition) | ≈0.00 | Negligible — HT did not avoid rivals |
| Mean HT percentile | 62.6th | vs 50th random baseline |

These weights are descriptive of Harris Teeter's revealed preferences, not prescriptive for all retailers. The framework supports weight personalisation for any facility type.

### Component Independence (Diagnostics)

| Correlation | ρ | Interpretation |
|-------------|---|---------------|
| P–A | 0.383 | Moderate — different spatial information |
| P–S | 0.065 | Near zero — independent |
| A–S | 0.040 | Near zero — independent |

**Variance decomposition (W1=0.4, W2=0.3, W3=0.3):**
- Market Potential: 44.5% of score variance
- Competition: 18.2%
- Accessibility: 16.6%
- Cross-covariance: 20.6%

All three components contribute meaningfully to candidate differentiation.

## Architecture

```
├── model_approach2.py       # Charlotte scoring model (Multi-source Dijkstra)
├── model_nashville.py       # Nashville scoring model (BJJ / CrossFit)
├── app.py                   # Flask API server
├── component_diagnostics.py # Component independence analysis
├── templates/
│   └── index.html           # Interactive map UI
├── data/
│   ├── candidates_osm.geojson              # 2,450 OSM commercial candidates
│   ├── charlotte_roads_drive.geojson       # Road network (282K nodes)
│   ├── mecklenburg_bg_with_acs.geojson     # 554 Census block groups
│   ├── mecklenburg_bg_population_with_density.geojson
│   ├── all_grocery_stores_combined.geojson  # HT + competitors
│   ├── harris_teeter_ground_truth.geojson   # 28 HT stores
│   └── charlotte_boundary.geojson           # Study area boundary
├── cache/                   # Auto-generated caches (parquet, joblib, npy)
└── paper_v4_with_diagnostics.tex  # IEEE conference paper (LaTeX)
```

## Algorithms

### Multi-source (Reverse) Dijkstra

The accessibility computation requires one-to-all shortest-path distances, not point-to-point queries. The key insight is that on an **undirected** road graph, d(A,B) = d(B,A). Instead of running Dijkstra from each of 2,450 candidates (forward), we run from each of 554 block groups (reverse). Since N_cands > N_BG, this reduces the number of SSSP calls by a factor of N_cands / N_BG = 4.42×.

Multi-source Dijkstra takes this further: all 554 BG sources are passed to a single `scipy.sparse.csgraph.dijkstra` call, which executes the source loop in compiled C rather than Python. This eliminates Python loop overhead and achieves 4.83× total speedup.

**Implementation in `model_approach2.py`:**
1. `load_all()` calls `precompute_bg_distance_matrix()` which runs one `csr_dijkstra()` call with all 554 BG node indices → returns (554 × 282,751) distance matrix
2. `score_all_candidates_like_ht()` snaps each candidate to its nearest road node, then **looks up** distances from the pre-computed matrix — no per-candidate Dijkstra call
3. The matrix is cached to disk as `bg_dist_matrix.npy` for fast reload

### Why Point-to-Point Algorithms Fail

A*, Bidirectional Dijkstra, and Contraction Hierarchies accelerate individual source→target queries via goal-directed pruning. But this pruning provides **no benefit** when the full distance vector from a source is needed. The correct one-to-all extension is PHAST (Delling et al., 2011), which requires C++ implementations not available in standard Python toolkits.

We empirically confirmed this: A* with haversine heuristic completed only 7 of 50 sampled queries within a 5-minute timeout on Charlotte's dense urban grid.

## Formulas

### Demand Weight
```
B_i = 0.4·pop̃_i + 0.3·ĩnc_i + 0.3·d̃ens_i
```
where ~ denotes percentile-robust normalisation to [0.01, 1.0].

### Huff Market Share
```
Share_ik = t_ik^(-β) / (t_ik^(-β) + Σ_{j ∈ C_K(i)} t_ij^(-β))
```
C_K(i) = K nearest competitors (default K=3). Uses Euclidean travel time.

### Accessibility (Hansen Index)
```
t̄_k = Σ(B_i · t_ik^road) / Σ(B_i)    (demand-weighted avg Dijkstra travel time)
A_k = exp(-α · t̄_k)                    (α = 0.05)
```

### Competition Saturation
```
S_k = (all stores in radius) / (population / 10,000)
```
Counts ALL stores (HT + competitors) — market is saturated regardless of brand.

### Composite Score
```
Score_k = W1·P̃_k + W2·Ã_k - W3·S̃_k
```
W1 + W2 + W3 = 1. Score ∈ [-W3, W1+W2]. All components normalised before combining.

## Parameters

| Symbol | Default | Description |
|--------|---------|-------------|
| β | 2.0 | Huff distance-decay exponent |
| α | 0.05 | Hansen accessibility decay rate |
| K | 3 | Nearest competitors in Huff denominator |
| r | 5 miles | Catchment radius |
| d_min | 2.5 miles | Minimum separation for diversity selection |
| Speed | 35 mph | Assumed uniform driving speed |
| Circuity | 1.30 | Dijkstra limit = r × circuity factor |

## Setup

```bash
pip install flask geopandas scipy networkx joblib pandas numpy
python app.py
```

Visit `http://localhost:5000` for the interactive map.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_DIR` | `cache/` | Cache directory |
| `FORCE_REBUILD_CACHE` | `0` | Set to `1` to rebuild all caches |
| `LOAD_ONLY_CACHE` | `0` | Set to `1` to fail if cache missing |
| `PRELOAD_MODELS` | `1` | Set to `0` for lazy loading |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Interactive map UI |
| `/health` | GET | Health check |
| `/recompute` | GET/POST | Re-score with custom parameters |
| `/blocks` | GET | Census block group GeoJSON |

### `/recompute` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `facility` | string | `charlotte_ht` | Facility type (`charlotte_ht`, `nash_bjj`, `nash_cf`) |
| `radius_miles` | float | 5.0 | Catchment radius |
| `beta` | float | 2.0 | Huff decay exponent |
| `K` | int | 3 | Nearest competitors |
| `W1` | float | 0.4 | Demand weight |
| `W2` | float | 0.3 | Accessibility weight |
| `W3` | float | 0.3 | Competition weight |

## Scalability

| City | Nodes | Multi-source time | Speedup |
|------|-------|-------------------|---------|
| Charlotte, NC | 282,751 | 43s | 4.83× |
| Atlanta, GA | 450,000 | 70s | 4.83× |
| Houston, TX | 900,000 | 2.3 min | 4.83× |
| Los Angeles, CA | 1,400,000 | 3.6 min | 4.83× |

**Nationwide (384 US MSAs):** ~1.3 hours on 8 cores, ~10 min on cloud (64 vCPU).

## Data Sources

All publicly available:
- **Road network:** OpenStreetMap via `osmnx` (drive-accessible roads)
- **Demographics:** US Census American Community Survey (block group level)
- **Candidates:** OSM commercial landuse polygons (`landuse=retail/commercial`, `shop=supermarket`), area-filtered 500–500,000 m²
- **Stores:** Compiled from OSM, SafeGraph, and manual geocoding
- **CRS:** NAD83/North Carolina (EPSG:32119) for metric computation

## References

- Dijkstra, E.W. (1959). A note on two problems in connexion with graphs.
- Hansen, W.G. (1959). How accessibility shapes land use.
- Huff, D.L. (1964). Defining and estimating a trading area.
- Hart, P.E. et al. (1968). A formal basis for heuristic determination of minimum cost paths. (A*)
- Delling, D. et al. (2011). PHAST: Hardware-accelerated shortest path trees.
- Geisberger, R. et al. (2008). Contraction Hierarchies.
- ReVelle, C.S. & Eiselt, H.A. (2005). Location analysis: A synthesis and survey.

## License

Academic use. Contact authors for commercial licensing.