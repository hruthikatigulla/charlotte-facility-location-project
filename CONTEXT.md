# Project Context: Multi-City Site Selection (Charlotte & Nashville)

## Runtime Architecture
- Flask app (`app.py`) loads model states once at startup:
  - `STATE_CHARLOTTE = model_approach2.load_all()`
  - `STATE_NASHVILLE = model_nashville.load_nashville()`
- `/recompute` endpoint selects scorer by `facility`:
  - `charlotte_ht` → `score_all_candidates_like_ht`
  - `nash_bjj` → `score_nashville_bjj`
  - `nash_cf` → `score_nashville_cf`
- Each scorer returns:
  - `top10` candidate sites (GeoDataFrame, lat/lon)
  - `heat_points` for UI heatmap
  - scored existing brand facilities

## Core Modeling Concepts

### Demand Units
- Census Block Groups (BGs)
- Geometry: polygons
- Demand point: BG centroid
- Attributes: population, income, density (if available)

### Robust Normalization
For any numeric array:
- Ignore non-finite values
- Use only values > 0
- Compute 1st–99th percentile bounds
- Clip to bounds, scale to [0,1]
- Final mapping: `0.01 + 0.99 * scaled`

### Block Weight Formula
```
block_weight = alpha* pop_w + beta* inc_w + gamma * dens_w
```

Where each term is robust-normalized.

## Competitor Travel Times (Euclidean)
- KDTree over competitor locations
- For each BG centroid, query K nearest competitors
- Convert distance to minutes using constant speed:
  - `SPEED = 35 mph`

Produces reusable matrix: `bg_tcomp[Nbg, K]`.

## Huff Gravity Model
For each BG:

```
A_new  = 1 / t_new^beta
A_comp = sum_k(1 / t_comp_k^beta)

share = A_new / (A_new + A_comp)
```

Candidate Huff potential:

```
potential = sum_bg(block_weight * share)
```

Then normalized via robust normalization to `potential_norm ∈ [0.01, 1.0]`.

## Local Competition Penalty
Within candidate buffer (radius miles):
- Count all stores (brand + competitors)
- Estimate population via BG ∩ buffer area weighting

```
stores_per_10k = stores / (pop_buf / 10000)
```

## Road Network & Reverse Dijkstra Accessibility

### Graph Construction (load-time)
- Undirected NetworkX graph
- Nodes: road LineString endpoints
- Edge weight: Euclidean distance between endpoints
- KDTree built over node coordinates
- Each BG centroid snapped to nearest road node → `bg_node_ids`

### Reverse Dijkstra (score-time)
For each candidate:
1. Snap candidate to nearest road node
2. Run:
   ```
   nx.single_source_dijkstra_path_length(G, cand_node)
   ```
3. Read distances to each BG node
4. Convert to travel time using same constant speed
5. Compute block-weighted mean travel time:
   ```
   weighted_time = sum(w * t) / sum(w)
   ```
6. Map to accessibility score:
   ```
   access_score_dj = clip(1 / (1 + t_hours), 0.01, 1.0)
   ```

Note: This is equivalent to BG→candidate distances because the graph is undirected.

## Final Scoring Formula
With defaults `W1 = W2 = W3 = 1`:

```
pair_score =
    W1 * potential_norm
  + W2 * access_score_dj
  - W3 * (lambda * stores_per_10k)
```

Heatmap intensity is min–max scaled `pair_score`.

## Spatial Diversity Filter
- Greedy top-N selection
- Enforces minimum separation distance (`min_sep_miles`)
- Prevents clustered recommendations

## Nashville-Specific Wrappers
- `score_nashville_bjj`:
  - Brand = BJJ gyms
  - Competitors = CrossFit + others
- `score_nashville_cf`:
  - Brand = CrossFit gyms
  - Competitors = BJJ + others
- Both call shared `_core_score`

---
This document is intended to be pasted at the start of a new ChatGPT session to restore full project context without re-explaining architecture or formulas.
