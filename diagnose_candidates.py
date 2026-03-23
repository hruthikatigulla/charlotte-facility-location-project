# =============================================================
# diagnose_candidates.py
#
# Answers three critical questions before paper decisions:
#
#   Q1. How were the 517 candidates generated?
#       (grid? block centroids? manual? commercial sites?)
#
#   Q2. Are existing HT stores inside the candidate pool?
#       (if yes, the model is trivially rewarding known answers)
#
#   Q3. What do Dijkstra access scores actually look like?
#       (variance, zero-rate, distribution)
#
# PLUS: checks candidate coverage of real HT locations
#       (how far is each HT store from its nearest candidate?)
#
# USAGE:
#   python diagnose_candidates.py
#
# OUTPUT printed to console + saved to results/diagnosis.txt
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_approach2 import (
    load_all,
    score_all_candidates_like_ht,
    MILE_M, CRS_M, CRS_LL,
    CANDIDATES_FILE,
)

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------

def fmt(title):
    return f"\n{'='*60}\n  {title}\n{'='*60}"


def diagnose():
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p(fmt("CANDIDATE POOL DIAGNOSIS"))
    p("Loading model state (uses cache if available)...")
    state = load_all()

    ht_m    = state["ht_m"].reset_index(drop=True)
    comp_m  = state["comp_m"].reset_index(drop=True)
    cands_m = state["cands_m"].reset_index(drop=True)
    cands_ll = state["cands_ll"].reset_index(drop=True)
    bg_m    = state["bg_m"].reset_index(drop=True)

    p(f"\nLoaded:")
    p(f"  HT stores    : {len(ht_m)}")
    p(f"  Competitors  : {len(comp_m)}")
    p(f"  Candidates   : {len(cands_m)}")
    p(f"  Block groups : {len(bg_m)}")

    # ===========================================================
    # Q1 — How were candidates generated?
    # ===========================================================
    p(fmt("Q1: HOW WERE CANDIDATES GENERATED?"))

    # Load raw candidates file to inspect columns
    try:
        raw_cands = gpd.read_file(CANDIDATES_FILE)
        p(f"\n  Raw candidates file: {CANDIDATES_FILE}")
        p(f"  Rows    : {len(raw_cands)}")
        p(f"  Columns : {list(raw_cands.columns)}")
        p(f"  CRS     : {raw_cands.crs}")
        p(f"  Geometry types: {raw_cands.geometry.geom_type.value_counts().to_dict()}")

        # Check if candidates look like block group centroids
        # by comparing to actual BG centroids
        bg_cent = bg_m.copy()
        if "cent" not in bg_cent.columns:
            bg_cent["cent"] = bg_cent.geometry.centroid
        bg_xy = np.c_[bg_cent["cent"].x.values, bg_cent["cent"].y.values]

        cand_xy = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]
        bg_tree  = cKDTree(bg_xy)
        dists, _ = bg_tree.query(cand_xy, k=1)
        dists_m  = dists  # already in metres (CRS_M)

        p(f"\n  Distance from each candidate to nearest BG centroid:")
        p(f"  Min  : {dists_m.min():.1f} m")
        p(f"  Mean : {dists_m.mean():.1f} m")
        p(f"  Max  : {dists_m.max():.1f} m")
        p(f"  Candidates within 10m of a BG centroid  : "
          f"{int(np.sum(dists_m <= 10))}")
        p(f"  Candidates within 100m of a BG centroid : "
          f"{int(np.sum(dists_m <= 100))}")

        if np.sum(dists_m <= 10) > len(cands_m) * 0.8:
            p("\n  FINDING: Candidates ARE block group centroids (>80% match within 10m)")
            p("  IMPLICATION: Candidates are sparse — one per BG, not a fine grid.")
            p("  Many real HT stores may fall between candidates.")
        elif np.sum(dists_m <= 100) > len(cands_m) * 0.5:
            p("\n  FINDING: Candidates are NEAR block group centroids (~100m tolerance)")
        else:
            p("\n  FINDING: Candidates do NOT appear to be BG centroids")
            p("  They may be a custom grid, commercial parcels, or intersections.")

        # Check if they look like a regular grid
        if len(cand_xy) > 10:
            xs = np.sort(np.unique(np.round(cand_xy[:, 0], -2)))  # round to 100m
            ys = np.sort(np.unique(np.round(cand_xy[:, 1], -2)))
            p(f"\n  Unique X positions (100m bins): {len(xs)}")
            p(f"  Unique Y positions (100m bins): {len(ys)}")
            if len(xs) * len(ys) > len(cands_m) * 0.5:
                p("  Pattern suggests a REGULAR GRID")
            else:
                p("  Pattern does NOT suggest a regular grid")

        # Non-geometry columns — any useful metadata?
        non_geom = [c for c in raw_cands.columns if c != "geometry"]
        if non_geom:
            p(f"\n  Non-geometry columns: {non_geom}")
            for col in non_geom[:5]:
                try:
                    vals = raw_cands[col].dropna()
                    if vals.dtype in [float, int] or pd.api.types.is_numeric_dtype(vals):
                        p(f"    {col}: min={vals.min():.2f}  "
                          f"mean={vals.mean():.2f}  max={vals.max():.2f}")
                    else:
                        p(f"    {col}: {vals.value_counts().head(5).to_dict()}")
                except Exception:
                    pass

    except Exception as e:
        p(f"  Could not read raw candidates file: {e}")

    # ===========================================================
    # Q2 — Are HT stores inside the candidate pool?
    # ===========================================================
    p(fmt("Q2: ARE EXISTING HT STORES IN THE CANDIDATE POOL?"))

    ht_xy   = np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values]
    cand_xy = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]
    cand_tree = cKDTree(cand_xy)

    dists_ht, nearest_idx = cand_tree.query(ht_xy, k=1)
    dists_ht_m  = dists_ht
    dists_ht_mi = dists_ht / MILE_M

    p(f"\n  Distance from each HT store to its nearest candidate:")
    p(f"  {'Store #':<10} {'Dist (m)':>10} {'Dist (mi)':>10} {'Assessment':>20}")
    p(f"  {'-'*55}")
    for i in range(len(ht_m)):
        assessment = ("IN POOL (<50m)" if dists_ht_m[i] < 50
                      else "VERY CLOSE (<200m)" if dists_ht_m[i] < 200
                      else "CLOSE (<0.5mi)" if dists_ht_mi[i] < 0.5
                      else "NEAR (0.5-1mi)" if dists_ht_mi[i] < 1.0
                      else "FAR (1-2mi)" if dists_ht_mi[i] < 2.0
                      else "VERY FAR (>2mi)")
        p(f"  {i+1:<10} {dists_ht_m[i]:>10.1f} {dists_ht_mi[i]:>10.3f} "
          f"{assessment:>20}")

    n_in_pool     = int(np.sum(dists_ht_m < 50))
    n_very_close  = int(np.sum(dists_ht_m < 200))
    n_within_half = int(np.sum(dists_ht_mi < 0.5))
    n_within_1mi  = int(np.sum(dists_ht_mi < 1.0))
    n_over_2mi    = int(np.sum(dists_ht_mi >= 2.0))

    p(f"\n  Summary:")
    p(f"  HT stores effectively IN candidate pool (<50m)  : {n_in_pool}/{len(ht_m)}")
    p(f"  HT stores within 200m of a candidate           : {n_very_close}/{len(ht_m)}")
    p(f"  HT stores within 0.5 miles of a candidate      : {n_within_half}/{len(ht_m)}")
    p(f"  HT stores within 1.0 mile of a candidate       : {n_within_1mi}/{len(ht_m)}")
    p(f"  HT stores MORE THAN 2 miles from any candidate : {n_over_2mi}/{len(ht_m)}")

    if n_in_pool > len(ht_m) * 0.5:
        p("\n  FINDING: Most HT stores ARE in the candidate pool.")
        p("  This is fine for validation — the model should rank them highly.")
        p("  But check that this doesn't inflate validation scores artificially.")
    elif n_over_2mi > len(ht_m) * 0.3:
        p(f"\n  CRITICAL FINDING: {n_over_2mi} HT stores are >2 miles from any candidate.")
        p("  The optimiser CANNOT reward these stores regardless of weights.")
        p("  This explains why mean_rank stays ~171-240 even with best weights.")
        p("  Fix: Add points near real HT locations to the candidate pool.")
    else:
        p(f"\n  FINDING: Candidates are sparse but not severely misaligned.")
        p(f"  Mean nearest-candidate distance = {dists_ht_mi.mean():.2f} miles.")

    # ===========================================================
    # Q3 — What do Dijkstra access scores look like?
    # ===========================================================
    p(fmt("Q3: DIJKSTRA ACCESS SCORE DISTRIBUTION"))
    p("\nRunning scorer to get access scores for all candidates...")
    p("(Using default weights — we just want the access_score_dj column)")

    try:
        _, _, _, all_scored = score_all_candidates_like_ht(
            state,
            radius_miles=5.0,
            beta=2.0,
            K=3,
            return_all=True,
        )

        if "access_score_dj" not in all_scored.columns:
            p("  ERROR: access_score_dj column not found in scored output.")
        else:
            acc = all_scored["access_score_dj"].values
            pot = all_scored["potential_norm"].values
            s10k = all_scored["stores_per_10k"].values
            score = all_scored["pair_score"].values

            p(f"\n  {'Metric':<25} {'Min':>8} {'Mean':>8} {'Median':>8} "
              f"{'Max':>8} {'Std':>8} {'Zero%':>8}")
            p(f"  {'-'*73}")
            for name, arr in [
                ("access_score_dj",  acc),
                ("potential_norm",   pot),
                ("stores_per_10k",   s10k),
                ("pair_score",       score),
            ]:
                zero_pct = 100.0 * np.sum(arr <= 0.011) / len(arr)
                p(f"  {name:<25} {arr.min():>8.3f} {arr.mean():>8.3f} "
                  f"{np.median(arr):>8.3f} {arr.max():>8.3f} "
                  f"{arr.std():>8.3f} {zero_pct:>7.1f}%")

            n_zero_access = int(np.sum(acc <= 0.011))
            p(f"\n  Candidates with access_score_dj near-zero (<=0.011): "
              f"{n_zero_access}/{len(acc)} ({100*n_zero_access/len(acc):.0f}%)")

            if n_zero_access > len(acc) * 0.3:
                p("\n  CRITICAL FINDING: >30% of candidates have zero accessibility.")
                p("  This means the road graph does not cover those areas.")
                p("  Likely cause: OSM road data clipped too tightly to boundary,")
                p("  or graph has disconnected components.")
                p("  When access_score_dj=0 for most candidates, W2 is useless")
                p("  — accessibility cannot distinguish candidates.")
            elif acc.std() < 0.05:
                p("\n  WARNING: Very low variance in access scores (std < 0.05).")
                p("  Accessibility barely differentiates candidates.")
                p("  This weakens it as a ranking factor.")
            else:
                p(f"\n  FINDING: Access scores have reasonable variance (std={acc.std():.3f}).")
                p("  Road graph coverage appears adequate.")

            # Correlation between components
            p(f"\n  Pearson correlations between score components:")
            from numpy import corrcoef
            for (n1, a1), (n2, a2) in [
                (("potential_norm", pot),  ("access_score_dj", acc)),
                (("potential_norm", pot),  ("stores_per_10k",  s10k)),
                (("access_score_dj", acc), ("stores_per_10k",  s10k)),
            ]:
                valid = np.isfinite(a1) & np.isfinite(a2)
                if valid.sum() > 5:
                    r = corrcoef(a1[valid], a2[valid])[0, 1]
                    p(f"  {n1} vs {n2}: r = {r:.3f}")

            # ===========================================================
            # BONUS: What rank do real HT stores get with DEFAULT weights?
            # ===========================================================
            p(fmt("BONUS: HT STORE RANKS WITH DEFAULT WEIGHTS"))
            scores_arr = all_scored["pair_score"].values
            order      = np.argsort(-scores_arr)
            rank_lookup = np.empty_like(order)
            rank_lookup[order] = np.arange(1, len(scores_arr) + 1)

            cand_xy_all = np.c_[
                all_scored.to_crs(CRS_M).geometry.x.values,
                all_scored.to_crs(CRS_M).geometry.y.values,
            ]
            cand_tree2 = cKDTree(cand_xy_all)

            p(f"\n  {'Store #':<10} {'Rank':>6} {'/ Total':>8} "
              f"{'Percentile':>12} {'Dist to cand (mi)':>20}")
            p(f"  {'-'*60}")
            all_ranks, all_pcts = [], []
            for i in range(len(ht_m)):
                hx = float(ht_m.iloc[i].geometry.x)
                hy = float(ht_m.iloc[i].geometry.y)
                d, nearest = cand_tree2.query([hx, hy], k=1)
                rank = int(rank_lookup[nearest])
                pct  = (1.0 - rank / len(scores_arr)) * 100
                all_ranks.append(rank)
                all_pcts.append(pct)
                p(f"  {i+1:<10} {rank:>6} {len(scores_arr):>8} "
                  f"{pct:>11.1f}% {d/MILE_M:>20.3f}")

            p(f"\n  Mean rank      : {np.mean(all_ranks):.1f} / {len(scores_arr)}")
            p(f"  Median rank    : {np.median(all_ranks):.1f} / {len(scores_arr)}")
            p(f"  Mean percentile: {np.mean(all_pcts):.1f}th")
            p(f"  Random baseline: 50th percentile")

            if np.mean(all_pcts) > 55:
                p(f"\n  GOOD: Model scores real HT stores above random baseline.")
            elif np.mean(all_pcts) > 45:
                p(f"\n  NEUTRAL: Model performs close to random baseline.")
                p(f"  Suggests the three components have limited predictive power")
                p(f"  OR the candidate grid doesn't align with HT locations.")
            else:
                p(f"\n  POOR: Model scores real HT stores BELOW random baseline.")
                p(f"  This is almost certainly a candidate coverage problem.")
                p(f"  Check Q2 results above.")

    except Exception as e:
        p(f"  Scorer failed: {e}")
        import traceback
        p(traceback.format_exc())

    # ===========================================================
    # SUMMARY AND RECOMMENDED ACTIONS
    # ===========================================================
    p(fmt("SUMMARY AND RECOMMENDED NEXT STEPS"))
    p("""
  Based on these diagnostics, take the following actions
  BEFORE making any paper decisions:

  If Q1 shows candidates are BG centroids:
    -> Regenerate candidates as a 0.5-mile grid over the metro area
    -> This gives ~2000-3000 candidates at fine resolution
    -> Ensures every HT location has a candidate within 0.25 miles

  If Q2 shows >5 HT stores are >2 miles from any candidate:
    -> The calibration is invalid — optimiser cannot reward those stores
    -> Add the 28 real HT coordinates directly to the candidate pool
       (for calibration/validation purposes only)
    -> Re-run weight_calibration.py after fixing

  If Q3 shows >30% of candidates have zero access score:
    -> Road graph has coverage gaps
    -> Check that charlotte_roads_drive.geojson covers the full metro
    -> Consider using a 1.5x boundary buffer when clipping roads

  If all three look reasonable:
    -> Proceed to paper writing with honest validation claims
    -> The model is a spatial screening tool, not a prediction model
    -> Frame the weight result (high W2) as a finding about HT strategy
    """)

    # Save report
    out_path = os.path.join("results", "diagnosis.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[Saved] Full diagnosis written to {out_path}")


if __name__ == "__main__":
    diagnose()