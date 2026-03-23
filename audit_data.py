# =============================================================
# audit_data.py
#
# Audits every data file used by the model.
# Reports: row counts, column ranges, CRS, geometry types,
#          missing values, and known issues.
#
# USAGE:
#   python audit_data.py
#
# Paste the full output here so we know exactly what to fix.
# =============================================================

import os
import sys
import json
import math

DATA_DIR = "data"

FILES = {
    # Exact filenames from model_approach2.py constants
    "1_stores":     "all_grocery_stores_combined.geojson",   # STORES_FILE
    "2_ht_truth":   "harris_teeter_ground_truth.geojson",    # HT_GROUND_TRUTH_FILE
    "3_bg_acs":     "mecklenburg_bg_with_acs.geojson",       # BG_ACS_FILE
    "4_bg_density": "mecklenburg_bg_population_with_density.geojson",  # BG_POPDENS_FILE
    "5_candidates": "candidates_scored.geojson",             # CANDIDATES_FILE
    "6_roads":      "charlotte_roads_drive.geojson",         # ROADS_FILE (size-limited)
    "7_boundary":   "charlotte_boundary.geojson",            # BOUNDARY_FILE
}

# ---------------------------------------------------------------

def _safe_float(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except:
        return None


def audit_geojson(label, path):
    print(f"\n{'='*60}")
    print(f"  {label.upper()}: {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"  FILE NOT FOUND: {path}")
        return

    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"  File size: {size_mb:.1f} MB")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    features = data.get("features", [])
    print(f"  Features: {len(features)}")

    if not features:
        print("  EMPTY FILE — no features")
        return

    # Geometry
    geom_types = {}
    null_geoms  = 0
    for feat in features:
        g = feat.get("geometry")
        if g is None:
            null_geoms += 1
        else:
            t = g.get("type","unknown")
            geom_types[t] = geom_types.get(t, 0) + 1

    print(f"  Geometry types: {geom_types}")
    if null_geoms:
        print(f"  NULL geometries: {null_geoms}")

    # CRS
    crs = data.get("crs", None)
    if crs:
        print(f"  CRS: {crs}")
    else:
        print(f"  CRS: not specified (assumed EPSG:4326)")

    # Properties
    props0 = features[0].get("properties", {}) or {}
    cols   = list(props0.keys())
    print(f"  Columns ({len(cols)}): {cols}")

    if not cols:
        print("  No property columns.")
        return

    # Per-column stats
    col_data = {c: [] for c in cols}
    col_str  = {c: [] for c in cols}

    for feat in features:
        p = feat.get("properties") or {}
        for c in cols:
            v = p.get(c)
            f = _safe_float(v)
            if f is not None:
                col_data[c].append(f)
            elif v is not None:
                col_str[c].append(str(v))

    print(f"\n  {'Column':<35} {'Type':<8} {'Count':>6} {'Min':>14} "
          f"{'Mean':>14} {'Max':>14} {'Nulls':>6} {'In[0,1]':>8}")
    print(f"  {'-'*110}")

    for c in cols:
        nums  = col_data[c]
        strs  = col_str[c]
        nulls = len(features) - len(nums) - len(strs)

        if nums:
            mn   = min(nums)
            mx   = max(nums)
            avg  = sum(nums) / len(nums)
            in01 = (mn >= -0.001) and (mx <= 1.001)
            # Flag suspicious values
            flags = []
            if mn < -1e6:
                flags.append("NEGATIVE_LARGE")
            if mx > 1e9:
                flags.append("OVERFLOW?")
            if len(set(round(x, 4) for x in nums)) == 1:
                flags.append("ALL_SAME")
            zeros = sum(1 for x in nums if x == 0)
            if zeros > len(nums) * 0.5:
                flags.append(f"{zeros/len(nums)*100:.0f}%_ZEROS")
            flag_str = " ".join(flags)
            print(f"  {c:<35} {'num':<8} {len(nums):>6} {mn:>14.4f} "
                  f"{avg:>14.4f} {mx:>14.4f} {nulls:>6} {str(in01):>8}"
                  f"  {flag_str}")
        elif strs:
            uniq = len(set(strs))
            sample = list(set(strs))[:3]
            print(f"  {c:<35} {'str':<8} {len(strs):>6} "
                  f"{'':>14} {'':>14} {'':>14} {nulls:>6} {'':>8}"
                  f"  unique={uniq} sample={sample}")
        else:
            print(f"  {c:<35} {'empty':<8} {'0':>6}  (all null)")

    # Bounding box from coordinates
    if features[0].get("geometry"):
        all_x, all_y = [], []
        for feat in features[:500]:  # sample first 500
            g = feat.get("geometry")
            if not g:
                continue
            t = g.get("type")
            coords = g.get("coordinates", [])
            try:
                if t == "Point":
                    all_x.append(coords[0])
                    all_y.append(coords[1])
                elif t in ("LineString", "MultiPoint"):
                    for c in coords:
                        all_x.append(c[0]); all_y.append(c[1])
                elif t in ("Polygon", "MultiLineString"):
                    for ring in coords:
                        for c in ring:
                            all_x.append(c[0]); all_y.append(c[1])
            except:
                pass

        if all_x:
            print(f"\n  Bounding box (sample of 500):")
            print(f"    X (lon): {min(all_x):.4f} to {max(all_x):.4f}")
            print(f"    Y (lat): {min(all_y):.4f} to {max(all_y):.4f}")
            # Sanity check for Charlotte, NC
            expected_lon = (-81.1, -80.6)
            expected_lat = (35.0,  35.6)
            lon_ok = (min(all_x) > expected_lon[0] and
                      max(all_x) < expected_lon[1])
            lat_ok = (min(all_y) > expected_lat[0] and
                      max(all_y) < expected_lat[1])
            if lon_ok and lat_ok:
                print(f"    Location check: OK (Charlotte metro area)")
            else:
                print(f"    Location check: WARNING — outside expected "
                      f"Charlotte bbox {expected_lon}, {expected_lat}")


# ---------------------------------------------------------------

def audit_stores(path):
    """Extra check: how many HT vs competitors, brand breakdown."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    features = data.get("features", [])
    brands   = {}
    for feat in features:
        p = feat.get("properties") or {}
        brand = str(p.get("brand", p.get("name", "unknown"))).strip().lower()
        brands[brand] = brands.get(brand, 0) + 1

    print(f"\n  Brand breakdown (top 15):")
    for b, n in sorted(brands.items(), key=lambda x: -x[1])[:15]:
        ht_flag = " <-- HT" if "harris" in b or b == "ht" else ""
        print(f"    {b:<40} {n:>4}{ht_flag}")


# ---------------------------------------------------------------

def main():
    print("DATA AUDIT REPORT")
    print("Charlotte HT Site Selection Project")
    print("="*60)

    for label, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)

        # Skip roads full audit — 45MB, just confirm it exists
        if label == "6_roads":
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024*1024)
                print(f"\n{'='*60}")
                print(f"  6_ROADS: {fname}")
                print(f"{'='*60}")
                print(f"  File size: {size_mb:.1f} MB")
                # Read first feature only to check geometry type
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    raw = f.read(8000)
                if '"LineString"' in raw or '"MultiLineString"' in raw:
                    print(f"  Geometry: LineString/MultiLineString (correct for roads)")
                else:
                    print(f"  WARNING: expected LineString geometry not found in first 8KB")
                print(f"  File exists and readable: YES")
            else:
                print(f"\n  6_ROADS: FILE NOT FOUND — Dijkstra will be disabled")
            continue

        audit_geojson(label, path)

        # Extra brand breakdown for stores file
        if label == "1_stores":
            audit_stores(path)

    print(f"\n{'='*60}")
    print(f"  AUDIT COMPLETE")
    print(f"{'='*60}")
    print("""
  What to look for in the output:
  - NEGATIVE_LARGE: broken computation (e.g. income_5mi_est)
  - ALL_SAME: column has no variance (useless for scoring)
  - 50%+_ZEROS: most values are zero (sparse data)
  - In[0,1]=False: not normalised
  - Location check WARNING: data is in wrong place
  - NULL geometries: features with no location
    """)


if __name__ == "__main__":
    main()