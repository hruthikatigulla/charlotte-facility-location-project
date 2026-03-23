from flask import Flask, request, jsonify, render_template
import logging
import os
import json

from model_approach2 import load_all as load_charlotte, score_all_candidates_like_ht
from model_nashville import load_nashville, score_nashville_bjj, score_nashville_cf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

STATE_CHARLOTTE = None
STATE_NASHVILLE = None


def _should_init_models_now() -> bool:
    if not app.debug:
        return True
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true"


def load_models(preload: bool = True):
    global STATE_CHARLOTTE, STATE_NASHVILLE
    if not preload:
        logger.info("PRELOAD_MODELS=0 → lazy loading enabled.")
        return
    logger.info("Loading Charlotte model...")
    STATE_CHARLOTTE = load_charlotte()
    logger.info("Charlotte model loaded.")
    logger.info("Loading Nashville model...")
    STATE_NASHVILLE = load_nashville()
    logger.info("Nashville model loaded.")
    logger.info("All models loaded.")


def ensure_model_loaded(facility: str):
    global STATE_CHARLOTTE, STATE_NASHVILLE
    if facility == "charlotte_ht":
        if STATE_CHARLOTTE is None:
            logger.info("Lazy-loading Charlotte model...")
            STATE_CHARLOTTE = load_charlotte()
        return STATE_CHARLOTTE
    if facility in ("nash_bjj", "nash_cf"):
        if STATE_NASHVILLE is None:
            logger.info("Lazy-loading Nashville model...")
            STATE_NASHVILLE = load_nashville()
        return STATE_NASHVILLE
    if STATE_CHARLOTTE is None:
        logger.info("Lazy-loading Charlotte model (default)...")
        STATE_CHARLOTTE = load_charlotte()
    return STATE_CHARLOTTE


def gdf_points_to_list(gdf_m):
    if gdf_m is None or gdf_m.empty:
        return []
    ll = gdf_m.to_crs(4326)
    return [[float(g.y), float(g.x)] for g in ll.geometry]


def _parse_float(raw, default):
    try:
        return float(raw)
    except Exception:
        return default


def _parse_int(raw, default):
    try:
        return int(raw)
    except Exception:
        return default


# ---- Boot ----
logger.info("Initializing application...")

PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "1").strip()
preload_flag   = PRELOAD_MODELS not in ("0", "false", "False", "no", "NO")

if _should_init_models_now():
    load_models(preload=preload_flag)
else:
    logger.info("Skipping model load in reloader parent process.")


# =============================================================
# ROUTES
# =============================================================

@app.get("/health")
def health():
    return jsonify({
        "ok":               True,
        "status":           "healthy",
        "preload_models":   preload_flag,
        "charlotte_loaded": STATE_CHARLOTTE is not None,
        "nashville_loaded": STATE_NASHVILLE is not None,
    }), 200


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recompute", methods=["GET", "POST"])
def recompute():
    raw      = (request.get_json(silent=True) or {}) if request.method == "POST" \
               else request.args
    facility = raw.get("facility", "charlotte_ht")
    logger.info(f"Recompute: facility={facility}")

    state  = ensure_model_loaded(facility)
    radius = _parse_float(raw.get("radius_miles", 5.0),  5.0)
    beta   = _parse_float(raw.get("beta",         2.0),  2.0)
    K      = _parse_int(  raw.get("K",             3),    3)
    W1     = _parse_float(raw.get("W1",            0.4),  0.4)
    W2     = _parse_float(raw.get("W2",            0.3),  0.3)
    W3     = _parse_float(raw.get("W3",            0.3),  0.3)

    if facility == "charlotte_ht":
        scorer = score_all_candidates_like_ht
    elif facility == "nash_bjj":
        scorer = score_nashville_bjj
    elif facility == "nash_cf":
        scorer = score_nashville_cf
    else:
        return jsonify({"ok": False, "error": f"Unknown facility '{facility}'"}), 400

    try:
        if facility == "charlotte_ht":
            top10, heat_points, target_gdf = scorer(
                state, radius_miles=radius, beta=beta, K=K, W1=W1, W2=W2, W3=W3)
        else:
            top10, heat_points, target_gdf = scorer(
                state, radius_miles=radius, beta=beta, K=K)
    except Exception as e:
        logger.error(f"Scoring error: {e}", exc_info=True)
        return jsonify({"ok": False, "error": f"Scoring failed: {e}"}), 500

    top10_payload = []
    for _, r in top10.iterrows():
        g = r.geometry
        top10_payload.append({
            "lat":             float(g.y),
            "lon":             float(g.x),
            "score":           float(r.get("pair_score",      0.0)),
            "stores_per_10k":  float(r.get("stores_per_10k",  0.0)),
            "income_med":      float(r.get("income_med",       0.0)),
            "access_score_dj": float(r.get("access_score_dj", 0.0)),
        })

    ht_payload = []
    if target_gdf is not None and not target_gdf.empty:
        ht_ll = target_gdf.to_crs(4326)
        for _, r in ht_ll.iterrows():
            g = r.geometry
            ht_payload.append({
                "lat":             float(g.y),
                "lon":             float(g.x),
                "score":           float(r.get("pair_score",      0.0)),
                "stores_per_10k":  float(r.get("stores_per_10k",  0.0)),
                "income_med":      float(r.get("income_med",       0.0)),
                "access_score_dj": float(r.get("access_score_dj", 0.0)),
            })

    if facility == "charlotte_ht":
        existing    = gdf_points_to_list(state["ht_m"])
        competitors = gdf_points_to_list(state["comp_m"])
    elif facility == "nash_bjj":
        existing    = gdf_points_to_list(state["bjj_m"])
        competitors = gdf_points_to_list(state["cf_m"]) + gdf_points_to_list(state["others_m"])
    else:
        existing    = gdf_points_to_list(state["cf_m"])
        competitors = gdf_points_to_list(state["bjj_m"]) + gdf_points_to_list(state["others_m"])

    return jsonify({
        "ok":          True,
        "heat_points": heat_points,
        "top10":       top10_payload,
        "ht_scored":   ht_payload,
        "existing_ht": existing,
        "competitors": competitors,
    })


@app.get("/blocks")
def blocks():
    facility   = request.args.get("facility", "charlotte_ht")
    cache_path = os.path.join("cache", f"blocks_{facility}.json")

    # ---- Serve from disk cache if available (fast path) ----
    if os.path.exists(cache_path):
        logger.info(f"[blocks] Serving {facility} from disk cache")
        with open(cache_path, "r") as f:
            return app.response_class(f.read(), mimetype="application/json")

    # ---- Build GeoJSON ----
    logger.info(f"[blocks] Building GeoJSON for {facility} (first time, will cache)")
    state = ensure_model_loaded(facility)
    bg    = state["bg_m"].to_crs(4326)
    cols  = set(bg.columns)

    features = []
    for _, row in bg.iterrows():
        geom      = row.geometry.__geo_interface__
        geo_id    = row["GEOID"]         if "GEOID"         in cols else None
        pop       = float(row["population"]   if "population"   in cols else 0) or 0
        area_sqmi = float(row["area_sqmi"]    if "area_sqmi"    in cols else 0) or 0
        pop_sqmi  = float(row["pop_per_sqmi"] if "pop_per_sqmi" in cols else 0) or 0

        if   "median_income" in cols: med_inc = float(row["median_income"] or 0)
        elif "income"        in cols: med_inc = float(row["income"]        or 0)
        else:                         med_inc = 0.0

        features.append({
            "type":     "Feature",
            "geometry": geom,
            "properties": {
                "GEOID":         geo_id,
                "population":    pop,
                "area_sqmi":     area_sqmi,
                "pop_per_sqmi":  pop_sqmi,
                "median_income": med_inc,
            },
        })

    result = {"type": "FeatureCollection", "features": features}

    # ---- Save to disk cache ----
    try:
        os.makedirs("cache", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)
        logger.info(f"[blocks] Cached to {cache_path}")
    except Exception as e:
        logger.warning(f"[blocks] Could not save cache: {e}")

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
```

Now **pre-build the cache before starting** so the first page load is instant. Run this once in your terminal:
```
python -c "
from app import app
with app.test_client() as c:
    c.get('/blocks?facility=charlotte_ht')
    c.get('/blocks?facility=nash_bjj')
    c.get('/blocks?facility=nash_cf')
print('Done')
"