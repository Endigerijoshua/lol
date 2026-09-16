# AGENTS.md — SIH26162 Industrial Fire Detection

Instructions for AI coding assistants working on this repo. Read this fully before
writing any code. When in doubt, prioritize the constraints in [NON-NEGOTIABLE RULES](#non-negotiable-rules).

## Project Overview

Smart India Hackathon 2026 (Software Edition) project. Problem ID **SIH26162** —
*"AI-Based Detection and Classification of Industrial Fires and Persistent Thermal
Sources Using NASA FIRMS, OSM & Satellite Data"*. Theme: Disaster Management.
Org: **NTRO**. Deadline: **30 September 2026**.

Goal: detect fire/thermal hotspots near industrial zones in India from **live NASA
FIRMS satellite data**, classify one-off fires vs. recurring **persistent thermal
sources**, rank by severity, and render everything on a live map dashboard
(Leaflet.js).

## Team Context (shapes ALL decisions)

- Beginner "vibecoder" BCA students. Heavy reliance on AI coding assistants. No deep
  ML / data-science background.
- Deliberate approach: **no custom CV, no hand-labelled datasets, no GPU**. The one
  sanctioned exception is the weak-label RandomForest in `scripts/train_classifier.py`:
  labels come from OUR OWN rule-based spatial join (`fire_type_rule`), the model is
  out-of-the-box scikit-learn RandomForestClassifier (n_estimators=100, no hyperparameter
  tuning) wrapped in CalibratedClassifierCV (isotonic, cv=5) so `predict_proba` gives
  calibrated probabilities, and the artifact is git-ignored. Everything else runs on
  off-the-shelf APIs + classical geospatial logic + out-of-the-box algorithms (e.g.
  sklearn DBSCAN).
- Priority order: (1) must run end-to-end and demo live, (2) must look technically
  credible to judges, (3) must stay debuggable by beginners.

## Tech Stack

| Layer      | Choice                                              | Why                                            |
| ---------- | --------------------------------------------------- | ---------------------------------------------- |
| Backend    | Python 3.11+ / FastAPI                              | Best geospatial lib support; simple async API  |
| Frontend   | Leaflet.js (plain JS)                               | Renders GeoJSON layers with minimal glue code  |
| Database   | SQLite (built-in `sqlite3` / SQLAlchemy)            | Zero setup. PostGIS upgrade only if bottleneck |
| GIS        | Shapely + GeoPandas                                 | Point-in-polygon, buffering, nearest-distance  |
| ML (naive) | scikit-learn DBSCAN + weak-label RandomForest (out-of-the-box) | DBSCAN groups recurring hotspots into "sites"; RF classifies fire_type from our own rule labels |

## Data Sources

### 1) NASA FIRMS (fire/thermal hotspots)
- `MAP_KEY` obtained (in `.env`, gitignored). ✅
- **PRIMARY SOURCE = CSV area API** (empirically verified — the WFS bbox filter returns
  0 features for every bbox tested, but the CSV area API respects the bbox correctly).
  Endpoint:
  ```
  https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{DATASET}/{bbox}/1
  ```
  - `DATASET` for VIIRS: `VIIRS_SNPP_NRT` (also `MODIS_NRT`, `VIIRS_NOAA20_NRT`,
    `VIIRS_NOAA21_NRT`).
  - `bbox` format: `west,south,east,north` (decimal, no CRS) — i.e. `68,6,97,37` for India.
  - The `/1` is the day-lookback window (1 = last 24h).
- Backend converts CSV → clean GeoJSON in `app/services/firms.py` using **pandas**
  (fields kept verbatim: `confidence`, `bright_ti4`, `frp`, `acq_date`, `acq_time`,
  `satellite`, `daynight`).
- Region coverage: `SouthEast_Asia` service includes India. India bbox used:
  `68,6,97,37` (verified — 208 live VIIRS fires returned on 2026-09-04).
- **India-only boundary filter (MUST KEEP)**: the FIRMS bbox is a *rectangle* and
  therefore also returns fires in neighbouring countries (Pakistan, China, Nepal,
  Bhutan, Bangladesh, Myanmar, Sri Lanka). `app/services/firms.py` clips every
  response to the real India polygon in `app/data/india_boundary.geojson`
  (Natural Earth 10m admin-0, public domain — includes the Andaman & Nicobar
  islands; the coarser 110m file was rejected because it drops NE border states
  and island territories). `filter_to_india_boundary()` runs inside
  `fetch_fires()`, so it applies to ALL live endpoints (`/api/fires`,
  `/api/flagged-fires`, `/api/thermal-sites`) AND the training-script backfill —
  one choke point, no bbox-only leaks. `india_boundary_buffer_deg` defaults to
  **0.0 (strict polygon)** — demo integrity requires ZERO out-of-boundary
  detections; only raise it if FIRMS ~375 m point error genuinely drops
  desired border fires (2026-09-06: 0 out-of-boundary in a live 1,258-fire
  window at 0.0).
- Refresh: ~15 min. Rate limit: 5,000 req / 10 min — safe to poll live during demo.
- CSV columns of interest: `latitude`, `longitude`, `confidence`
  (l/n/h = low/nominal/high), `bright_ti4` (brightness), `frp` (Fire Radiative Power;
  higher = more severe), `acq_date`, `acq_time`, `satellite`, `instrument`, `daynight`.
- WFS fallback (Do NOT use for bbox queries — bbox param returns 0): WFS works
  un-bounded, e.g.
  `https://firms.modaps.eosdis.nasa.gov/mapserver/wfs/SouthEast_Asia/{MAP_KEY}/?SERVICE=WFS&REQUEST=GetFeature&VERSION=2.0.0&TYPENAME=ms:fires_snpp_24hrs&outputformat=geojson`
  Valid TYPENAMEs (from GetCapabilities): `ms:fires_snpp_24hrs`, `ms:fires_modis_24hrs`,
  `ms:fires_noaa20_24hrs`, `ms:fires_noaa21_24hrs` (+ `_7days`, `ms:fires_landsat_24hrs`).

### 2) OSM via Overpass API (industrial zones)
- Free, no API key.
- Industrial land-use: `way["landuse"="industrial"]`; named facilities: also pull
  `way["industrial"="*"]` / `building="industrial"` where relevant.
- Query pattern: `[out:json];(way["landuse"="industrial"](bbox););out body geom;`

### 3) WRI Global Power Plant Database (second industrial source)
- Free, no API key. CSV → point GeoJSON in `app/services/powerplants.py`.
- URL:
  ```
  https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv
  ```
  (verified 200; ~1,589 India rows; `country` column is ISO3, filter on `"IND"`).
- Cache: `power_plants_cache.json` (720h TTL), git-ignored.
- Any fire within **1 km** of a plant is tagged industrial; `industrial_match_source`
  records whether proximity came from OSM (`osm`), the power-plant DB (`power_plant_db`),
  or both (`both`).
- Mining polygons (`landuse=quarry`, `man_made=mineshaft`) are fetched via the same
  tiled `osm.py` machinery into a `mining_zones_cache.json` (10,942 India zones) and
  classify a fire as the `mining` rule class (see Architecture §1).

### 4) NASA GIBS satellite imagery (frontend base layer, render only)
- Free WMTS, no key; used purely as a Leaflet base-layer toggle — NO raster
  processing (see Non-Negotiable Rules).
- Template:
  ```
  https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{YYYY-MM-DD}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg
  ```
  Layers: `MODIS_Terra_CorrectedReflectance_TrueColor`, `VIIRS_SNPP_CorrectedReflectance_TrueColor`
  (both verified 200 for a recent date; date ≈ yesterday since GIBS lags ~1 day).

### 5) VIIRS Nightfire (VNF) — gas-flare evidence (NOAA EO Group)
- The nightly VNF product is **license-gated since 2025-01-10** (free account + Data Use
  License application; interim access = ezCSV only) and has **no queryable API** (FIRMS does
  not carry it). Do NOT build on nightly VNF. Report covers verified dates:
  `eogdata.mines.edu/global_flare_data/2024_flare_summary_v20250730_j01.kml`.
- Instead we use the **VNF-derived annual global gas-flare catalog** (freely downloadable,
  no auth): 14,092 global flare sites (2024) → filtered to the FIRMS India bbox at parse
  time in `app/services/flares.py` → **423 India flare sites** cached (`flares_cache.json`,
  720h TTL, git-ignored).
- Rule use (NOT an ML class): an industrial `fire_type_rule` fire within
  **FLARE_BUFFER_METERS = 2 km** of a catalog site gets `gas_flare = True` +
  `distance_to_flare` + `flare_site_name`. Measured on accumulated `fire_history`
  (2026-09-06): 252/1924 detections (13.1% of all, 60% of the industrial class) — strong
  supporting evidence, and explainable in the summary (".· near a known gas-flare site
  (NASA VIIRS Nightfire)"). Frontend has a "Gas-flare sites (VIIRS Nightfire)" overlay.

## Architecture / Core Logic ("the AI")

1. **Spatial join (rule-based fire type)**: for each FIRMS point, measure distance to
   OSM industrial polygons (`landuse=industrial`), WRI power-plant points, vegetation
   polygons (`natural=wood`, `natural=scrub`, `landuse=forest`), and OSM mining polygons
   (`landuse=quarry`, `man_made=mineshaft`) with a per-fire UTM projection. Produces
   `fire_type_rule`:
   - `industrial` — within **1 km** of an industrial polygon OR a power plant (checked
     first); `industrial_match_source` records `osm` / `power_plant_db` / `both`, and
     `near_power_plant` + `power_plant_name` + `power_plant_distance_m` detail the match
   - `mining` — not industrial, but within **1 km** of a mining polygon; recorded via
     `near_mining` + `distance_to_mining` + `mining_site_name` + `mining_zone_type`
     (quarry / mine). Checked second (industrial wins, mining beats forest/other).
     Volume measured on history: ~8.6% (1,924 rows) — clears the 5% balance guard.
   - `forest` — not industrial or mining, but within **3 km** of a vegetation polygon (OSM forest
     polygons are conservative, so the wider radius is a fairer "near natural vegetation")
   - `other_natural` — neither (agricultural/grass burning, isolated events)
   - Industrial fires within **2 km** of a known VNF gas-flare site additionally get
     `gas_flare = True` (+ `distance_to_flare`, `flare_site_name`) — a sub-label of
     industrial, never a competing class (see Data Sources §5).
2. **Persistence detection**: store each detection (SQLite) keyed by location+time;
   if a hotspot repeats at/near the same coordinates across days/weeks, flag it as a
   **"persistent thermal source"** (term is literally in the SIH title) — e.g.
   `"🔴 Persistent thermal source — Nth occurrence in X days"`.
3. **ML fire type (weak-label classifier)**: `scripts/train_classifier.py` rebuilds a
   RandomForestClassifier from `fire_history` rows using the rule-based `fire_type_rule`
   as labels and 9 features (`confidence`, `bright_ti4`, `frp`, `daynight`, `hour`,
   `distance_to_industrial`, `distance_to_mining`, `distance_to_vegetation`,
   `occurrence_count`). The RF is
   wrapped in `CalibratedClassifierCV` (isotonic, cv=5) so `predict_proba` outputs
   calibrated probabilities (`fire_type_ml_confidence` = max calibrated prob), not raw
   vote fractions. `app/services/ml.py` loads `models/fire_classifier.pkl` at runtime and
   annotates `fire_type_ml` + `fire_type_ml_confidence`. The ML output is a parallel
   field — the rule output is always kept.
4. **Explainable verdicts (no training)**: `app/services/summary.py` turns the
   rule output into plain language after the whole pipeline runs — every fire gets
   `summary` ("Industrial Fire — ≈340 m from a known industrial zone", plus a
   "Recurring — detected N times in the past 14 days…" note for persistent sources),
   `summary_headline`, `summary_detail`, `summary_persistent`, `summary_icon`
   (🏭 / ⛏ / 🌲 / 🌾) and `explanation` — a one-line "why we think this" naming the
   exact rule (1-km industrial / 1-km mining / 3-km vegetation / neither). `main.py` calls it last,
   after ML, so it sees the persisted counts too.
5. **Severity scoring (rule-based, no training)**:
   - `confidence` (low/nominal/high)
   - `brightness` / `frp` (higher = more severe)
   - recurrence count (more occurrences = higher persistence severity)
6. **Optional stretch — DBSCAN**: cluster repeated hotspot coordinates over time into
   auto-named "sites" (out-of-the-box sklearn, `fit` on historical coords only).

## MVP Feature Scope (build top→bottom; each must work before next)

- [ ] 1. Backend endpoint: fetch live FIRMS data (India bbox) → clean GeoJSON — **✅ API-level done; see status section**
- [ ] 2. Backend endpoint: fetch OSM industrial polygons (same area)
- [ ] 3. Spatial join: label fire as "near industrial" + distance to nearest zone
- [ ] 4. Frontend map: fire points (color by confidence/severity) + industrial polygon
       layer, sidebar listing flagged detections
- [ ] 5. Persistence tracking: daily detections in DB, recurrence detection, persistent
       thermal source surfacing
- [x] 5b. Weak-label ML classifier: `scripts/train_classifier.py` (rule labels → RF →
       `models/fire_classifier.pkl`), `fire_type_ml` served alongside `fire_type_rule`
       in `/api/flagged-fires`
- [ ] 6. (Stretch) DBSCAN clustering → named "sites"
- [ ] 7. (Stretch) Severity ranking / dashboard sort by risk score

## Non-Negotiable Rules

- **NO custom ML/CV model training** beyond the sanctioned weak-label pipeline described
  in [Architecture](#architecture--core-logic-the-ai). Concretely: no custom deep models,
  no CV object detection, no hand-labelled datasets, no GPU. The ONLY allowed training is
  `scripts/train_classifier.py` (RandomForestClassifier fit on labels produced by our own
  spatial rule from `fire_history`; out-of-the-box sklearn, no hyperparameter tuning —
  only a `CalibratedClassifierCV` isotonic wrapper so `predict_proba` is meaningful).
  If more classes or more signal are ever needed, extend the rule + features first —
  never a new learned model.
- **NO raw satellite raster processing** (Sentinel/Landsat imagery). FIRMS already
  pre-processes hotspots; that is the whole point.
- **The trained model is a generated artifact, not source code.** `models/` is git-ignored;
  `scripts/train_classifier.py` regenerates it. Never commit `.pkl` files. Feature encoding
  must stay shared between training and serving (`app/services/ml.py` is the single source
  of truth for `FEATURE_COLUMNS`).
- **Training balance guard**: `train_classifier.py` refuses to write a model when any class
  is under `--min-class-fraction` (5%) of the training set. Do not disable this to "make it
  train" — a lopsided model is worse than a clear `fire_type_rule`-only answer.
- Keep the DB simple: **SQLite first**. Do not add PostGIS/PostgreSQL unless SQLite is
  proven to be the bottleneck.
- Every dependency change must be justified and recorded in `requirements.txt` (pinned)
  or `pyproject.toml`.

## Repository Layout (adopt as scaffolding)

```
sih-fire-detection/
├── AGENTS.md
├── README.md                 # quickstart, setup steps, how to demo
├── requirements.txt          # pinned
├── .env.example              # MAP_KEY etc. (never commit real keys)
├── app/
│   ├── main.py               # FastAPI app entry, CORS, routes
│   ├── config.py             # env config (MAP_KEY, bbox, thresholds, state bboxes)
│   ├── services/
│   │   ├── firms.py          # FIRMS CSV area API fetch → clean GeoJSON
│   │   ├── osm.py            # Overpass industrial + vegetation zones fetch (tiled, cached)
│   │   ├── powerplants.py    # WRI power-plant DB fetch → point GeoJSON (cached)
│   │   ├── spatial.py        # rule-based fire_type (industrial/forest/other_natural)
│   │   ├── persistence.py    # recurrence detection & severity scoring
│   │   ├── ml.py             # fire_type_ml: load model + shared feature encoding
│   │   └── clustering.py     # (stretch) DBSCAN site grouping
│   ├── db.py                 # SQLite schema + access helpers
│   └── models.py             # Pydantic schemas / dataclasses
├── scripts/
│   └── train_classifier.py   # weak-label training pipeline → models/fire_classifier.pkl
├── models/                   # git-ignored generated artifacts (the .pkl lives here)
├── static/
│   └── index.html + app.js   # Leaflet map + sidebar
└── tests/
```

## Commands (verify/adjust during setup)

- Install: `pip install -r requirements.txt`
- Run: `uvicorn app.main:app --reload` (default http://localhost:8000)
- Map: open `http://localhost:8000/` in a browser
- Lint/format: `ruff check .` and `ruff format .` (add ruff to requirements)
- Tests: `pytest`

## Coding Conventions

- Python 3.11+, type hints on all signatures.
- All external API calls via `httpx` (async) where possible; timeouts always set.
- GeoJSON flows through the app as source of truth: FIRMS → GeoJSON, backend enriches →
  GeoJSON, Leaflet renders GeoJSON.
- Clearly separate code that must be debuggable by beginners (keep pure logic functions
  small, one responsibility each).
- Never log or commit API keys. Use `.env` + `python-dotenv`; commit only `.env.example`.
- Comments in code are fine but keep code self-explanatory first.

## Known Gaps / Things to Confirm During Setup

1. ~~FIRMS MAP_KEY~~ — **done**, in `.env` (`FIRMS_MAP_KEY`).
2. **FIRMS region + bbox** — **done**: use `api/area/csv` with bbox `68,6,97,37`
   (verified 208 live VIIRS India fires on 2026-09-04). WFS bbox filter is broken
   (returns 0); do not rely on it for bbox queries.
3. **Overpass query working set** — confirm tag choice (`landuse=industrial`) returns
   sensible polygons for India; tune bbox size for response limits.
4. **DBSCAN** — decide `eps`/`min_samples` sensible for kilometer-scale coordinates.

## Definition of Done

- `uvicorn app.main:app` starts cleanly.
- FIRMS endpoint returns non-empty GeoJSON for the India bbox.
- OSM endpoint returns industrial polygons.
- Map renders both layers; fire points color-coded, sidebar populated.
- Persistence endpoint reports recurrence counts / persistent sources over the demo DB.
- ML endpoint exposes both `fire_type_rule` and `fire_type_ml`; `pytest tests/test_ml.py`
  passes once `scripts/train_classifier.py` has been run at least once.
- `pytest` passes; `ruff` clean.

## Current Status (keep updated)

- [x] FIRMS MAP_KEY obtained (stored in `.env`)
- [x] Confirmed FIRMS region name / bbox for India (`api/area/csv`, bbox `68,6,97,37`)
- [x] Backend scaffold created (FastAPI project structure)
- [x] FIRMS fetch endpoint working (`GET /api/fires`, default `FIRMS_DAYS=3` —
       lookback configurable via `?days= N` too)
- [x] OSM Overpass fetch working (tiled state bboxes, 24h file cache, kumi mirror)
- [x] Spatial join logic working (UTM-projected nearest-distance; 1 km near flag)
- [x] Frontend map rendering both layers (Leaflet; fires color-coded, zones translucent,
       sidebar with flagged + persistent sections)
- [x] Persistence tracking implemented (`fire_history` SQLite table accumulates every
       detection on `/api/fires` and `/api/flagged-fires`; flags `persistent_thermal_source`
       for 2+ distinct days within 300 m in last 14 days, with `occurrence_count`; 22 tests, ruff clean)
- [x] OSM coverage expanded to 16 states (west_bengal, odisha, chhattisgarh,
       tamil_nadu, kerala, karnataka, madhya_pradesh, assam, rajasthan, andhra_pradesh,
       telangana, punjab, uttar_pradesh since the original 3); vegetation layer uses
       `natural=wood`, `natural=scrub`, `landuse=forest`. Industrial cache 29,374 ways,
       vegetation cache 150,842 ways. Overpass hardened: 4 mirrors (kumi,
       overpass-api.de, overpass.private.coffee, maps.mail.ru), 2 retries, per-tile
       `max_concurrency`, parallel tile fetch with a failures counter.
- [x] WRI power-plant layer wired in: `app/services/powerplants.py` (CSV → point GeoJSON,
       1,589 India plants, 720h cache `power_plants_cache.json`), `/api/power-plants`,
       and the spatial rule now treats a fire within 1 km of a plant as industrial with
       `industrial_match_source` = `osm` / `power_plant_db` / `both` + `near_power_plant`,
       `power_plant_name`, `power_plant_distance_m`. Frontend renders a power-plant
       overlay and a NASA GIBS true-color base-layer toggle (OSM/MODIS/VIIRS, render-only).
       OSM is the DEFAULT base layer; GIBS is strictly opt-in via the Layer control. GIBS
       dates can lack an India granule (404 → black tile gaps), so `pickGIBSDate()` in
       `static/app.js` probes the last 7 days and registers the toggle only for the newest
       date with verified z5 India tiles — never hardcode "yesterday".
- [x] Weak-label ML classifier trained + wired in: `scripts/train_classifier.py` (balance
       guard ≥5%/class, always backfills FIRMS `days=5` — the area API caps day range at
       [1..5]); `app/services/ml.py` (predict + shared encoding); `fire_type_rule`,
       `fire_type_ml` and calibrated `fire_type_ml_confidence` on `/api/flagged-fires`.
       Current training set: n=1730 (industrial 17.3%, forest 15.5%, other_natural 67.2%);
       test accuracy 1.00 with `distance_to_vegetation` (0.420) and
       `distance_to_industrial` (0.403) dominant — expected for a weak-label demo model.
       RF is wrapped in CalibratedClassifierCV (isotonic, cv=5): held-out mean confidence
       0.968→0.997, Brier 0.0054→0.0003, log-loss 0.0334→0.0028. max_depth /
       min_samples_leaf grid re-run: defaults remain best — no tuning change.
- [x] Explainable verdicts added server-side: `app/services/summary.py` annotates
       every fire with plain-language `summary` (+ `summary_headline`,
       `summary_detail`, `summary_persistent`, `summary_icon`) and a one-line
       `explanation` ("why we think this"). Frontend shows the summary big, bold and
       iconed (🏭/🌲/🌾); raw metrics (distance, confidence, frp, brightness) are
       collapsed under an expandable "Details" section in popups and sidebar cards.
- [x] VNF gas-flare evidence layer: nightly VIIRS Nightfire is license-gated + has no
       queryable API (reported), so we use the free VNF-derived 2024 annual flare catalog
       (`app/services/flares.py`, 423 India sites, 720h cache, `/api/flares`). Industrial
       fires within 2 km of a catalog site get `gas_flare` + `distance_to_flare` +
       `flare_site_name` (sub-label of industrial, not an ML class). Measured on history:
       252/1924 (13.1% of all, 60% of industrial). Frontend "Gas-flare sites (VIIRS
       Nightfire)" overlay + orange gas-flare markers. 58 tests, ruff clean.
- [x] Mining-zone 4th class: `osm.py` `get_mining_zones()` fetches `landuse=quarry` +
       `man_made=mineshaft` over the 16-state tiles (10,942 India zones,
       `mining_zones_cache.json`, 720h TTL, `/api/mining-zones`). Spatial rule labels
       non-industrial fires within 1 km of a mining polygon as the new `mining` class
       (`near_mining` + `distance_to_mining` + `mining_site_name` + `mining_zone_type`;
       industrial wins, mining beats forest/other). `distance_to_mining` added to the 9
       ML features and the weak-label RF retrained: 1,924 rows (industrial 420 / mining
       166 / forest 233 / other 1105; mining 8.6% — clears the 5% guard), held-out
       accuracy 1.00, `distance_to_mining` importance 0.170. Summary adds
       "Mining/Quarry Fire" ⛏ + explanation. Frontend mining overlay + brown markers +
       "⛏ Mining" filter tab. 63 tests, ruff clean.
- [x] India-only boundary filter on LIVE endpoints: `app/services/firms.py`
       `filter_to_india_boundary()` clips every `fetch_fires()` response to the real
       India polygon (`app/data/india_boundary.geojson`, Natural Earth 10m admin-0
       incl. Andaman & Nicobar; 110m rejected — drops NE states + islands);
       `india_boundary_buffer_deg` defaults to **0.0 (strict polygon)**. Applies to
       `/api/fires`, `/api/flagged-fires`, `/api/thermal-sites` AND the training
       backfill (single choke point inside `fetch_fires`), so no neighbor
       (PK/CN/NP/BT/BD/MM/LK) detections leak. Verified live 2026-09-06: stale server
       (pre-fix process) leaked 679/1258 fires outside India; after restart,
       `/api/fires` + `/api/flagged-fires` returned 579 fires, 0 out-of-boundary.
       67 tests, ruff clean.
- [x] Frontend filter-tab → map fix + command-console redesign: each fire maps to
       exactly ONE category (`persistent → flare → industrial → mining → forest →
       other_natural`) shared by the sidebar AND a dedicated Leaflet layer group per
       category; filter tabs show/hide whole groups (no re-fetch, no per-marker
       re-filtering). Visual pass: ember/amber severity ramp, cyan chrome, Chakra Petch
       + IBM Plex Mono + Inter, radar HUD + scanlines, pulsing LIVE badge + persistent
       markers, count-up stats, hover-to-highlight markers, fade/scale tab transitions.
       All colors tunable via CSS custom properties in `:root`.
- [x] Overpass resilience hardening: `fetch_region()` now treats a HTTP 200 with a
       non-JSON body (proxy error pages, tarpits) as a mirror failure and falls
       through to the next mirror instead of aborting the whole tile fetch
       (`app/services/osm.py`; caught `ValueError`, `last_error` kept for the final
       RuntimeError). Timeouts: per-request `OVERPASS_REQUEST_TIMEOUT_SECONDS=12`
       (confirmed passed through, not the 180 s client default); falling back across
       all mirrors on `httpx.RequestError`/429/502/503/504. Tests added in
       `tests/test_osm_resilience.py` (timeout→next, 500→next, garbage-body→next,
       all-mirrors-fail→RuntimeError, timeout propagation, and app import does no
       network work) — 88 tests, ruff clean (includes new dashboard features below).
- [x] Dashboard feature pack:
  - **Relative "time since detection"** (`static/app.js` `acqTimestamp`/`timeSinceText`,
    FIRMS `acq_time` is UTC HHMM with 2400 = next-dn midnight). Every card + popup
    shows "12 min ago"; refreshes every 45 s via `setInterval(refreshAges)` with **no
    reload**; `< 30 min` detections get a `.age.fresh` green accent (fires under half an
    hour are scannable at a glance).
  - **Severity (FRP) sort**: `#sort` select (Priority / Severity (FRP) / Newest /
    Nearest). FRP is surfaced on every card via the `.meta` row (`FRP 42.3 MW`, ember
    accent) — no longer buried in the Details expander.
  - **Unregistered persistent source** (new derived class, highest-value alert): a fire
    that is `persistent_thermal_source` AND matches no named WRI power plant
    (`app/services/persistence.py:_unregistered_persistent` → `unregistered_persistent`
    prop; `summary_unregistered` note in `app/services/summary.py`). Frontend category
    precedence `unregistered → persistent → flare → …` with 🚨 filter tab, ember
    markers, `.unreg` framing "Unregistered — no known facility match (WRI power-plant
    database)".
  - **Multi-day diff view**: "Show" select (`#view-new`) = All / New since last refresh
    / New since yesterday. Backend: `app/db.py:annotate_new_since_last_refresh` (called
    in `main.py` before `record_featurecollection`, so a fire is "new" when its
    (lat,lon,date,time,satellite) key wasn't stored before) + `is_new_since_yesterday`
    (acq_date ≥ today UTC). A status chip shows "New since X: N of M detections".
  - Tests: `tests/test_diff.py` (9) + unregistered/is_new cases in
    `tests/test_persistence.py` + summary note tests — 88 total, ruff clean.
- [ ] DBSCAN clustering (stretch)
- [ ] Deployed somewhere accessible for demo