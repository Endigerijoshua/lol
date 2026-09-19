"""SIH26162 — Industrial fire detection via NASA FIRMS + OSM satellite data.

FastAPI app entry point. Serves the Leaflet dashboard (static) and the API
endpoints that fetch + enrich live fire data.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Check runs before importing app.services / app.config — those modules use
# PEP 604 union syntax (`dict | None`).
_MIN_PYTHON = (3, 11)
if sys.version_info < _MIN_PYTHON:
    raise SystemExit(
        f"EmberMap requires Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ "
        f"(found {sys.version_info.major}.{sys.version_info.minor}). "
        "Install Python 3.11 or newer and retry."
    )

# pyrefly: ignore [missing-import]
import httpx

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Query

# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware

# pyrefly: ignore [missing-import]
from fastapi.responses import FileResponse, HTMLResponse

# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles

from . import db
from .config import repo_path, settings
from .services import (
    clustering,
    firms,
    flares,
    ml,
    osm,
    persistence,
    powerplants,
    spatial,
    summary,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the reference-layer caches in the background at boot.

    The OSM zone caches (industrial / vegetation / mining) are git-ignored and
    rebuilt from Overpass on first use. Without this, the first dashboard
    request cold-builds them and stalls for minutes while the page shows only
    skeleton loaders ("data is not fetching"). Kicking off the build here means
    a fresh boot assembles missing caches *before* the user opens the page,
    so the first real request serves from cache in ~seconds. Each layer's
    getter already short-circuits on a fresh cache, so warm boots are instant.
    """
    logger.info("lifespan: starting background reference-layer warm-up …")
    warmup = asyncio.create_task(_warm_reference_caches())
    try:
        yield
    finally:
        if not warmup.done():
            warmup.cancel()
        await asyncio.gather(warmup, return_exceptions=True)


async def _warm_reference_caches() -> None:
    """Prefetch all five reference layers; a failing layer never aborts the rest.

    Each getter reads its local cache first (fresh => no network), so on a warm
    boot this only reads disk. On a cold boot it builds whatever is missing.
    When every layer loads, the parsed shapely + STRtree reference index is
    prebuilt too (spatial.register_reference_bundle) so the first live request
    reuses it instead of paying the ~20-30 s parse. Exceptions are logged and
    swallowed: warm-up is best-effort and must never crash server startup.
    """

    async def one(name: str, coro) -> dict | None:
        try:
            fc = await coro
            logger.info(
                "[warm-up] %s ready (%d features)",
                name,
                len(fc.get("features", [])),
            )
            return fc
        except Exception as exc:  # noqa: BLE001 — best-effort, never crash boot
            logger.warning("[warm-up] %s failed: %s", name, exc)
            return None

    industrial_fc, vegetation_fc, mining_fc, power_plants_fc, flares_fc = (
        await asyncio.gather(
            one("industrial", osm.get_industrial_zones()),
            one("vegetation", osm.get_vegetation_zones()),
            one("mining", osm.get_mining_zones()),
            one("power-plants", powerplants.get_power_plants()),
            one("flares", flares.get_flares()),
        )
    )

    # Prebuild the parsed shapely + STRtree reference index so the first live
    # /api/flagged-fires request after boot doesn't pay the ~20-30 s shape()
    # parse over the ~150k vegetation polygons. Only register when all five
    # layers are present — a partial bundle would silently mislabel fires, and
    # the live request will rebuild it correctly from whatever it can load.
    layers = (industrial_fc, vegetation_fc, power_plants_fc, flares_fc, mining_fc)
    if all(fc is not None for fc in layers):
        try:
            spatial.register_reference_bundle(
                _reference_cache_version(), *layers
            )
            logger.info("[warm-up] reference index prebuilt")
        except Exception as exc:  # noqa: BLE001 — index build is best-effort
            logger.warning("[warm-up] reference index prebuild failed: %s", exc)
    else:
        logger.info(
            "[warm-up] skipping reference index prebuild (a layer is missing)"
        )


app = FastAPI(
    title="SIH26162 Industrial Fire Detection",
    description="Live industrial fire + persistent thermal source detection using NASA FIRMS and OSM.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


PLACEHOLDER_INDEX = """<!doctype html>
<html><head><meta charset="utf-8"><title>SIH26162</title></head>
<body><h1>SIH26162 — Industrial Fire Detection</h1>
<p>Backend is running. Frontend map is coming in a later milestone.</p>
<p>Try <a href="/api/fires">/api/fires</a> for live NASA FIRMS hotspots (GeoJSON).</p>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    path = STATIC_DIR / "landing.html"
    if path.exists():
        return FileResponse(path)
    return PLACEHOLDER_INDEX


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    path = STATIC_DIR / "index.html"
    if path.exists():
        return FileResponse(path)
    return PLACEHOLDER_INDEX


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/history/daily")
async def history_daily(days: int = Query(14, ge=1, le=30)) -> dict:
    """Per-day FIRMS detection counts from the local fire_history table."""
    return {"days": db.daily_counts(days)}


@app.get("/api/fires")
async def get_fires(days: int | None = None) -> dict:
    """Fetch live FIRMS hotspots (India bbox) and return clean GeoJSON.

    `days` overrides the FIRMS lookback window (default from settings/.env).
    Every returned detection is also appended to the local SQLite fire_history.
    """
    try:
        fires_fc = await firms.fetch_fires(days=days)
        db.record_featurecollection(fires_fc)
        return fires_fc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"NASA FIRMS API error: {exc}"
        ) from exc


@app.get("/api/industrial-zones")
async def get_industrial_zones() -> dict:
    """Fetch or serve cached OSM industrial-zone polygons as GeoJSON."""
    try:
        return await osm.get_industrial_zones()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/vegetation-zones")
async def get_vegetation_zones() -> dict:
    """Fetch or serve cached OSM forest/vegetation polygons as GeoJSON."""
    try:
        return await osm.get_vegetation_zones()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/mining-zones")
async def get_mining_zones() -> dict:
    """Fetch or serve cached OSM mining polygons (quarries + mine shafts) as GeoJSON."""
    try:
        return await osm.get_mining_zones()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/power-plants")
async def get_power_plants() -> dict:
    """India power plants (WRI Global Power Plant Database) as point GeoJSON."""
    try:
        return await powerplants.get_power_plants()
    except (RuntimeError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/flares")
async def get_flares() -> dict:
    """Known gas-flare sites from NASA VIIRS Nightfire (annual catalog) as
    point GeoJSON — supporting evidence for the `gas_flare` sub-label."""
    try:
        return await flares.get_flares()
    except (RuntimeError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/flagged-fires")
async def get_flagged_fires(days: int | None = None) -> dict:
    """Live fires annotated with rule-based + ML fire type and persistence."""
    try:
        fires_fc = await firms.fetch_fires(days=days)
        db.record_featurecollection(fires_fc)
        logger.info(
            "[flagged-fires] %d fires fetched", len(fires_fc.get("features", []))
        )
        (
            industrial_fc,
            vegetation_fc,
            power_plants_fc,
            flares_fc,
            mining_fc,
        ) = await _reference_layers()
        logger.info(
            "[flagged-fires] reference layers loaded: %d industrial, %d vegetation, %d power plants, %d flares, %d mining",
            len(industrial_fc.get("features", [])),
            len(vegetation_fc.get("features", [])),
            len(power_plants_fc.get("features", [])),
            len(flares_fc.get("features", [])),
            len(mining_fc.get("features", [])),
        )
        logger.info("[flagged-fires] START spatial.annotate_fires")
        spatial.annotate_fires(
            fires_fc,
            industrial_fc,
            vegetation_fc,
            power_plants_fc,
            flares_fc,
            mining_fc,
            cache_version=_reference_cache_version(),
        )
        logger.info("[flagged-fires] END spatial.annotate_fires")
        logger.info("[flagged-fires] START persistence.annotate_persistence")
        persistence.annotate_persistence(fires_fc)
        logger.info("[flagged-fires] END persistence.annotate_persistence")
        logger.info("[flagged-fires] START ml.annotate_fire_type_ml")
        ml.annotate_fire_type_ml(fires_fc)
        logger.info("[flagged-fires] END ml.annotate_fire_type_ml")
        logger.info("[flagged-fires] START summary.add_summary")
        summary.add_summary(fires_fc)
        logger.info("[flagged-fires] END summary.add_summary")
        logger.info(
            "[flagged-fires] complete: %d fires", len(fires_fc.get("features", []))
        )
        return fires_fc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Upstream API error: {exc}"
        ) from exc


async def _reference_layers() -> tuple[dict, dict, dict, dict, dict]:
    """Fetch industrial + vegetation + mining zones, power plants and flare sites."""
    logger.info("_reference_layers: START fetch")
    (
        industrial_fc,
        vegetation_fc,
        power_plants_fc,
        flares_fc,
        mining_fc,
    ) = await asyncio.gather(
        osm.get_industrial_zones(),
        osm.get_vegetation_zones(),
        powerplants.get_power_plants(),
        flares.get_flares(),
        osm.get_mining_zones(),
    )
    logger.info(
        "_reference_layers: END fetch (%d industrial, %d vegetation, %d power plants, %d flares, %d mining)",
        len(industrial_fc.get("features", [])),
        len(vegetation_fc.get("features", [])),
        len(power_plants_fc.get("features", [])),
        len(flares_fc.get("features", [])),
        len(mining_fc.get("features", [])),
    )
    return industrial_fc, vegetation_fc, power_plants_fc, flares_fc, mining_fc


def _reference_cache_version() -> str:
    """Fingerprint the reference-layer disk caches so spatial can reuse its
    parsed geometry + STRtree while the underlying cache files are unchanged.

    Uses file size + mtime (cheap); any refresh of a cache file changes it, so
    the next request rebuilds the reference index instead of serving stale
    geometry. This removes a ~20-30 s `shape()` parse from every request.
    """
    parts = []
    for name in (
        "industrial_cache_file",
        "vegetation_cache_file",
        "power_plants_cache_file",
        "flares_cache_file",
        "mining_cache_file",
    ):
        path = repo_path(getattr(settings, name, name))
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime:.0f}")
        except OSError:
            parts.append(f"{name}:missing")
    return "|".join(parts)


@app.get("/api/thermal-sites")
async def get_thermal_sites(days: int | None = None) -> dict:
    """DBSCAN-cluster persistent recurrences into named industrial sites."""
    try:
        fires_fc = await firms.fetch_fires(days=days)
        db.record_featurecollection(fires_fc)
        (
            industrial_fc,
            vegetation_fc,
            power_plants_fc,
            flares_fc,
            mining_fc,
        ) = await _reference_layers()
        spatial.annotate_fires(
            fires_fc,
            industrial_fc,
            vegetation_fc,
            power_plants_fc,
            flares_fc,
            mining_fc,
            cache_version=_reference_cache_version(),
        )
        persistence.annotate_persistence(fires_fc)
        ml.annotate_fire_type_ml(fires_fc)
        return clustering.cluster_persistent_fires(fires_fc, industrial_fc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Upstream API error: {exc}"
        ) from exc


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
