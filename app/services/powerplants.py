"""WRI Global Power Plant Database — curated industrial-infrastructure points.

Secondary industrial layer complementary to OSM polygons. The WRI dataset is a
curated CSV (free, no API key) of ~34,000 power plants worldwide; we filter to
India and serve the plants as point GeoJSON. A fire within 1 km of any plant is
treated as industrial (see spatial.py), and the matching source is recorded so
the demo can show we fuse a curated infrastructure database with open map data
instead of relying on OSM alone.
"""

import io
import json
import logging
import time
from pathlib import Path

import httpx
import pandas as pd

from ..config import repo_path, settings

logger = logging.getLogger(__name__)

POWER_PLANTS_URL = (
    "https://raw.githubusercontent.com/wri/global-power-plant-database/"
    "master/output_database/global_power_plant_database.csv"
)

# The WRI `country` column uses ISO3 codes; India is "IND".
INDIA_COUNTRY_CODE = "IND"

# Fields copied onto each GeoJSON feature. Verbose WRI fields (owner, fuel1, ...)
# are dropped to keep the layer small for the browser.
PROPERTY_FIELDS = ("gppd_idnr", "name", "capacity_mw", "primary_fuel", "country_long")


def _cache_path() -> Path:
    return repo_path(settings.power_plants_cache_file)


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    fresh = age_seconds < settings.power_plants_cache_max_age_hours * 3600
    if fresh:
        logger.info("power plants cache is fresh (%.1f h old)", age_seconds / 3600)
    else:
        logger.info("power plants cache is stale (%.1f h old)", age_seconds / 3600)
    return fresh


def read_cache() -> dict | None:
    path = _cache_path()
    if not _cache_is_fresh(path):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read power plants cache: %s", exc)
        return None
    return data.get("feature_collection")


def write_cache(feature_collection: dict) -> None:
    path = _cache_path()
    payload = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": POWER_PLANTS_URL,
        "filter": f"country={INDIA_COUNTRY_CODE}",
        "feature_collection": feature_collection,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger.info(
        "cached %s power plant feature(s) to %s",
        len(feature_collection["features"]),
        path,
    )


def _json_value(value):
    """Convert a pandas/numpy scalar into a plain JSON-safe Python value."""
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


def csv_to_geojson(csv_text: str) -> dict:
    """Parse the WRI CSV into a point FeatureCollection restricted to India."""
    df = pd.read_csv(io.StringIO(csv_text))
    india = df[df["country"] == INDIA_COUNTRY_CODE]
    features = []
    dropped = 0
    for _, row in india.iterrows():
        lon, lat = row["longitude"], row["latitude"]
        if pd.isna(lon) or pd.isna(lat):
            dropped += 1
            continue
        props = {field: _json_value(row[field]) for field in PROPERTY_FIELDS}
        props["source"] = "power_plant_db"
        gppd_id = props.get("gppd_idnr")
        feature_id = f"wri-pp-{gppd_id}" if gppd_id is not None else f"wri-pp-{len(features)}"
        features.append(
            {
                "type": "Feature",
                "id": feature_id,
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)],
                },
                "properties": props,
            }
        )
    if dropped:
        logger.warning("dropped %s India power plant(s) with missing coordinates", dropped)
    logger.info("WRI power plant CSV -> GeoJSON: %s India plant(s)", len(features))
    return {"type": "FeatureCollection", "features": features}


async def fetch_power_plants(client: httpx.AsyncClient | None = None) -> dict:
    """Download the WRI CSV (India only), parse it, and cache the result."""
    logger.info("downloading power plants CSV: %s", POWER_PLANTS_URL)
    closer = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=settings.external_request_timeout_seconds,
            follow_redirects=True,
        )
        closer = True
    try:
        response = await client.get(POWER_PLANTS_URL)
        response.raise_for_status()
        feature_collection = csv_to_geojson(response.text)
        write_cache(feature_collection)
        return feature_collection
    finally:
        if closer:
            await client.aclose()


async def get_power_plants(client: httpx.AsyncClient | None = None) -> dict:
    """Return India power plants as point GeoJSON (cache-first, then download)."""
    cached = read_cache()
    if cached is not None:
        return cached
    return await fetch_power_plants(client)
