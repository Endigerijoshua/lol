"""VIIRS Nightfire (VNF) derived global gas-flare catalog — known flare sites.

The nightly VIIRS Nightfire product at NOAA's Earth Observation Group
(eogdata.mines.edu) is license-gated since 2025-01-10 and has no queryable API,
so it cannot be streamed live. But EOG freely publishes an annual global
gas-flare site catalog derived from VNF (the same nightly detections rolled up
per site, with radio/lat for every flare that burned that year). We use it
exactly like the WRI power-plant DB: a curated reference layer of coordinates
where gas flares are known to burn. A FIRMS hotspot that is already industrial
by our rule AND falls within 2 km of one of these sites gets a `gas_flare`
sub-label (see spatial.py) — supporting evidence, not a separate ML class.

Source: 2024 flare-summary KML (14,092 global sites). Filtering to the FIRMS
India bbox happens at parse time, so only the India subset (~423 sites) is
cached and served.
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from ..config import repo_path, settings

logger = logging.getLogger(__name__)

FLARES_URL = "https://eogdata.mines.edu/global_flare_data/2024_flare_summary_v20250730_j01.kml"
CATALOG_YEAR = 2024
_NS = "{http://www.opengis.net/kml/2.2}"


def _cache_path() -> Path:
    return repo_path(settings.flares_cache_file)


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    fresh = age_seconds < settings.flares_cache_max_age_hours * 3600
    if fresh:
        logger.info("flares cache is fresh (%.1f h old)", age_seconds / 3600)
    else:
        logger.info("flares cache is stale (%.1f h old)", age_seconds / 3600)
    return fresh


def read_cache() -> dict | None:
    path = _cache_path()
    if not _cache_is_fresh(path):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read flares cache: %s", exc)
        return None
    return data.get("feature_collection")


def write_cache(feature_collection: dict) -> None:
    path = _cache_path()
    payload = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": FLARES_URL,
        "filter": f"bbox={settings.firms_bbox}",
        "feature_collection": feature_collection,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger.info(
        "cached %s flare site(s) to %s",
        len(feature_collection["features"]),
        path,
    )


def _india_bbox() -> tuple[float, float, float, float]:
    """west, south, east, north from settings.firms_bbox ("68,6,97,37")."""
    west, south, east, north = (float(v) for v in settings.firms_bbox.split(","))
    return west, south, east, north


def kml_to_geojson(kml_text: str) -> dict:
    """Parse the VNF flare-summary KML into India-bbox point GeoJSON.

    Placemarks are `Point` with a name like
    "IND_UNKNOWN_2024_72.6394E_21.1073N_v0.2"; the country prefix is derived
    from the first underscore-delimited token.
    """
    west, south, east, north = _india_bbox()
    try:
        root = ET.fromstring(kml_text)
    except ET.ParseError as exc:
        logger.warning("could not parse VNF flare KML: %s", exc)
        return {"type": "FeatureCollection", "features": []}
    features = []
    for placemark in root.findall(f".//{_NS}Placemark"):
        coords_el = placemark.find(f".//{_NS}Point/{_NS}coordinates")
        if coords_el is None or not coords_el.text:
            continue
        try:
            lon, lat, *_ = (float(v) for v in coords_el.text.strip().split(","))
        except ValueError:
            continue
        if not (west <= lon <= east and south <= lat <= north):
            continue
        name = (placemark.findtext(f"{_NS}name") or "").strip()
        country = name.split("_", 1)[0] if name else None
        features.append(
            {
                "type": "Feature",
                "id": f"vnf-flare-{len(features)}",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "name": name or None,
                    "country": country,
                    "year": CATALOG_YEAR,
                    "source": "viirs_nightfire",
                },
            }
        )
    logger.info("VNF flare KML -> GeoJSON: %s India flare site(s)", len(features))
    return {"type": "FeatureCollection", "features": features}


async def fetch_flares(client: httpx.AsyncClient | None = None) -> dict:
    """Download the 2024 VNF flare KML, parse it, and cache the result."""
    logger.info("downloading VNF flare KML: %s", FLARES_URL)
    closer = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=settings.external_request_timeout_seconds,
            follow_redirects=True,
        )
        closer = True
    try:
        response = await client.get(FLARES_URL)
        response.raise_for_status()
        feature_collection = kml_to_geojson(response.text)
        write_cache(feature_collection)
        return feature_collection
    finally:
        if closer:
            await client.aclose()


async def get_flares(client: httpx.AsyncClient | None = None) -> dict:
    """Return known gas-flare sites as point GeoJSON (cache-first, then download)."""
    cached = read_cache()
    if cached is not None:
        return cached
    return await fetch_flares(client)
