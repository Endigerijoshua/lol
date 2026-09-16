"""OpenStreetMap zone fetch via the Overpass API (industrial + vegetation + mining).

Queries industrial (`landuse=industrial`), vegetation (`natural=wood`,
`natural=scrub`, `landuse=forest`) and mining (`landuse=quarry`,
`man_made=mineshaft`) polygons for a set of regional state bboxes (whole-India
box is too slow / times out), converts the Overpass JSON response into GeoJSON
FeatureCollections, and caches each layer to its own local file so Overpass is
not hit on every request.

The primary mirror at overpass-api.de consistently rejected requests with 406
during development, so queries go straight to the kumi mirror. Queries stay to a
minimal tag set per tile: every extra union clause measurably increases Overpass
504 rates (verified while expanding coverage to more states). Each mirror is
retried a few times with a short backoff because transient 504s are common under
load and a later attempt usually succeeds (only mirror exhausts do we skip a tile).
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from ..config import repo_path, settings

logger = logging.getLogger(__name__)

OVERPASS_URLS = (
    "https://lz4.overpass-api.de/api/interpreter",               # fast (3-5 s)
    "https://overpass.openstreetmap.fr/api/interpreter",          # fast (3-5 s)
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",   # fast (3-5 s)
    "https://overpass-api.de/api/interpreter",                    # medium
    "https://z.overpass-api.de/api/interpreter",                  # slow (6-20 s)
    "https://overpass.kumi.systems/api/interpreter",              # very slow (30 s+)
)

OVERPASS_RETRIES = 1
OVERPASS_RETRY_BACKOFF_SECONDS = (2.0,)

# Per-request HTTP timeout. Fast mirrors respond in 1-3s; drop slow/hung mirrors quickly.
OVERPASS_REQUEST_TIMEOUT_SECONDS = 12.0

OVERPASS_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "SIH26162-FireDetection/1.0 (Mozilla/5.0 compatible)",
}

OVERPASS_QUERY = "[out:json][timeout:{timeout}];({queries});out body geom;"

# Per-layer Overpass tag queries. Each entry is a union block of `way[...]`
# clauses; the bbox placeholders are filled per tile by build_query().
# Note Overpass bbox order is south,west,north,east.
ZONE_QUERIES = {
    "industrial": ('way["landuse"="industrial"]({south},{west},{north},{east});',),
    "vegetation": (
        'way["natural"="wood"]({south},{west},{north},{east});',
        'way["natural"="scrub"]({south},{west},{north},{east});',
        'way["landuse"="forest"]({south},{west},{north},{east});',
    ),
    "mining": (
        'way["landuse"="quarry"]({south},{west},{north},{east});',
        'way["man_made"="mineshaft"]({south},{west},{north},{east});',
    ),
}

CACHE_FILE_SETTINGS = {
    "industrial": "industrial_cache_file",
    "vegetation": "vegetation_cache_file",
    "mining": "mining_cache_file",
}


def parse_state_bbox(bbox: str) -> tuple[float, float, float, float]:
    """Parse a state bbox 'west,south,east,north' into floats."""
    parts = [float(p.strip()) for p in bbox.split(",")]
    if len(parts) != 4 or not all(
        a < b for a, b in ((parts[0], parts[2]), (parts[1], parts[3]))
    ):
        raise ValueError(f"invalid state bbox: {bbox}")
    return parts[0], parts[1], parts[2], parts[3]


def tile_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    cols: int = 2,
    rows: int = 2,
) -> list[tuple[float, float, float, float]]:
    """Split a bbox into a cols-by-rows grid of smaller tiles.

    Whole-state boxes are too slow for Overpass (timeouts/504s), so queries are
    issued per tile. Tile seams may overlap by chance; results are de-duplicated
    downstream by OSM id.
    """
    tiles: list[tuple[float, float, float, float]] = []
    for col in range(cols):
        for row in range(rows):
            tile_west = west + (east - west) * col / cols
            tile_east = west + (east - west) * (col + 1) / cols
            tile_south = south + (north - south) * row / rows
            tile_north = south + (north - south) * (row + 1) / rows
            tiles.append((tile_west, tile_south, tile_east, tile_north))
    return tiles


def build_query(
    west: float,
    south: float,
    east: float,
    north: float,
    zone_type: str,
    timeout: int,
) -> str:
    """Build the Overpass query for one tile of one zone layer."""
    if zone_type not in ZONE_QUERIES:
        raise ValueError(f"unknown zone type: {zone_type}")
    union = "\n".join(
        clause.format(south=south, west=west, north=north, east=east)
        for clause in ZONE_QUERIES[zone_type]
    )
    return OVERPASS_QUERY.format(timeout=timeout, queries=union)


def _way_to_geometry(element: dict) -> dict | None:
    points = element.get("geometry")
    if not isinstance(points, list) or len(points) < 4:
        return None
    coords = [[float(p["lon"]), float(p["lat"])] for p in points]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    if len(coords) < 4:
        return None
    return {"type": "Polygon", "coordinates": [coords]}


def osm_elements_to_featurecollection(elements: list[dict]) -> dict:
    """Convert Overpass `way` elements into a GeoJSON polygon FeatureCollection.

    Only ways with a valid closed polygon coordinate ring are kept.
    """
    features = []
    seen_ids = set()
    for element in elements:
        if element.get("type") != "way":
            continue
        osm_id = element.get("id")
        if osm_id is None or osm_id in seen_ids:
            continue
        geom = _way_to_geometry(element)
        if geom is None:
            continue
        seen_ids.add(osm_id)
        tags = element.get("tags", {})
        features.append(
            {
                "type": "Feature",
                "id": f"osm-way-{osm_id}",
                "geometry": geom,
                "properties": {
                    "osm_type": "way",
                    "osm_id": osm_id,
                    "name": tags.get("name"),
                    "landuse": tags.get("landuse"),
                    "industrial": tags.get("industrial"),
                    "natural": tags.get("natural"),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _cache_path(zone_type: str) -> Path:
    return repo_path(getattr(settings, CACHE_FILE_SETTINGS[zone_type]))


def _cache_is_fresh(path: Path, zone_type: str) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    fresh = age_seconds < settings.industrial_cache_max_age_hours * 3600
    if fresh:
        logger.info(
            "%s zones cache is fresh (%.1f h old)", zone_type, age_seconds / 3600
        )
    else:
        logger.info(
            "%s zones cache is stale (%.1f h old)", zone_type, age_seconds / 3600
        )
    return fresh


def read_cache(zone_type: str) -> dict | None:
    path = _cache_path(zone_type)
    if not _cache_is_fresh(path, zone_type):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s zones cache: %s", zone_type, exc)
        return None
    return data.get("feature_collection")


def write_cache(zone_type: str, feature_collection: dict) -> None:
    payload = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "states": list(settings.industrial_states),
        "feature_collection": feature_collection,
    }
    path = _cache_path(zone_type)
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger.info(
        "cached %s %s zone feature(s) to %s",
        len(feature_collection["features"]),
        zone_type,
        path,
    )


async def fetch_region(
    west: float,
    south: float,
    east: float,
    north: float,
    zone_type: str,
    client: httpx.AsyncClient,
    timeout: int,
) -> list[dict]:
    """Query mirrors in order, retrying transient failures, until one succeeds."""
    query = build_query(west, south, east, north, zone_type, timeout)
    last_error: Exception | None = None
    for base_url in OVERPASS_URLS:
        for attempt in range(OVERPASS_RETRIES):
            try:
                logger.info(
                    "querying %s (attempt %d/%d) for %s bbox %s,%s,%s,%s",
                    base_url,
                    attempt + 1,
                    OVERPASS_RETRIES,
                    zone_type,
                    west,
                    south,
                    east,
                    north,
                )
                response = await client.get(
                    base_url,
                    params={"data": query},
                    headers=OVERPASS_HEADERS,
                    timeout=OVERPASS_REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                try:
                    return response.json().get("elements", [])
                except ValueError as exc:
                    # A 200 with a non-JSON body (proxy error pages, HTML
                    # captcha/tarpit responses, etc.) is still a mirror failure —
                    # fall through to the next mirror instead of killing the tile.
                    last_error = exc
                    logger.warning(
                        "%s returned a non-JSON body (%s) — treating as mirror failure",
                        base_url,
                        exc,
                    )
                    break
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_error = exc
                retriable = isinstance(exc, httpx.RequestError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code in (429, 502, 503, 504)
                )
                if retriable and attempt + 1 < OVERPASS_RETRIES:
                    sleep_s = OVERPASS_RETRY_BACKOFF_SECONDS[
                        min(attempt, len(OVERPASS_RETRY_BACKOFF_SECONDS) - 1)
                    ]
                    logger.warning(
                        "%s failed (%s) — retrying in %ss", base_url, exc, sleep_s
                    )
                    await asyncio.sleep(sleep_s)
                    continue
                logger.warning("%s failed (%s) — trying next mirror", base_url, exc)
                break
    raise RuntimeError(
        f"Overpass unavailable for {zone_type} bbox "
        f"{west},{south},{east},{north}: {last_error}"
    )


async def _fetch_one(
    zone_type: str,
    state: str,
    tile: tuple[float, float, float, float],
    client: httpx.AsyncClient,
    timeout: int,
    semaphore: asyncio.Semaphore,
    failures: dict,
) -> list[dict]:
    """Fetch one tile under a shared concurrency limit; logs and skips on failure."""
    async with semaphore:
        try:
            return await fetch_region(*tile, zone_type, client, timeout)
        except RuntimeError as exc:
            failures["count"] += 1
            logger.error(
                "skipping %s tile %s of %s after %s",
                zone_type,
                tile,
                state,
                exc,
            )
            return []


async def _fetch_all_tiles(
    zone_type: str, client: httpx.AsyncClient, timeout: int, max_concurrency: int
) -> tuple[list[dict], int]:
    """Query every state bbox tile for the layer, respecting a concurrency cap."""
    semaphore = asyncio.Semaphore(max_concurrency)
    failures = {"count": 0}
    tasks = []
    for state, bbox in settings.industrial_states.items():
        west, south, east, north = parse_state_bbox(bbox)
        for tile in tile_bbox(west, south, east, north):
            tasks.append(
                _fetch_one(zone_type, state, tile, client, timeout, semaphore, failures)
            )
    results = await asyncio.gather(*tasks)
    elements = [element for group in results for element in group]
    return elements, failures["count"]


_in_flight_fetches: dict[str, asyncio.Task] = {}


async def get_zones(
    zone_type: str,
    client: httpx.AsyncClient | None = None,
    max_concurrency: int = 4,
) -> dict:
    """Return one zone layer as a GeoJSON FeatureCollection.

    Serves from a fresh local cache if available; otherwise queries Overpass per
    state, split into tiles, and writes a new cache file. Individual tile
    failures are logged and skipped so a partial dataset still caches and the
    demo keeps working.
    """
    cached = read_cache(zone_type)
    if cached is not None:
        return cached

    # Deduplicate concurrent fetch calls for the same zone layer
    if zone_type in _in_flight_fetches and not _in_flight_fetches[zone_type].done():
        return await _in_flight_fetches[zone_type]

    task = asyncio.create_task(_get_zones_impl(zone_type, client, max_concurrency))
    _in_flight_fetches[zone_type] = task
    try:
        return await task
    finally:
        _in_flight_fetches.pop(zone_type, None)


async def _get_zones_impl(
    zone_type: str,
    client: httpx.AsyncClient | None = None,
    max_concurrency: int = 4,
) -> dict:
    closer = False
    if client is None:
        client = httpx.AsyncClient(timeout=float(settings.overpass_timeout))
        closer = True
    try:
        elements, failures = await _fetch_all_tiles(
            zone_type, client, settings.overpass_timeout, max_concurrency
        )
        logger.info(
            "%s: %s ways total, %s tile(s) failed",
            zone_type,
            len(elements),
            failures,
        )
        feature_collection = osm_elements_to_featurecollection(elements)
        write_cache(zone_type, feature_collection)
        return feature_collection
    finally:
        if closer:
            await client.aclose()


async def get_industrial_zones(
    client: httpx.AsyncClient | None = None,
    max_concurrency: int = 4,
) -> dict:
    """Industrial land-use polygons (landuse=industrial) as GeoJSON."""
    return await get_zones("industrial", client, max_concurrency)


async def get_vegetation_zones(
    client: httpx.AsyncClient | None = None,
    max_concurrency: int = 4,
) -> dict:
    """Forest/vegetation polygons (wood, scrub, forest) as GeoJSON."""
    return await get_zones("vegetation", client, max_concurrency)


async def get_mining_zones(
    client: httpx.AsyncClient | None = None,
    max_concurrency: int = 4,
) -> dict:
    """Mining polygons (quarries + mine shafts) as GeoJSON."""
    return await get_zones("mining", client, max_concurrency)
