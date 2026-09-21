"""NASA FIRMS live fire/thermal hotspot fetch.

Uses the NASA FIRMS "area" API (CSV) and converts the response into clean GeoJSON.

Empirical notes (verified while building, keep for anyone touching this):
- The FIRMS WFS bbox filter returns 0 features for any bbox -> unusable for India.
- The unrestricted WFS `fires_snpp_24hrs` layer under-reports India (~16 points on
  2026-09-04) vs the CSV area API (208 points, same box/date) for the India box.
  Hence the CSV area API is the correct source.
- Region for India: `SouthEast_Asia`; India bbox used: `68,6,97,37`.

India boundary filter:
- The FIRMS area API bbox is a *rectangle*, so it always includes fires in
  neighbouring countries (Pakistan, China, Nepal, Bhutan, Bangladesh, Myanmar,
  Sri Lanka). Every response is therefore clipped to a real India boundary
  polygon (`app/data/india_boundary.geojson`, Natural Earth 10m admin-0) before
  it is returned. A tight rectangle is NOT enough - it leaks along the borders.
- A tiny outward buffer (default 0.03 deg ~= 3 km, `india_boundary_buffer_deg`)
  absorbs FIRMS point geolocation error (~375 m for VIIRS) so genuine border
  fires on the India side are not dropped.

Reference: https://firms.modaps.eosdis.nasa.gov/api/area/csv
"""

import io
import json
import logging
from functools import lru_cache
from pathlib import Path

import httpx
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon, shape

from ..config import get_settings, settings

logger = logging.getLogger(__name__)

FIRMS_AREA_URL = settings.firms_area_url

# Natural Earth 10m admin-0 boundary for India (public domain). Contains islands
# (Andaman & Nicobar) as extra polygons. 110m was tried first but is too coarse:
# it drops the NE border states and the island territories.
INDIA_BOUNDARY_FILE = Path(__file__).resolve().parent.parent / "data" / "india_boundary.geojson"

# Fields copied verbatim from each FIRMS CSV row onto the GeoJSON feature.
PROPERTY_FIELDS = (
    "confidence",
    "bright_ti4",
    "frp",
    "acq_date",
    "acq_time",
    "satellite",
    "daynight",
)


def parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    """Parse 'west,south,east,north' into floats, validating global bounds."""
    parts = [float(p.strip()) for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError(f"bbox must be 'west,south,east,north', got: {bbox}")
    west, south, east, north = parts
    if not (-180 <= west < east <= 180) or not (-90 <= south < north <= 90):
        raise ValueError(f"invalid bbox: {bbox}")
    return west, south, east, north


def build_url(api_key: str, dataset: str, bbox: str, days: int) -> str:
    if not api_key:
        raise ValueError("FIRMS_MAP_KEY is not set. Add it to .env (see .env.example).")
    parsed = parse_bbox(bbox)
    bbox_comma = ",".join(str(v) for v in parsed)
    return FIRMS_AREA_URL.format(key=api_key, dataset=dataset, bbox=bbox_comma, days=days)


def _json_value(value):
    """Convert a pandas/numpy scalar into a plain JSON-safe Python value."""
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (str, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    return value


@lru_cache(maxsize=1)
def _india_boundary() -> Polygon | MultiPolygon:
    """Load the India boundary polygon from the bundled Natural Earth GeoJSON."""
    if not INDIA_BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"India boundary GeoJSON not found: {INDIA_BOUNDARY_FILE}. "
            "It ships with the repo; do not delete it."
        )
    data = json.loads(INDIA_BOUNDARY_FILE.read_text(encoding="utf-8"))
    geometry = data["geometry"]
    poly = shape(geometry)
    if poly.geom_type == "Polygon":
        poly = MultiPolygon([poly])
    return poly


def is_point_in_india(lon: float, lat: float, buffer_deg: float | None = None) -> bool:
    """Return True if (lon, lat) falls inside (or within buffer_deg of) India.

    A small outward buffer absorbs FIRMS point geolocation error so border fires
    on the India side are not falsely dropped.
    """
    if buffer_deg is None:
        buffer_deg = get_settings().india_boundary_buffer_deg
    boundary = _india_boundary()
    if boundary.contains(Point(lon, lat)):
        return True
    if buffer_deg > 0:
        return boundary.buffer(buffer_deg).contains(Point(lon, lat))
    return False


def filter_to_india_boundary(fc: dict, buffer_deg: float | None = None) -> dict:
    """Drop every fire outside the real India boundary polygon.

    The FIRMS area-API bbox is a rectangle that includes neighbouring countries;
    this is the actual "India-only" filter and is applied to every live response.
    """
    features = fc.get("features", [])
    total = len(features)
    kept = [
        f
        for f in features
        if is_point_in_india(*f["geometry"]["coordinates"], buffer_deg=buffer_deg)
    ]
    dropped = total - len(kept)
    if dropped:
        logger.info(
            "India boundary filter dropped %s of %s FIRMS detections (outside India polygon)",
            dropped,
            total,
        )
    return {"type": "FeatureCollection", "features": kept}


def csv_to_geojson(csv_text: str) -> dict:
    """Convert FIRMS area-API CSV text into a GeoJSON FeatureCollection.

    Logs the input row count vs. produced feature count so any dropped
    detection is never silent.
    """
    df = pd.read_csv(io.StringIO(csv_text))
    total_rows = len(df)

    features = []
    dropped = 0
    for _, row in df.iterrows():
        lon, lat = row["longitude"], row["latitude"]
        if pd.isna(lon) or pd.isna(lat):
            dropped += 1
            continue
        props = {field: _json_value(row[field]) for field in PROPERTY_FIELDS}
        features.append(
            {
                "type": "Feature",
                "id": len(features),
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": props,
            }
        )

    if dropped:
        logger.warning(
            "FIRMS CSV->GeoJSON dropped %s of %s rows (missing coordinates)",
            dropped,
            total_rows,
        )
    logger.info(
        "FIRMS CSV -> GeoJSON: %s rows parsed, %s features produced",
        total_rows,
        len(features),
    )
    return {"type": "FeatureCollection", "features": features}


async def fetch_fires(
    api_key: str | None = None,
    dataset: str | None = None,
    bbox: str | None = None,
    days: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Fetch live FIRMS hotspot detections for the configured India bbox.

    Returns a GeoJSON FeatureCollection. Raises ValueError on missing key,
    httpx.HTTPStatusError on a non-200 response, and httpx.RequestError on
    network failures.
    """
    cfg = get_settings()
    api_key = api_key or cfg.firms_map_key
    dataset = dataset or cfg.firms_dataset
    bbox = bbox or cfg.firms_bbox
    days = days or cfg.firms_days

    url = build_url(api_key, dataset, bbox, days)
    logger.info("fetching FIRMS CSV: %s", url)

    closer = False
    if client is None:
        client = httpx.AsyncClient(timeout=settings.firms_request_timeout_seconds)
        closer = True
    try:
        response = await client.get(url)
        response.raise_for_status()
        return filter_to_india_boundary(csv_to_geojson(response.text))
    finally:
        if closer:
            await client.aclose()
