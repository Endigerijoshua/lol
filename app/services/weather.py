"""Open-Meteo current-weather fetch for the directional risk indicator.

Open-Meteo is a free, no-key weather API. This module pulls a single current
observation (temperature, humidity, wind speed/direction) for one lat/lon and
returns a small plain dict:

    {
        "temperature_c": float|None,
        "humidity_pct": float|None,
        "wind_speed_kmh": float|None,
        "wind_direction_deg": float|None,
    }

It is deliberately a best-effort helper: every failure (network, HTTP status,
bad JSON, missing fields) is swallowed and returns None, so the fire pipeline
keeps serving the hotspot — just without a risk overlay. A tiny in-memory TTL
cache keyed on rounded (~0.1 deg) coordinates stops the same storm fetch from
being repeated for every nearby hotspot.
"""

import logging
import time

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_REQUEST_TIMEOUT_SECONDS = 10.0

# "current" variables requested from the forecast API (default units: °C, %,
# km/h and degrees).
CURRENT_VARIABLES = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"

_WEATHER_CACHE: dict[tuple[float, float], tuple[float, dict]] = {}


def clear_cache() -> None:
    """Drop all cached weather (used by tests)."""
    _WEATHER_CACHE.clear()


def _cache_key(lat: float, lon: float) -> tuple[float, float]:
    """Round to ~10 km grid so nearby hotspots share one fetch."""
    return round(lat, 1), round(lon, 1)


def _parse_payload(data) -> dict:
    """Coerce the Open-Meteo `current` block into our flat weather dict."""
    current = (data or {}).get("current") or {}
    return {
        "temperature_c": _optional_float(current.get("temperature_2m")),
        "humidity_pct": _optional_float(current.get("relative_humidity_2m")),
        "wind_speed_kmh": _optional_float(current.get("wind_speed_10m")),
        "wind_direction_deg": _optional_float(current.get("wind_direction_10m")),
    }


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value


async def fetch_weather(
    lat: float,
    lon: float,
    client: httpx.AsyncClient | None = None,
) -> dict | None:
    """Fetch current weather for (lat, lon); None on any failure.

    `client` is injectable for tests. A process-local cache keyed on ~0.1 deg
    rounded coordinates suppresses repeat calls within the TTL.
    """
    key = _cache_key(lat, lon)
    now = time.monotonic()
    hit = _WEATHER_CACHE.get(key)
    if hit and now - hit[0] < settings.weather_cache_ttl_seconds:
        return dict(hit[1])

    closer = False
    if client is None:
        client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS)
        closer = True
    try:
        response = await client.get(
            OPEN_METEO_URL,
            params={
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "current": CURRENT_VARIABLES,
            },
        )
        response.raise_for_status()
        weather = _parse_payload(response.json())
        if all(value is None for value in weather.values()):
            logger.warning(
                "Open-Meteo returned no usable current weather for (%.2f, %.2f)",
                lat,
                lon,
            )
            return None
        _WEATHER_CACHE[key] = (now, weather)
        return dict(weather)
    except (httpx.HTTPError, ValueError):
        logger.warning(
            "Open-Meteo weather fetch failed for (%.2f, %.2f) — risk overlay "
            "skipped for this hotspot",
            lat,
            lon,
        )
        return None
    finally:
        if closer:
            await client.aclose()
