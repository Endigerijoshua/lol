"""Directional risk indicator: a simple wind-based spread cone per hotspot.

This is a HEURISTIC, not a fire-behavior simulation. For each high-priority
detection (a persistent thermal source, or a high-FRP fire) we pull current
wind/temperature/humidity from Open-Meteo and draw a cone:

- the cone points along the wind *spread* direction (where the wind blows TO),
  180° from the "from" direction Open-Meteo reports
- cone length scales UP with wind speed
- cone width (half-angle) scales DOWN with wind speed — faster wind means a
  longer, narrower, more directional spread
- risk intensity (0..1) is a weighted linear blend of wind, nearby vegetation
  density (reuses the OSM `vegetation_distance_m` computed by spatial.py;
  closer = more fuel = higher risk) and a dryness factor from temperature +
  humidity (hot + dry = higher risk)

Every pure function here is unit-testable; the network fetch (`weather.py`)
never raises, so a failed weather lookup degrades gracefully — the hotspot
still renders, it just has no risk overlay.
"""

import logging
import math

from shapely.geometry import shape

from ..config import settings
from . import weather

logger = logging.getLogger(__name__)

# Weights of the explainable linear risk blend (sum ≈ 1.0).
WIND_WEIGHT = 0.34
VEGETATION_WEIGHT = 0.33
DRYNESS_WEIGHT = 0.33

_METERS_PER_DEGREE_LAT = 111320.0
_CONE_ARC_POINTS = 8

_CARDINALS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _wind_factor(wind_speed_kmh) -> float:
    """Normalized wind strength 0..1 (None → neutral 0.5)."""
    speed = _coerce_float(wind_speed_kmh)
    if speed is None:
        return 0.5
    return clamp(speed / settings.risk_wind_speed_max_kmh)


def cone_length_m(wind_speed_kmh) -> float:
    """Spread-cone length (m): more wind → longer cone."""
    if _coerce_float(wind_speed_kmh) is None:
        return settings.risk_cone_min_length_m
    return settings.risk_cone_min_length_m + (
        settings.risk_cone_max_length_m - settings.risk_cone_min_length_m
    ) * _wind_factor(wind_speed_kmh)


def cone_half_angle_deg(wind_speed_kmh) -> float:
    """Cone half-width (deg): more wind → narrower, more directional spread."""
    if _coerce_float(wind_speed_kmh) is None:
        return (settings.risk_cone_min_half_angle_deg + settings.risk_cone_max_half_angle_deg) / 2
    return settings.risk_cone_max_half_angle_deg - (
        settings.risk_cone_max_half_angle_deg - settings.risk_cone_min_half_angle_deg
    ) * _wind_factor(wind_speed_kmh)


def dryness_factor(temperature_c, humidity_pct) -> float:
    """0..1 dryness from temperature + humidity (hot + dry → 1)."""
    temp = _coerce_float(temperature_c)
    hum = _coerce_float(humidity_pct)
    temp_term = 0.0 if temp is None else clamp((temp - 15.0) / 25.0)
    hum_term = 0.0 if hum is None else clamp((100.0 - hum) / 60.0)
    if temp is None and hum is None:
        return 0.5
    if temp is None:
        return hum_term
    if hum is None:
        return temp_term
    return 0.5 * temp_term + 0.5 * hum_term


def vegetation_factor(vegetation_distance_m) -> float:
    """0..1 vegetation density from `vegetation_distance_m` (closer → 1).

    No vegetation measurement (None) means no extra fuel signal → 0.
    """
    dist = _coerce_float(vegetation_distance_m)
    if dist is None:
        return 0.0
    return clamp(1.0 - dist / settings.risk_vegetation_distance_m)


def risk_score(
    wind_speed_kmh,
    temperature_c,
    humidity_pct,
    vegetation_distance_m,
) -> float:
    """Weighted linear risk intensity in 0..1 (explainable, no ML)."""
    score = (
        WIND_WEIGHT * _wind_factor(wind_speed_kmh)
        + VEGETATION_WEIGHT * vegetation_factor(vegetation_distance_m)
        + DRYNESS_WEIGHT * dryness_factor(temperature_c, humidity_pct)
    )
    return clamp(score)


def risk_tier(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def spread_direction_deg(wind_direction_deg) -> float:
    """Where the wind blows TO; Open-Meteo reports the "from" direction."""
    direction = _coerce_float(wind_direction_deg)
    if direction is None:
        return 0.0
    return (direction + 180.0) % 360.0


def cardinal(degrees) -> str:
    """Compass name (N/NE/…) for a clockwise-from-north bearing."""
    if _coerce_float(degrees) is None:
        return "\u2014"
    index = round((float(degrees) % 360.0) / 45.0) % 8
    return _CARDINALS[index]


def _point_at(lat: float, lon: float, distance_m: float, bearing_deg: float):
    """(new_lon, new_lat) `distance_m` away on `bearing_deg` (0=North)."""
    d_lat = distance_m / _METERS_PER_DEGREE_LAT
    d_lon = distance_m / (_METERS_PER_DEGREE_LAT * max(math.cos(math.radians(lat)), 0.2))
    bearing = math.radians(bearing_deg % 360.0)
    return (lon + d_lon * math.sin(bearing), lat + d_lat * math.cos(bearing))


def cone_geometry(
    lat: float,
    lon: float,
    length_m: float,
    half_angle_deg: float,
    spread_deg: float,
) -> dict:
    """GeoJSON Polygon (lon/lat) for a wind-direction wedge.

    Shape: a sector with its apex at the hotspot and a rounded outer arc
    between `spread - half_angle` and `spread + half_angle`, at `length_m`.
    """
    left_bearing = spread_deg - half_angle_deg
    right_bearing = spread_deg + half_angle_deg
    apex = (lon, lat)
    ring = [apex, list(_point_at(lat, lon, length_m, left_bearing))]
    for i in range(1, _CONE_ARC_POINTS):
        fraction = i / _CONE_ARC_POINTS
        bearing = left_bearing + (right_bearing - left_bearing) * fraction
        ring.append(list(_point_at(lat, lon, length_m, bearing)))
    ring.append(list(_point_at(lat, lon, length_m, right_bearing)))
    ring.append(apex)
    return {"type": "Polygon", "coordinates": [ring]}


def high_priority(props: dict, frp_threshold_mw: float | None = None) -> bool:
    """A persistent thermal source or a high-FRP fire gets a risk cone."""
    threshold = settings.risk_frp_threshold_mw if frp_threshold_mw is None else frp_threshold_mw
    if props.get("persistent_thermal_source"):
        return True
    frp = _coerce_float(props.get("frp"))
    return frp is not None and frp >= threshold


def build_directional_risk(props: dict, weather_data: dict, lat: float, lon: float) -> dict:
    """Assemble the full `directional_risk` payload for one fire."""
    wind_speed = weather_data.get("wind_speed_kmh")
    wind_from = weather_data.get("wind_direction_deg")
    spread = spread_direction_deg(wind_from)
    length_m = cone_length_m(wind_speed)
    half_angle_deg = cone_half_angle_deg(wind_speed)
    veg_distance = props.get("vegetation_distance_m")
    score = risk_score(
        wind_speed,
        weather_data.get("temperature_c"),
        weather_data.get("humidity_pct"),
        veg_distance,
    )
    return {
        "method": "heuristic",
        "weather": weather_data,
        "spread_direction_deg": round(spread, 1),
        "cone_length_m": round(length_m, 0),
        "cone_half_angle_deg": round(half_angle_deg, 1),
        "vegetation_factor": round(vegetation_factor(veg_distance), 2),
        "dryness_factor": round(
            dryness_factor(
                weather_data.get("temperature_c"),
                weather_data.get("humidity_pct"),
            ),
            2,
        ),
        "risk_score": round(score, 2),
        "risk_tier": risk_tier(score),
        "geometry": cone_geometry(lat, lon, length_m, half_angle_deg, spread),
    }


async def annotate_directional_risk(fires_fc: dict) -> dict:
    """Add `directional_risk` to high-priority fires in a FeatureCollection.

    Only persistent-thermal-source and high-FRP fires are touched (the "high
    priority" set). Weather failures leave the feature untouched so the hotspot
    still renders without the overlay — this never raises. Mutates and returns
    the input FeatureCollection.
    """
    touched = 0
    for feature in fires_fc.get("features", []):
        props = feature.get("properties") or {}
        if not high_priority(props):
            continue
        try:
            point = shape(feature.get("geometry"))
        except (ValueError, TypeError):
            continue
        if point.is_empty or point.geom_type != "Point":
            continue
        weather_data = await weather.fetch_weather(point.y, point.x)
        if not weather_data:
            continue
        props["directional_risk"] = build_directional_risk(props, weather_data, point.y, point.x)
        touched += 1
    logger.info(
        "risk.annotate_directional_risk: %d/%d fires got a directional risk indicator",
        touched,
        len(fires_fc.get("features", [])),
    )
    return fires_fc
