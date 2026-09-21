"""Tests for the directional risk indicator heuristic.

The indicator must:
- point the cone along the wind *spread* direction (180° from the
  meteorological "from" direction),
- scale cone length with wind speed and width inversely,
- blend vegetation density + dryness (temperature/humidity) into a bounded
  risk intensity,
- and degrade gracefully when Open-Meteo data is missing/failing.
"""

import pytest

from app.services import risk


@pytest.fixture(autouse=True)
def _fresh_weather_cache():
    """Isolate the module-level weather cache between tests."""
    from app.services import weather

    weather.clear_cache()
    yield
    weather.clear_cache()


def _props(**overrides):
    base = {
        "frp": 5.0,
        "persistent_thermal_source": False,
        "vegetation_distance_m": 800.0,
    }
    base.update(overrides)
    return base


def _feature(frp, persistent=False, **overrides):
    props = _props(frp=frp, persistent_thermal_source=persistent, **overrides)
    return {
        "type": "Feature",
        "id": 0,
        "geometry": {"type": "Point", "coordinates": [77.0, 22.0]},
        "properties": props,
    }


def _fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


# --- spread direction -----------------------------------------------------


def test_spread_direction_is_opposite_wind_from():
    assert risk.spread_direction_deg(0) == 180.0  # from N -> toward S
    assert risk.spread_direction_deg(90) == 270.0  # from E -> toward W
    assert risk.spread_direction_deg(180) == 0.0  # from S -> toward N
    assert risk.spread_direction_deg(270) == 90.0  # from W -> toward E


def test_cone_points_along_spread_direction_south():
    """Wind from north (0°) → cone points south: apex above the outer arc."""
    geom = risk.cone_geometry(22.0, 77.0, 2000, 20, spread_deg=180.0)
    ring = geom["coordinates"][0]
    apex_lat = ring[0][1]
    apex_lon = ring[0][0]
    # Outer-arc midpoint (index 5 with the default _CONE_ARC_POINTS) sits due
    # south of the apex: lower lat, same lon, ~2000 m away.
    mid_lat, mid_lon = ring[5][1], ring[5][0]
    assert mid_lat < apex_lat
    assert abs(mid_lon - apex_lon) < 1e-9
    assert abs((apex_lat - mid_lat) * 111320.0 - 2000) < 5


def test_cone_points_along_spread_direction_east():
    """Wind from west (270°) → spread toward east (90°): lon increases."""
    geom = risk.cone_geometry(22.0, 77.0, 2000, 20, spread_deg=90.0)
    ring = geom["coordinates"][0]
    apex_lat, apex_lon = ring[0][1], ring[0][0]
    mid_lat, mid_lon = ring[5][1], ring[5][0]
    assert mid_lon > apex_lon
    assert abs(mid_lat - apex_lat) < 1e-9
    assert abs((mid_lon - apex_lon) * 111320.0 * 0.927 - 2000) < 10


def test_cone_geometry_is_a_valid_closed_polygon():
    geom = risk.cone_geometry(22.0, 77.0, 2000, 20, spread_deg=180.0)
    ring = geom["coordinates"][0]
    assert ring[0] == ring[-1]  # closed ring
    assert len(ring) >= 5  # apex + two flanks + arc + close


# --- wind scaling ----------------------------------------------------------


def test_cone_length_scales_with_wind_speed():
    assert risk.cone_length_m(5) < risk.cone_length_m(30) < risk.cone_length_m(120)
    # Clamped at the configured maximum for very strong wind.
    assert risk.cone_length_m(200) == pytest.approx(risk.settings.risk_cone_max_length_m)


def test_cone_width_scales_inversely_with_wind_speed():
    # Faster wind -> narrower, more directional cone.
    assert risk.cone_half_angle_deg(5) > risk.cone_half_angle_deg(30)
    assert risk.cone_half_angle_deg(200) == pytest.approx(
        risk.settings.risk_cone_min_half_angle_deg
    )


# --- risk intensity blend --------------------------------------------------


def test_vegetation_density_boosts_risk():
    hot_dry_close = risk.risk_score(10, 25, 50, 20)  # deep in vegetation
    hot_dry_far = risk.risk_score(10, 25, 50, 5000)  # far from vegetation
    assert hot_dry_close > hot_dry_far


def test_dryness_boosts_risk():
    # Same wind + vegetation, only weather differs.
    hot_dry = risk.risk_score(10, 40, 20, 800)
    cool_humid = risk.risk_score(10, 10, 90, 800)
    assert hot_dry > cool_humid


def test_wind_boosts_risk():
    assert risk.risk_score(60, 20, 50, 800) > risk.risk_score(0, 20, 50, 800)


def test_risk_score_is_bounded():
    cases = [
        (60, 45, 10, 0),
        (0, 0, 100, 10000),
        (120, 50, 100, 50),
        (None, None, None, None),
        (5, 25, 60, 500),
    ]
    for wind, temp, hum, veg in cases:
        score = risk.risk_score(wind, temp, hum, veg)
        assert 0.0 <= score <= 1.0


def test_risk_tiers():
    assert risk.risk_tier(0.9) == "high"
    assert risk.risk_tier(0.5) == "medium"
    assert risk.risk_tier(0.1) == "low"


def test_cardinal_compass_names():
    assert risk.cardinal(0) == "N"
    assert risk.cardinal(90) == "E"
    assert risk.cardinal(180) == "S"
    assert risk.cardinal(270) == "W"
    assert risk.cardinal(315) == "NW"


# --- high-priority selection -------------------------------------------------


def test_high_priority_persistent_or_high_frp():
    assert risk.high_priority(_props(persistent_thermal_source=True, frp=0))
    assert risk.high_priority(_props(frp=30), frp_threshold_mw=20.0)
    assert not risk.high_priority(_props(frp=5), frp_threshold_mw=20.0)
    assert not risk.high_priority(_props(frp=None))
    assert not risk.high_priority(_props(persistent_thermal_source=False, frp=0))


# --- payload assembly -------------------------------------------------------


def test_build_directional_risk_payload():
    data = risk.build_directional_risk(
        _props(vegetation_distance_m=250),
        {
            "temperature_c": 34.0,
            "humidity_pct": 30.0,
            "wind_speed_kmh": 24.0,
            "wind_direction_deg": 200.0,
        },
        lat=22.0,
        lon=77.0,
    )
    assert data["method"] == "heuristic"
    assert data["spread_direction_deg"] == pytest.approx(20.0)  # 200 + 180 % 360
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_tier"] in ("low", "medium", "high")
    assert data["geometry"]["type"] == "Polygon"
    assert data["cone_length_m"] > 0


# --- annotation pipeline (graceful degradation) ------------------------------


class _AsyncWeather:
    """Stub risk.weather: fetch_weather returning a fixed result."""

    def __init__(self, result):
        self.result = result

    async def fetch_weather(self, lat, lon):
        return self.result


@pytest.mark.asyncio
async def test_annotate_only_touches_high_priority_fires(monkeypatch):
    weather_data = {
        "temperature_c": 31.0,
        "humidity_pct": 41.0,
        "wind_speed_kmh": 24.0,
        "wind_direction_deg": 200.0,
    }
    monkeypatch.setattr(risk, "weather", _AsyncWeather(weather_data))
    fc = _fc(_feature(30.0), _feature(0.0, persistent=True), _feature(5.0))
    await risk.annotate_directional_risk(fc)
    high_frp = fc["features"][0]["properties"]
    persistent = fc["features"][1]["properties"]
    calm = fc["features"][2]["properties"]
    assert "directional_risk" in high_frp
    assert "directional_risk" in persistent
    assert "directional_risk" not in calm


@pytest.mark.asyncio
async def test_annotate_weather_failure_degrades_gracefully(monkeypatch):
    """Open-Meteo unavailable → hotspots render with NO risk overlay, no crash."""
    monkeypatch.setattr(risk, "weather", _AsyncWeather(None))
    fc = _fc(_feature(50.0), _feature(1.0, persistent=True))
    await risk.annotate_directional_risk(fc)
    for feature in fc["features"]:
        assert "directional_risk" not in feature["properties"]


@pytest.mark.asyncio
async def test_annotate_weather_with_empty_dict_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(risk, "weather", _AsyncWeather({}))
    fc = _fc(_feature(50.0))
    await risk.annotate_directional_risk(fc)
    assert "directional_risk" not in fc["features"][0]["properties"]


@pytest.mark.asyncio
async def test_annotate_does_not_raise_on_empty_collection():
    assert await risk.annotate_directional_risk(_fc()) == _fc()


def test_risk_module_exposes_settings_bounds():
    # Sanity: defaults keep the heuristic sane & explainable.
    assert risk.settings.risk_cone_min_length_m < risk.settings.risk_cone_max_length_m
    assert risk.settings.risk_cone_min_half_angle_deg < risk.settings.risk_cone_max_half_angle_deg
    assert risk.settings.risk_wind_speed_max_kmh > 0
