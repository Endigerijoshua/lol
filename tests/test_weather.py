"""Tests for the Open-Meteo weather fetch (directional risk indicator).

The fetch must parse the `current` block into a flat dict and — critically —
NEVER raise: network errors, HTTP errors and unusable payloads all collapse to
None so the fire pipeline keeps serving hotspots without a risk overlay.
"""

import asyncio

import httpx
import pytest
from httpx import MockTransport, Response

from app.services import weather


@pytest.fixture(autouse=True)
def _fresh_weather_cache():
    weather.clear_cache()
    yield
    weather.clear_cache()


def _payload(**overrides):
    current = {
        "time": "2026-09-16T12:00",
        "temperature_2m": 31.2,
        "relative_humidity_2m": 41.0,
        "wind_speed_10m": 24.1,
        "wind_direction_10m": 200.0,
    }
    current.update(overrides)
    return {"current": current}


async def _fetch(handler, lat=21.5, lon=76.5):
    async with httpx.AsyncClient(transport=MockTransport(handler)) as client:
        return await weather.fetch_weather(lat, lon, client=client)


def test_parses_current_weather_block():
    result = asyncio.run(_fetch(lambda request: Response(200, json=_payload())))
    assert result == {
        "temperature_c": 31.2,
        "humidity_pct": 41.0,
        "wind_speed_kmh": 24.1,
        "wind_direction_deg": 200.0,
    }


def test_request_asks_for_the_right_current_variables():
    seen = {}

    def handler(request):
        seen["params"] = request.url.params
        return Response(200, json=_payload())

    asyncio.run(_fetch(handler))
    assert seen["params"]["latitude"] == "21.5"
    assert "wind_speed_10m" in seen["params"]["current"]
    assert "wind_direction_10m" in seen["params"]["current"]
    assert "temperature_2m" in seen["params"]["current"]
    assert "relative_humidity_2m" in seen["params"]["current"]


def test_network_error_degrades_to_none():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    assert asyncio.run(_fetch(handler)) is None


def test_http_error_degrades_to_none():
    assert asyncio.run(_fetch(lambda request: Response(503))) is None


def test_missing_current_block_degrades_to_none():
    assert asyncio.run(_fetch(lambda request: Response(200, json={}))) is None


def test_empty_current_block_degrades_to_none():
    assert asyncio.run(_fetch(lambda request: Response(200, json={"current": {}}))) is None


def test_partial_field_returns_none_for_missing_only():
    result = asyncio.run(
        _fetch(
            lambda request: Response(200, json=_payload(temperature_2m=None, wind_speed_10m=None))
        )
    )
    assert result["temperature_c"] is None
    assert result["humidity_pct"] == 41.0
    assert result["wind_speed_kmh"] is None
    assert result["wind_direction_deg"] == 200.0


def test_cache_dedupes_nearby_coordinates():
    """Fires within the same ~0.1° grid cell share one Open-Meteo call."""
    calls = []

    def handler(request):
        calls.append(request.url)
        return Response(200, json=_payload())

    async def run():
        async with httpx.AsyncClient(transport=MockTransport(handler)) as client:
            await weather.fetch_weather(20.04, 80.02, client=client)
            await weather.fetch_weather(20.02, 80.05, client=client)  # same cell

    asyncio.run(run())
    assert len(calls) == 1
