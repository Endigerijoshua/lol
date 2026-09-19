"""Tests for the startup reference-layer warm-up (app/main.py lifespan)."""

import asyncio

from app.main import _reference_cache_version, _warm_reference_caches
from app.services import flares, osm, powerplants, spatial


def _run(coro) -> None:
    asyncio.run(coro)


def test_warm_reference_caches_survives_failure(monkeypatch):
    """One failing layer must not abort the rest of the warm-up or the index."""

    async def ok_fc(n=3):
        # Valid point geometry so the reference bundle could build if reached.
        return {
            "type": "FeatureCollection",
            "features": [
                {"id": i, "geometry": {"type": "Point", "coordinates": [77.2, 20.6]}}
                for i in range(n)
            ],
        }

    async def failing():
        raise RuntimeError("overpass down")

    spatial._REFERENCE_CACHE.clear()

    # Patch each public getter on its service module. `mining` fails; the rest
    # succeed.
    attempted = []

    def patch(name: str, getter, module, failure: bool):
        real = getter

        async def replacement(*a, **kw):
            attempted.append(name)
            if failure:
                return await failing()
            return await ok_fc()

        monkeypatch.setattr(module, real.__name__, replacement)

    patch("industrial", osm.get_industrial_zones, osm, False)
    patch("vegetation", osm.get_vegetation_zones, osm, False)
    patch("mining", osm.get_mining_zones, osm, True)
    patch("power-plants", powerplants.get_power_plants, powerplants, False)
    patch("flares", flares.get_flares, flares, False)

    # Warm-up must return normally even though the mining layer raises.
    _run(_warm_reference_caches())

    # Every layer was attempted, including the failing one.
    assert set(attempted) == {
        "industrial",
        "vegetation",
        "mining",
        "power-plants",
        "flares",
    }
    # Because mining failed, no (potentially partial) reference bundle may be
    # registered — a partial bundle would silently mislabel fires.
    assert spatial._REFERENCE_CACHE == {}
    # A real failure surfaced only as a log, not as a raised exception.


def test_warm_reference_caches_prebuilds_index(monkeypatch):
    """With every layer up, warm-up prebuilds the parsed spatial index."""

    async def ok_fc(n=3):
        return {
            "type": "FeatureCollection",
            "features": [
                {"id": i, "geometry": {"type": "Point", "coordinates": [77.2, 20.6]}}
                for i in range(n)
            ],
        }

    for module, name in (
        (osm, "get_industrial_zones"),
        (osm, "get_vegetation_zones"),
        (osm, "get_mining_zones"),
        (powerplants, "get_power_plants"),
        (flares, "get_flares"),
    ):
        monkeypatch.setattr(module, name, ok_fc)

    spatial._REFERENCE_CACHE.clear()
    try:
        _run(_warm_reference_caches())
        key = _reference_cache_version()
        assert key in spatial._REFERENCE_CACHE
        bundle = spatial._REFERENCE_CACHE[key]
        assert len(bundle["vegetation_polygons"]) == 3
    finally:
        spatial._REFERENCE_CACHE.clear()