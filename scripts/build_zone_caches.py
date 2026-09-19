"""Build all reference-layer caches ahead of the demo.

The OSM zone caches (industrial / vegetation / mining) are git-ignored and
rebuilt from Overpass on first use. On a fresh checkout the *first* dashboard
load cold-builds them and takes minutes. Run this once on the demo machine so
the first page load is fast (~10 s, FIRMS-bound):

    python scripts/build_zone_caches.py

It writes whichever of the five reference caches are missing or stale:
industrial_zones_cache.json, vegetation_zones_cache.json,
mining_zones_cache.json, power_plants_cache.json, flares_cache.json.
Re-running is a no-op while caches are fresh (720 h TTL). A failing layer is
logged and skipped; the script exits non-zero if any layer failed.
"""

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from app.services import flares, osm, powerplants

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_zone_caches")


async def main() -> int:
    failures: list[str] = []

    async def build_one(name: str, coro) -> None:
        try:
            fc = await coro
            logger.info(
                "%s cache ready (%d features)",
                name,
                len(fc.get("features", [])),
            )
        except Exception as exc:  # noqa: BLE001 — keep going, report at the end
            failures.append(name)
            logger.warning("%s cache build failed: %s", name, exc)

    async with httpx.AsyncClient(timeout=60.0) as client:
        await asyncio.gather(
            build_one("industrial", osm.get_zones("industrial", client)),
            build_one("vegetation", osm.get_zones("vegetation", client)),
            build_one("mining", osm.get_zones("mining", client)),
            build_one("power-plants", powerplants.get_power_plants(client=client)),
            build_one("flares", flares.get_flares(client=client)),
        )

    if failures:
        logger.error("caches failed to build: %s", ", ".join(failures))
        return 1
    logger.info("all reference caches ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))