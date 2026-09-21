"""Fetch vegetation zones from Overpass state by state, saving incrementally to vegetation_zones_cache.json."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from app.config import settings
from app.services import osm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_vegetation")


async def main():
    logger.info(
        "Starting vegetation zone incremental fetch across %d states...",
        len(settings.industrial_states),
    )

    # Load existing elements if cache file already exists
    all_elements = []
    seen_ids = set()
    cache_path = Path(settings.vegetation_cache_file)
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            fc = raw.get("feature_collection", {})
            logger.info("Loaded existing cache with %d features", len(fc.get("features", [])))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read existing cache: %s", e)

    semaphore = asyncio.Semaphore(6)
    failures = {"count": 0}

    async with httpx.AsyncClient(timeout=15.0) as client:
        for state, bbox in settings.industrial_states.items():
            t0 = time.time()
            west, south, east, north = osm.parse_state_bbox(bbox)
            tiles = osm.tile_bbox(west, south, east, north)

            tasks = [
                osm._fetch_one("vegetation", state, tile, client, 60, semaphore, failures)
                for tile in tiles
            ]
            results = await asyncio.gather(*tasks)
            state_elements = [el for grp in results for el in grp]

            for el in state_elements:
                eid = el.get("id")
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    all_elements.append(el)

            fc = osm.osm_elements_to_featurecollection(all_elements)
            osm.write_cache("vegetation", fc)
            logger.info(
                "State %s done in %.1fs (+%d elements, total %d features saved to %s)",
                state,
                time.time() - t0,
                len(state_elements),
                len(fc["features"]),
                settings.vegetation_cache_file,
            )

    logger.info("Incremental fetch complete! Total features: %d", len(fc["features"]))


if __name__ == "__main__":
    asyncio.run(main())
