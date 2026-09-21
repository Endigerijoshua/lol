import json
import logging
import os
import random
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from app.services import ml, persistence, spatial, summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VERSION = "repro|1:1:1"


def distributions(fires):
    from collections import Counter

    counts = Counter()
    for f in fires["features"]:
        p = f["properties"]
        counts[p.get("fire_type_rule")] = counts.get(p.get("fire_type_rule", "?"), 0) + 1
        p.setdefault("fire_type_ml", None)
    ml_counts = Counter(f["properties"].get("fire_type_ml") for f in fires["features"])
    return dict(counts), dict(ml_counts)


def load_fc(name):
    path = os.path.join(ROOT, f"{name}_zones_cache.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("feature_collection", data)


def build_fake_fires(n):
    """Synthetically scatter n fires across India to stress the join."""
    fires = {"type": "FeatureCollection", "features": []}
    rng = random.Random(42)
    for i in range(n):
        lon = rng.uniform(68, 97)
        lat = rng.uniform(6, 37)
        fires["features"].append(
            {
                "type": "Feature",
                "id": i,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "confidence": rng.choice(["l", "n", "h"]),
                    "bright_ti4": rng.uniform(290, 340),
                    "frp": rng.uniform(0, 50),
                    "acq_date": "2026-09-05",
                    "acq_time": f"{rng.randint(0, 23):02d}{rng.randint(0, 59):02d}",
                    "daynight": rng.choice(["D", "N"]),
                    "satellite": "NPP",
                },
            }
        )
    return fires


def main():
    print("loading reference layers from disk cache...")
    industrial_fc = load_fc("industrial")
    vegetation_fc = load_fc("vegetation")
    with open(os.path.join(ROOT, "power_plants_cache.json"), encoding="utf-8") as f:
        power_plants_fc = json.load(f).get("feature_collection")
    print(
        f"industrial: {len(industrial_fc['features'])}, "
        f"vegetation: {len(vegetation_fc['features'])}, "
        f"powerplants: {len(power_plants_fc['features'])}"
    )

    n_fires = 1097
    fires = build_fake_fires(n_fires)
    print(f"generated {n_fires} synthetic fires")

    t = time.perf_counter()
    print("spatial.annotate_fires (cold: builds reference index)")
    spatial.annotate_fires(
        fires, industrial_fc, vegetation_fc, power_plants_fc, cache_version=VERSION
    )
    print(f"  spatial cold done in {time.perf_counter() - t:.2f}s")
    rule_dist, _ = distributions(fires)
    print(f"  fire_type_rule dist: {rule_dist}")

    t = time.perf_counter()
    print("spatial.annotate_fires (warm: reuses cached reference index)")
    spatial.annotate_fires(
        fires, industrial_fc, vegetation_fc, power_plants_fc, cache_version=VERSION
    )
    print(f"  spatial warm done in {time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    print("persistence.annotate_persistence")
    persistence.annotate_persistence(fires)
    print(f"  persistence done in {time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    print("ml.annotate_fire_type_ml")
    ml.annotate_fire_type_ml(fires)
    print(f"  ml done in {time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    print("summary.add_summary")
    summary.add_summary(fires)
    print(f"  summary done in {time.perf_counter() - t:.2f}s")

    print("ALL STAGES COMPLETED")


if __name__ == "__main__":
    main()
