"""Spatial join: label each FIRMS fire point by proximity to OSM zones + power plants.

Pure Shapely logic — no model training. Each fire gets a rule-based
`fire_type_rule` field:
- "industrial"    — within 1 km of an OSM industrial-zone polygon OR a known
                    power plant (WRI Global Power Plant Database point); checked
                    first (industrial is the higher-value signal for this problem)
- "mining"        — not industrial, but within 1 km of an OSM mining polygon
                    (landuse=quarry / man_made=mineshaft)
- "forest"        — not industrial or mining, but within 3 km of a
                    forest/vegetation polygon
- "other_natural" — not near any of the above (e.g. agricultural burning, grassland)

The industrial match source is recorded per fire (`industrial_match_source`):
"osm", "power_plant_db", or "both" — so the dashboard can show that we fuse a
curated infrastructure database with open map data, not OSM alone.

Backward-compatible fields are kept: `near_industrial` + `distance_m` (meters to
the NO NEAREST industrial feature — OSM polygon or power plant, whichever is
closer; None when nothing is within 20 km), plus `near_vegetation` +
`vegetation_distance_m`, and `near_power_plant` + `power_plant_distance_m`.

Distances are measured with a per-fire UTM transverse-Mercator projection, so
meter values stay meaningful regardless of longitude (raw lon/lat degrees are
not linear in meters).

All geometries are (lon, lat) order, matching both FIRMS GeoJSON and our OSM
conversion.
"""

import logging
import time
from functools import lru_cache

import pyproj
from shapely.errors import GEOSException
from shapely.geometry import shape
from shapely.ops import transform
from shapely.strtree import STRtree

from ..config import settings

logger = logging.getLogger(__name__)

NEAR_BUFFER_METERS = settings.near_buffer_meters
VEGETATION_BUFFER_METERS = settings.vegetation_buffer_meters
FLARE_BUFFER_METERS = settings.flare_buffer_meters
MINING_BUFFER_METERS = settings.mining_buffer_meters
SEARCH_RADIUS_METERS = settings.search_radius_meters
_SEARCH_PAD_DEGREES = 0.3


def _utm_epsg_for_longitude(lon: float) -> int:
    """Pick a UTM zone EPSG code (northern hemisphere) for a longitude."""
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone


@lru_cache(maxsize=16)
def _transformer_for_zone(zone: int):
    """Return a cached EPSG:4326 → UTM transformer for a longitude.

    Creating a pyproj Transformer is surprisingly expensive (~10s of µs) and
    doing it once per candidate polygon was eating minutes on the 150k-polygon
    vegetation layer. Caching per UTM zone cuts that to a handful of objects.
    """
    return pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{zone}", always_xy=True)


def _utm_distance_meters(point, polygon: "object") -> float:
    """Project both geometries to the point's UTM zone and measure meters."""
    zone = _utm_epsg_for_longitude(point.x)
    transformer = _transformer_for_zone(zone)
    projected_point = transform(transformer.transform, point)
    projected_polygon = transform(transformer.transform, polygon)
    return float(projected_polygon.distance(projected_point))


def _nearest_distance_meters(point, polygons: list, tree: STRtree | None = None) -> float | None:
    """Nearest distance (meters) from point to any polygon, or None if too far.

    `tree` may be prebuilt and reused across many points — building the STRtree
    once per annotate pass instead of per fire is a large speedup (the
    vegetation set is ~150k polygons). Uses STRtree.query_nearest (a native
    GEOS nearest-neighbor lookup) to pull back a single candidate instead of
    buffering the point and measuring every polygon within the pad — measuring
    every candidate re-projected in UTM was minutes per pass.
    """
    if not polygons:
        return None
    if tree is None:
        tree = STRtree(polygons)
    candidates_index = tree.query_nearest(
        point, max_distance=_SEARCH_PAD_DEGREES, all_matches=False
    )
    if len(candidates_index) == 0:
        return None
    nearest = _utm_distance_meters(point, polygons[int(candidates_index[0])])
    return nearest if nearest <= SEARCH_RADIUS_METERS else None


def _nearest_power_plant(
    point,
    features: list,
    points: list | None = None,
    tree: STRtree | None = None,
) -> tuple[float | None, dict | None]:
    """Nearest power-plant point (meters) + its properties, or (None, None)."""
    if not features:
        return None, None
    if points is None:
        points = [shape(feature["geometry"]) for feature in features]
    if tree is None:
        tree = STRtree(points)
    candidates_index = tree.query_nearest(
        point, max_distance=_SEARCH_PAD_DEGREES, all_matches=False
    )
    if len(candidates_index) == 0:
        return None, None
    i = int(candidates_index[0])
    distance_m = _utm_distance_meters(point, points[i])
    if distance_m > SEARCH_RADIUS_METERS:
        return None, None
    return round(distance_m, 1), features[i]["properties"] or {}


def _nearest_flare(
    point,
    features: list,
    points: list | None = None,
    tree: STRtree | None = None,
) -> tuple[float | None, dict | None]:
    """Nearest known gas-flare site (VNF catalog) in meters + its properties."""
    if not features:
        return None, None
    if points is None:
        points = [shape(feature["geometry"]) for feature in features]
    if tree is None:
        tree = STRtree(points)
    candidates_index = tree.query_nearest(
        point, max_distance=_SEARCH_PAD_DEGREES, all_matches=False
    )
    if len(candidates_index) == 0:
        return None, None
    i = int(candidates_index[0])
    distance_m = _utm_distance_meters(point, points[i])
    if distance_m > SEARCH_RADIUS_METERS:
        return None, None
    return round(distance_m, 1), features[i]["properties"] or {}


def _nearest_mining(
    point,
    features: list,
    polygons: list | None = None,
    tree: STRtree | None = None,
) -> tuple[float | None, dict | None]:
    """Nearest mining polygon (meters) + its properties, or (None, None)."""
    if not features:
        return None, None
    if polygons is None:
        polygons = [shape(feature["geometry"]) for feature in features]
    if tree is None:
        tree = STRtree(polygons)
    candidates_index = tree.query_nearest(
        point, max_distance=_SEARCH_PAD_DEGREES, all_matches=False
    )
    if len(candidates_index) == 0:
        return None, None
    i = int(candidates_index[0])
    distance_m = _utm_distance_meters(point, polygons[i])
    if distance_m > SEARCH_RADIUS_METERS:
        return None, None
    return round(distance_m, 1), features[i]["properties"] or {}


def nearest_zone_info(
    point, industrial_fc: dict, search_meters: int = SEARCH_RADIUS_METERS
) -> dict | None:
    """Nearest industrial-zone feature name, osm id, and distance.

    Returns None when no industrial polygon is within `search_meters`. Used by
    the clustering service to name recurring-fire "sites" after the industrial
    zone they sit next to.
    """
    features = industrial_fc.get("features", [])
    if not features:
        return None
    polygons, tree = _get_spatial_index(industrial_fc)
    if not polygons or tree is None:
        return None
    candidates_index = tree.query_nearest(
        point, max_distance=_SEARCH_PAD_DEGREES, all_matches=False
    )
    if len(candidates_index) == 0:
        return None
    i = int(candidates_index[0])
    poly = polygons[i]
    feature = features[i]
    distance_m = _utm_distance_meters(point, poly)
    if distance_m > search_meters:
        return None
    props = feature["properties"] or {}
    return {
        "name": props.get("name"),
        "osm_id": props.get("osm_id"),
        "distance_m": round(distance_m, 1),
    }


_SPATIAL_INDEX_CACHE: dict[int, tuple[list, STRtree | None]] = {}


def _get_spatial_index(fc: dict | None) -> tuple[list, STRtree | None]:
    if not fc or not fc.get("features"):
        return [], None
    key = id(fc)
    cached = _SPATIAL_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    polygons = [shape(feature["geometry"]) for feature in fc["features"]]
    tree = STRtree(polygons) if polygons else None
    _SPATIAL_INDEX_CACHE[key] = (polygons, tree)
    return polygons, tree


def _extract_point(feature: dict):
    """Return the feature's Point geometry, or None if it is unusable."""
    try:
        point = shape(feature["geometry"])
    except (GEOSException, TypeError, ValueError):
        logger.warning("skipping malformed fire feature %s", feature.get("id"))
        return None
    if point.is_empty or point.geom_type != "Point":
        return None
    return point


# Parsed+indexed reference layers, keyed by the caller's `cache_version`
# fingerprint (main.py derives it from the reference-layer cache-file
# size+mtime). Without a fingerprint (e.g. tests passing small inline layers)
# nothing is cached, so two different FeatureCollections with the same length
# can never accidentally share geometry. Rebuilding with `shape()` over the
# ~150k vegetation polygons costs ~20-30 s, so a live demo must reuse it.
_REFERENCE_CACHE: dict[str, dict] = {}


def _build_reference_bundle(
    industrial_fc: dict,
    vegetation_fc: dict,
    power_plants_fc: dict,
    flares_fc: dict,
    mining_fc: dict,
) -> dict:
    """Parse the reference layers into shapely geometry + STRtree."""
    industrial_polygons = [shape(feature["geometry"]) for feature in industrial_fc["features"]]
    vegetation_polygons = [shape(feature["geometry"]) for feature in vegetation_fc["features"]]
    power_plant_points = [shape(feature["geometry"]) for feature in power_plants_fc["features"]]
    flare_points = [shape(feature["geometry"]) for feature in flares_fc["features"]]
    mining_features = mining_fc["features"]
    mining_polygons = [shape(feature["geometry"]) for feature in mining_features]
    return {
        "industrial_polygons": industrial_polygons,
        "vegetation_polygons": vegetation_polygons,
        "power_plant_features": power_plants_fc["features"],
        "power_plant_points": power_plant_points,
        "flare_features": flares_fc["features"],
        "flare_points": flare_points,
        "mining_features": mining_features,
        "mining_polygons": mining_polygons,
        "industrial_tree": STRtree(industrial_polygons),
        "vegetation_tree": STRtree(vegetation_polygons),
        "power_plant_tree": STRtree(power_plant_points) if power_plant_points else None,
        "flare_tree": STRtree(flare_points) if flare_points else None,
        "mining_tree": STRtree(mining_polygons) if mining_polygons else None,
    }


def annotate_fires(
    fires_fc: dict,
    industrial_fc: dict,
    vegetation_fc: dict | None = None,
    power_plants_fc: dict | None = None,
    flares_fc: dict | None = None,
    mining_fc: dict | None = None,
    cache_version: str | None = None,
) -> dict:
    """Add rule-based fire type + proximity flags to every fire feature.

    Mutates and returns the input FeatureCollection (fires_fc). Classification
    priority: industrial first (1 km buffer around OSM industrial polygons OR
    power-plant points), then mining (1 km buffer around OSM mining polygons),
    then forest (3 km vegetation buffer — OSM forest/wood polygons are
    conservative, so a wider radius is a fairer "near natural vegetation"
    definition), else other_natural. `distance_m` is the distance to whichever
    industrial feature (OSM polygon or power plant) is nearest.

    An industrial fire also within `FLARE_BUFFER_METERS` (2 km) of a known
    VIIRS Nightfire gas-flare site gets `gas_flare` set on its properties — a
    sub-label within industrial (evidence from the free VNF-derived annual flare
    catalog), never a competing class.

    `cache_version` (optional) is a caller-held fingerprint identifying the
    reference layers. Pass it from the live endpoints (main.py computes it from
    the reference-layer cache files) so the parsed geometry + STRtree are built
    once and reused until the cache files actually change.
    """
    vegetation_fc = vegetation_fc or {"type": "FeatureCollection", "features": []}
    power_plants_fc = power_plants_fc or {"type": "FeatureCollection", "features": []}
    flares_fc = flares_fc or {"type": "FeatureCollection", "features": []}
    mining_fc = mining_fc or {"type": "FeatureCollection", "features": []}
    power_plant_points, power_plant_tree = _get_spatial_index(power_plants_fc)

    if cache_version is not None:
        bundle = _REFERENCE_CACHE.get(cache_version)
        if bundle is None:
            logger.info(
                "spatial.annotate_fires: building reference index for %s (%.0fs)",
                cache_version,
                len(industrial_fc.get("features", [])),
            )
            bundle = _build_reference_bundle(
                industrial_fc, vegetation_fc, power_plants_fc, flares_fc, mining_fc
            )
            _REFERENCE_CACHE.clear()  # only one live dataset in memory at a time
            _REFERENCE_CACHE[cache_version] = bundle
        else:
            logger.info(
                "spatial.annotate_fires: reusing cached reference index (%s)",
                cache_version,
            )
        industrial_polygons = bundle["industrial_polygons"]
        vegetation_polygons = bundle["vegetation_polygons"]
        power_plant_points = bundle["power_plant_points"]
        flare_features = bundle["flare_features"]
        flare_points = bundle["flare_points"]
        mining_features = bundle["mining_features"]
        mining_polygons = bundle["mining_polygons"]
        industrial_tree = bundle["industrial_tree"]
        vegetation_tree = bundle["vegetation_tree"]
        power_plant_tree = bundle["power_plant_tree"]
        flare_tree = bundle["flare_tree"]
        mining_tree = bundle["mining_tree"]
    else:
        industrial_polygons = [shape(feature["geometry"]) for feature in industrial_fc["features"]]
        vegetation_polygons = [shape(feature["geometry"]) for feature in vegetation_fc["features"]]
        power_plant_points = [shape(feature["geometry"]) for feature in power_plants_fc["features"]]
        flare_features = flares_fc["features"]
        flare_points = [shape(feature["geometry"]) for feature in flare_features]
        mining_features = mining_fc["features"]
        mining_polygons = [shape(feature["geometry"]) for feature in mining_features]
        industrial_tree = STRtree(industrial_polygons)
        vegetation_tree = STRtree(vegetation_polygons)
        power_plant_tree = STRtree(power_plant_points) if power_plant_points else None
        flare_tree = STRtree(flare_points) if flare_points else None
        mining_tree = STRtree(mining_polygons) if mining_polygons else None

    _t0 = time.perf_counter()
    logger.info(
        "spatial.annotate_fires: START (%d fires; %d industrial, %d vegetation, %d power plants, %d flares, %d mining)",
        len(fires_fc.get("features", [])),
        len(industrial_polygons),
        len(vegetation_polygons),
        len(power_plant_points),
        len(flare_points),
        len(mining_polygons),
    )

    for i, feature in enumerate(fires_fc["features"]):
        prop = feature["properties"]
        prop["near_industrial"] = False
        prop["distance_m"] = None
        prop["near_vegetation"] = False
        prop["vegetation_distance_m"] = None
        prop["near_power_plant"] = False
        prop["power_plant_distance_m"] = None
        prop["industrial_match_source"] = None
        prop["fire_type_rule"] = "other_natural"
        prop["gas_flare"] = False
        prop["distance_to_flare"] = None
        prop["flare_site_name"] = None
        prop["near_mining"] = False
        prop["distance_to_mining"] = None
        prop["mining_site_name"] = None

        point = _extract_point(feature)
        if point is None:
            continue

        industrial_distance = _nearest_distance_meters(point, industrial_polygons, industrial_tree)
        power_plant_distance, power_plant_props = _nearest_power_plant(
            point,
            power_plants_fc["features"],
            power_plant_points,
            power_plant_tree,
        )

        if industrial_distance is not None:
            prop["distance_m"] = round(industrial_distance, 1)
        if power_plant_distance is not None and (
            industrial_distance is None or power_plant_distance < prop["distance_m"]
        ):
            prop["distance_m"] = round(power_plant_distance, 1)

        near_osm_industrial = (
            industrial_distance is not None and industrial_distance <= NEAR_BUFFER_METERS
        )
        near_power_plant = (
            power_plant_distance is not None and power_plant_distance <= NEAR_BUFFER_METERS
        )
        prop["near_power_plant"] = near_power_plant
        prop["near_industrial"] = near_osm_industrial or near_power_plant
        if power_plant_distance is not None:
            prop["power_plant_distance_m"] = round(power_plant_distance, 1)
            if power_plant_props:
                prop["power_plant_name"] = power_plant_props.get("name")
        prop["industrial_match_source"] = (
            "both"
            if near_osm_industrial and near_power_plant
            else "osm"
            if near_osm_industrial
            else "power_plant_db"
            if near_power_plant
            else None
        )

        mining_distance, mining_props = _nearest_mining(
            point, mining_features, mining_polygons, mining_tree
        )
        if mining_distance is not None:
            prop["distance_to_mining"] = mining_distance
            if mining_distance <= MINING_BUFFER_METERS:
                prop["near_mining"] = True
                if mining_props:
                    prop["mining_site_name"] = mining_props.get("name") or (
                        "quarry" if mining_props.get("landuse") == "quarry" else "mine"
                    )
                    prop["mining_zone_type"] = (
                        "quarry" if mining_props.get("landuse") == "quarry" else "mine"
                    )

        if prop["near_industrial"]:
            prop["fire_type_rule"] = "industrial"
        elif prop["near_mining"]:
            prop["fire_type_rule"] = "mining"

        vegetation_distance = _nearest_distance_meters(point, vegetation_polygons, vegetation_tree)
        if vegetation_distance is not None:
            prop["vegetation_distance_m"] = round(vegetation_distance, 1)
            if vegetation_distance <= VEGETATION_BUFFER_METERS:
                prop["near_vegetation"] = True
                if prop["fire_type_rule"] == "other_natural":
                    prop["fire_type_rule"] = "forest"

        if prop["fire_type_rule"] == "industrial":
            flare_distance, flare_props = _nearest_flare(
                point, flare_features, flare_points, flare_tree
            )
            if flare_distance is not None:
                prop["distance_to_flare"] = flare_distance
                if flare_distance <= FLARE_BUFFER_METERS:
                    prop["gas_flare"] = True
                    if flare_props:
                        prop["flare_site_name"] = flare_props.get("name")

    logger.info(
        "spatial.annotate_fires: END (%d fires processed in %.2fs)",
        len(fires_fc.get("features", [])),
        time.perf_counter() - _t0,
    )
    return fires_fc
