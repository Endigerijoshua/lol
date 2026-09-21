"""DBSCAN site grouping — turns repeated persistent hotspots into named sites.

Applied as-is from scikit-learn, NEVER trained (per project rules): we just feed
the coordinates of fires already flagged `persistent_thermal_source=true` into
the off-the-shelf DBSCAN algorithm so a recurring blaze shows as ONE named site
instead of many sidebar entries.

Distances are real ground meters: sklearn's `haversine` metric expects (lat,
lon) in radians, and eps is scaled by the Earth radius — so `eps_m=500` means
~500 m on the ground, no degree/UTM juggling per point.
"""

import logging

import numpy as np
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

from . import spatial

logger = logging.getLogger(__name__)

EARTH_RADIUS_METERS = 6371000.0


def _site_name(centroid: Point, industrial_fc: dict) -> tuple[str, dict | None]:
    """Name a site after its nearest industrial zone, if any is close enough."""
    zone_info = spatial.nearest_zone_info(centroid, industrial_fc)
    if zone_info is None:
        return f"Site at {centroid.y:.2f}, {centroid.x:.2f}", None
    name = zone_info["name"] or "industrial zone"
    distance_part = f" ({zone_info['distance_m'] / 1000:.1f} km)" if zone_info["name"] else ""
    label = name if zone_info["name"] else "industrial zone"
    return f"Site near {label}{distance_part}", zone_info


def cluster_persistent_fires(
    features_fc: dict,
    industrial_fc: dict,
    eps_m: int | None = None,
    min_samples: int | None = None,
) -> dict:
    """DBSCAN-cluster persistent_thermal_source features into site features.

    Returns a FeatureCollection of site centroids with site metadata, plus a
    `meta` object reporting persisted/recurrence counts:
    - `persistent_count`: recurring detections fed to DBSCAN
    - `unclustered`: recurrences with no neighbor within `eps` (kept out of sites)
    Non-persistent features are ignored entirely.
    """
    from ..config import settings

    eps_m = eps_m or settings.clustering_eps_m
    min_samples = min_samples or settings.clustering_min_samples

    persistent = [
        f for f in features_fc["features"] if f["properties"].get("persistent_thermal_source")
    ]

    meta = {"persistent_count": len(persistent), "unclustered": 0}
    if len(persistent) < min_samples:
        return {"type": "FeatureCollection", "features": [], "meta": meta}

    radians = np.radians(
        [[f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]] for f in persistent]
    )
    labels = DBSCAN(
        eps=eps_m / EARTH_RADIUS_METERS,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(radians)

    site_features = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        members = [f for f, lab in zip(persistent, labels) if lab == label]
        lon = sum(f["geometry"]["coordinates"][0] for f in members) / len(members)
        lat = sum(f["geometry"]["coordinates"][1] for f in members) / len(members)
        centroid = Point(lon, lat)
        site_name, zone_info = _site_name(centroid, industrial_fc)
        site_features.append(
            {
                "type": "Feature",
                "id": f"site-{len(site_features)}",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "site_name": site_name,
                    "member_count": len(members),
                    "total_occurrences": sum(
                        m["properties"].get("occurrence_count") or 0 for m in members
                    ),
                    "max_occurrence_count": max(
                        m["properties"].get("occurrence_count") or 0 for m in members
                    ),
                    "near_industrial_zone": (zone_info["name"] if zone_info else None),
                    "distance_m": zone_info["distance_m"] if zone_info else None,
                    "member_fire_ids": [m.get("id") for m in members],
                },
            }
        )

    meta["unclustered"] = sum(1 for label in labels if label == -1)
    logger.info(
        "DBSCAN: %s persistent point(s) -> %s site(s), %s unclustered",
        meta["persistent_count"],
        len(site_features),
        meta["unclustered"],
    )
    return {"type": "FeatureCollection", "features": site_features, "meta": meta}
