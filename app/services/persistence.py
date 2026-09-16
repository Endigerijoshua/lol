"""Persistence detection: recurring hotspots = "persistent thermal sources".

A fire is a persistent thermal source when the same location has been detected
on 2+ separate days (within `persistence_radius_m` meters) in the last
`persistence_lookback_days` days. Purely classical lookup against the SQLite
history — no model training.

Annotation added to every fire feature's properties:
- `persistent_thermal_source` (bool)
- `occurrence_count` (int) — number of distinct days seen in the window
- `unregistered_persistent` (bool) — persistent AND no matching named facility
  in the WRI power-plant database (the highest-value alert: a recurring
  industrial-looking heat source with no accounted-for owner)
- `is_new_since_yesterday` (bool) — acquired today (new since the previous day)
"""

import datetime as dt
import logging

from shapely.geometry import shape

from .. import db
from ..config import settings

logger = logging.getLogger(__name__)


def occurrence_count(
    lat: float,
    lon: float,
    radius_m: int | None = None,
    lookback_days: int | None = None,
) -> int:
    """Number of distinct past days with a detection within `radius_m` of a point."""
    radius_m = radius_m or settings.persistence_radius_m
    lookback_days = lookback_days or settings.persistence_lookback_days
    return len(db.distinct_days_near(lat, lon, radius_m, lookback_days))


def is_persistent(count: int, min_occurrences: int | None = None) -> bool:
    min_occurrences = min_occurrences or settings.persistence_min_occurrences
    return count >= min_occurrences


def _is_new_since_yesterday(props: dict) -> bool:
    """True when the detection's acquisition date is today (UTC).

    "New since the previous day" — lets the dashboard show just what changed.
    Returns False for missing / non-ISO dates.
    """
    acq_date = props.get("acq_date")
    if not acq_date:
        return False
    try:
        return dt.date.fromisoformat(str(acq_date)) >= db.today()
    except ValueError:
        return False


def _unregistered_persistent(props: dict) -> bool:
    """True when a fire is persistent but matches NO named WRI facility.

    A fire is "registered" only when the spatial join matched it to a named power
    plant (`near_power_plant` + `power_plant_name`). Persistent + no such match =
    an unregistered persistent source. OSM landuse polygons carry no owner, so
    they do not count as a "named facility".
    """
    if not props.get("persistent_thermal_source"):
        return False
    return not (
        props.get("near_power_plant") and props.get("power_plant_name")
    )


def annotate_persistence(features_fc: dict) -> dict:
    """Add `persistent_thermal_source` and `occurrence_count` to every feature.

    Also annotates `unregistered_persistent` (persistence × spatial facility
    match) and `is_new_since_yesterday`. Mutates and returns the input
    FeatureCollection.
    """
    for feature in features_fc["features"]:
        prop = feature["properties"]
        prop["unregistered_persistent"] = False
        prop["is_new_since_yesterday"] = _is_new_since_yesterday(prop)
        try:
            point = shape(feature["geometry"])
        except (ValueError, TypeError):
            prop["persistent_thermal_source"] = False
            prop["occurrence_count"] = 0
            continue
        if point.is_empty or point.geom_type != "Point":
            prop["persistent_thermal_source"] = False
            prop["occurrence_count"] = 0
            continue
        count = occurrence_count(point.y, point.x)
        prop["occurrence_count"] = count
        prop["persistent_thermal_source"] = is_persistent(count)
        prop["unregistered_persistent"] = _unregistered_persistent(prop)
    return features_fc
