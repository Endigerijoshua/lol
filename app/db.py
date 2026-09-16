"""SQLite storage for fire detection history.

Each detection returned by the FIRMS endpoints is appended to the `fire_history`
table so recurrence (persistent thermal source) checks have days of data to match
against. Pure stdlib `sqlite3` — no ORM, no setup.

The table is de-duplicated by (latitude, longitude, acq_date, acq_time, satellite),
so calling the endpoints repeatedly (e.g. with different lookback windows) does not
inflate occurrence counts.
"""

import datetime as dt
import logging
import math
import sqlite3
from contextlib import contextmanager

from .config import repo_path, settings

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS fire_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    acq_date TEXT NOT NULL,
    acq_time INTEGER,
    confidence TEXT,
    bright_ti4 REAL,
    frp REAL,
    satellite TEXT,
    daynight TEXT,
    recorded_at TEXT NOT NULL,
    UNIQUE (latitude, longitude, acq_date, acq_time, satellite)
);
CREATE INDEX IF NOT EXISTS idx_fire_history_location
    ON fire_history (latitude, longitude, acq_date);
"""


def db_path() -> str:
    return str(repo_path(settings.fire_history_db))


def today() -> dt.date:
    """UTC date, used as the reference point for persistence lookback windows."""
    return dt.datetime.now(dt.timezone.utc).date()


@contextmanager
def get_connection():
    conn = sqlite3.connect(db_path())
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_featurecollection(fc: dict) -> int:
    """Insert every FIRMS feature in `fc` into fire_history.

    Returns the number of new rows actually inserted (duplicates ignored).
    """
    rows = []
    for feature in fc.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords or not props.get("acq_date"):
            continue
        lon, lat = coords[0], coords[1]
        if lon is None or lat is None:
            continue
        rows.append(
            (
                float(lat),
                float(lon),
                str(props["acq_date"]),
                props.get("acq_time"),
                props.get("confidence"),
                props.get("bright_ti4"),
                props.get("frp"),
                props.get("satellite"),
                props.get("daynight"),
                dt.datetime.now(dt.timezone.utc).isoformat(),
            )
        )
    return _insert_many(rows)


def _feature_key(
    lat: float, lon: float, acq_date: str, acq_time, satellite
) -> tuple:
    """Dedup key mirroring the fire_history UNIQUE constraint."""
    return (float(lat), float(lon), str(acq_date), acq_time, satellite)


def existing_feature_keys(fc: dict) -> set[tuple]:
    """Subset of `fc`'s dedup keys that are already present in fire_history.

    Used to mark "new since last refresh": a detection is new when its key is
    NOT already stored before this fetch is recorded.
    """
    want = []
    for feature in fc.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates")
        if coords and props.get("acq_date"):
            lon, lat = coords[0], coords[1]
            if lon is None or lat is None:
                continue
            want.append(
                _feature_key(
                    lat, lon, props["acq_date"], props.get("acq_time"),
                    props.get("satellite"),
                )
            )
    if not want:
        return set()
    found: set[tuple] = set()
    with get_connection() as conn:
        for lat, lon, acq_date, acq_time, satellite in want:
            row = conn.execute(
                """
                SELECT 1 FROM fire_history
                WHERE latitude = ? AND longitude = ?
                  AND acq_date = ? AND acq_time IS ? AND satellite IS ?
                """,
                (lat, lon, acq_date, acq_time, satellite),
            ).fetchone()
            if row:
                found.add((lat, lon, acq_date, acq_time, satellite))
    return found


def annotate_new_since_last_refresh(fc: dict) -> dict:
    """Set `is_new_since_last_refresh` on every feature before it is recorded.

    A detection is "new" when it is not already present in fire_history — i.e.
    it appeared since the previous refresh/fetch of the feed. Call this BEFORE
    `record_featurecollection` so the comparison is against the prior snapshot.
    Mutates and returns the input FeatureCollection.
    """
    existing = existing_feature_keys(fc)
    for feature in fc.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates")
        is_new = False
        if coords and props.get("acq_date"):
            lon, lat = coords[0], coords[1]
            if lon is not None and lat is not None:
                key = _feature_key(
                    lat, lon, props["acq_date"], props.get("acq_time"),
                    props.get("satellite"),
                )
                is_new = key not in existing
        props["is_new_since_last_refresh"] = is_new
    return fc


def _insert_many(rows: list[tuple]) -> int:
    if not rows:
        return 0
    with get_connection() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO fire_history (
                latitude, longitude, acq_date, acq_time, confidence,
                bright_ti4, frp, satellite, daynight, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    inserted = cur.rowcount
    if inserted:
        logger.info("stored %s new detection(s) in fire_history", inserted)
    return inserted


def distinct_days_near(
    lat: float, lon: float, radius_m: float, lookback_days: int
) -> list[str]:
    """Distinct acq_dates of past detections within `radius_m` of (lat, lon).

    A coarse degree-based bounding box filters in SQL, then the exact
    haversine distance is checked in Python so results are in real meters.
    `lookback_days` counts back from today (inclusive).
    """
    pad_m = radius_m / 111320.0
    lon_pad = pad_m / max(math.cos(math.radians(lat)), 0.2)
    cutoff = (today() - dt.timedelta(days=lookback_days - 1)).isoformat()

    days: set[str] = set()
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT DISTINCT acq_date, latitude, longitude
            FROM fire_history
            WHERE latitude BETWEEN ? AND ?
              AND longitude BETWEEN ? AND ?
              AND acq_date >= ?
            """,
            (lat - pad_m, lat + pad_m, lon - lon_pad, lon + lon_pad, cutoff),
        )
        for acq_date, row_lat, row_lon in cur.fetchall():
            if _haversine_meters(lat, lon, row_lat, row_lon) <= radius_m:
                days.add(acq_date)
    return sorted(days)


def daily_counts(days: int = 14) -> list[dict]:
    """Detection counts per UTC day for the last `days` days.

    Returns a dense, forward-filled list of ``{"date": "YYYY-MM-DD", "count": n}``
    entries (zero-filled for days with no detections) ordered oldest → newest, so
    the frontend can render a bar/line chart without filling gaps itself.
    """
    days = max(1, min(int(days), 30))
    cutoff = (today() - dt.timedelta(days=days - 1)).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT acq_date, COUNT(*) AS n
            FROM fire_history
            WHERE acq_date >= ?
            GROUP BY acq_date
            """,
            (cutoff,),
        )
        counts = dict(cur.fetchall())
    daily = []
    for i in range(days):
        d = (today() - dt.timedelta(days=days - 1 - i)).isoformat()
        daily.append({"date": d, "count": int(counts.get(d, 0))})
    return daily


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in meters."""
    radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))
