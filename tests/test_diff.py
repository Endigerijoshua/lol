"""Tests for the multi-day / refresh diff view (`is_new_since_last_refresh`).

A detection is "new since last refresh" when its (lat, lon, acq_date, acq_time,
satellite) key is NOT already stored in fire_history before the current fetch
is recorded. This powers the dashboard's "what changed since the last fetch"
view without any frontend bookkeeping.
"""

import datetime as dt

import pytest

from app import db
from app.services import persistence


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "test_diff.db"
    monkeypatch.setattr(db, "db_path", lambda: str(db_file))
    return db_file


def _feature(lat, lon, acq_date, acq_time=1234, satellite="T"):
    return {
        "type": "Feature",
        "id": 0,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "acq_date": acq_date,
            "acq_time": acq_time,
            "satellite": satellite,
            "confidence": "n",
        },
    }


def _fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


def _marked(fc):
    return [f["properties"]["is_new_since_last_refresh"] for f in fc["features"]]


def test_first_fetch_marks_everything_new(temp_db):
    day = db.today().isoformat()
    fc = _fc(_feature(22.5, 72.5, day), _feature(23.5, 73.5, day))
    db.annotate_new_since_last_refresh(fc)
    assert _marked(fc) == [True, True]


def test_same_key_after_record_is_not_new(temp_db):
    day = db.today().isoformat()
    fc = _fc(_feature(22.5, 72.5, day))
    db.annotate_new_since_last_refresh(fc)
    assert _marked(fc) == [True]
    db.record_featurecollection(fc)
    db.annotate_new_since_last_refresh(fc)
    assert _marked(fc) == [False]


def test_new_time_or_satellite_counts_as_new(temp_db):
    day = db.today().isoformat()
    fc = _fc(_feature(22.5, 72.5, day, acq_time=1234))
    db.annotate_new_since_last_refresh(fc)
    assert _marked(fc) == [True]
    db.record_featurecollection(fc)
    new_time = _fc(_feature(22.5, 72.5, day, acq_time=1235))
    db.annotate_new_since_last_refresh(new_time)
    assert _marked(new_time) == [True]
    new_sat = _fc(_feature(22.5, 72.5, day, satellite="N"))
    db.annotate_new_since_last_refresh(new_sat)
    assert _marked(new_sat) == [True]


def test_new_location_among_known_is_new(temp_db):
    day = db.today().isoformat()
    known = _fc(_feature(22.5, 72.5, day))
    db.annotate_new_since_last_refresh(known)
    db.record_featurecollection(known)
    mixed = _fc(_feature(22.5, 72.5, day), _feature(25.5, 80.5, day))
    db.annotate_new_since_last_refresh(mixed)
    assert _marked(mixed) == [False, True]


def test_marked_false_when_coordinates_missing():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 0,
                "geometry": {"type": "Point", "coordinates": [None, None]},
                "properties": {"acq_date": "2026-09-09"},
            }
        ],
    }
    db.annotate_new_since_last_refresh(fc)
    assert fc["features"][0]["properties"]["is_new_since_last_refresh"] is False


def test_annotate_called_before_record_keeps_refresh_semantics(temp_db):
    """The common main.py flow: annotate -> record. Second refresh is empty."""
    day = db.today().isoformat()
    first = _fc(_feature(21.0, 71.0, day))
    db.annotate_new_since_last_refresh(first)
    db.record_featurecollection(first)
    second = _fc(_feature(21.0, 71.0, day))
    db.annotate_new_since_last_refresh(second)
    db.record_featurecollection(second)
    assert _marked(second) == [False]
    assert db.record_featurecollection(_fc()) == 0


def test_is_new_since_yesterday_combines_with_persistence(temp_db):
    """Both diff fields survive a full persistence annotation pass."""
    yesterday = (db.today() - dt.timedelta(days=1)).isoformat()
    _record = db.record_featurecollection
    _record(
        _fc(
            _feature(22.5, 72.5, yesterday),
            _feature(22.5, 72.5, (db.today() - dt.timedelta(days=8)).isoformat()),
        )
    )
    today = db.today().isoformat()
    fc = _fc(_feature(22.5, 72.5, today))
    db.annotate_new_since_last_refresh(fc)
    db.record_featurecollection(fc)
    persistence.annotate_persistence(fc)
    props = fc["features"][0]["properties"]
    assert props["persistent_thermal_source"] is True
    assert props["is_new_since_yesterday"] is True
    assert props["is_new_since_last_refresh"] is True