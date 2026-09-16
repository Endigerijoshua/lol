"""Tests for SQLite fire_history accumulation and persistence detection."""

import datetime as dt

import pytest

from app import db
from app.services import persistence


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "test_history.db"
    monkeypatch.setattr(db, "db_path", lambda: str(db_file))
    return db_file


def _feature(lat, lon, acq_date, satellite="T"):
    return {
        "type": "Feature",
        "id": 0,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "acq_date": acq_date,
            "acq_time": 1234,
            "satellite": satellite,
            "confidence": "n",
        },
    }


def _record(fc):
    return db.record_featurecollection({"type": "FeatureCollection", "features": fc})


def _days_ago(*offsets):
    return [(db.today() - dt.timedelta(days=d)).isoformat() for d in offsets]


def test_haversine_known_distance():
    meters = db._haversine_meters(22.0, 72.0, 23.0, 72.0)
    assert 110_000 < meters < 112_000


def test_records_accumulate_across_calls_and_dedupe(temp_db):
    day = db.today().isoformat()
    first = _record([_feature(22.5, 72.5, day)])
    second = _record([_feature(22.5, 72.5, day)])
    assert first == 1
    assert second == 0
    with db.get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM fire_history").fetchone()[0]
    assert count == 1


def test_two_days_within_300m_reads_as_2_occurrences(temp_db):
    lat, lon = 22.5, 72.5
    _record([_feature(lat + 0.002, lon, d) for d in _days_ago(1, 5)])
    count = persistence.occurrence_count(lat, lon)
    assert count == 2
    assert persistence.is_persistent(count) is True


def test_annotate_sets_persistence_fields_for_recurring_source(temp_db):
    lat, lon = 20.0, 76.0
    _record([_feature(lat + 0.001, lon, d) for d in _days_ago(1, 8)])
    fc = {
        "type": "FeatureCollection",
        "features": [_feature(lat, lon, db.today().isoformat())],
    }
    persistence.annotate_persistence(fc)
    props = fc["features"][0]["properties"]
    assert props["persistent_thermal_source"] is True
    assert props["occurrence_count"] == 2


def test_single_day_or_too_far_is_not_persistent(temp_db):
    lat, lon = 21.0, 77.0
    today = db.today().isoformat()
    _record([_feature(lat, lon, today)])
    _record([_feature(lat + 0.005, lon, today)])
    fc = {
        "type": "FeatureCollection",
        "features": [_feature(lat, lon, today)],
    }
    persistence.annotate_persistence(fc)
    props = fc["features"][0]["properties"]
    assert props["persistent_thermal_source"] is False
    assert props["occurrence_count"] == 1


def test_occurrences_outside_14day_window_not_counted(temp_db):
    lat, lon = 23.0, 78.0
    _record([_feature(lat + 0.001, lon, d) for d in _days_ago(15, 16)])
    count = persistence.occurrence_count(lat, lon)
    assert count == 0


def _annotate(persistent, **override):
    props = {
        "persistent_thermal_source": persistent,
        "near_power_plant": False,
        "power_plant_name": None,
    }
    props.update(override)
    return props


def test_unregistered_persistent_when_no_facility_match():
    props = _annotate(persistent=True)
    assert persistence._unregistered_persistent(props) is True


def test_unregistered_persistent_false_when_named_plant_match():
    props = _annotate(
        persistent=True, near_power_plant=True, power_plant_name="Panipat"
    )
    assert persistence._unregistered_persistent(props) is False


def test_unregistered_false_when_not_persistent():
    props = _annotate(persistent=False)
    assert persistence._unregistered_persistent(props) is False


def test_persistent_unregistered_annotated_end_to_end(temp_db):
    lat, lon = 20.0, 76.0
    _record([_feature(lat + 0.001, lon, d) for d in _days_ago(1, 8)])
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 0,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "acq_date": db.today().isoformat(),
                    "acq_time": 1234,
                    "satellite": "T",
                    "confidence": "n",
                    "near_power_plant": False,
                    "power_plant_name": None,
                },
            }
        ],
    }
    persistence.annotate_persistence(fc)
    props = fc["features"][0]["properties"]
    assert props["persistent_thermal_source"] is True
    assert props["unregistered_persistent"] is True


def test_registered_persistent_annotated_end_to_end(temp_db):
    lat, lon = 20.0, 76.0
    _record([_feature(lat + 0.001, lon, d) for d in _days_ago(1, 8)])
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 0,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "acq_date": db.today().isoformat(),
                    "acq_time": 1234,
                    "satellite": "T",
                    "confidence": "n",
                    "near_power_plant": True,
                    "power_plant_name": "Panipat",
                },
            }
        ],
    }
    persistence.annotate_persistence(fc)
    props = fc["features"][0]["properties"]
    assert props["persistent_thermal_source"] is True
    assert props["unregistered_persistent"] is False


def test_is_new_since_yesterday_for_today_yesterday_and_garbage():
    assert persistence._is_new_since_yesterday(
        {"acq_date": db.today().isoformat()}
    ) is True
    old = (db.today() - dt.timedelta(days=1)).isoformat()
    assert persistence._is_new_since_yesterday({"acq_date": old}) is False
    assert persistence._is_new_since_yesterday({"acq_date": "not-a-date"}) is False
    assert persistence._is_new_since_yesterday({}) is False
