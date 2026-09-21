"""End-to-end wiring test: the heuristic directional-risk cone flow.

Proves `/api/flagged-fires` runs the full pipeline (FIRMS → boundaries →
spatial → persistence → directional risk → ML → summary) with a stubbed
weather provider, and that high-FRP fires come back with `directional_risk`
plus the mandated "Directional Risk Indicator" labeling in `summary_risk`.
"""

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app

WEATHER = {
    "temperature_c": 34.0,
    "humidity_pct": 25.0,
    "wind_speed_kmh": 40.0,
    "wind_direction_deg": 200.0,
}


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "risk_pipeline.db"
    monkeypatch.setattr(db, "db_path", lambda: str(db_file))
    return db_file


@pytest.fixture()
def live_flagged_pipeline(monkeypatch):
    """Stub FIRMS + reference layers + ML, keep the real risk/summary code."""
    from app.config import settings

    monkeypatch.setattr(settings, "firms_map_key", "test-key")

    fires = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 1,
                "geometry": {"type": "Point", "coordinates": [77.0, 22.0]},
                "properties": {
                    "confidence": "h",
                    "bright_ti4": 340.0,
                    "frp": 60.0,
                    "acq_date": "2026-09-16",
                    "acq_time": 600,
                    "satellite": "N",
                    "daynight": "D",
                },
            },
            {
                "type": "Feature",
                "id": 2,
                "geometry": {"type": "Point", "coordinates": [78.0, 23.0]},
                "properties": {
                    "confidence": "l",
                    "bright_ti4": 320.0,
                    "frp": 1.0,
                    "acq_date": "2026-09-16",
                    "acq_time": 700,
                    "satellite": "N",
                    "daynight": "D",
                },
            },
        ],
    }

    async def fake_fetch_fires(days=None):
        return fires

    async def fake_reference_layers():
        empty = {"type": "FeatureCollection", "features": []}
        return (empty, empty, empty, empty, empty)

    def fake_ml(fc):
        for feature in fc["features"]:
            feature["properties"]["fire_type_ml"] = "other_natural"
            feature["properties"]["fire_type_ml_confidence"] = 0.9
        return fc

    async def fake_weather_fetch(lat, lon):
        return dict(WEATHER)

    monkeypatch.setattr("app.main.firms.fetch_fires", fake_fetch_fires)
    monkeypatch.setattr("app.main._reference_layers", fake_reference_layers)
    monkeypatch.setattr("app.main.ml.annotate_fire_type_ml", fake_ml)
    monkeypatch.setattr("app.main.risk.weather.fetch_weather", fake_weather_fetch)


def test_flagged_fires_includes_directional_risk(live_flagged_pipeline, temp_db):
    client = TestClient(app)
    resp = client.get("/api/flagged-fires")
    assert resp.status_code == 200, resp.text
    features = {f["id"]: f["properties"] for f in resp.json()["features"]}

    high = features[1]
    calms = features[2]
    assert "directional_risk" in high
    assert high["directional_risk"]["method"] == "heuristic"
    assert high["directional_risk"]["spread_direction_deg"] == pytest.approx(20.0)
    assert high["directional_risk"]["geometry"]["type"] == "Polygon"
    # Low-FRP, non-persistent fire stays clean.
    assert "directional_risk" not in calms

    assert "summary_risk" in high
    assert "Directional Risk Indicator" in high["summary_risk"]
    assert "Estimated spread direction (heuristic)" in high["summary_risk"]


def test_flagged_fires_survives_weather_failure(live_flagged_pipeline, temp_db, monkeypatch):
    """Open-Meteo down → hotspot still returns, just without the overlay."""

    async def bad_weather_fetch(lat, lon):
        return None

    monkeypatch.setattr("app.main.risk.weather.fetch_weather", bad_weather_fetch)
    client = TestClient(app)
    resp = client.get("/api/flagged-fires")
    assert resp.status_code == 200, resp.text
    features = {f["id"]: f["properties"] for f in resp.json()["features"]}
    assert "directional_risk" not in features[1]
    assert features[1]["summary_headline"]  # the rest of the pipeline still ran
