import math

import pytest

from app.services import ml

SAMPLE_PROPS = {
    "confidence": "n",
    "bright_ti4": 310.0,
    "frp": 5.0,
    "daynight": "D",
    "acq_time": 1430,
}


@pytest.mark.skipif(not ml.MODEL_FILE.exists(), reason="models/fire_classifier.pkl not trained yet")
def test_predict_from_props_returns_valid_class():
    prediction = ml.predict(SAMPLE_PROPS)
    assert prediction in ml.VALID_CLASSES


@pytest.mark.skipif(not ml.MODEL_FILE.exists(), reason="models/fire_classifier.pkl not trained yet")
def test_predict_near_industrial_is_industrial():
    props = dict(SAMPLE_PROPS, distance_m=250.0, near_industrial=True)
    assert ml.predict(props) == "industrial"


def test_features_from_props_has_expected_shape():
    vector = ml.features_from_props(SAMPLE_PROPS)
    assert len(vector) == len(ml.FEATURE_COLUMNS)
    assert vector.dtype == float


def test_features_missing_values_filled():
    vector = ml.features_from_props({"confidence": None, "acq_time": None})
    for value in vector:
        assert not math.isnan(value)  # no NaN leaks into the model input
    assert vector[0] == 1.0  # unknown confidence maps to nominal


def test_annotate_fire_type_ml_graceful_without_model():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [72.0, 22.0]},
                "properties": dict(SAMPLE_PROPS),
            }
        ],
    }
    ml.annotate_fire_type_ml(fc)
    prop = fc["features"][0]["properties"]
    assert "fire_type_ml" in prop
    if ml.MODEL_FILE.exists():
        assert prop["fire_type_ml"] in ml.VALID_CLASSES
    else:
        assert prop["fire_type_ml"] is None
