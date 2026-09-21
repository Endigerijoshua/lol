"""Weak-label ML classifier for fire type.

Loads a RandomForestClassifier trained offline by scripts/train_classifier.py
(the "weak labels" come from our own rule-based spatial join, so we never train
on hand-labelled data) and predicts one of three classes:

- "industrial"     — near an OSM industrial zone
- "mining"         — near an OSM mining zone (quarry / mine shaft)
- "forest"         — near OSM vegetation/forest polygons
- "other_natural"  — neither

The classifier is wrapped in sklearn's CalibratedClassifierCV (isotonic,
cv=5), so `predict_proba` returns calibrated probabilities rather than raw
vote fractions. The model file is generated at build/train time and
git-ignored; predict() raises a clear error when it is missing. Feature
encoding lives here too so the training script and the live endpoint use
exactly the same numbers.
"""

import logging
import time
from functools import lru_cache

import joblib
import numpy as np

from ..config import repo_path
from . import spatial

logger = logging.getLogger(__name__)

VALID_CLASSES = ("industrial", "mining", "forest", "other_natural")

# Order of columns passed to the model. MUST match scripts/train_classifier.py.
FEATURE_COLUMNS = (
    "confidence",
    "bright_ti4",
    "frp",
    "daynight",
    "hour",
    "distance_to_industrial",
    "distance_to_mining",
    "distance_to_vegetation",
    "occurrence_count",
)

MODEL_FILE = repo_path("models/fire_classifier.pkl")

_CONFIDENCE_ENCODING = {"l": 0, "n": 1, "h": 2, "low": 0, "nominal": 1, "high": 2}
_DAYNIGHT_ENCODING = {"N": 0, "D": 1}
_NONE_DISTANCE_FILL = float(spatial.SEARCH_RADIUS_METERS)


class ModelNotAvailable(RuntimeError):
    """Raised when the trained classifier file has not been built yet."""


def _expect(number) -> float:
    """Coerce a possibly-null scalar into a float, filling missing values."""
    if number is None:
        return 0.0
    try:
        value = float(number)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _hour_from_acq_time(acq_time) -> float:
    """FIRMS acq_time is HHMM (e.g. 2330). Extract the hour component."""
    if acq_time is None:
        return 0.0
    try:
        return float(int(acq_time) // 100)
    except (TypeError, ValueError):
        return 0.0


def features_from_props(properties: dict) -> np.ndarray:
    """Encode a fire feature's properties into the model's feature values."""
    confidence = _CONFIDENCE_ENCODING.get(properties.get("confidence"), 1)
    daynight = _DAYNIGHT_ENCODING.get(properties.get("daynight"), 1)
    distance_to_industrial = properties.get("distance_m")
    if distance_to_industrial is None:
        distance_to_industrial = _NONE_DISTANCE_FILL
    distance_to_mining = properties.get("distance_to_mining")
    if distance_to_mining is None:
        distance_to_mining = _NONE_DISTANCE_FILL
    distance_to_vegetation = properties.get("vegetation_distance_m")
    if distance_to_vegetation is None:
        distance_to_vegetation = _NONE_DISTANCE_FILL

    return np.asarray(
        [
            float(confidence),
            _expect(properties.get("bright_ti4")),
            _expect(properties.get("frp")),
            float(daynight),
            _hour_from_acq_time(properties.get("acq_time")),
            float(distance_to_industrial),
            float(distance_to_mining),
            float(distance_to_vegetation),
            _expect(properties.get("occurrence_count")),
        ],
        dtype=float,
    )


@lru_cache(maxsize=1)
def load_model():
    """Load the trained classifier once per process, cached afterwards."""
    if not MODEL_FILE.exists():
        raise ModelNotAvailable(
            f"trained model not found at {MODEL_FILE}. Run scripts/train_classifier.py first."
        )
    model = joblib.load(MODEL_FILE)
    logger.info("loaded fire classifier from %s", MODEL_FILE)
    return model


def predict(features) -> str:
    """Predict fire type for one sample.

    `features` is either a properties dict (encoded automatically) or an
    array-like with exactly `len(FEATURE_COLUMNS)` values.
    """
    model = load_model()
    if isinstance(features, dict):
        sample = features_from_props(features).reshape(1, -1)
    else:
        sample = np.asarray(features, dtype=float).reshape(1, -1)
    prediction = model.predict(sample)[0]
    return str(prediction)


def predict_from_props(properties: dict) -> str:
    """Convenience wrapper: predict fire type straight from GeoJSON properties."""
    return predict(features_from_props(properties))


def predict_proba_from_props(properties: dict) -> np.ndarray:
    """Calibrated class probabilities (one value per VALID_CLASSES entry)."""
    model = load_model()
    return model.predict_proba(features_from_props(properties).reshape(1, -1))[0]


def annotate_fire_type_ml(features_fc: dict) -> dict:
    """Add `fire_type_ml` and `fire_type_ml_confidence` to every feature.

    `fire_type_ml_confidence` is the max calibrated probability from
    CalibratedClassifierCV, so it is a trustworthy confidence rather than a raw
    vote fraction. Mutates and returns the input FeatureCollection. Missing
    model file is logged and handled gracefully so the rule-based dashboard
    still renders.
    """
    try:
        model = load_model()
    except ModelNotAvailable as exc:
        logger.warning("%s", exc)
        for feature in features_fc["features"]:
            feature["properties"]["fire_type_ml"] = None
            feature["properties"]["fire_type_ml_confidence"] = None
        return features_fc

    _t0 = time.perf_counter()
    logger.info(
        "ml.annotate_fire_type_ml: START (%d fires)",
        len(features_fc.get("features", [])),
    )
    features = features_fc["features"]
    if not features:
        logger.info("ml.annotate_fire_type_ml: no fires to classify")
        return features_fc
    samples = np.vstack([features_from_props(feature["properties"]) for feature in features])
    probas = model.predict_proba(samples)
    classes = np.asarray(model.classes_)
    for feature, probs in zip(features_fc["features"], probas):
        properties = feature["properties"]
        properties["fire_type_ml"] = str(classes[int(np.argmax(probs))])
        properties["fire_type_ml_confidence"] = float(probs.max())
    logger.info(
        "ml.annotate_fire_type_ml: END (%d fires in %.2fs)",
        len(features_fc.get("features", [])),
        time.perf_counter() - _t0,
    )
    return features_fc
