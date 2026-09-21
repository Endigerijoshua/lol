"""Train a weak-label RandomForestClassifier for four-way fire classification.

Weak labels: for every historical fire in `fire_history`, the fire type is derived
from our existing rule-based spatial join (spatial.annotate_fires), giving us
"industrial" / "mining" / "forest" / "other_natural" ground truth without any hand labelling.

Model: out-of-the-box scikit-learn RandomForestClassifier (n_estimators=100)
wrapped in CalibratedClassifierCV (isotonic, cv=5) so `predict_proba` outputs are
properly calibrated probabilities rather than raw vote fractions. No custom tuning,
no deep learning, no GPU — consistent with the project's rules.

Output: models/fire_classifier.pkl (a generated artifact, git-ignored).

Backfill: the script ALWAYS fetches FIRMS before training (regardless of how many
rows `fire_history` already holds), so each re-run folds in any newer detections and
the sample keeps growing. `--fetch-days` defaults to 5 because the FIRMS area API
hard-caps the day range at [1..5] (verified: days=10/20/30 all return HTTP 400
"Invalid day range. Expects [1..5]"). Variety therefore comes from the growing
multi-day history plus the multi-state OSM coverage, not from a longer single fetch.

Guardrail: if any class makes up under `--min-class-fraction` (default 5%) of the
training set, the script stops WITHOUT writing a model — a model trained on such
lopsided data would predict one class almost always and look broken.

Run (from repo root):
    python scripts/train_classifier.py
"""

import argparse
import asyncio
import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db
from app.services import (
    firms,
    ml,
    osm,
    persistence,
    powerplants,
    spatial,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BACKFILL_DAYS = 5  # FIRMS area API max; verified [1..5] (10/20/30 -> HTTP 400)

MODEL_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "fire_classifier.pkl"


def load_history_as_featurecollection() -> dict:
    """Read every row of fire_history into a FIRMS-style GeoJSON FeatureCollection."""
    features = []
    with sqlite3.connect(db.db_path()) as conn:
        rows = conn.execute(
            "SELECT latitude, longitude, acq_date, acq_time, confidence, "
            "bright_ti4, frp, satellite, daynight FROM fire_history"
        ).fetchall()
    for i, (
        lat,
        lon,
        acq_date,
        acq_time,
        confidence,
        bright_ti4,
        frp,
        satellite,
        daynight,
    ) in enumerate(rows):
        features.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "acq_date": acq_date,
                    "acq_time": acq_time,
                    "confidence": confidence,
                    "bright_ti4": bright_ti4,
                    "frp": frp,
                    "satellite": satellite,
                    "daynight": daynight,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_training_set(fires_fc: dict) -> tuple[list, list]:
    """Return (X, y): raw feature vectors + rule-based fire_type labels."""
    X, y = [], []
    for feature in fires_fc["features"]:
        props = feature["properties"]
        label = props.get("fire_type_rule")
        if label is None:
            continue
        X.append(list(ml.features_from_props(props)))
        y.append(label)
    return X, y


def report_class_balance(y: list[str]) -> None:
    counts = Counter(y)
    total = len(y)
    print("\n=== Class balance (rule-based weak labels) ===")
    for label in spatial_fire_classes_order():
        count = counts.get(label, 0)
        print(f"  {label:<15} {count:>5}  ({count / total * 100:.1f}%)")


def spatial_fire_classes_order() -> tuple[str, ...]:
    return ("industrial", "mining", "forest", "other_natural")


def compare_calibration(
    X_test,
    y_test,
    baseline_model,
    calibrated_model,
) -> None:
    """Held-out sanity check: raw vote fractions vs isotonic-calibrated probs.

    Prints accuracy, mean max-probability (confidence), multiclass Brier score and
    log loss for both models so we can see calibration actually improving the
    probability outputs without hurting test accuracy.
    """
    print("\n=== Calibration check: raw RF vs isotonic-calibrated (held-out) ===")
    rows = [("model", "acc", "mean_conf", "brier", "log_loss")]
    for name, model in (
        ("raw RF (vote fraction)", baseline_model),
        ("isotonic calibrated", calibrated_model),
    ):
        proba = model.predict_proba(X_test)
        classes = list(model.classes_)
        y_index = np.array([classes.index(label) for label in y_test])
        one_hot = np.zeros_like(proba)
        one_hot[np.arange(len(y_test)), y_index] = 1.0
        brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
        rows.append(
            (
                name,
                f"{accuracy_score(y_test, model.predict(X_test)):.4f}",
                f"{float(proba.max(axis=1).mean()):.4f}",
                f"{brier:.4f}",
                f"{log_loss(y_test, proba, labels=classes):.4f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-days", type=int, default=BACKFILL_DAYS)
    parser.add_argument("--min-class-fraction", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    fires_fc = load_history_as_featurecollection()
    print(f"fire_history rows loaded: {len(fires_fc['features'])}")

    print(
        f"backfilling FIRMS for the last {args.fetch_days} day(s) regardless of "
        "row count (area API caps the window at [1..5])."
    )
    fetched = asyncio.run(firms.fetch_fires(days=args.fetch_days))
    new_rows = db.record_featurecollection(fetched)
    print(f"new rows recorded from backfill: {new_rows}")
    fires_fc = load_history_as_featurecollection()
    print(f"fire_history rows after backfill: {len(fires_fc['features'])}")

    industrial_fc = asyncio.run(osm.get_industrial_zones())
    vegetation_fc = asyncio.run(osm.get_vegetation_zones())
    power_plants_fc = asyncio.run(powerplants.get_power_plants())
    mining_fc = asyncio.run(osm.get_mining_zones())
    spatial.annotate_fires(
        fires_fc,
        industrial_fc,
        vegetation_fc,
        power_plants_fc,
        flares_fc=None,
        mining_fc=mining_fc,
    )
    persistence.annotate_persistence(fires_fc)

    X, y = build_training_set(fires_fc)
    print(f"usable training examples: {len(X)}")
    if len(X) == 0:
        print("ERROR: no rows produced a rule-based label. Aborting.")
        return 2

    report_class_balance(y)
    counts = Counter(y)
    lowest_fraction = min(counts.values()) / len(y)
    if lowest_fraction < args.min_class_fraction:
        print(
            "\nFATAL: class balance is severely skewed — '"
            + lowest_class(counts)
            + "' has only "
            + f"{lowest_fraction * 100:.1f}% of examples (min allowed "
            + f"{args.min_class_fraction * 100:.0f}%)."
        )
        print(
            "NOT writing a model: it would predict the majority class almost "
            "always and look broken. Grow the dataset first (e.g. fetch more "
            "days), then re-run."
        )
        return 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print("\n=== Training RandomForestClassifier (n_estimators=100) ===")
    baseline = RandomForestClassifier(n_estimators=100, random_state=args.random_state).fit(
        X_train, y_train
    )

    print("=== ... wrapped in CalibratedClassifierCV (isotonic, cv=5) ===")
    model = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=100, random_state=args.random_state),
        method="isotonic",
        cv=5,
    )
    model.fit(X_train, y_train)

    compare_calibration(X_test, y_test, baseline, model)

    print("\n=== Classification report (test split) ===")
    print(classification_report(y_test, model.predict(X_test)))

    print("=== Feature importances (first calibration fold) ===")
    for name, importance in sorted(
        zip(
            ml.FEATURE_COLUMNS,
            model.calibrated_classifiers_[0].estimator.feature_importances_,
        ),
        key=lambda pair: pair[1],
        reverse=True,
    ):
        print(f"  {name:<24} {importance:.3f}")

    print(f"\n=== Saving model to {MODEL_OUTPUT} ===")
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    print(f"Saved. Expected classes: {list(model.classes_)}")
    return 0


def lowest_class(counts: Counter) -> str:
    return min(counts, key=counts.get)


if __name__ == "__main__":
    sys.exit(main())
