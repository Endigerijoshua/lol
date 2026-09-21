"""Human-readable classification summaries + explanations.

Adds a plain-language `summary` to every fire alongside the machine fields so
judges and viewers understand a classification without reading raw numbers.

Fields added to each fire's properties:
- `summary_headline` — short label ("Industrial Fire", "Forest Fire", …)
- `summary_detail` — the distance/context phrase ("≈340 m from a known
  industrial zone", "deep within a vegetation zone", …)
- `summary_persistent` — only when `persistent_thermal_source`; a recurrence note
- `summary` — headline + detail + (persistent note), the string used on the UI
- `summary_icon` — emoji hint for the frontend
- `summary_risk` — only when `directional_risk`; a Directional Risk Indicator
  line ("…Estimated spread direction (heuristic)…")
- `explanation` — one-line "why we think this" (the rule that fired)
"""

import logging
import time

from ..config import settings
from .risk import cardinal
from .spatial import (
    MINING_BUFFER_METERS,
    NEAR_BUFFER_METERS,
    VEGETATION_BUFFER_METERS,
)

logger = logging.getLogger(__name__)

NEAR_KM = NEAR_BUFFER_METERS / 1000  # industrial: 1 km
VEGETATION_KM = VEGETATION_BUFFER_METERS / 1000  # forest: 3 km
MINING_KM = MINING_BUFFER_METERS / 1000  # mining: 1 km

HEADLINES = {
    "industrial": "Industrial Fire",
    "mining": "Mining/Quarry Fire",
    "forest": "Forest Fire",
    "other_natural": "Crop/Agricultural Burning",
}

ICONS = {
    "industrial": "\U0001f3ed",  # factory
    "mining": "\u26cf\ufe0f",  # pickaxe
    "forest": "\U0001f332",  # evergreen tree
    "other_natural": "\U0001f33e",  # ear of rice
}


def _format_distance(meters: float | None) -> str:
    """ "≈340 m" or "≈1.2 km"; empty when unknown."""
    if meters is None:
        return ""
    if meters >= 1000:
        return f"\u2248{meters / 1000:.1f} km"
    return f"\u2248{round(meters)} m"


def _industrial_detail(props: dict) -> str:
    """Distance/source phrase for the industrial headline."""
    plant_name = props.get("power_plant_name")
    source = props.get("industrial_match_source")
    d = props.get("distance_m")
    if d is not None and d < 10:
        base = "burning inside a mapped industrial facility"
    else:
        base = (
            f"{_format_distance(d)} from a known industrial zone"
            if d
            else "near a known industrial zone"
        )
    if source == "power_plant_db" and plant_name:
        return f"\u2248{_format_distance(props.get('power_plant_distance_m'))} from the {plant_name} power plant"
    if source == "both":
        suffix = f" and the {plant_name} power plant" if plant_name else " and a mapped power plant"
        return base + suffix
    return base


def _mining_detail(props: dict) -> str:
    """Distance/context phrase for the mining headline."""
    name = props.get("mining_site_name")
    zone_type = props.get("mining_zone_type") or "mining"
    d = props.get("distance_to_mining")
    if d is not None and d < 10:
        return "burning inside a mapped quarry/mine"
    where = f" the {name} {zone_type}" if name else f" a known {zone_type}"
    return f"{_format_distance(d) or 'near'} from{where}"


def _explanation(props: dict) -> str:
    """One line saying why the rule picked this class."""
    rule = props.get("fire_type_rule")
    source = props.get("industrial_match_source")
    plant_name = props.get("power_plant_name")
    if props.get("gas_flare"):
        flare_name = props.get("flare_site_name")
        flare_hint = f" {flare_name}" if flare_name else ""
        return (
            f"Classified as industrial because it sits within {NEAR_KM:.0f} km of "
            "an industrial/plant match, and it is also within 2 km of a known "
            f"gas-flare site{flare_hint} from NASA VIIRS Nightfire (2024) \u2014 "
            "consistent with a flare."
        )
    if rule == "industrial":
        if source == "power_plant_db":
            what = (
                f"the {plant_name} power plant (WRI)" if plant_name else "a mapped WRI power plant"
            )
        elif source == "both":
            what = (
                f"both a mapped industrial zone and the {plant_name} power plant"
                if plant_name
                else "both a mapped industrial zone and a power plant"
            )
        else:
            what = "a mapped OSM industrial zone"
        return (
            f"Classified as industrial because it sits within {NEAR_KM:.0f} km of {what}, "
            "confirmed by a live NASA FIRMS hotspot."
        )
    if rule == "mining":
        name = props.get("mining_site_name")
        zone_type = props.get("mining_zone_type") or "mining"
        location = f" the {name} {zone_type}" if name else f" a mapped OSM {zone_type}"
        return (
            f"Classified as mining because it is not near industry but sits within "
            f"{MINING_KM:.0f} km of{location} (OSM quarry/mine shaft)."
        )
    if rule == "forest":
        return (
            f"Classified as forest because it is not near industry but sits within "
            f"{VEGETATION_KM:.0f} km of mapped natural vegetation (OSM forest/scrub)."
        )
    return (
        "No industrial zone or mapped vegetation is within range, so this is most likely "
        "crop/agricultural burning or an isolated natural event."
    )


def _persistent_note(props: dict, rule: str) -> str:
    occurrences = props.get("occurrence_count") or 0
    lookback = settings.persistence_lookback_days
    if rule == "industrial":
        tail = "an ongoing industrial source rather than a one-time incident"
    elif rule == "mining":
        tail = "ongoing heat at a mining/quarry site rather than a one-off event"
    else:
        tail = "a persistent burning area rather than a one-off event"
    return (
        f"Recurring \u2014 detected {occurrences} times in the past {lookback} days, "
        f"suggesting {tail}."
    )


def _fmt_metric(value, suffix: str = "") -> str:
    """Compact number formatting that tolerates None values."""
    if value is None:
        return "\u2014"
    return f"{value:g}{suffix}"


def _direction_note(props: dict) -> str | None:
    """Plain-language directional-risk line (only when a cone was computed).

    The exact labels "Directional Risk Indicator" and "Estimated spread
    direction (heuristic)" are mandated by the product brief — never reword
    them into "prediction" / "predicted path".
    """
    risk = props.get("directional_risk")
    if not risk:
        return None
    weather = risk.get("weather") or {}
    detail = (
        f"wind {_fmt_metric(weather.get('wind_speed_kmh'))} km/h from "
        f"{_fmt_metric(weather.get('wind_direction_deg'), '°')}"
    )
    if _fmt_metric(weather.get("temperature_c")) != "\u2014":
        detail += f", {_fmt_metric(weather.get('temperature_c'), '°C')}"
    if _fmt_metric(weather.get("humidity_pct")) != "\u2014":
        detail += f", {_fmt_metric(weather.get('humidity_pct'), '%')} humidity"
    return (
        f"Directional Risk Indicator — Estimated spread direction "
        f"(heuristic): toward {cardinal(risk.get('spread_direction_deg'))}; "
        f"{detail}. Not a validated fire-behavior model."
    )


def add_summary(fires_fc: dict) -> dict:
    """Add the plain-language `summary`, `explanation` and helpers to every fire.

    Mutates and returns the input FeatureCollection.
    """
    _t0 = time.perf_counter()
    logger.info("summary.add_summary: START (%d fires)", len(fires_fc.get("features", [])))
    for feature in fires_fc["features"]:
        prop = feature["properties"]
        rule = prop.get("fire_type_rule")
        if rule not in HEADLINES:
            rule = "other_natural"
            prop["fire_type_rule"] = rule

        headline = HEADLINES[rule]
        detail = (
            _industrial_detail(prop)
            if rule == "industrial"
            else _mining_detail(prop)
            if rule == "mining"
            else "deep within a vegetation zone"
            if rule == "forest" and (prop.get("vegetation_distance_m") or 0) < 500
            else f"{_format_distance(prop.get('vegetation_distance_m'))} from the nearest vegetation"
            if rule == "forest"
            else "no nearby industrial or forest activity"
        )
        if prop.get("gas_flare"):
            detail += " \u00b7 near a known gas-flare site (NASA VIIRS Nightfire)"

        persistent = ""
        if prop.get("persistent_thermal_source"):
            persistent = _persistent_note(prop, rule)

        unregistered = ""
        if prop.get("unregistered_persistent"):
            unregistered = "Unregistered \u2014 no known facility match (WRI power-plant database)."

        prop["summary_headline"] = headline
        prop["summary_detail"] = detail
        prop["summary_icon"] = ICONS[rule]
        prop["summary_unregistered"] = unregistered or None
        prop["summary_risk"] = _direction_note(prop)
        prop["summary"] = (
            f"{headline} \u2014 {detail}"
            + (f" {persistent}" if persistent else "")
            + (f" {unregistered}" if unregistered else "")
        )
        prop["summary_persistent"] = persistent or None
        prop["explanation"] = _explanation(prop)
    logger.info(
        "summary.add_summary: END (%d fires in %.2fs)",
        len(fires_fc.get("features", [])),
        time.perf_counter() - _t0,
    )
    return fires_fc
