from app.services import summary


def _props(**overrides):
    base = {
        "fire_type_rule": "other_natural",
        "distance_m": None,
        "near_industrial": False,
        "industrial_match_source": None,
        "near_power_plant": False,
        "power_plant_distance_m": None,
        "power_plant_name": None,
        "near_vegetation": False,
        "vegetation_distance_m": None,
        "persistent_thermal_source": False,
        "occurrence_count": 0,
    }
    base.update(overrides)
    return base


def _summarize(prop):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": prop,
            }
        ],
    }
    summary.add_summary(fc)
    return fc["features"][0]["properties"]


def test_industrial_fire_summary_and_explanation():
    prop = _summarize(
        _props(
            fire_type_rule="industrial",
            near_industrial=True,
            industrial_match_source="osm",
            distance_m=340,
        )
    )
    assert prop["summary_headline"] == "Industrial Fire"
    assert prop["summary_icon"] == "\U0001f3ed"
    assert "from a known industrial zone" in prop["summary"]
    assert "\u2248340 m" in prop["summary"]
    assert "within 1 km" in prop["explanation"]
    assert "industrial" in prop["explanation"]


def test_industrial_burning_inside_zone():
    prop = _summarize(_props(fire_type_rule="industrial", near_industrial=True, distance_m=0))
    assert "burning inside a mapped industrial facility" in prop["summary"]


def test_industrial_match_from_power_plant_names_it():
    prop = _summarize(
        _props(
            fire_type_rule="industrial",
            near_industrial=True,
            industrial_match_source="power_plant_db",
            distance_m=150,
            near_power_plant=True,
            power_plant_distance_m=150.0,
            power_plant_name="Panipat",
        )
    )
    assert "Panipat power plant" in prop["summary"]
    assert "Panipat power plant" in prop["explanation"]


def test_forest_deep_within():
    prop = _summarize(
        _props(
            fire_type_rule="forest",
            near_vegetation=True,
            vegetation_distance_m=40,
        )
    )
    assert prop["summary_headline"] == "Forest Fire"
    assert prop["summary_icon"] == "\U0001f332"
    assert "deep within a vegetation zone" in prop["summary"]
    assert "within 3 km" in prop["explanation"]


def test_forest_at_range_shows_distance():
    prop = _summarize(
        _props(fire_type_rule="forest", near_vegetation=True, vegetation_distance_m=2400)
    )
    assert "\u22482.4 km" in prop["summary"]
    assert "deep within" not in prop["summary"]


def test_other_natural_crop_burning():
    prop = _summarize(
        _props(fire_type_rule="other_natural", distance_m=None, vegetation_distance_m=None)
    )
    assert prop["summary_headline"] == "Crop/Agricultural Burning"
    assert prop["summary_icon"] == "\U0001f33e"
    assert "no nearby industrial or forest activity" in prop["summary"]
    assert "most likely crop/agricultural burning" in prop["explanation"]


def test_mining_fire_summary_and_explanation():
    prop = _summarize(
        _props(
            fire_type_rule="mining",
            near_mining=True,
            distance_to_mining=820,
            mining_site_name="Kothagudem",
            mining_zone_type="mine",
        )
    )
    assert prop["summary_headline"] == "Mining/Quarry Fire"
    assert prop["summary_icon"] == "\u26cf\ufe0f"
    assert "Kothagudem mine" in prop["summary"]
    assert "\u2248820 m" in prop["summary"]
    assert "within 1 km" in prop["explanation"]
    assert "quarry/mine shaft" in prop["explanation"]


def test_persistent_note_appended():
    prop = _summarize(
        _props(
            fire_type_rule="industrial",
            near_industrial=True,
            industrial_match_source="osm",
            distance_m=340,
            persistent_thermal_source=True,
            occurrence_count=4,
        )
    )
    assert "Recurring" in prop["summary"]
    assert "detected 4 times in the past 14 days" in prop["summary"]
    assert "ongoing industrial source" in prop["summary"]
    assert prop["summary_persistent"]


def test_gas_flare_sub_label_mentioned():
    prop = _summarize(
        _props(
            fire_type_rule="industrial",
            near_industrial=True,
            industrial_match_source="osm",
            distance_m=340,
            gas_flare=True,
            distance_to_flare=620,
            flare_site_name="IND_X_2024_72.5E_22.4N_v0.2",
        )
    )
    assert "gas-flare site" in prop["summary"]
    assert "VIIRS Nightfire" in prop["summary"]
    assert "VIIRS Nightfire" in prop["explanation"]


def test_unregistered_persistent_note_in_summary():
    prop = _summarize(
        _props(
            fire_type_rule="industrial",
            near_industrial=True,
            distance_m=340,
            persistent_thermal_source=True,
            occurrence_count=4,
            unregistered_persistent=True,
        )
    )
    assert prop["summary_unregistered"]
    assert "Unregistered" in prop["summary_unregistered"]
    assert "no known facility match" in prop["summary_unregistered"]
    assert "Unregistered" in prop["summary"]


def test_registered_persistent_has_no_unregistered_note():
    prop = _summarize(
        _props(
            fire_type_rule="industrial",
            near_industrial=True,
            distance_m=340,
            persistent_thermal_source=True,
            occurrence_count=3,
            unregistered_persistent=False,
            power_plant_name="Panipat",
        )
    )
    assert prop["summary_unregistered"] is None
    assert "Unregistered" not in prop["summary"]


def test_all_summary_fields_present():
    prop = _summarize(_props())
    for key in (
        "summary",
        "summary_headline",
        "summary_detail",
        "summary_icon",
        "summary_persistent",
        "summary_risk",
        "explanation",
    ):
        assert key in prop
    assert prop["summary_risk"] is None


def test_directional_risk_summary_uses_mandated_labels():
    prop = _summarize(
        _props(
            directional_risk={
                "spread_direction_deg": 180.0,
                "weather": {
                    "wind_speed_kmh": 24.0,
                    "wind_direction_deg": 0.0,
                    "temperature_c": 31.0,
                    "humidity_pct": 41.0,
                },
            }
        )
    )
    assert prop["summary_risk"]
    assert "Directional Risk Indicator" in prop["summary_risk"]
    assert "Estimated spread direction" in prop["summary_risk"]
    assert "heuristic" in prop["summary_risk"]
    assert "toward S" in prop["summary_risk"]
    assert "validated fire-behavior model" in prop["summary_risk"].lower()
    assert "prediction" not in prop["summary_risk"].lower()


def test_directional_risk_summary_tolerates_missing_weather():
    prop = _summarize(_props(directional_risk={"spread_direction_deg": None, "weather": {}}))
    assert prop["summary_risk"]
    assert "Directional Risk Indicator" in prop["summary_risk"]
