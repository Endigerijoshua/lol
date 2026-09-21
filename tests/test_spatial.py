from app.services import spatial

# (lon, lat) order — as FIRMS and our OSM conversion output.
PLANT_POLY = {
    "type": "Feature",
    "id": "osm-way-1",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [72.50, 22.40],
                [72.60, 22.40],
                [72.60, 22.50],
                [72.50, 22.50],
                [72.50, 22.40],
            ]
        ],
    },
    "properties": {"landuse": "industrial", "name": "Test Plant"},
}

# A fire just outside the plant polygon (~1.1 km south).
FIRE_INSIDE = {
    "type": "Feature",
    "id": 0,
    "geometry": {"type": "Point", "coordinates": [72.55, 22.45]},
    "properties": {},
}
FIRE_500M_AWAY = {
    "type": "Feature",
    "id": 1,
    "geometry": {"type": "Point", "coordinates": [72.55, 22.505]},
    "properties": {},
}
FIRE_1_3KM_AWAY = {
    "type": "Feature",
    "id": 2,
    "geometry": {"type": "Point", "coordinates": [72.55, 22.513]},
    "properties": {},
}
FIRE_FAR_AWAY = {
    "type": "Feature",
    "id": 3,
    "geometry": {"type": "Point", "coordinates": [93.0, 12.0]},
    "properties": {},
}

# Forest polygon overlapping the plant so a point inside both exists.
FOREST_POLY = {
    "type": "Feature",
    "id": "osm-way-2",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [72.15, 22.40],
                [72.75, 22.40],
                [72.75, 22.50],
                [72.15, 22.50],
                [72.15, 22.40],
            ]
        ],
    },
    "properties": {"landuse": "forest", "name": "Test Forest"},
}

# A fire inside the forest polygon, far from the plant.
FIRE_IN_FOREST = {
    "type": "Feature",
    "id": 4,
    "geometry": {"type": "Point", "coordinates": [72.15, 22.45]},
    "properties": {},
}

# A fire inside the forest polygon AND inside the plant polygon.
FIRE_IN_BOTH = {
    "type": "Feature",
    "id": 5,
    "geometry": {"type": "Point", "coordinates": [72.55, 22.49]},
    "properties": {},
}

FOREST_FC = {"type": "FeatureCollection", "features": [FOREST_POLY]}

# WRI-style power-plant points. One sits right next to the OSM polygon (so a
# fire there matches both sources); one sits far from any OSM polygon.
POWER_PLANT_NEAR = {
    "type": "Feature",
    "id": "wri-pp-1",
    "geometry": {"type": "Point", "coordinates": [72.55, 22.506]},
    "properties": {"name": "Halo Thermal", "source": "power_plant_db"},
}
POWER_PLANT_SOLO = {
    "type": "Feature",
    "id": "wri-pp-2",
    "geometry": {"type": "Point", "coordinates": [74.05, 15.0]},
    "properties": {"name": "Solo Thermal", "source": "power_plant_db"},
}

# A fire ~440 m south of the solo power plant, far from any OSM polygon.
FIRE_SOLO_NEAR_PLANT = {
    "type": "Feature",
    "id": 6,
    "geometry": {"type": "Point", "coordinates": [74.05, 15.004]},
    "properties": {},
}

# A mining polygon (landuse=quarry) far from the plant (~45 km east of the
# industrial/forest cluster) with a fire inside it.
MINING_POLY = {
    "type": "Feature",
    "id": "osm-way-m1",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [73.00, 22.40],
                [73.10, 22.40],
                [73.10, 22.50],
                [73.00, 22.50],
                [73.00, 22.40],
            ]
        ],
    },
    "properties": {"landuse": "quarry", "name": "Test Quarry"},
}
FIRE_NEAR_MINING = {
    "type": "Feature",
    "id": 7,
    "geometry": {"type": "Point", "coordinates": [73.05, 22.45]},
    "properties": {},
}
MINING_FC = {"type": "FeatureCollection", "features": [MINING_POLY]}

# A mining polygon overlapping the plant + forest cluster to test priority.
MINING_POLY_OVERLAP = {
    "type": "Feature",
    "id": "osm-way-m2",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [72.45, 22.35],
                [72.75, 22.35],
                [72.75, 22.55],
                [72.45, 22.55],
                [72.45, 22.35],
            ]
        ],
    },
    "properties": {"landuse": "quarry", "name": "Overlap Quarry"},
}
MINING_OVERLAP_FC = {"type": "FeatureCollection", "features": [MINING_POLY_OVERLAP]}


def _run(fires, vegetation_fc=None, power_plants_fc=None, flares_fc=None, mining_fc=None):
    fc = {"type": "FeatureCollection", "features": fires}
    zones = {"type": "FeatureCollection", "features": [PLANT_POLY]}
    return spatial.annotate_fires(fc, zones, vegetation_fc, power_plants_fc, flares_fc, mining_fc)[
        "features"
    ]


def test_fire_inside_zone_flagged():
    feats = _run([FIRE_INSIDE])
    assert feats[0]["properties"]["near_industrial"] is True
    assert feats[0]["properties"]["distance_m"] < 1000


def test_fire_within_1km_flagged():
    feats = _run([FIRE_500M_AWAY])
    prop = feats[0]["properties"]
    assert prop["near_industrial"] is True
    assert 400 <= prop["distance_m"] <= 1000


def test_fire_over_1km_not_flagged():
    feats = _run([FIRE_1_3KM_AWAY])
    prop = feats[0]["properties"]
    assert prop["near_industrial"] is False
    assert prop["distance_m"] > 1000


def test_fire_far_away_not_flagged():
    feats = _run([FIRE_FAR_AWAY])
    prop = feats[0]["properties"]
    assert prop["near_industrial"] is False
    assert prop["distance_m"] is None


def test_all_features_have_annotation_fields():
    feats = _run(
        [
            FIRE_INSIDE,
            FIRE_500M_AWAY,
            FIRE_1_3KM_AWAY,
            FIRE_FAR_AWAY,
            FIRE_SOLO_NEAR_PLANT,
        ]
    )
    for feat in feats:
        assert "near_industrial" in feat["properties"]
        assert "distance_m" in feat["properties"]
        assert "fire_type_rule" in feat["properties"]
        assert "near_vegetation" in feat["properties"]
        assert "vegetation_distance_m" in feat["properties"]
        assert "near_power_plant" in feat["properties"]
        assert "power_plant_distance_m" in feat["properties"]
        assert "industrial_match_source" in feat["properties"]
        assert "near_mining" in feat["properties"]
        assert "distance_to_mining" in feat["properties"]
        assert "mining_site_name" in feat["properties"]
    assert sum(f["properties"]["near_industrial"] for f in feats) == 2


def test_fire_near_power_plant_only_is_industrial():
    pp = {"type": "FeatureCollection", "features": [POWER_PLANT_SOLO]}
    feats = _run([FIRE_SOLO_NEAR_PLANT], power_plants_fc=pp)
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "industrial"
    assert prop["near_industrial"] is True
    assert prop["near_power_plant"] is True
    assert prop["industrial_match_source"] == "power_plant_db"
    assert prop["power_plant_distance_m"] <= 500
    assert prop["power_plant_name"] == "Solo Thermal"


def test_fire_near_power_plant_and_osm_marks_both():
    pp = {"type": "FeatureCollection", "features": [POWER_PLANT_NEAR]}
    feats = _run([FIRE_500M_AWAY], power_plants_fc=pp)
    prop = feats[0]["properties"]
    assert prop["near_industrial"] is True
    assert prop["industrial_match_source"] == "both"
    assert prop["near_power_plant"] is True
    assert prop["distance_m"] <= 300  # power plant is the nearer source


def test_power_plant_within_radar_but_over_1km_not_industrial():
    plant = {
        "type": "Feature",
        "id": "wri-pp-3",
        "geometry": {"type": "Point", "coordinates": [72.55, 22.463]},
        "properties": {"name": "Far Plant", "source": "power_plant_db"},
    }
    pp = {"type": "FeatureCollection", "features": [plant]}
    feats = _run([FIRE_INSIDE], power_plants_fc=pp)
    prop = feats[0]["properties"]
    assert prop["near_power_plant"] is False
    assert prop["power_plant_distance_m"] > 1000  # within 20 km search radius
    assert prop["industrial_match_source"] == "osm"


def test_fire_inside_industrial_zone_fire_type_rule_industrial():
    feats = _run([FIRE_INSIDE])
    assert feats[0]["properties"]["fire_type_rule"] == "industrial"
    assert feats[0]["properties"]["near_industrial"] is True


def test_fire_in_forest_only_is_forest():
    feats = _run([FIRE_IN_FOREST], FOREST_FC)
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "forest"
    assert prop["near_industrial"] is False
    assert prop["near_vegetation"] is True
    assert prop["vegetation_distance_m"] <= 1000


def test_fire_near_everything_stays_industrial():
    feats = _run([FIRE_IN_BOTH], FOREST_FC)
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "industrial"
    assert prop["near_industrial"] is True
    assert prop["near_vegetation"] is True


def test_fire_far_from_all_is_other_natural():
    feats = _run([FIRE_FAR_AWAY], FOREST_FC)
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "other_natural"
    assert prop["near_industrial"] is False
    assert prop["near_vegetation"] is False


# Known gas-flare site (VNF catalog) ~550 m north of the industrial polygon.
FLARE_SITE = {
    "type": "Feature",
    "id": "vnf-flare-1",
    "geometry": {"type": "Point", "coordinates": [72.55, 22.455]},
    "properties": {
        "name": "IND_X_2024_72.55E_22.455N_v0.2",
        "source": "viirs_nightfire",
    },
}
FLARE_FAR = {
    "type": "Feature",
    "id": "vnf-flare-2",
    "geometry": {"type": "Point", "coordinates": [74.1, 15.1]},
    "properties": {"name": "FAR_FLARE_15KM", "source": "viirs_nightfire"},
}


def test_industrial_fire_near_flare_site_gets_gas_flare_sub_label():
    flares_fc = {"type": "FeatureCollection", "features": [FLARE_SITE]}
    feats = _run([FIRE_INSIDE], flares_fc=flares_fc)
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "industrial"
    assert prop["gas_flare"] is True
    assert 0 <= prop["distance_to_flare"] <= 2000
    assert prop["flare_site_name"] == FLARE_SITE["properties"]["name"]


def test_industrial_fire_far_from_flare_not_gas_flare():
    flares_fc = {"type": "FeatureCollection", "features": [FLARE_FAR]}
    feats = _run(
        [FIRE_SOLO_NEAR_PLANT],
        power_plants_fc={"type": "FeatureCollection", "features": [POWER_PLANT_SOLO]},
        flares_fc=flares_fc,
    )
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "industrial"
    assert prop["gas_flare"] is False
    assert prop["distance_to_flare"] is not None  # measured but beyond 2 km


def test_non_industrial_fire_not_gas_flare():
    flares_fc = {"type": "FeatureCollection", "features": [FLARE_SITE]}
    feats = _run([FIRE_IN_FOREST], FOREST_FC, flares_fc=flares_fc)
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "forest"
    assert prop["gas_flare"] is False
    assert "distance_to_flare" in prop


def test_all_features_have_gas_flare_fields():
    feats = _run([FIRE_INSIDE, FIRE_FAR_AWAY])
    for feat in feats:
        props = feat["properties"]
        assert "gas_flare" in props
        assert "distance_to_flare" in props
        assert "flare_site_name" in props


# A fire in the forest AND the overlap quarry but outside the plant polygon.
FIRE_IN_FOREST_AND_MINING = {
    "type": "Feature",
    "id": 8,
    "geometry": {"type": "Point", "coordinates": [72.70, 22.45]},
    "properties": {},
}


def test_fire_near_mining_only_is_mining():
    feats = _run([FIRE_NEAR_MINING], mining_fc=MINING_FC)
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "mining"
    assert prop["near_industrial"] is False
    assert prop["near_mining"] is True
    assert prop["distance_to_mining"] <= 1000
    assert prop["mining_site_name"] == "Test Quarry"
    assert prop["mining_zone_type"] == "quarry"


def test_fire_far_from_mining_not_mining():
    feats = _run([FIRE_FAR_AWAY], mining_fc=MINING_FC)
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "other_natural"
    assert prop["near_mining"] is False
    assert prop["distance_to_mining"] is None


def test_fire_near_mining_and_industrial_stays_industrial():
    feats = _run([FIRE_IN_BOTH], FOREST_FC, mining_fc=MINING_OVERLAP_FC)
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "industrial"
    assert prop["near_industrial"] is True
    assert prop["near_mining"] is True  # recorded as proximity even when industrial wins


def test_fire_near_mining_and_forest_is_mining():
    feats = _run([FIRE_IN_FOREST_AND_MINING], FOREST_FC, mining_fc=MINING_OVERLAP_FC)
    prop = feats[0]["properties"]
    assert prop["fire_type_rule"] == "mining"  # mining beats forest when not industrial
    assert prop["near_vegetation"] is True  # still records the vegetation proximity


def test_forest_and_other_natural_never_overlap():
    """Every detection maps to exactly one fire_type_rule; forest and
    agriculture (other_natural) are complementary buckets — a fire can never
    appear in both. (Rules: industrial 1 km > mining 1 km > forest 3 km veg >
    other_natural fallback.)"""
    feats = _run(
        [FIRE_IN_FOREST, FIRE_FAR_AWAY, FIRE_IN_BOTH, FIRE_NEAR_MINING, FIRE_INSIDE],
        FOREST_FC,
        power_plants_fc={"type": "FeatureCollection", "features": [POWER_PLANT_SOLO]},
        flares_fc={"type": "FeatureCollection", "features": [FLARE_SITE]},
        mining_fc=MINING_OVERLAP_FC,
    )
    forest_ids, natural_ids = set(), set()
    for feat in feats:
        rule = feat["properties"]["fire_type_rule"]
        assert rule in ("industrial", "mining", "forest", "other_natural")
        if rule == "forest":
            forest_ids.add(feat["id"])
        elif rule == "other_natural":
            natural_ids.add(feat["id"])
    assert forest_ids.isdisjoint(natural_ids)


def test_forest_and_other_natural_fields_are_complementary():
    """The per-fire flags used by the frontend buckets can't both fire: a fire
    is 'forest' iff flagged near_vegetation (and not industrial/mining), and
    'other_natural' never has near_vegetation — so the two UI lists can't share
    a detection even by side-effect."""
    feats = _run([FIRE_IN_FOREST, FIRE_FAR_AWAY, FIRE_IN_BOTH], FOREST_FC)
    for feat in feats:
        prop = feat["properties"]
        rule = prop["fire_type_rule"]
        near_veg = prop["near_vegetation"]
        if rule == "forest":
            assert near_veg is True
            assert prop["near_industrial"] is False
            assert prop["near_mining"] is False
        elif rule == "other_natural":
            assert near_veg is False


# ── Frontend category-mapping mirror (static/app.js fireCategory()).
# The sidebar buckets each fire into exactly ONE tab; forest & agriculture must
# never both list the same detection. Kept in sync with the JS precedence:
# unregistered → persistent → flare → industrial → mining → forest → other.
def _frontend_category(props):
    if props.get("unregistered_persistent"):
        return "unregistered"
    if props.get("persistent_thermal_source"):
        return "persistent"
    if props.get("gas_flare"):
        return "flare"
    rule = props.get("fire_type_rule") or "other_natural"
    if rule == "industrial" or props.get("near_industrial"):
        return "industrial"
    if rule == "mining" or props.get("near_mining"):
        return "mining"
    if rule == "forest" or props.get("near_vegetation"):
        return "forest"
    return "other_natural"


def test_frontend_category_mapping_is_exclusive():
    ALL = (
        "unregistered",
        "persistent",
        "flare",
        "industrial",
        "mining",
        "forest",
        "other_natural",
    )
    cases = [
        {"fire_type_rule": "forest", "near_vegetation": True},
        {"fire_type_rule": "other_natural", "near_vegetation": False},
        {
            "fire_type_rule": "forest",
            "near_vegetation": True,
            "persistent_thermal_source": True,
            "occurrence_count": 3,
        },
        {"fire_type_rule": "forest", "near_vegetation": True, "unregistered_persistent": True},
        {
            "fire_type_rule": "other_natural",
            "near_vegetation": False,
            "gas_flare": True,
        },  # malformed data: precedence still picks ONE
        {"fire_type_rule": "forest"},  # missing flag: rule takes precedence
        {},  # empty props → agriculture
        {"fire_type_rule": "other_natural", "near_vegetation": True},  # legacy data
    ]
    buckets = {}
    for i, props in enumerate(cases):
        cat = _frontend_category(props)
        assert cat in ALL
        buckets.setdefault(cat, set()).add(i)

    assert buckets["forest"].isdisjoint(buckets["other_natural"])
    assert (
        _frontend_category({"fire_type_rule": "other_natural", "near_vegetation": True}) == "forest"
    )  # hardened fallback keeps agriculture free of forest flags
