from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(name: str | Path) -> Path:
    """Resolve a (possibly relative) file setting against the repo root.

    Keeps the DB, cache and .env files in a fixed place regardless of the
    working directory the app was launched from, so a fresh run from anywhere
    behaves identically to one started in the repo root.
    """
    path = Path(name)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), env_file_encoding="utf-8")

    firms_map_key: str = ""
    firms_dataset: str = "VIIRS_SNPP_NRT"
    firms_bbox: str = "68,6,97,37"
    firms_days: int = 3
    india_boundary_buffer_deg: float = 0.0
    buffer_meters: int = 1000
    fire_history_db: str = "fire_history.db"
    persistence_radius_m: int = 300
    persistence_min_occurrences: int = 2
    persistence_lookback_days: int = 14
    clustering_eps_m: int = 500
    clustering_min_samples: int = 2
    industrial_states: dict[str, str] = {
        "gujarat": "67.5,20,75,25",
        "jharkhand": "83,21.5,88,25.5",
        "maharashtra": "72,15,81,22.5",
        "west_bengal": "86,21.5,89.5,27",
        "odisha": "82,17.5,87.5,22.5",
        "chhattisgarh": "80,17.5,84,24",
        "tamil_nadu": "76,8,80.5,13.5",
        "kerala": "74.5,8.5,77.5,12.5",
        "karnataka": "74,11.5,78.5,18.5",
        "madhya_pradesh": "74,21.5,82.5,27",
        "assam": "89.5,24,96,28",
        "rajasthan": "69.5,23.3,78,30.2",
        "andhra_pradesh": "76.9,12.6,84.8,19.2",
        "telangana": "77.2,15.8,81.4,19.9",
        "punjab": "73.9,29.5,76.9,32.6",
        "uttar_pradesh": "77.1,23.9,84.4,30.4",
    }
    overpass_timeout: int = 180
    industrial_cache_file: str = "industrial_zones_cache.json"
    vegetation_cache_file: str = "vegetation_zones_cache.json"
    mining_cache_file: str = "mining_zones_cache.json"
    industrial_cache_max_age_hours: int = 720
    power_plants_cache_file: str = "power_plants_cache.json"
    power_plants_cache_max_age_hours: int = 720
    flares_cache_file: str = "flares_cache.json"
    flares_cache_max_age_hours: int = 720
    risk_frp_threshold_mw: float = 10.0
    risk_cone_min_length_m: int = 500
    risk_cone_max_length_m: int = 5000
    risk_wind_speed_max_kmh: float = 60.0
    risk_cone_min_half_angle_deg: int = 10
    risk_cone_max_half_angle_deg: int = 45
    risk_vegetation_distance_m: int = 1000
    weather_cache_ttl_seconds: int = 600
    # Buffer distances (meters) for spatial classification rules
    near_buffer_meters: int = 1000
    vegetation_buffer_meters: int = 3000
    flare_buffer_meters: int = 2000
    mining_buffer_meters: int = 1000
    search_radius_meters: int = 20000
    # Overpass mirror list and retry policy
    overpass_urls: list[str] = [
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    overpass_retries: int = 1
    overpass_retry_backoff_seconds: float = 2.0
    overpass_request_timeout_seconds: float = 12.0
    # HTTP timeouts for external data fetches
    firms_request_timeout_seconds: float = 60.0
    external_request_timeout_seconds: float = 120.0
    # FIRMS CSV area API URL template
    firms_area_url: str = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{dataset}/{bbox}/{days}"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
