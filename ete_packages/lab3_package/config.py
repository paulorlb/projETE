"""
Constants shared across all lab3_package modules.

Override any constant in the notebook after importing if you need a different value.
"""
from __future__ import annotations

RANDOM_SEED: int = 42
N_PERMUTATIONS: int = 999

REQUIRED_PACKAGES: list[str] = [
    "numpy",
    "pandas",
    "geopandas",
    "shapely",
    "matplotlib",
    "libpysal",
    "esda",
    "statsmodels",
    "spreg",
]

OPTIONAL_PACKAGES: list[str] = [
    "pyogrio",
    "fiona",
    "mapclassify",
    "splot",
    "scipy",
    "sklearn",
    "networkx",
    "contextily",
    "osmnx",
]

# --- Coordinate reference systems ---
PROJECTED_CRS: str = "EPSG:3763"
GEOGRAPHIC_CRS: str = "EPSG:4326"

# --- GeoPackage file name (searched inside data/ or project root) ---
GPKG_FILENAME: str = "ETE_Lab.gpkg"

# --- Expected layer names (token-based fallbacks are tried if exact match fails) ---
LISTINGS_LAYER_EXPECTED: str = "PrimeYield_HousingListingsDataClean"
ZONES_LAYER_EXPECTED: str = "M0105_M0110_C2021_e_casasapo_ZonesPlaces_clean"
FREGUESIA_LAYER_EXPECTED: str = "BGRI21_CONT_FREG_0105_0110"
MUNICIPAL_LAYER_EXPECTED: str = "CAOP24_CONT_MUNI"
MUNICIPAL_BOUNDARY_FALLBACK_EXPECTED: str = "cont_municipios"

# --- Listing-level column names used across modules ---
TARGET_PRICE: str = "price_eur"
TARGET_UNIT_PRICE: str = "unit_price_eur_m2"
AREA_VAR: str = "area_living_m2"
ZONE_ID: str = "zone_id"

# --- Municipal-level target and indicator column lists ---
MUNICIPAL_PRICE_TARGETS: list[str] = [
    "sales_median_eur_m2_2024_total",
    "sales_median_eur_m2_2024_existing",
    "rent_median_eur_m2_2023",
    "sales_median_eur_m2_2024_new",
]

MUNICIPAL_SOCIO_ECONOMIC_INDICATORS: list[str] = [
    "population_density",
    "share_age_25_64",
    "share_age_65plus",
    "foreign_nationality_share",
    "share_education_higher",
    "employment_rate",
    "tertiary_sector_employment_share",
    "vacant_total_share",
    "secondary_residence_share",
    "recent_to_old_building_ratio",
]

MUNICIPAL_GEOGRAPHIC_INDICATORS: list[str] = [
    "area_ha",
    "perimetro_km",
    "n_freguesias",
    "nuts1",
    "nuts2",
    "nuts3",
    "share_artificialized",
    "share_green_land",
    "poi_density_total",
    "essential_poi_density",
    "leisure_tourism_poi_density",
    "tourism_beds_density",
]
