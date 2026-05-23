"""Configuration constants for Topic 4 spatial-analysis labs.

The constants in this module are deliberately conservative. They document the
expected course files and variables, but the IO layer still validates the
runtime GeoPackage before any analysis is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

RANDOM_STATE: Final[int] = 20260521

RAW_GEOGRAPHIC_CRS: Final[str] = "EPSG:4326"
METRIC_CRS: Final[str] = "EPSG:3763"

DEFAULT_GPKG_NAME: Final[str] = "ETE_Lab.gpkg"
DEFAULT_OUTPUT_DIR: Final[str] = "outputs/topic4"

EXPECTED_LAYERS: Final[dict[str, str]] = {
    "listings": "PrimeYield_HousingListingsDataClean",
    "zones": "M0105_M0110_C2021_e_casasapo_ZonesPlaces_clean",
    "parishes": "BGRI21_CONT_FREG_0105_0110",
    "municipalities": "CAOP24_CONT_MUNI",
}

LISTING_REQUIRED_COLUMNS: Final[list[str]] = [
    "listing_id",
    "listing_year",
    "price_eur",
    "area_living_m2",
    "unit_price_eur_m2",
    "property_type_std",
    "typology_bucket_std",
    "condition_std",
    "preservation_class_std",
    "coordinate_quality_flag",
    "coordinate_is_valid_for_point_analysis",
    "coordinate_within_study_area",
    "coordinate_duplicate_count",
    "coordinate_duplicate_is_suspicious",
    "zone_match_method",
    "zone_match_distance_m",
    "zone_id",
    "municipality_name",
    "geometry",
]

ZONE_REQUIRED_COLUMNS: Final[list[str]] = [
    "zone_id",
    "zone_code",
    "id_name",
    "zone_name",
    "municipality_code",
    "municipality_name",
    "geometry",
]

MUNICIPAL_TARGETS: Final[list[str]] = [
    "rent_median_eur_m2_2023",
    "sales_median_eur_m2_2024_total",
    "sales_median_eur_m2_2024_new",
    "sales_median_eur_m2_2024_existing",
]

MUNICIPAL_COVARIATES: Final[list[str]] = [
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
    "share_artificialized",
    "share_green_land",
    "poi_density_total",
    "essential_poi_density",
    "leisure_tourism_poi_density",
    "tourism_beds_density",
    "beds_per_resident",
    "share_tourism_beds_al",
]

DEFAULT_AVEIRO_OUTCOME: Final[str] = "unit_price_eur_m2"
DEFAULT_GROUP_KEY: Final[str] = "zone_id"

HEDONIC_NUMERIC_CONTROLS: Final[list[str]] = ["area_living_m2"]
HEDONIC_CATEGORICAL_CONTROLS: Final[list[str]] = [
    "property_type_std",
    "typology_bucket_std",
    "condition_std",
    "preservation_class_std",
    "listing_year",
]

SPATIAL_FEATURE_COLUMNS: Final[list[str]] = [
    "x_pttm06_m",
    "y_pttm06_m",
    "x_centered_km",
    "y_centered_km",
    "distance_to_aveiro_centre_km",
    "distance_to_ilhavo_centre_km",
    "zone_area_km2",
    "zone_listing_count",
    "zone_listing_density_km2",
]

ID_COLUMNS: Final[list[str]] = [
    "listing_id",
    "zone_id",
    "zone_code",
    "zone_name",
    "municipality_code",
    "municipality_name",
    "parish_id",
    "parish_name",
]

LEAKAGE_SUFFIX: Final[str] = "__leakage_sensitive"

# Approximate civic centres used only for transparent distance features.
# These are not theoretical amenities and should not be interpreted causally.
REFERENCE_CENTRES_WGS84: Final[dict[str, tuple[float, float]]] = {
    "aveiro_centre": (-8.6538, 40.6405),  # lon, lat
    "ilhavo_centre": (-8.6659, 40.6019),  # lon, lat
}

DEFAULT_KNN_K: Final[int] = 8
DEFAULT_PERMUTATIONS: Final[int] = 999

PLOT_DPI: Final[int] = 160
FIGSIZE_MAP: Final[tuple[int, int]] = (8, 8)
FIGSIZE_TABLE: Final[tuple[int, int]] = (8, 4)


@dataclass(frozen=True)
class LayerExpectation:
    """Expected layer role and minimum identifying columns."""

    role: str
    layer_name: str
    required_columns: tuple[str, ...]


LAYER_EXPECTATIONS: Final[tuple[LayerExpectation, ...]] = (
    LayerExpectation("listings", EXPECTED_LAYERS["listings"], tuple(LISTING_REQUIRED_COLUMNS)),
    LayerExpectation("zones", EXPECTED_LAYERS["zones"], tuple(ZONE_REQUIRED_COLUMNS)),
    LayerExpectation("municipalities", EXPECTED_LAYERS["municipalities"], tuple(["dtmn", *MUNICIPAL_TARGETS])),
)
