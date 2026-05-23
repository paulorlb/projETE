"""Data preparation, joins, transformations, and feature engineering."""

from __future__ import annotations

from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from . import config
from .utils import LabDataError, require_columns, safe_divide, warn_if


def _to_metric(gdf: gpd.GeoDataFrame, metric_crs: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame in the metric CRS."""

    if gdf.crs is None:
        raise LabDataError("GeoDataFrame has no CRS; cannot perform metric operations.")
    return gdf if str(gdf.crs) == str(metric_crs) else gdf.to_crs(metric_crs)


def summarise_coordinate_quality(
    listings: gpd.GeoDataFrame,
    zone_key: str = config.DEFAULT_GROUP_KEY,
    metric_crs: str = config.METRIC_CRS,
) -> pd.DataFrame:
    """Summarise listing-coordinate QA flags and zone-assignment quality."""

    require_columns(
        listings,
        [
            "coordinate_quality_flag",
            "coordinate_is_valid_for_point_analysis",
            "coordinate_within_study_area",
            "coordinate_duplicate_count",
            "coordinate_duplicate_is_suspicious",
            "zone_match_method",
            zone_key,
        ],
        "listings",
    )

    rows: list[dict[str, object]] = []
    n = len(listings)
    rows.append({"metric": "listing_rows", "value": n, "detail": "Rows in loaded listing layer"})

    for value, count in listings["coordinate_quality_flag"].value_counts(dropna=False).items():
        rows.append({"metric": "coordinate_quality_flag", "value": int(count), "detail": str(value)})

    bool_cols = [
        "coordinate_is_valid_for_point_analysis",
        "coordinate_within_study_area",
        "coordinate_duplicate_is_suspicious",
    ]
    for col in bool_cols:
        rows.append(
            {
                "metric": col,
                "value": int(listings[col].fillna(False).sum()),
                "detail": f"True values out of {n}",
            }
        )

    for value, count in listings["zone_match_method"].value_counts(dropna=False).items():
        rows.append({"metric": "zone_match_method", "value": int(count), "detail": str(value)})

    rows.extend(
        [
            {
                "metric": "unique_zones_represented",
                "value": int(listings[zone_key].nunique(dropna=True)),
                "detail": zone_key,
            },
            {
                "metric": "max_duplicate_coordinate_count",
                "value": int(listings["coordinate_duplicate_count"].max()),
                "detail": "Maximum exact-coordinate duplicate count",
            },
        ]
    )

    if "zone_match_distance_m" in listings.columns:
        dist = pd.to_numeric(listings["zone_match_distance_m"], errors="coerce")
        rows.append(
            {
                "metric": "nearest_zone_distance_m_max",
                "value": float(dist.max(skipna=True)) if dist.notna().any() else np.nan,
                "detail": "Only meaningful for nearest-zone fallback matches",
            }
        )

    return pd.DataFrame(rows)


def _filter_valid_listings(
    listings: gpd.GeoDataFrame,
    dropna_required: list[str] | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Apply required point-analysis filters and return drop counts."""

    df = listings.copy()
    counts: dict[str, int] = {"initial_rows": len(df)}

    if "coordinate_is_valid_for_point_analysis" in df:
        mask = df["coordinate_is_valid_for_point_analysis"].fillna(False).astype(bool)
        counts["dropped_invalid_coordinates"] = int((~mask).sum())
        df = df.loc[mask].copy()

    if "coordinate_within_study_area" in df:
        mask = df["coordinate_within_study_area"].fillna(False).astype(bool)
        counts["dropped_outside_study_area"] = int((~mask).sum())
        df = df.loc[mask].copy()

    if dropna_required:
        existing = [c for c in dropna_required if c in df.columns]
        before = len(df)
        df = df.dropna(subset=existing).copy()
        counts["dropped_missing_required"] = int(before - len(df))

    if {"price_eur", "area_living_m2"}.issubset(df.columns):
        before = len(df)
        df = df.loc[(df["price_eur"] > 0) & (df["area_living_m2"] > 0)].copy()
        counts["dropped_nonpositive_price_or_area"] = int(before - len(df))

    counts["final_rows"] = len(df)
    return df, counts


def _add_projected_coordinates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add projected x/y columns in metres."""

    out = gdf.copy()
    out["x_pttm06_m"] = out.geometry.x
    out["y_pttm06_m"] = out.geometry.y
    out["x_centered_km"] = (out["x_pttm06_m"] - out["x_pttm06_m"].mean()) / 1000.0
    out["y_centered_km"] = (out["y_pttm06_m"] - out["y_pttm06_m"].mean()) / 1000.0
    return out


def _attach_zone_context(
    listings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame | None,
    zone_key: str = config.DEFAULT_GROUP_KEY,
) -> gpd.GeoDataFrame:
    """Attach zone area and listing-count context without assuming fragile names."""

    out = listings.copy()
    if zones is None or zone_key not in out.columns or zone_key not in zones.columns:
        return out

    z = zones[[zone_key, "geometry", *[c for c in ["zone_name", "zone_code", "municipality_name"] if c in zones.columns]]].copy()
    z["zone_area_km2"] = z.geometry.area / 1_000_000.0

    zone_counts = out.groupby(zone_key).size().rename("zone_listing_count").reset_index()
    z_attr = z.drop(columns="geometry").merge(zone_counts, on=zone_key, how="left")
    z_attr["zone_listing_count"] = z_attr["zone_listing_count"].fillna(0).astype(int)
    z_attr["zone_listing_density_km2"] = safe_divide(z_attr["zone_listing_count"], z_attr["zone_area_km2"])

    cols_to_add = [c for c in z_attr.columns if c != zone_key and c not in out.columns]
    out = out.merge(z_attr[[zone_key, *cols_to_add]], on=zone_key, how="left")
    return out


def _attach_parish_context(
    listings: gpd.GeoDataFrame,
    parishes: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame:
    """Spatially join parish identifiers if parish polygons are available."""

    if parishes is None:
        return listings
    if listings.crs is None or parishes.crs is None:
        return listings

    left = listings.copy()
    right = parishes.to_crs(left.crs)[[c for c in ["DTMNFR21", "DTMN21", "freguesia", "geometry"] if c in parishes.columns]]
    if "geometry" not in right.columns:
        return left

    try:
        joined = gpd.sjoin(left, right, how="left", predicate="within")
    except Exception:
        joined = gpd.sjoin(left, right, how="left", predicate="intersects")
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    if "DTMNFR21" in joined.columns:
        joined = joined.rename(columns={"DTMNFR21": "parish_id"})
    if "freguesia" in joined.columns:
        joined = joined.rename(columns={"freguesia": "parish_name"})
    return joined


def prepare_aveiro_model_table(
    listings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame | None,
    parishes: gpd.GeoDataFrame | None = None,
    model_spec: Mapping[str, Any] | None = None,
    metric_crs: str = config.METRIC_CRS,
) -> gpd.GeoDataFrame:
    """Prepare the Track A listing-level modelling table."""

    spec = dict(model_spec or {})
    outcome = spec.get("outcome", config.DEFAULT_AVEIRO_OUTCOME)
    dropna_required = list(spec.get("dropna_required", [outcome, "area_living_m2", "zone_id"]))

    require_columns(listings, [outcome, "area_living_m2", "geometry"], "listings")
    gdf = _to_metric(listings, metric_crs)
    gdf, filter_counts = _filter_valid_listings(gdf, dropna_required=dropna_required)
    gdf = _add_projected_coordinates(gdf)
    gdf = _attach_zone_context(gdf, zones.to_crs(metric_crs) if zones is not None else None)
    gdf = _attach_parish_context(gdf, parishes.to_crs(metric_crs) if parishes is not None else None)

    if "listing_year" in gdf.columns:
        gdf["listing_year"] = gdf["listing_year"].astype("Int64").astype("string")

    gdf.attrs["filter_counts"] = filter_counts
    gdf.attrs["outcome"] = outcome
    gdf.attrs["unit_of_analysis"] = spec.get("unit_of_analysis", "listing")
    return gdf


def summarise_aveiro_model_table(
    model_table: gpd.GeoDataFrame,
    model_spec: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a compact summary of the prepared Track A modelling table."""

    spec = dict(model_spec or {})
    outcome = spec.get("outcome", model_table.attrs.get("outcome", config.DEFAULT_AVEIRO_OUTCOME))
    rows: list[dict[str, object]] = [
        {"metric": "rows", "value": len(model_table), "detail": "modelling sample"},
        {"metric": "outcome", "value": outcome, "detail": "dependent variable"},
    ]

    for col, label in [
        ("listing_year", "listing years"),
        ("municipality_name", "municipalities"),
        ("zone_id", "zones represented"),
        ("parish_id", "parishes represented"),
        ("property_type_std", "property types"),
    ]:
        if col in model_table.columns:
            rows.append(
                {
                    "metric": label,
                    "value": int(model_table[col].nunique(dropna=True)),
                    "detail": ", ".join(map(str, sorted(model_table[col].dropna().astype(str).unique())[:8])),
                }
            )

    if outcome in model_table.columns:
        s = pd.to_numeric(model_table[outcome], errors="coerce")
        rows.extend(
            [
                {"metric": f"{outcome}_mean", "value": float(s.mean()), "detail": "mean"},
                {"metric": f"{outcome}_median", "value": float(s.median()), "detail": "median"},
                {"metric": f"{outcome}_missing", "value": int(s.isna().sum()), "detail": "missing outcome values"},
            ]
        )

    for k, v in (model_table.attrs.get("filter_counts") or {}).items():
        rows.append({"metric": k, "value": v, "detail": "filter audit"})

    return pd.DataFrame(rows)


def _reference_centres_metric(metric_crs: str) -> gpd.GeoDataFrame:
    """Return package-defined reference centres in metric CRS."""

    records = []
    for name, (lon, lat) in config.REFERENCE_CENTRES_WGS84.items():
        records.append({"centre": name, "geometry": Point(lon, lat)})
    return gpd.GeoDataFrame(records, geometry="geometry", crs=config.RAW_GEOGRAPHIC_CRS).to_crs(metric_crs)


def build_aveiro_spatial_features(
    model_table: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame | None = None,
    parishes: gpd.GeoDataFrame | None = None,
    feature_spec: Mapping[str, Any] | None = None,
    random_state: int = config.RANDOM_STATE,
    metric_crs: str = config.METRIC_CRS,
) -> gpd.GeoDataFrame:
    """Add transparent spatial/contextual features to the Track A model table.

    Outcome-derived zone summaries are explicitly marked as leakage-sensitive and
    are excluded by the modelling helpers unless they are recomputed inside a
    training fold by a custom workflow.
    """

    spec = dict(feature_spec or {})
    gdf = _to_metric(model_table, metric_crs).copy()

    if spec.get("include_projected_coordinates", True):
        if "x_pttm06_m" not in gdf.columns or "y_pttm06_m" not in gdf.columns:
            gdf = _add_projected_coordinates(gdf)

    if spec.get("include_distance_to_reference_centres", True):
        centres = _reference_centres_metric(metric_crs)
        for _, row in centres.iterrows():
            col = f"distance_to_{row['centre']}_km"
            gdf[col] = gdf.geometry.distance(row.geometry) / 1000.0

    if zones is not None:
        gdf = _attach_zone_context(gdf, zones.to_crs(metric_crs))

    if parishes is not None and "parish_id" not in gdf.columns:
        gdf = _attach_parish_context(gdf, parishes.to_crs(metric_crs))

    leakage_cols: list[str] = []
    if spec.get("include_zone_neighbourhood_summaries", False) and "zone_id" in gdf.columns:
        outcomes = spec.get("neighbourhood_summary_outcomes", [])
        for outcome in outcomes:
            if outcome in gdf.columns:
                col = f"zone_mean_{outcome}{config.LEAKAGE_SUFFIX}"
                zone_mean = gdf.groupby("zone_id")[outcome].mean().rename(col).reset_index()
                gdf = gdf.merge(zone_mean, on="zone_id", how="left")
                leakage_cols.append(col)

    if spec.get("include_zone_indicators", True) and "zone_id" in gdf.columns:
        gdf["zone_id_cat"] = gdf["zone_id"].astype("string")
    if spec.get("include_municipality_indicator", True) and "municipality_name" in gdf.columns:
        gdf["municipality_name_cat"] = gdf["municipality_name"].astype("string")

    gdf.attrs.update(model_table.attrs)
    gdf.attrs["leakage_sensitive_features"] = leakage_cols
    return gdf


def summarise_spatial_features(
    features: gpd.GeoDataFrame,
    return_object: bool = False,
) -> pd.DataFrame:
    """Summarise available spatial and leakage-sensitive feature columns."""

    rows: list[dict[str, object]] = []
    leakage_cols = set(features.attrs.get("leakage_sensitive_features", []))

    for col in features.columns:
        if col == "geometry":
            continue
        if col in leakage_cols or col.endswith(config.LEAKAGE_SUFFIX):
            kind = "leakage_sensitive_outcome_summary"
        elif col in config.SPATIAL_FEATURE_COLUMNS or col.startswith("distance_to_"):
            kind = "spatial_numeric"
        elif col.endswith("_cat"):
            kind = "spatial_categorical"
        elif col in config.HEDONIC_NUMERIC_CONTROLS:
            kind = "hedonic_numeric"
        elif col in config.HEDONIC_CATEGORICAL_CONTROLS:
            kind = "hedonic_categorical"
        else:
            continue
        rows.append(
            {
                "column": col,
                "feature_type": kind,
                "dtype": str(features[col].dtype),
                "missing": int(features[col].isna().sum()),
                "unique": int(features[col].nunique(dropna=True)),
                "used_by_default": not (col in leakage_cols or col.endswith(config.LEAKAGE_SUFFIX)),
            }
        )

    out = pd.DataFrame(rows).sort_values(["feature_type", "column"]).reset_index(drop=True)
    return out


def prepare_municipal_market_table(
    municipalities: gpd.GeoDataFrame,
    municipal_spec: Mapping[str, Any] | None = None,
    metric_crs: str = config.METRIC_CRS,
) -> gpd.GeoDataFrame:
    """Prepare the Track B municipality-level table."""

    spec = dict(municipal_spec or {})
    outcomes = list(spec.get("outcomes", config.MUNICIPAL_TARGETS[:2]))
    covariates = [c for c in spec.get("candidate_covariates", config.MUNICIPAL_COVARIATES) if c in municipalities.columns]

    require_columns(municipalities, ["dtmn", "municipio", "geometry", *outcomes], "municipalities")
    gdf = _to_metric(municipalities, metric_crs).copy()

    before = len(gdf)
    gdf = gdf.dropna(subset=outcomes).copy()
    gdf.attrs["dropped_missing_outcomes"] = int(before - len(gdf))
    gdf.attrs["outcomes"] = outcomes
    gdf.attrs["candidate_covariates"] = covariates
    gdf.attrs["unit_of_analysis"] = "municipality"

    # Ensure numeric covariates are numeric.
    for col in outcomes + covariates:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    return gdf
