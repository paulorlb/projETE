"""Input/output and schema-inspection functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import pandas as pd

from . import config
from .utils import LabDataError, as_path, optional_import, require_columns, require_file


def _list_layers(gpkg_path: str | Path) -> list[tuple[str, str | None]]:
    """List GeoPackage layers using pyogrio when available, then Fiona."""

    gpkg = require_file(gpkg_path, "GeoPackage")
    try:
        pyogrio = optional_import("pyogrio", "fast GeoPackage layer listing")
        raw = pyogrio.list_layers(gpkg)
        return [(str(row[0]), str(row[1]) if len(row) > 1 else None) for row in raw]
    except Exception:
        try:
            import fiona

            return [(name, None) for name in fiona.listlayers(gpkg)]
        except Exception as exc:
            raise LabDataError(f"Could not list layers in {gpkg}: {exc}") from exc


def read_geopackage_layer(
    gpkg_path: str | Path,
    layer: str,
    *,
    target_crs: str | None = None,
    require_geometry: bool = True,
) -> gpd.GeoDataFrame:
    """Read a GeoPackage layer and optionally reproject it."""

    gpkg = require_file(gpkg_path, "GeoPackage")
    try:
        gdf = gpd.read_file(gpkg, layer=layer)
    except Exception as exc:
        raise LabDataError(f"Could not read layer '{layer}' from {gpkg}: {exc}") from exc

    if require_geometry and "geometry" not in gdf.columns:
        raise LabDataError(f"Layer '{layer}' does not contain a geometry column.")
    if require_geometry and gdf.crs is None:
        raise LabDataError(f"Layer '{layer}' has no CRS. Define it before metric operations.")

    if target_crs is not None and require_geometry:
        if gdf.crs is None:
            raise LabDataError(f"Layer '{layer}' cannot be reprojected because CRS is missing.")
        if str(gdf.crs) != str(target_crs):
            gdf = gdf.to_crs(target_crs)
    return gdf


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV with an actionable error message."""

    p = require_file(path, "CSV")
    try:
        return pd.read_csv(p, **kwargs)
    except Exception as exc:
        raise LabDataError(f"Could not read CSV {p}: {exc}") from exc


def inspect_geopackage(
    gpkg_path: str | Path,
    layer_hints: Mapping[str, str] | None = None,
    sample_rows: int = 5,
) -> pd.DataFrame:
    """Inspect GeoPackage layers, CRS, geometry type, columns, and candidate keys."""

    gpkg = require_file(gpkg_path, "GeoPackage")
    hints = dict(layer_hints or {})
    rows: list[dict[str, Any]] = []

    for layer_name, listed_geom in _list_layers(gpkg):
        try:
            gdf = gpd.read_file(gpkg, layer=layer_name)
        except Exception as exc:
            rows.append(
                {
                    "layer": layer_name,
                    "read_ok": False,
                    "error": str(exc),
                }
            )
            continue

        role = next((k for k, v in hints.items() if v == layer_name), None)
        candidate_keys = [
            col
            for col in ["listing_id", "zone_id", "dtmn", "DTMN21", "DTMNFR21", "geo"]
            if col in gdf.columns
        ]
        key_uniqueness = {
            col: bool(gdf[col].is_unique) for col in candidate_keys if col in gdf.columns
        }
        rows.append(
            {
                "layer": layer_name,
                "role_hint": role,
                "read_ok": True,
                "geometry_type": ", ".join(sorted(map(str, gdf.geom_type.dropna().unique()))),
                "listed_geometry_type": listed_geom,
                "crs": str(gdf.crs) if gdf.crs is not None else None,
                "row_count": int(len(gdf)),
                "column_count": int(len(gdf.columns)),
                "columns": list(gdf.columns),
                "candidate_keys": candidate_keys,
                "key_uniqueness": key_uniqueness,
                "has_empty_geometry": bool(gdf.geometry.is_empty.any()) if "geometry" in gdf else None,
                "has_missing_geometry": bool(gdf.geometry.isna().any()) if "geometry" in gdf else None,
                "error": None,
            }
        )
    return pd.DataFrame(rows)


def find_layer_by_role(
    inventory: pd.DataFrame,
    role: str,
    layer_hints: Mapping[str, str] | None = None,
) -> str | None:
    """Find the layer name for a role using hints first, then inventory role hints."""

    hints = dict(layer_hints or {})
    if role in hints and hints[role] in set(inventory["layer"]):
        return hints[role]

    match = inventory.loc[inventory.get("role_hint", pd.Series(dtype=str)) == role, "layer"]
    if len(match):
        return str(match.iloc[0])

    # Conservative fallbacks based on known column signatures.
    for _, row in inventory.iterrows():
        cols = set(row.get("columns") or [])
        geom = str(row.get("geometry_type") or "").lower()
        layer = str(row["layer"])
        if role == "listings" and {"listing_id", "unit_price_eur_m2", "zone_id"}.issubset(cols):
            return layer
        if role == "zones" and {"zone_id", "zone_name"}.issubset(cols) and "polygon" in geom:
            return layer
        if role == "parishes" and {"DTMNFR21", "freguesia"}.issubset(cols):
            return layer
        if role == "municipalities" and {"dtmn", "sales_median_eur_m2_2024_total"}.issubset(cols):
            return layer
    return None


def load_layers(
    gpkg_path: str | Path,
    layer_hints: Mapping[str, str] | None = None,
    metric_crs: str = config.METRIC_CRS,
    raw_geographic_crs: str = config.RAW_GEOGRAPHIC_CRS,
) -> dict[str, gpd.GeoDataFrame]:
    """Load and validate the GeoPackage layers needed by the lab blueprint.

    Listing points are reprojected to ``metric_crs`` for all downstream metric
    operations. Raw longitude/latitude columns remain available if present.
    """

    inventory = inspect_geopackage(gpkg_path, layer_hints=layer_hints)
    layers: dict[str, gpd.GeoDataFrame] = {}

    for role in ["listings", "zones", "parishes", "municipalities"]:
        layer_name = find_layer_by_role(inventory, role, layer_hints)
        if layer_name is None:
            continue
        target_crs = metric_crs
        gdf = read_geopackage_layer(gpkg_path, layer_name, target_crs=target_crs)
        layers[role] = gdf

    if "listings" in layers:
        require_columns(layers["listings"], config.LISTING_REQUIRED_COLUMNS, "listing layer")
    if "zones" in layers:
        require_columns(layers["zones"], config.ZONE_REQUIRED_COLUMNS, "zone layer")
    if "municipalities" in layers:
        missing_targets = [c for c in config.MUNICIPAL_TARGETS if c not in layers["municipalities"].columns]
        if missing_targets:
            raise LabDataError(f"municipality layer is missing target variables: {missing_targets}")

    return layers


def compare_inventory_to_schema_notes(
    inventory: pd.DataFrame,
    schema_path: str | Path | None = None,
    municipality_metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compare runtime inventory with count and variable hints in markdown notes.

    This is intentionally heuristic: runtime GeoPackage inspection is the source
    of truth for modelling, while markdown notes are semantic documentation.
    """

    rows: list[dict[str, object]] = []
    if schema_path is not None and as_path(schema_path).exists():
        text = as_path(schema_path).read_text(encoding="utf-8", errors="ignore")
        expected_counts = {
            "PrimeYield_HousingListingsDataClean": ["Clean listing rows", "listing rows"],
            "M0105_M0110_C2021_e_casasapo_ZonesPlaces_clean": ["Full zone rows", "zone rows"],
        }
        for layer, labels in expected_counts.items():
            observed = inventory.loc[inventory["layer"] == layer, "row_count"]
            observed_count = int(observed.iloc[0]) if len(observed) else None
            note = "not parsed"
            for label in labels:
                if label in text:
                    import re

                    pattern = rf"{label}[^0-9`]*`?([0-9]+)`?"
                    match = re.search(pattern, text)
                    if match:
                        note = int(match.group(1))
                        break
            rows.append(
                {
                    "source": "PrimeYield schema note",
                    "item": layer,
                    "runtime_value": observed_count,
                    "documented_value": note,
                    "status": "match" if observed_count == note else "check",
                }
            )

    if municipality_metadata_path is not None and as_path(municipality_metadata_path).exists():
        text = as_path(municipality_metadata_path).read_text(encoding="utf-8", errors="ignore")
        for var in config.MUNICIPAL_TARGETS:
            present = any(var in (cols or []) for cols in inventory["columns"])
            documented = var in text
            rows.append(
                {
                    "source": "municipality metadata note",
                    "item": var,
                    "runtime_value": present,
                    "documented_value": documented,
                    "status": "match" if present == documented else "check",
                }
            )

    return pd.DataFrame(rows)
