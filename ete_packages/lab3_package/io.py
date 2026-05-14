"""
GeoPackage I/O and analysis-stack import helpers.

All GeoPackage access is implemented through sqlite3 (no geopandas dependency)
to support the pre-loading layer inspection step.  Actual vector loading uses
geopandas, imported lazily via import_analysis_stack().
"""
from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

from .config import PROJECTED_CRS


# ---------------------------------------------------------------------------
# Analysis-stack lazy importers
# ---------------------------------------------------------------------------

def import_analysis_stack():
    """Import and return (numpy, pandas, geopandas, matplotlib.pyplot)."""
    missing = [
        name for name in ["numpy", "pandas", "geopandas", "matplotlib"]
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ImportError(
            "These packages are required for the data-preparation sections: "
            + ", ".join(missing)
            + ". Install them in the notebook kernel before running analysis cells."
        )
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    return np, pd, gpd, plt


def import_model_stack():
    """Import and return (statsmodels.api, esda.Moran, libpysal.weights.lag_spatial, w_subset)."""
    missing = [
        name for name in ["statsmodels", "esda", "libpysal"]
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ImportError(
            "Modelling sections require these packages: " + ", ".join(missing)
        )
    import statsmodels.api as sm
    from esda import Moran
    from libpysal.weights import lag_spatial
    try:
        from libpysal.weights import w_subset
    except ImportError:
        from libpysal.weights.set_operations import w_subset
    return sm, Moran, lag_spatial, w_subset


def import_lisa_stack():
    """Import and return esda.Moran_Local for LISA analysis."""
    if importlib.util.find_spec("esda") is None:
        raise ImportError(
            "esda is required for LISA analysis. "
            "Install it in the notebook kernel before running LISA cells."
        )
    from esda import Moran_Local
    return Moran_Local


# ---------------------------------------------------------------------------
# Raw sqlite3 helpers (no geopandas)
# ---------------------------------------------------------------------------

def quote_sql_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sqlite_scalar(conn: sqlite3.Connection, sql: str, params=()):
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    return None if row is None else row[0]


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[dict]:
    sql = f"PRAGMA table_info({quote_sql_identifier(table_name)})"
    return [
        {"name": row[1], "type": row[2], "not_null": bool(row[3]), "primary_key": bool(row[5])}
        for row in conn.execute(sql).fetchall()
    ]


# ---------------------------------------------------------------------------
# GeoPackage layer inspection
# ---------------------------------------------------------------------------

def inspect_gpkg_layers(gpkg_path: Path) -> list[dict]:
    """Return metadata for every layer in *gpkg_path* using only sqlite3."""
    if not gpkg_path.exists():
        raise FileNotFoundError(gpkg_path)

    with sqlite3.connect(gpkg_path) as conn:
        contents = conn.execute(
            "SELECT table_name, data_type, identifier, description, srs_id "
            "FROM gpkg_contents ORDER BY table_name"
        ).fetchall()

        geom_rows = conn.execute(
            "SELECT table_name, column_name, geometry_type_name, srs_id "
            "FROM gpkg_geometry_columns"
        ).fetchall()
        geom_by_table = {row[0]: row for row in geom_rows}

        srs_rows = conn.execute(
            "SELECT srs_id, organization, organization_coordsys_id, definition "
            "FROM gpkg_spatial_ref_sys"
        ).fetchall()
        srs_by_id = {row[0]: row for row in srs_rows}

        layers = []
        for table_name, data_type, identifier, description, content_srs_id in contents:
            geom_row = geom_by_table.get(table_name)
            geom_col = geom_row[1] if geom_row else None
            geom_type = geom_row[2] if geom_row else None
            geom_srs_id = geom_row[3] if geom_row else content_srs_id
            srs_row = srs_by_id.get(geom_srs_id)
            if srs_row and srs_row[1] and srs_row[2]:
                crs = f"{srs_row[1]}:{srs_row[2]}"
            elif geom_srs_id is not None:
                crs = f"srs_id={geom_srs_id}"
            else:
                crs = None

            row_count = sqlite_scalar(
                conn, f"SELECT COUNT(*) FROM {quote_sql_identifier(table_name)}"
            )
            cols = table_columns(conn, table_name)
            layers.append(
                {
                    "layer_name": table_name,
                    "data_type": data_type,
                    "geometry_column": geom_col,
                    "geometry_type": geom_type,
                    "crs": crs,
                    "row_count": row_count,
                    "columns": cols,
                }
            )
    return layers


def detect_layer(
    layers: list[dict],
    exact_name: str | None = None,
    required_tokens: list[str] | None = None,
) -> str | None:
    """Return the layer name that matches *exact_name* first, then token heuristics."""
    layer_names = [
        layer["layer_name"] if isinstance(layer, dict) else str(layer)
        for layer in layers
    ]
    if exact_name and exact_name in layer_names:
        return exact_name
    if required_tokens:
        tokens = [t.upper() for t in required_tokens]
        matches = [n for n in layer_names if any(t in n.upper() for t in tokens)]
        return matches[0] if matches else None
    return None


def summarize_layer_columns(layer: dict) -> str:
    return ", ".join(col["name"] for col in layer["columns"])


def layer_summary_rows(layers: list[dict]) -> list[dict]:
    return [
        {
            "layer_name": layer["layer_name"],
            "geometry_type": layer["geometry_type"],
            "crs": layer["crs"],
            "row_count": layer["row_count"],
            "n_columns": len(layer["columns"]),
        }
        for layer in layers
    ]


# ---------------------------------------------------------------------------
# CRS and geometry helpers
# ---------------------------------------------------------------------------

def require_projected_crs(gdf, expected_crs: str = PROJECTED_CRS, label: str = "GeoDataFrame") -> bool:
    """Raise ValueError if *gdf* is not in *expected_crs*."""
    crs_text = str(getattr(gdf, "crs", None))
    if expected_crs.lower() not in crs_text.lower():
        raise ValueError(
            f"{label} must be in {expected_crs} before metric operations; found {crs_text}"
        )
    return True


def read_gpkg_layer(gpkg_path: Path, layer_name: str | None, label: str):
    """Load a GeoDataFrame from *gpkg_path*; raises informative errors on missing layer."""
    if layer_name is None:
        raise ValueError(f"No GeoPackage layer detected for {label}.")
    _, _, gpd, _ = import_analysis_stack()
    return gpd.read_file(gpkg_path, layer=layer_name)


def repair_invalid_geometries(gdf, label: str) -> tuple:
    """Repair invalid geometries and return (repaired_gdf, repair_report_dict)."""
    if gdf is None:
        return None, {
            "label": label, "rows": 0,
            "invalid_before": None, "invalid_after": None, "repair": "not loaded",
        }
    result = gdf.copy()
    invalid_before = int((~result.geometry.is_valid).sum())
    repair_method = "none"
    if invalid_before:
        try:
            from shapely.validation import make_valid
            result.geometry = result.geometry.apply(
                lambda geom: make_valid(geom) if geom is not None else geom
            )
            repair_method = "shapely.validation.make_valid"
        except Exception:
            result.geometry = result.geometry.buffer(0)
            repair_method = "buffer(0)"
    invalid_after = int((~result.geometry.is_valid).sum())
    return result, {
        "label": label,
        "rows": len(result),
        "invalid_before": invalid_before,
        "invalid_after": invalid_after,
        "repair": repair_method,
    }


def to_metric_crs(gdf, label: str, target_crs: str = PROJECTED_CRS):
    """Reproject *gdf* to *target_crs*; raises if the source CRS is unknown."""
    if gdf is None:
        return None
    if gdf.crs is None:
        raise ValueError(
            f"{label} has no CRS. Do not assign a CRS silently; inspect the data source "
            "before metric operations."
        )
    return gdf.to_crs(target_crs)
