"""Tidy reporting, summaries, and export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import pandas as pd

from . import config
from .paths import ensure_output_dir
from .utils import module_version


def summarise_qa(
    file_status: pd.DataFrame | None = None,
    inventory: pd.DataFrame | None = None,
    schema_check: pd.DataFrame | None = None,
    coordinate_qa: pd.DataFrame | None = None,
    aveiro_model_table: gpd.GeoDataFrame | None = None,
    feature_summary: pd.DataFrame | None = None,
    global_diagnostics: pd.DataFrame | None = None,
    gwr_results: Mapping[str, Any] | None = None,
    cv_results: Mapping[str, Any] | None = None,
    residual_diagnostics: pd.DataFrame | None = None,
    municipal_cv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Collect the main QA outputs into a concise checklist table."""

    rows: list[dict[str, object]] = []

    if file_status is not None and not file_status.empty:
        rows.append(
            {
                "section": "files",
                "status": "ok" if bool(file_status["exists"].all()) else "check",
                "detail": f"{int(file_status['exists'].sum())}/{len(file_status)} expected paths exist",
            }
        )

    if inventory is not None and not inventory.empty:
        layer_names = ", ".join(inventory["layer"].astype(str).tolist())
        rows.append({"section": "geopackage_inventory", "status": "ok", "detail": layer_names})

    if schema_check is not None and not schema_check.empty:
        n_check = int((schema_check["status"] != "match").sum()) if "status" in schema_check else 0
        rows.append({"section": "schema_notes", "status": "ok" if n_check == 0 else "check", "detail": f"{n_check} documented/runtime mismatches"})

    if coordinate_qa is not None and not coordinate_qa.empty:
        rows.append({"section": "coordinate_qa", "status": "ok", "detail": f"{len(coordinate_qa)} QA rows reported"})

    if aveiro_model_table is not None:
        rows.append({"section": "track_a_model_table", "status": "ok", "detail": f"{len(aveiro_model_table)} listing rows retained"})

    if feature_summary is not None and not feature_summary.empty:
        excluded = int((feature_summary.get("used_by_default", pd.Series(dtype=bool)) == False).sum())
        rows.append({"section": "spatial_features", "status": "ok", "detail": f"{len(feature_summary)} features summarised; {excluded} excluded by default"})

    if global_diagnostics is not None and not global_diagnostics.empty:
        rows.append({"section": "global_reference", "status": "ok", "detail": f"{len(global_diagnostics)} diagnostic rows"})

    if gwr_results is not None:
        rows.append({"section": "gwr", "status": gwr_results.get("status", "unknown"), "detail": gwr_results.get("reason", "fitted or attempted")})

    if cv_results is not None:
        metrics = cv_results.get("fold_metrics", pd.DataFrame()) if isinstance(cv_results, Mapping) else pd.DataFrame()
        rows.append({"section": "validation", "status": cv_results.get("status", "unknown"), "detail": f"{len(metrics)} fold-metric rows"})

    if residual_diagnostics is not None and not residual_diagnostics.empty:
        rows.append({"section": "residual_spatial_autocorrelation", "status": "ok", "detail": f"{len(residual_diagnostics)} diagnostic rows"})

    if municipal_cv is not None and not municipal_cv.empty:
        rows.append({"section": "track_b_municipal_extension", "status": "ok", "detail": f"{len(municipal_cv)} validation rows"})

    return pd.DataFrame(rows)


def summarise_outputs(*objects: Any, names: list[str] | None = None) -> pd.DataFrame:
    """Return a small summary of notebook objects produced by the lab."""

    rows: list[dict[str, object]] = []
    for i, obj in enumerate(objects):
        name = names[i] if names and i < len(names) else f"object_{i + 1}"
        if isinstance(obj, (pd.DataFrame, gpd.GeoDataFrame)):
            detail = f"{len(obj)} rows x {len(obj.columns)} columns"
        elif isinstance(obj, Mapping):
            detail = f"mapping keys: {', '.join(map(str, obj.keys()))}"
        else:
            detail = type(obj).__name__
        rows.append({"name": name, "type": type(obj).__name__, "detail": detail})
    return pd.DataFrame(rows)


def export_lab_outputs(
    outputs: Mapping[str, Any],
    output_dir: str | Path = config.DEFAULT_OUTPUT_DIR,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Export selected tables and GeoDataFrames to files.

    Tables are written as CSV. GeoDataFrames are written as GeoPackage layers.
    Objects that are not tabular are skipped with a note.
    """

    out_dir = ensure_output_dir(output_dir)
    rows: list[dict[str, object]] = []

    for name, obj in outputs.items():
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
        if isinstance(obj, gpd.GeoDataFrame):
            path = out_dir / f"{safe_name}.gpkg"
            if path.exists() and not overwrite:
                rows.append({"name": name, "status": "skipped", "path": str(path), "detail": "exists"})
                continue
            obj.to_file(path, layer=safe_name, driver="GPKG")
            rows.append({"name": name, "status": "written", "path": str(path), "detail": "GeoPackage"})
        elif isinstance(obj, pd.DataFrame):
            path = out_dir / f"{safe_name}.csv"
            if path.exists() and not overwrite:
                rows.append({"name": name, "status": "skipped", "path": str(path), "detail": "exists"})
                continue
            obj.to_csv(path, index=False)
            rows.append({"name": name, "status": "written", "path": str(path), "detail": "CSV"})
        else:
            rows.append({"name": name, "status": "skipped", "path": None, "detail": type(obj).__name__})

    return pd.DataFrame(rows)


def package_versions(packages: list[str] | None = None) -> pd.DataFrame:
    """Return package versions for reproducibility footers."""

    packages = packages or [
        "python",
        "pandas",
        "geopandas",
        "shapely",
        "pyproj",
        "sklearn",
        "libpysal",
        "esda",
        "mgwr",
    ]
    return pd.DataFrame({"package": packages, "version": [module_version(p) for p in packages]})
