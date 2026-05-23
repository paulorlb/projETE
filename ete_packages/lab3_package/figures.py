"""Maps, charts, and diagnostic visualisations for Topic 5."""

from __future__ import annotations

from typing import Mapping

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .utils import LabDataError


def plot_aveiro_orientation(
    model_table: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame | None = None,
    outcome: str = config.DEFAULT_AVEIRO_OUTCOME,
    metric_crs: str = config.METRIC_CRS,
):
    """Plot zones and listing points coloured by the selected outcome."""

    if outcome not in model_table.columns:
        raise LabDataError(f"Outcome column not found for map: {outcome}")

    gdf = model_table if str(model_table.crs) == str(metric_crs) else model_table.to_crs(metric_crs)
    fig, ax = plt.subplots(figsize=config.FIGSIZE_MAP)
    if zones is not None:
        z = zones if str(zones.crs) == str(metric_crs) else zones.to_crs(metric_crs)
        z.boundary.plot(ax=ax, linewidth=0.6)
    gdf.plot(ax=ax, column=outcome, markersize=14, alpha=0.75, legend=True)
    ax.set_title(f"Aveiro-Ílhavo listings by {outcome}")
    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax


def plot_gwr_surface_if_available(
    gwr_results: Mapping,
    zones: gpd.GeoDataFrame | None,
    coefficient: str,
    metric_crs: str = config.METRIC_CRS,
):
    """Plot a fitted GWR coefficient surface as points over zones, when available."""

    if gwr_results.get("status") != "fitted":
        fig, ax = plt.subplots(figsize=config.FIGSIZE_TABLE)
        ax.text(0.02, 0.5, f"GWR not plotted: {gwr_results.get('reason', 'not fitted')}", va="center")
        ax.set_axis_off()
        return fig, ax

    params = gwr_results.get("params")
    if params is None or coefficient not in params.columns:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_TABLE)
        ax.text(0.02, 0.5, f"Coefficient not available: {coefficient}", va="center")
        ax.set_axis_off()
        return fig, ax

    # The GWR result stores coefficient rows aligned to the input feature index.
    # Coordinates are not duplicated in the result to avoid stale geometries, so
    # this plot is a scaffold unless the caller attaches geometry separately.
    fig, ax = plt.subplots(figsize=config.FIGSIZE_TABLE)
    params[coefficient].plot(kind="hist", bins=25, ax=ax)
    ax.set_title(f"Distribution of local GWR coefficient: {coefficient}")
    ax.set_xlabel("local coefficient")
    fig.tight_layout()
    return fig, ax


def plot_cv_metric_summary(cv_summary: pd.DataFrame, metric: str = "mean_rmse"):
    """Plot a compact bar chart from ``summarise_cv_comparison`` output."""

    if metric not in cv_summary.columns:
        raise LabDataError(f"Metric column not found: {metric}")
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = cv_summary["strategy"].astype(str) + "\n" + cv_summary["model_family"].astype(str)
    ax.bar(labels, cv_summary[metric])
    ax.set_ylabel(metric)
    ax.set_title("Validation comparison")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, ax
