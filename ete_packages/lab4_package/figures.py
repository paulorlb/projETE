"""Maps, charts, and diagnostic visualisations for Topic 5."""

from __future__ import annotations

from typing import Mapping

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
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


def plot_gwr_coefficient_map(
    gwr_results: Mapping,
    features: gpd.GeoDataFrame,
    coefficient: str = "area_living_m2",
    model_key: str = "gwr",
    metric_crs: str = config.METRIC_CRS,
    zones: gpd.GeoDataFrame | None = None,
) -> tuple:
    """Plot spatial distribution and histogram of a GWR or MGWR local coefficient.

    Parameters
    ----------
    model_key : "gwr" (default) or "mgwr"
    zones : optional zone polygons drawn as background outlines on the map
    """

    if model_key == "mgwr":
        mgwr_data = gwr_results.get("mgwr")
        if mgwr_data is None or mgwr_data.get("status") != "fitted":
            reason = (mgwr_data or {}).get("reason", "MGWR not fitted")
            fig, ax = plt.subplots(figsize=config.FIGSIZE_TABLE)
            ax.text(0.02, 0.5, f"MGWR not plotted: {reason}", va="center")
            ax.set_axis_off()
            return fig, ax
        params = mgwr_data["params"]
        bw_map = mgwr_data.get("bandwidths", {})
        bw_val = bw_map.get(coefficient, "?")
        model_label = "MGWR"
    else:
        if gwr_results.get("status") != "fitted":
            fig, ax = plt.subplots(figsize=config.FIGSIZE_TABLE)
            ax.text(0.02, 0.5, f"GWR not plotted: {gwr_results.get('reason', 'not fitted')}", va="center")
            ax.set_axis_off()
            return fig, ax
        params = gwr_results.get("params")
        bw_val = gwr_results.get("bandwidth", "?")
        model_label = "GWR"

    if params is None or coefficient not in params.columns:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_TABLE)
        ax.text(0.02, 0.5, f"Coefficient '{coefficient}' not in {model_label} params", va="center")
        ax.set_axis_off()
        return fig, ax

    idx = params.index
    try:
        gdf_pts = features.loc[idx].copy()
    except KeyError:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_TABLE)
        ax.text(0.02, 0.5, "Cannot join GWR params to features: index mismatch", va="center")
        ax.set_axis_off()
        return fig, ax

    if str(gdf_pts.crs) != str(metric_crs):
        gdf_pts = gdf_pts.to_crs(metric_crs)

    coef_vals = params[coefficient].values
    gdf_pts = gdf_pts.copy()
    gdf_pts["_local_coef"] = coef_vals

    coef_min = float(np.nanmin(coef_vals))
    coef_max = float(np.nanmax(coef_vals))
    coef_range = coef_max - coef_min

    # Choose colormap and normalisation:
    # - If zero is inside the data range: diverging palette centred exactly at zero.
    # - If all values are same sign: sequential palette (no false zero-crossing implied).
    if coef_range < 1e-10:
        cmap, norm = "plasma", None
    elif coef_min < 0.0 < coef_max:
        abs_max = max(abs(coef_min), abs(coef_max))
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)
        cmap = "RdBu_r"
    elif coef_min >= 0.0:
        norm = mcolors.Normalize(vmin=coef_min, vmax=coef_max)
        cmap = "YlOrRd"
    else:
        norm = mcolors.Normalize(vmin=coef_min, vmax=coef_max)
        cmap = "YlGnBu_r"

    fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(14, 7))

    if zones is not None:
        z = zones if str(zones.crs) == str(metric_crs) else zones.to_crs(metric_crs)
        z.boundary.plot(ax=ax_map, linewidth=0.5, color="grey", alpha=0.6)

    plot_kwargs: dict = dict(
        ax=ax_map, column="_local_coef", markersize=14, alpha=0.7,
        cmap=cmap, legend=True,
        legend_kwds={"label": f"local β ({coefficient})", "shrink": 0.65},
    )
    if norm is not None:
        plot_kwargs["norm"] = norm
    gdf_pts.plot(**plot_kwargs)

    coef_range_note = ""
    if coef_range < 1e-6:
        coef_range_note = "  [near-constant — bandwidth ≈ N]"
    ax_map.set_title(
        f"{model_label} local coefficient: {coefficient}\n(bandwidth = {bw_val}){coef_range_note}",
        fontsize=10,
    )
    ax_map.set_axis_off()

    s = pd.Series(coef_vals)
    s.plot(kind="hist", bins=30, ax=ax_hist, color="steelblue", edgecolor="white")
    ax_hist.axvline(float(s.median()), color="red", linestyle="--",
                    label=f"median = {s.median():.3f}")
    # Zero reference line — only draw if it falls within the plotted range
    xlim_lo, xlim_hi = ax_hist.get_xlim()
    if xlim_lo < 0.0 < xlim_hi:
        ax_hist.axvline(0.0, color="black", linewidth=0.8, linestyle="-", label="zero")
    ax_hist.set_xlabel(f"local β ({coefficient})")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"{model_label} coefficient distribution: {coefficient}", fontsize=10)
    ax_hist.legend(fontsize=9)

    fig.tight_layout()
    return fig, (ax_map, ax_hist)


def _extract_feature_names(preprocessor) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer, sklearn-version-agnostic."""

    # Prefer the ColumnTransformer method (sklearn >= 1.0); its names have transformer
    # prefixes like "num__area_living_m2" and "cat__zone_id_cat_Z001".
    try:
        return preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        pass

    # Fallback: reconstruct manually using each sub-transformer.
    names: list[str] = []
    for trans_name, transformer, cols in preprocessor.transformers_:
        if trans_name == "remainder":
            continue
        if trans_name == "num":
            names.extend([f"num__{c}" for c in cols])
        elif trans_name == "cat":
            ohe = transformer.steps[-1][1]
            try:
                cat = ohe.get_feature_names_out(cols).tolist()
            except AttributeError:
                cat = ohe.get_feature_names(cols).tolist()
            names.extend([f"cat__{n}" for n in cat])
    return names


def plot_zone_dummy_coefficients(
    global_reference: Mapping,
    zones: gpd.GeoDataFrame,
    feature_set: str = "hedonic_plus_spatial_features",
    metric_crs: str = config.METRIC_CRS,
) -> tuple:
    """Map zone fixed-effect (dummy) coefficients from the global Ridge model onto zone polygons."""

    models = global_reference.get("models", {})
    pipe = models.get(feature_set)
    if pipe is None:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_MAP)
        ax.text(0.02, 0.5, f"No model found for feature_set='{feature_set}'", va="center")
        ax.set_axis_off()
        return fig, ax

    try:
        preprocessor = pipe.named_steps["preprocess"]
        estimator = pipe.named_steps["model"]

        feature_names = _extract_feature_names(preprocessor)
        coefs = np.asarray(estimator.coef_).ravel()

        if len(coefs) != len(feature_names):
            raise LabDataError(
                f"Coef/feature length mismatch: {len(coefs)} coefs vs {len(feature_names)} features.\n"
                f"First features: {feature_names[:5]}"
            )

        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})

        # Feature names may be prefixed "cat__zone_id_cat_<val>" or bare "zone_id_cat_<val>"
        zone_mask = coef_df["feature"].str.contains("zone_id_cat", na=False)
        zone_coefs = coef_df[zone_mask].copy()

        if zone_coefs.empty:
            fig, ax = plt.subplots(figsize=config.FIGSIZE_MAP)
            ax.text(
                0.02, 0.5,
                "No zone_id_cat dummy coefficients found.\n"
                f"Available features (first 10): {feature_names[:10]}",
                va="center", fontsize=8, wrap=True,
            )
            ax.set_axis_off()
            return fig, ax

        # Extract the raw zone_id value from the feature name regardless of prefix style
        extracted = zone_coefs["feature"].str.extract(r"zone_id_cat_(.+)$")[0]
        zone_coefs = zone_coefs.copy()
        zone_coefs["zone_id"] = extracted.astype(str)

        z = zones.copy()
        if str(z.crs) != str(metric_crs):
            z = z.to_crs(metric_crs)
        z["zone_id"] = z["zone_id"].astype(str)
        z = z.merge(zone_coefs[["zone_id", "coefficient"]], on="zone_id", how="left")

        has_coef = int(z["coefficient"].notna().sum())
        n_zones = len(z)

        # Diverging colormap centred at zero for the zone map
        vals = zone_coefs["coefficient"].dropna().to_numpy()
        v_min, v_max = float(vals.min()), float(vals.max())
        if v_min < 0.0 < v_max:
            abs_max = max(abs(v_min), abs(v_max))
            zone_norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-abs_max, vmax=abs_max)
            zone_cmap = "RdBu_r"
        else:
            zone_norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
            zone_cmap = "YlOrRd" if v_min >= 0.0 else "YlGnBu_r"

        fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(14, 7))

        z.plot(
            ax=ax_map, column="coefficient", cmap=zone_cmap, norm=zone_norm, legend=True,
            missing_kwds={"color": "lightgrey", "label": "reference (not encoded)"},
            legend_kwds={"label": "zone coefficient (std. units)", "shrink": 0.65},
        )
        z.boundary.plot(ax=ax_map, linewidth=0.4, color="black", alpha=0.5)
        ax_map.set_title(
            f"Zone fixed-effect coefficients\n(Ridge, {feature_set}, {has_coef}/{n_zones} zones encoded)",
            fontsize=10,
        )
        ax_map.set_axis_off()

        sorted_z = zone_coefs.sort_values("coefficient")
        colors = ["#d73027" if v < 0 else "#1a9850" for v in sorted_z["coefficient"]]
        ax_bar.barh(sorted_z["zone_id"], sorted_z["coefficient"], color=colors)
        ax_bar.axvline(0.0, color="black", linewidth=0.8)
        ax_bar.set_xlabel("standardised coefficient (relative to regularised baseline)")
        ax_bar.set_title("Zone dummy coefficients (sorted)", fontsize=10)
        ax_bar.tick_params(axis="y", labelsize=7)

        fig.tight_layout()
        return fig, (ax_map, ax_bar)

    except Exception as exc:
        fig, ax = plt.subplots(figsize=config.FIGSIZE_MAP)
        ax.text(0.02, 0.4, f"Zone dummy map failed:\n{exc}", va="center", fontsize=9)
        ax.set_axis_off()
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
