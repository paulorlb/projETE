"""
Exploratory analysis: choropleth maps, point maps, LISA cluster maps, and ranked diagnostic tables.
"""
from __future__ import annotations

from .config import MUNICIPAL_PRICE_TARGETS, RANDOM_SEED
from .io import import_analysis_stack
from .prep import existing_columns, available_share_columns

# Standard LISA cluster colours (classic GeoDa palette)
_LISA_COLORS = {
    "HH": "#d7191c",  # red
    "LL": "#2c7bb6",  # blue
    "LH": "#abd9e9",  # light blue
    "HL": "#fdae61",  # orange-salmon
    "ns": "#d3d3d3",  # light grey
}
_LISA_LEGEND = [
    ("HH", "High–High"),
    ("HL", "High–Low"),
    ("LH", "Low–High"),
    ("LL", "Low–Low"),
    ("ns", "Not significant (p ≥ 0.05)"),
]


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def plot_layer_map(gdf, column: str, title: str, cmap: str = "viridis", missing_color: str = "lightgrey") -> None:
    _, _, _, plt = import_analysis_stack()
    if gdf is None or column not in gdf.columns:
        print(f"Map skipped: {column!r} is not available.")
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    gdf.plot(
        column=column,
        ax=ax,
        legend=True,
        cmap=cmap,
        missing_kwds={"color": missing_color, "label": "Missing/no observations"},
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_title(title)
    ax.set_axis_off()
    plt.show()


def plot_point_map(gdf, column: str, title: str, sample_size: int = 2000, cmap: str = "viridis") -> None:
    _, _, _, plt = import_analysis_stack()
    if gdf is None or column not in gdf.columns:
        print(f"Point map skipped: {column!r} is not available.")
        return
    plot_df = gdf.dropna(subset=[column])
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_df.plot(column=column, ax=ax, legend=True, cmap=cmap, markersize=12, alpha=0.75)
    ax.set_title(title)
    ax.set_axis_off()
    plt.show()


def plot_lisa_map(gdf, lisa_df, title: str) -> None:
    """Draw a LISA cluster map from a compute_lisa() result joined on gdf.index."""
    _, _, _, plt = import_analysis_stack()
    from matplotlib.patches import Patch

    if lisa_df is None or lisa_df.empty:
        print(f"LISA map skipped: no results available.")
        return

    plot_gdf = gdf.join(lisa_df[["cluster"]], how="left")
    plot_gdf["cluster"] = plot_gdf["cluster"].fillna("ns")
    plot_gdf["_color"] = plot_gdf["cluster"].map(_LISA_COLORS).fillna(_LISA_COLORS["ns"])

    fig, ax = plt.subplots(figsize=(9, 8))
    plot_gdf.plot(color=plot_gdf["_color"], ax=ax, edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_axis_off()

    # Only show legend entries that actually appear in the data
    present = set(plot_gdf["cluster"].unique())
    handles = [
        Patch(facecolor=_LISA_COLORS[k], label=lbl)
        for k, lbl in _LISA_LEGEND
        if k in present
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Ranked diagnostic tables
# ---------------------------------------------------------------------------

def high_price_support_table(gdf, id_cols: list[str], min_n: int = 3, top_n: int = 10):
    _, pd, _, _ = import_analysis_stack()
    if "n_listings" not in gdf.columns:
        return pd.DataFrame()
    df = gdf.loc[gdf["n_listings"] >= min_n].copy()
    if df.empty:
        return pd.DataFrame()

    if "median_unit_price_eur_m2" in df.columns:
        df["rank_unit_price"] = df["median_unit_price_eur_m2"].rank(ascending=False, method="min")
        df["unit_price_percentile"] = df["median_unit_price_eur_m2"].rank(pct=True)
    if "median_price_eur" in df.columns:
        df["rank_total_price"] = df["median_price_eur"].rank(ascending=False, method="min")
        df["total_price_percentile"] = df["median_price_eur"].rank(pct=True)

    sort_cols = existing_columns(df, ["rank_unit_price", "rank_total_price"])
    if sort_cols:
        df = df.sort_values(sort_cols)

    diagnostic_cols = existing_columns(
        df,
        id_cols + [
            "n_listings",
            "median_price_eur", "mean_price_eur", "price_eur_iqr",
            "median_unit_price_eur_m2", "mean_unit_price_eur_m2", "unit_price_eur_m2_iqr",
            "median_area_living_m2", "area_living_m2_iqr",
            "mean_condition_score", "median_condition_score",
            "fallback_assignment_share",
            "price_outlier_share", "unit_price_outlier_share", "area_outlier_share",
            "small_n_lt5", "small_n_lt10",
            "rank_unit_price", "rank_total_price",
            "unit_price_percentile", "total_price_percentile",
        ],
    )
    composition_cols = (
        available_share_columns(df, "property_type_std")
        + available_share_columns(df, "typology_bucket_std")
        + available_share_columns(df, "condition_std")
        + available_share_columns(df, "preservation_class_std")
        + available_share_columns(df, "listing_year")
    )
    return df[diagnostic_cols + composition_cols].head(top_n)


def high_price_municipal_table(gdf, min_nonmissing: bool = True, top_n: int = 15):
    _, pd, _, _ = import_analysis_stack()
    target = "sales_median_eur_m2_2024_total"
    if target not in gdf.columns:
        targets = existing_columns(gdf, MUNICIPAL_PRICE_TARGETS)
        if not targets:
            return pd.DataFrame()
        target = targets[0]

    df = gdf.copy()
    if min_nonmissing:
        df = df.loc[df[target].notna()].copy()
    df["rank_price"] = pd.to_numeric(df[target], errors="coerce").rank(ascending=False, method="min")

    optional_cols = [
        "dtmn", "municipio", target, "rank_price",
        "population_density", "share_education_higher", "employment_rate",
        "secondary_residence_share", "vacant_total_share",
        "share_artificialized", "share_green_land",
        "poi_density_total", "essential_poi_density",
        "leisure_tourism_poi_density", "tourism_beds_density",
        "n_freguesias", "area_ha", "nuts2", "nuts3",
    ]
    purchasing_power_like = [
        col for col in gdf.columns
        if any(token in col.lower() for token in ["income", "rendimento", "purchas", "power", "poder"])
    ]
    cols = existing_columns(df, optional_cols + purchasing_power_like)
    return df[cols].sort_values("rank_price").head(top_n)
