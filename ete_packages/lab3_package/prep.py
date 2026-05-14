"""
Data preparation, cleaning, feature engineering, and spatial aggregation.

All functions are pure (no side-effects on inputs) and defer heavy imports to
import_analysis_stack() so the module can be imported before the analysis stack
is installed.
"""
from __future__ import annotations

import importlib

from .config import (
    TARGET_PRICE, TARGET_UNIT_PRICE, AREA_VAR, ZONE_ID,
)
from .io import import_analysis_stack


# ---------------------------------------------------------------------------
# Column utilities
# ---------------------------------------------------------------------------

def existing_columns(df, columns: list[str]) -> list[str]:
    """Return only the elements of *columns* that exist in *df*."""
    return [col for col in columns if col in df.columns]


# ---------------------------------------------------------------------------
# QA / missingness diagnostics
# ---------------------------------------------------------------------------

def missingness_summary(df, columns=None):
    _, pd, _, _ = import_analysis_stack()
    if columns is None:
        columns = list(df.columns)
    n = len(df)
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        missing = int(df[col].isna().sum())
        rows.append({
            "variable": col,
            "n_rows": n,
            "n_missing": missing,
            "pct_missing": round(100 * missing / n, 2) if n else None,
            "n_non_missing": n - missing,
        })
    return pd.DataFrame(rows)


def value_counts_table(df, column: str, include_na: bool = True):
    _, pd, _, _ = import_analysis_stack()
    if column not in df.columns:
        return pd.DataFrame([{"variable": column, "value": "not present", "count": None, "share": None}])
    counts = df[column].value_counts(dropna=not include_na)
    n = counts.sum()
    return pd.DataFrame({
        "variable": column,
        "value": counts.index.astype(str),
        "count": counts.values,
        "share": [round(v / n, 4) if n else None for v in counts.values],
    })


def duplicate_summary(df, id_col: str):
    _, pd, _, _ = import_analysis_stack()
    if id_col not in df.columns:
        return pd.DataFrame([{"id_column": id_col, "n_duplicate_rows": None, "n_duplicated_ids": None}])
    duplicated_mask = df[id_col].duplicated(keep=False)
    duplicated_ids = df.loc[duplicated_mask, id_col].dropna().nunique()
    return pd.DataFrame([{
        "id_column": id_col,
        "n_rows": len(df),
        "n_duplicate_rows": int(duplicated_mask.sum()),
        "n_duplicated_ids": int(duplicated_ids),
    }])


def positive_value_summary(df, columns: list[str]):
    _, pd, _, _ = import_analysis_stack()
    rows = []
    for col in existing_columns(df, columns):
        numeric = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "variable": col,
            "n_non_missing": int(numeric.notna().sum()),
            "n_positive": int((numeric > 0).sum()),
            "n_zero": int((numeric == 0).sum()),
            "n_negative": int((numeric < 0).sum()),
            "n_non_positive_or_missing": int((~(numeric > 0)).sum()),
            "min": numeric.min(),
            "median": numeric.median(),
            "max": numeric.max(),
        })
    return pd.DataFrame(rows)


def numeric_summary(df, columns: list[str]):
    _, pd, _, _ = import_analysis_stack()
    cols = existing_columns(df, columns)
    if not cols:
        return pd.DataFrame()
    summary = df[cols].apply(pd.to_numeric, errors="coerce").describe().T.reset_index()
    return summary.rename(columns={"index": "variable"})


# ---------------------------------------------------------------------------
# Log-transformation helpers
# ---------------------------------------------------------------------------

def add_log_if_positive(df, source_col: str, log_col: str | None = None):
    """Add log(source_col) where values are strictly positive; NaN elsewhere."""
    np, pd, _, _ = import_analysis_stack()
    if source_col not in df.columns:
        return df
    if log_col is None:
        log_col = f"log_{source_col}"
    result = df.copy()
    values = pd.to_numeric(result[source_col], errors="coerce")
    result[log_col] = np.where(values > 0, np.log(values), np.nan)
    return result


def add_price_logs(df):
    """Add log-transformed price, unit price, and area columns."""
    result = df.copy()
    for col in [TARGET_PRICE, TARGET_UNIT_PRICE, AREA_VAR]:
        if col in result.columns:
            result = add_log_if_positive(result, col)
    return result


# ---------------------------------------------------------------------------
# Outlier flags
# ---------------------------------------------------------------------------

def add_iqr_outlier_flag(df, source_col: str, flag_col: str | None = None, multiplier: float = 1.5):
    """Add a boolean flag for IQR-based outliers in *source_col*."""
    _, pd, _, _ = import_analysis_stack()
    if source_col not in df.columns:
        return df
    result = df.copy()
    flag_col = flag_col or f"{source_col}_outlier_iqr"
    values = pd.to_numeric(result[source_col], errors="coerce")
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        result[flag_col] = False
        return result
    result[flag_col] = values.lt(q1 - multiplier * iqr) | values.gt(q3 + multiplier * iqr)
    return result


def add_listing_outlier_flags(df):
    """Add IQR-outlier flags for price, unit price, and area."""
    result = df.copy()
    result = add_iqr_outlier_flag(result, TARGET_PRICE, "price_eur_outlier_iqr")
    result = add_iqr_outlier_flag(result, TARGET_UNIT_PRICE, "unit_price_eur_m2_outlier_iqr")
    result = add_iqr_outlier_flag(result, AREA_VAR, "area_living_m2_outlier_iqr")
    return result


# ---------------------------------------------------------------------------
# Condition score
# ---------------------------------------------------------------------------

def infer_condition_score(df):
    """Add condition_score only when observed labels support a defensible ordinal order."""
    _, pd, _, _ = import_analysis_stack()
    result = df.copy()
    decision = {
        "condition_score_created": False,
        "source_column": None,
        "mapping": None,
        "reason": None,
    }

    if "preservation_class_std" in result.columns:
        observed = set(result["preservation_class_std"].dropna().astype(str).unique())
        mapping = {"Used_26plus": 1, "Used_11_25": 2, "Used_0_10": 3, "New": 4}
        if observed and observed.issubset(set(mapping)):
            result["condition_score"] = result["preservation_class_std"].map(mapping).astype("float")
            decision = {
                "condition_score_created": True,
                "source_column": "preservation_class_std",
                "mapping": mapping,
                "reason": "Labels encode a clear new-to-older preservation order.",
            }
            return result, decision

    if "condition_std" in result.columns:
        observed = set(result["condition_std"].dropna().astype(str).str.lower().unique())
        mapping = {"used": 1, "new": 2}
        if observed and observed.issubset(set(mapping)):
            result["condition_score"] = (
                result["condition_std"].astype(str).str.lower().map(mapping).astype("float")
            )
            decision = {
                "condition_score_created": True,
                "source_column": "condition_std",
                "mapping": mapping,
                "reason": "Labels support only a binary used/new ordering. Use alongside category shares.",
            }
            return result, decision

    decision["reason"] = (
        "Observed condition labels do not support a defensible ordinal score; use shares/dummies."
    )
    return result, decision


# ---------------------------------------------------------------------------
# Category shares
# ---------------------------------------------------------------------------

def slugify_label(value: str) -> str:
    import re
    import unicodedata
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return text or "missing"


def category_share_table(df, group_col: str, cat_col: str, prefix: str | None = None):
    _, pd, _, _ = import_analysis_stack()
    if group_col not in df.columns or cat_col not in df.columns:
        return None
    prefix = prefix or cat_col
    tmp = df[[group_col, cat_col]].copy()
    tmp[cat_col] = tmp[cat_col].fillna("missing").astype(str)
    table = pd.crosstab(tmp[group_col], tmp[cat_col], normalize="index")
    table.columns = [f"{prefix}_share_{slugify_label(col)}" for col in table.columns]
    return table.reset_index()


def add_category_shares(base, listings_df, group_col: str, cat_cols: list[str]):
    result = base.copy()
    for cat_col in cat_cols:
        shares = category_share_table(listings_df, group_col, cat_col)
        if shares is not None:
            result = result.merge(shares, on=group_col, how="left")
    share_cols = [col for col in result.columns if "_share_" in col]
    result[share_cols] = result[share_cols].fillna(0)
    return result


# ---------------------------------------------------------------------------
# Listing aggregation
# ---------------------------------------------------------------------------

def aggregate_listing_support(listings_df, group_col: str):
    """Aggregate listing-level data to a spatial support (zone, parish, etc.)."""
    _, pd, _, _ = import_analysis_stack()
    if group_col not in listings_df.columns:
        raise ValueError(f"Group column {group_col!r} is not present.")

    df = listings_df.copy()
    if "zone_match_method" in df.columns:
        df["is_nearest_zone_fallback"] = (
            df["zone_match_method"].astype(str).str.lower().eq("nearest_zone")
        )

    agg_spec = {}
    id_col = "listing_id" if "listing_id" in df.columns else None
    if id_col:
        agg_spec["n_listings"] = (id_col, "count")
    else:
        df["_row_counter"] = 1
        agg_spec["n_listings"] = ("_row_counter", "sum")

    optional_aggs = [
        (TARGET_PRICE, "median_price_eur", "median"),
        (TARGET_PRICE, "mean_price_eur", "mean"),
        (TARGET_UNIT_PRICE, "median_unit_price_eur_m2", "median"),
        (TARGET_UNIT_PRICE, "mean_unit_price_eur_m2", "mean"),
        (AREA_VAR, "median_area_living_m2", "median"),
        ("log_price_eur", "mean_log_price_eur", "mean"),
        ("log_price_eur", "median_log_price_eur", "median"),
        ("log_unit_price_eur_m2", "mean_log_unit_price_eur_m2", "mean"),
        ("log_unit_price_eur_m2", "median_log_unit_price_eur_m2", "median"),
        ("log_area_living_m2", "mean_log_area_living_m2", "mean"),
        ("condition_score", "mean_condition_score", "mean"),
        ("condition_score", "median_condition_score", "median"),
        ("is_nearest_zone_fallback", "fallback_assignment_share", "mean"),
        ("price_eur_outlier_iqr", "price_outlier_share", "mean"),
        ("unit_price_eur_m2_outlier_iqr", "unit_price_outlier_share", "mean"),
        ("area_living_m2_outlier_iqr", "area_outlier_share", "mean"),
    ]
    for source, output, func in optional_aggs:
        if source in df.columns:
            agg_spec[output] = (source, func)

    base = df.groupby(group_col, dropna=False).agg(**agg_spec).reset_index()

    for source, prefix in [
        (TARGET_PRICE, "price_eur"),
        (TARGET_UNIT_PRICE, "unit_price_eur_m2"),
        (AREA_VAR, "area_living_m2"),
    ]:
        if source in df.columns:
            quantiles = (
                df.groupby(group_col, dropna=False)[source]
                .quantile([0.25, 0.75])
                .unstack()
                .rename(columns={0.25: f"{prefix}_q25", 0.75: f"{prefix}_q75"})
                .reset_index()
            )
            quantiles[f"{prefix}_iqr"] = quantiles[f"{prefix}_q75"] - quantiles[f"{prefix}_q25"]
            base = base.merge(quantiles, on=group_col, how="left")

    base["small_n_lt3"] = base["n_listings"] < 3
    base["small_n_lt5"] = base["n_listings"] < 5
    base["small_n_lt10"] = base["n_listings"] < 10

    category_cols = existing_columns(
        df,
        ["condition_std", "preservation_class_std", "property_type_std",
         "typology_bucket_std", "listing_year"],
    )
    base = add_category_shares(base, df, group_col, category_cols)
    return base


def fill_support_count_flags(gdf):
    """Fill NaN listing counts and ensure small-N flag columns are present."""
    result = gdf.copy()
    if "n_listings" in result.columns:
        result["n_listings"] = result["n_listings"].fillna(0).astype(int)
    else:
        result["n_listings"] = 0
    for threshold in [3, 5, 10]:
        result[f"small_n_lt{threshold}"] = result["n_listings"] < threshold
    if "fallback_assignment_share" in result.columns:
        result["fallback_assignment_share"] = result["fallback_assignment_share"].fillna(0)
    share_cols = [col for col in result.columns if "_share_" in col]
    if share_cols:
        result[share_cols] = result[share_cols].fillna(0)
    return result


def spatial_join_with_fallback(left_gdf, right_gdf, columns: list[str], predicate: str = "within"):
    _, _, gpd, _ = import_analysis_stack()
    try:
        return gpd.sjoin(left_gdf, right_gdf[columns + ["geometry"]], how="left", predicate=predicate)
    except TypeError:
        return gpd.sjoin(left_gdf, right_gdf[columns + ["geometry"]], how="left", op=predicate)


# ---------------------------------------------------------------------------
# Model-frame utilities
# ---------------------------------------------------------------------------

def available_share_columns(df, contains: str | None = None) -> list[str]:
    cols = [col for col in df.columns if "_share_" in col]
    if contains:
        cols = [col for col in cols if contains.lower() in col.lower()]
    return cols


def add_dummy_columns(df, source_col: str, prefix: str | None = None, drop_first: bool = True):
    _, pd, _, _ = import_analysis_stack()
    if source_col not in df.columns:
        return df, []
    result = df.copy()
    dummies = pd.get_dummies(
        result[source_col], prefix=prefix or source_col, drop_first=drop_first, dtype=float
    )
    result = pd.concat([result, dummies], axis=1)
    return result, list(dummies.columns)


def select_share_predictors(
    df, prefix: str, reference_contains: str | None = None, max_cols: int | None = None
) -> list[str]:
    cols = sorted([col for col in df.columns if col.startswith(prefix) and "_share_" in col])
    if reference_contains:
        ref = [col for col in cols if reference_contains.lower() in col.lower()]
        if ref:
            cols = [col for col in cols if col != ref[0]]
    elif cols:
        cols = cols[1:]
    if max_cols is not None:
        cols = cols[:max_cols]
    return cols


def valid_numeric_predictors(df, candidate_cols: list[str]) -> list[str]:
    """Return only columns with sufficient numeric variation to enter a regression."""
    _, pd, _, _ = import_analysis_stack()
    valid = []
    for col in candidate_cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() < 3:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        valid.append(col)
    return valid


def prepare_model_frame(df, y_col: str, x_cols: list[str], min_n_col=None, min_n: int = 3):
    """Filter to complete cases, drop collinear predictors, and return (model_df, x_valid)."""
    np, pd, _, _ = import_analysis_stack()
    if y_col not in df.columns:
        raise ValueError(f"Dependent variable {y_col!r} is not present.")

    model_cols = [y_col] + list(x_cols)
    if min_n_col and min_n_col in df.columns:
        model_cols.append(min_n_col)

    model_df = df[model_cols].copy()
    for col in [y_col] + list(x_cols):
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    if min_n_col and min_n_col in model_df.columns:
        model_df = model_df.loc[model_df[min_n_col] >= min_n].copy()

    model_df = model_df.dropna(subset=[y_col] + list(x_cols)).copy()
    x_valid = valid_numeric_predictors(model_df, x_cols)
    model_df = model_df.dropna(subset=[y_col] + x_valid).copy()

    if x_valid:
        retained = []
        for col in x_valid:
            trial = retained + [col]
            matrix = model_df[trial].astype(float).values
            if np.linalg.matrix_rank(matrix) == len(trial):
                retained.append(col)
        dropped = [col for col in x_valid if col not in retained]
        if dropped:
            print(f"Dropped collinear predictors: {dropped}")
        x_valid = retained

    if len(model_df) <= len(x_valid) + 2:
        raise ValueError(
            f"Model sample too small after filtering: n={len(model_df)}, k={len(x_valid)}."
        )
    return model_df, x_valid
