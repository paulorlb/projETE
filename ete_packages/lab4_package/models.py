"""Model wrappers and diagnostics for the Topic 5 lab."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .utils import LabDataError, OptionalDependencyError, clip_n_splits, module_version, require_columns
from .weights import SimpleWeights, build_weights_suite, knn_weights, moran_i, queen_weights


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def _metric_record(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
    }


def _safe_one_hot_encoder() -> OneHotEncoder:
    """Create an OneHotEncoder across scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _default_feature_columns(
    df: pd.DataFrame,
    outcome: str,
    include_spatial: bool = True,
    include_leakage_sensitive: bool = False,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical default model columns."""

    forbidden = {
        outcome,
        "geometry",
        "listing_id",
        "price_eur" if outcome != "price_eur" else "",
        "unit_price_eur_m2" if outcome != "unit_price_eur_m2" else "",
    }
    leakage_cols = set(getattr(df, "attrs", {}).get("leakage_sensitive_features", []))
    numeric_candidates = [
        c
        for c in [*config.HEDONIC_NUMERIC_CONTROLS, *(config.SPATIAL_FEATURE_COLUMNS if include_spatial else [])]
        if c in df.columns and c not in forbidden
    ]
    numeric_candidates += [c for c in df.columns if include_spatial and c.startswith("distance_to_") and c not in forbidden]

    categorical_candidates = [
        c
        for c in config.HEDONIC_CATEGORICAL_CONTROLS
        if c in df.columns and c not in forbidden
    ]
    if include_spatial:
        categorical_candidates += [c for c in ["municipality_name_cat", "zone_id_cat"] if c in df.columns]

    if include_leakage_sensitive:
        numeric_candidates += [c for c in leakage_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_candidates = [c for c in numeric_candidates if c not in leakage_cols and not c.endswith(config.LEAKAGE_SUFFIX)]

    # Unique, stable order.
    numeric = list(dict.fromkeys(numeric_candidates))
    categorical = list(dict.fromkeys(categorical_candidates))
    return numeric, categorical


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str], standardize_numeric: bool = True) -> ColumnTransformer:
    """Build a deterministic preprocessing transformer."""

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if standardize_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _safe_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise LabDataError("No usable feature columns were found for the requested model.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_estimator(model_family: str, random_state: int) -> Any:
    """Create a model estimator from a small supported family."""

    name = model_family.lower()
    if name in {"regularized_linear", "regularized_linear_or_ols_baseline", "ridge", "linear"}:
        return RidgeCV(alphas=np.logspace(-3, 3, 13))
    if name in {"random_forest", "random_forest_or_gradient_boosting_baseline", "ml", "rf"}:
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
    if name in {"lightgbm", "lgbm", "lgb", "light_gbm"}:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise OptionalDependencyError("lightgbm", "pip install lightgbm") from exc
        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    raise LabDataError(f"Unsupported model_family: {model_family}")


def _build_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_family: str,
    random_state: int,
    standardize_numeric: bool = True,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _build_preprocessor(numeric_cols, categorical_cols, standardize_numeric)),
            ("model", _build_estimator(model_family, random_state)),
        ]
    )


def fit_global_reference_model(
    features: gpd.GeoDataFrame,
    model_spec: Mapping[str, Any],
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Any]:
    """Fit global reference models for the requested feature sets."""

    spec = dict(model_spec)
    outcome = spec.get("outcome", config.DEFAULT_AVEIRO_OUTCOME)
    require_columns(features, [outcome], "features")

    data = features.dropna(subset=[outcome]).copy()
    y = pd.to_numeric(data[outcome], errors="coerce")
    valid = y.notna()
    data = data.loc[valid].copy()
    y = y.loc[valid].to_numpy(dtype=float)

    fitted: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    for feature_set in spec.get("feature_sets", ["hedonic_controls_only", "hedonic_plus_spatial_features"]):
        include_spatial = feature_set != "hedonic_controls_only"
        numeric_cols, categorical_cols = _default_feature_columns(
            data,
            outcome=outcome,
            include_spatial=include_spatial,
            include_leakage_sensitive=False,
        )
        family = spec.get("model_family", "regularized_linear")
        pipe = _build_pipeline(
            numeric_cols,
            categorical_cols,
            model_family=family,
            random_state=random_state,
            standardize_numeric=bool(spec.get("standardize_numeric", True)),
        )
        pipe.fit(data, y)
        yhat = pipe.predict(data)
        record = {
            "feature_set": feature_set,
            "model_family": family,
            "n": len(y),
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            **_metric_record(y, yhat),
        }
        metrics.append(record)
        fitted[feature_set] = pipe

        pred = pd.DataFrame(
            {
                "row_index": data.index,
                "feature_set": feature_set,
                "y_true": y,
                "y_pred": yhat,
                "residual": y - yhat,
            },
            index=data.index,
        )
        if "listing_id" in data.columns:
            pred["listing_id"] = data["listing_id"].to_numpy()
        predictions.append(pred)

    return {
        "status": "fitted",
        "outcome": outcome,
        "models": fitted,
        "metrics": pd.DataFrame(metrics),
        "predictions": pd.concat(predictions, axis=0).reset_index(drop=True),
    }


def diagnose_global_reference_model(
    global_reference: Mapping[str, Any],
    features: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame | None = None,
    metric_crs: str = config.METRIC_CRS,
) -> pd.DataFrame:
    """Compute basic residual diagnostics for fitted global reference models."""

    pred = global_reference.get("predictions")
    if pred is None or len(pred) == 0:
        return pd.DataFrame([{"status": "skipped", "reason": "No global-reference predictions available"}])

    rows: list[dict[str, Any]] = []
    for feature_set, part in pred.groupby("feature_set"):
        idx = part["row_index"].to_numpy()
        g = features.loc[idx].copy()
        if g.crs is None:
            rows.append({"feature_set": feature_set, "diagnostic": "moran_knn_8", "status": "skipped", "reason": "missing CRS"})
            continue
        try:
            w = knn_weights(g, k=8, metric_crs=metric_crs, name="listing_knn_8")
            mi = moran_i(part["residual"].to_numpy(), w, permutations=199)
            rows.append(
                {
                    "feature_set": feature_set,
                    "diagnostic": "moran_knn_8",
                    "weights": w.name,
                    "status": "computed",
                    **mi,
                }
            )
        except Exception as exc:
            rows.append({"feature_set": feature_set, "diagnostic": "moran_knn_8", "status": "skipped", "reason": str(exc)})

    metrics = global_reference.get("metrics", pd.DataFrame())
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "feature_set": row["feature_set"],
                    "diagnostic": "in_sample_fit",
                    "weights": None,
                    "status": "computed",
                    "rmse": row.get("rmse"),
                    "mae": row.get("mae"),
                    "r2": row.get("r2"),
                    "n": row.get("n"),
                }
            )
    return pd.DataFrame(rows)


def _numeric_gwr_frame(features: gpd.GeoDataFrame, outcome: str, covariates: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.Index]:
    """Create numeric matrices for GWR from selected covariates."""

    df = features[[outcome, "geometry", *[c for c in covariates if c in features.columns]]].dropna(subset=[outcome]).copy()
    if len(df) < 160:
        raise LabDataError("GWR skipped: fewer than 160 complete observations after initial filtering.")

    X_parts: list[pd.DataFrame] = []
    names: list[str] = []
    for col in covariates:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            X_parts.append(pd.DataFrame({col: pd.to_numeric(df[col], errors="coerce")}, index=df.index))
            names.append(col)
        else:
            dummies = pd.get_dummies(df[col].astype("string"), prefix=col, drop_first=True)
            X_parts.append(dummies)
            names.extend(dummies.columns.tolist())

    if not X_parts:
        raise LabDataError("GWR skipped: no usable covariates found.")
    X = pd.concat(X_parts, axis=1)
    valid = X.notna().all(axis=1) & df[outcome].notna()
    df = df.loc[valid]
    X = X.loc[valid]
    y = pd.to_numeric(df[outcome], errors="coerce").to_numpy().reshape((-1, 1))
    coords = np.column_stack([df.geometry.x.to_numpy(), df.geometry.y.to_numpy()])
    X_arr = X.to_numpy(dtype=float)

    # Standardise numeric matrix to reduce local conditioning problems.
    X_arr = (X_arr - X_arr.mean(axis=0)) / np.where(X_arr.std(axis=0) == 0, 1, X_arr.std(axis=0))
    return coords, y, X_arr, X.columns.tolist(), df.index


def fit_gwr_if_available(
    features: gpd.GeoDataFrame,
    gwr_spec: Mapping[str, Any],
    random_state: int = config.RANDOM_STATE,
    metric_crs: str = config.METRIC_CRS,
) -> dict[str, Any]:
    """Fit GWR (and optionally MGWR) using the PySAL ``mgwr`` package.

    Both GWR and MGWR use adaptive bisquare kernels with AICc bandwidth
    selection.  MGWR is fitted when ``mgwr_extension`` in the spec is not one
    of the sentinel strings that mark it as conceptual-only.

    Returns a structured skip dict when ``mgwr`` is unavailable or when
    fitting fails, so the notebook never raises an unhandled exception.
    """

    spec = dict(gwr_spec)
    if not spec.get("enabled", True):
        return {"status": "skipped", "reason": "GWR disabled by specification"}

    try:
        from mgwr.gwr import GWR, MGWR
        from mgwr.sel_bw import Sel_BW
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": (
                "PySAL 'mgwr' package is not installed or could not be imported. "
                "Install with: pip install mgwr"
            ),
            "detail": str(exc),
        }

    outcome = spec.get("outcome", config.DEFAULT_AVEIRO_OUTCOME)
    covariates = list(spec.get("covariates", config.HEDONIC_NUMERIC_CONTROLS))

    _MGWR_SKIP_SENTINELS = {
        "conceptual_only_unless_available",
        "conceptual_only",
        "off",
        "false",
        "no",
        "skip",
    }
    run_mgwr = str(spec.get("mgwr_extension", "conceptual_only_unless_available")).lower() not in _MGWR_SKIP_SENTINELS

    try:
        gdf = features if str(features.crs) == str(metric_crs) else features.to_crs(metric_crs)
        coords, y, X, names, idx = _numeric_gwr_frame(gdf, outcome, covariates)

        # ── GWR ────────────────────────────────────────────────────────────
        gwr_selector = Sel_BW(coords, y, X, kernel="bisquare", fixed=False, spherical=False)
        bw = gwr_selector.search(criterion="AICc")
        gwr_model = GWR(coords, y, X, bw=bw, kernel="bisquare", fixed=False, spherical=False)
        gwr_result = gwr_model.fit()
        params = pd.DataFrame(gwr_result.params, columns=["Intercept", *names], index=idx)

        out: dict[str, Any] = {
            "status": "fitted",
            "outcome": outcome,
            "bandwidth": bw,
            "covariate_names": names,
            "params": params,
            "model": gwr_model,
            "result": gwr_result,
            "index": idx,
            "summary": str(gwr_result.summary()) if hasattr(gwr_result, "summary") else None,
            "mgwr": None,
        }

        # ── MGWR (optional) ─────────────────────────────────────────────────
        # MGWR back-fitting is sensitive to near-singular local design matrices.
        # Binary (0/1) dummy columns are the most common cause; use only numeric
        # covariates in gwr_spec["covariates"] to avoid this.
        if run_mgwr:
            # Detect binary columns in X and warn when there are many (typical of OHE expansion).
            # A single binary column (e.g. is_apartment) is tolerable; 3+ sparse dummies
            # cause near-singular local matrices and should be avoided.
            binary_cols = [names[j] for j in range(X.shape[1]) if np.unique(X[:, j]).size <= 2]
            if len(binary_cols) > 1:
                out["mgwr"] = {
                    "status": "skipped",
                    "reason": (
                        f"MGWR skipped: {len(binary_cols)} binary/dummy columns detected "
                        f"({binary_cols}). Multiple sparse dummies cause near-singular local "
                        "matrices. Use at most one binary structural control and ensure all "
                        "other covariates are continuous."
                    ),
                }
            else:
                try:
                    # minimum bandwidth of 20 avoids singular local matrices at small samples.
                    min_bw = max(20, int(len(y) * 0.02))
                    mgwr_selector = Sel_BW(
                        coords, y, X,
                        multi=True, kernel="bisquare", fixed=False, spherical=False,
                    )
                    mgwr_bws = mgwr_selector.search(multi_bw_min=[min_bw], criterion="AICc")
                    mgwr_model = MGWR(
                        coords, y, X,
                        mgwr_selector,
                        kernel="bisquare", fixed=False, sigma2_v1=True, spherical=False,
                    )
                    mgwr_result = mgwr_model.fit()
                    mgwr_params = pd.DataFrame(
                        mgwr_result.params, columns=["Intercept", *names], index=idx
                    )
                    # mgwr_bws is an array of per-variable bandwidths (one per column
                    # including intercept).
                    bw_arr = np.asarray(mgwr_bws).ravel()
                    bw_map = dict(zip(["Intercept", *names], bw_arr.tolist()))
                    out["mgwr"] = {
                        "status": "fitted",
                        "bandwidths": bw_map,
                        "covariate_names": names,
                        "params": mgwr_params,
                        "model": mgwr_model,
                        "result": mgwr_result,
                        "summary": str(mgwr_result.summary()) if hasattr(mgwr_result, "summary") else None,
                    }
                except np.linalg.LinAlgError as mgwr_exc:
                    out["mgwr"] = {
                        "status": "skipped",
                        "reason": (
                            f"MGWR fitting failed (singular matrix): {mgwr_exc}. "
                            "Tip: use only continuous numeric covariates in gwr_spec['covariates']."
                        ),
                    }
                except Exception as mgwr_exc:
                    out["mgwr"] = {
                        "status": "skipped",
                        "reason": f"MGWR fitting failed: {mgwr_exc}",
                    }

        return out

    except Exception as exc:
        return {"status": "skipped", "reason": f"GWR fitting failed: {exc}"}


def summarise_gwr_results(gwr_results: Mapping[str, Any]) -> pd.DataFrame:
    """Return a tidy summary of fitted or skipped GWR and MGWR results.

    For GWR, all coefficients share a single global bandwidth.
    For MGWR, each coefficient has its own per-variable bandwidth, shown in
    the ``bandwidth`` column.
    """

    if gwr_results.get("status") != "fitted":
        return pd.DataFrame(
            [
                {
                    "model": "GWR",
                    "status": gwr_results.get("status", "unknown"),
                    "reason": gwr_results.get("reason", "No GWR result"),
                    "detail": gwr_results.get("detail"),
                }
            ]
        )

    rows: list[dict[str, Any]] = []

    # ── GWR coefficient summary ─────────────────────────────────────────────
    gwr_bw = gwr_results.get("bandwidth")
    params = gwr_results["params"]
    for col in params.columns:
        s = params[col]
        rows.append(
            {
                "model": "GWR",
                "status": "fitted",
                "coefficient": col,
                "bandwidth": gwr_bw,
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
                "mean": float(s.mean()),
            }
        )

    # ── MGWR coefficient summary ────────────────────────────────────────────
    mgwr = gwr_results.get("mgwr")
    if mgwr is not None:
        if mgwr.get("status") != "fitted":
            rows.append(
                {
                    "model": "MGWR",
                    "status": mgwr.get("status", "unknown"),
                    "reason": mgwr.get("reason", "MGWR not fitted"),
                    "coefficient": None,
                    "bandwidth": None,
                }
            )
        else:
            bw_map: dict[str, float] = mgwr.get("bandwidths", {})
            mgwr_params = mgwr["params"]
            for col in mgwr_params.columns:
                s = mgwr_params[col]
                rows.append(
                    {
                        "model": "MGWR",
                        "status": "fitted",
                        "coefficient": col,
                        "bandwidth": bw_map.get(col),
                        "min": float(s.min()),
                        "q25": float(s.quantile(0.25)),
                        "median": float(s.median()),
                        "q75": float(s.quantile(0.75)),
                        "max": float(s.max()),
                        "mean": float(s.mean()),
                    }
                )

    return pd.DataFrame(rows)


def _make_cv_splits(
    data: pd.DataFrame,
    strategy: Mapping[str, Any],
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct train/test indices for a CV strategy."""

    strategy_type = strategy.get("type")
    if strategy_type == "random_kfold":
        kf = KFold(n_splits=clip_n_splits(n_splits, len(data)), shuffle=True, random_state=random_state)
        return list(kf.split(data))

    if strategy_type == "group_kfold":
        group_key = strategy.get("group_key")
        if not group_key or group_key not in data.columns:
            if strategy.get("optional", False):
                return []
            raise LabDataError(f"Group CV requested but group_key is missing: {group_key}")
        groups = data[group_key].astype("string").fillna("missing").to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return []
        gkf = GroupKFold(n_splits=clip_n_splits(n_splits, len(data), n_groups))
        return list(gkf.split(data, groups=groups))

    if strategy_type == "spatial_clusters":
        if not {"x_pttm06_m", "y_pttm06_m"}.issubset(data.columns):
            return []
        n_clusters = clip_n_splits(n_splits, len(data))
        coords = data[["x_pttm06_m", "y_pttm06_m"]].to_numpy()
        clusters = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(coords)
        gkf = GroupKFold(n_splits=n_clusters)
        return list(gkf.split(data, groups=clusters))

    raise LabDataError(f"Unsupported CV strategy type: {strategy_type}")


def _cv_feature_columns(data: pd.DataFrame, outcome: str, feature_set: str) -> tuple[list[str], list[str], list[str]]:
    """Select safe CV feature columns and report excluded leakage-sensitive columns."""

    include_spatial = feature_set != "hedonic_controls_only"
    numeric, categorical = _default_feature_columns(
        data,
        outcome=outcome,
        include_spatial=include_spatial,
        include_leakage_sensitive=False,
    )
    leakage = [
        c
        for c in data.columns
        if c.endswith(config.LEAKAGE_SUFFIX) or c in getattr(data, "attrs", {}).get("leakage_sensitive_features", [])
    ]
    return numeric, categorical, sorted(set(leakage))


def fit_ml_models_with_importance(
    features: gpd.GeoDataFrame,
    ml_spec: Mapping[str, Any],
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Any]:
    """Fit Ridge, Random Forest, and LightGBM on the full dataset.

    Returns in-sample metrics, raw feature importances (sorted), and the
    fitted pipeline objects.  Feature names are derived from the sklearn
    ColumnTransformer so they are consistent with CV pipelines.
    """

    outcome = ml_spec.get("outcome", config.DEFAULT_AVEIRO_OUTCOME)
    require_columns(features, [outcome], "features")

    data = features.dropna(subset=[outcome]).copy()
    y = pd.to_numeric(data[outcome], errors="coerce")
    valid = y.notna()
    data, y = data.loc[valid].copy(), y.loc[valid].to_numpy(dtype=float)

    feature_set = ml_spec.get("feature_set", "hedonic_plus_spatial_features")
    numeric_cols, categorical_cols, excluded = _cv_feature_columns(data, outcome, feature_set)

    families: list[str] = ["regularized_linear", "random_forest", "lightgbm"]
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for family in families:
        try:
            pipe = _build_pipeline(numeric_cols, categorical_cols, family, random_state)
            pipe.fit(data, y)
            yhat = pipe.predict(data)
            metric_rows.append({"model": family, "n": len(y), **_metric_record(y, yhat)})
            fitted[family] = pipe

            # Recover feature names from the ColumnTransformer
            pre: ColumnTransformer = pipe.named_steps["preprocess"]
            feat_names: list[str] = []
            if "num" in pre.named_transformers_:
                feat_names.extend(numeric_cols)
            if "cat" in pre.named_transformers_:
                ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["onehot"]
                feat_names.extend(ohe.get_feature_names_out(categorical_cols).tolist())

            mdl = pipe.named_steps["model"]
            if hasattr(mdl, "feature_importances_"):
                raw = mdl.feature_importances_
            elif hasattr(mdl, "coef_"):
                raw = np.abs(mdl.coef_)
            else:
                raw = np.zeros(len(feat_names))

            total = raw.sum() or 1.0
            for fn, imp in zip(feat_names, raw):
                importance_rows.append({"model": family, "feature": fn, "importance": float(imp / total)})

        except Exception as exc:
            metric_rows.append({"model": family, "n": 0, "rmse": np.nan, "mae": np.nan,
                                 "r2": np.nan, "status": "failed", "reason": str(exc)})

    imp_df = pd.DataFrame(importance_rows)

    return {
        "status": "computed",
        "outcome": outcome,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "excluded_leakage_sensitive_features": excluded,
        "in_sample_metrics": pd.DataFrame(metric_rows),
        "feature_importances": imp_df,
        "fitted_models": fitted,
        "n": int(len(y)),
    }


def compare_random_and_spatial_cv(
    features: gpd.GeoDataFrame,
    ml_spec: Mapping[str, Any],
    cv_spec: Mapping[str, Any],
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Any]:
    """Compare random and spatial/block validation strategies."""

    outcome = ml_spec.get("outcome", config.DEFAULT_AVEIRO_OUTCOME)
    require_columns(features, [outcome], "features")

    data = features.dropna(subset=[outcome]).copy()
    y = pd.to_numeric(data[outcome], errors="coerce")
    valid = y.notna()
    data = data.loc[valid].copy()
    y = y.loc[valid].to_numpy(dtype=float)

    feature_set = ml_spec.get("feature_set", "hedonic_plus_spatial_features")
    numeric_cols, categorical_cols, excluded = _cv_feature_columns(data, outcome, feature_set)
    strategies = list(cv_spec.get("strategies", [{"name": "random_cv", "type": "random_kfold"}]))
    n_splits = int(ml_spec.get("n_splits", 5))

    model_families = [
        ml_spec.get("baseline_model_family", "regularized_linear"),
        ml_spec.get("model_family", "random_forest"),
    ]
    model_families.extend(ml_spec.get("additional_model_families", []))
    model_families = list(dict.fromkeys(model_families))

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []

    for strategy in strategies:
        strategy_name = str(strategy.get("name", strategy.get("type", "cv")))
        splits = _make_cv_splits(data, strategy, n_splits=n_splits, random_state=random_state)
        if not splits:
            skipped.append({"strategy": strategy_name, "reason": "No valid splits; optional strategy skipped"})
            continue

        for model_family in model_families:
            for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
                train = data.iloc[train_idx]
                test = data.iloc[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]

                pipe = _build_pipeline(
                    numeric_cols,
                    categorical_cols,
                    model_family=model_family,
                    random_state=random_state,
                    standardize_numeric=True,
                )
                pipe.fit(train, y_train)
                y_pred = pipe.predict(test)
                metrics = _metric_record(y_test, y_pred)
                metric_rows.append(
                    {
                        "strategy": strategy_name,
                        "strategy_type": strategy.get("type"),
                        "model_family": model_family,
                        "fold": fold_id,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        **metrics,
                    }
                )
                pred = pd.DataFrame(
                    {
                        "row_index": test.index,
                        "strategy": strategy_name,
                        "model_family": model_family,
                        "fold": fold_id,
                        "y_true": y_test,
                        "y_pred": y_pred,
                        "residual": y_test - y_pred,
                    },
                    index=test.index,
                )
                for id_col in ["listing_id", "zone_id", "parish_id"]:
                    if id_col in test.columns:
                        pred[id_col] = test[id_col].to_numpy()
                pred_rows.append(pred)

    predictions = pd.concat(pred_rows, axis=0).reset_index(drop=True) if pred_rows else pd.DataFrame()
    return {
        "status": "computed" if metric_rows else "skipped",
        "outcome": outcome,
        "feature_set": feature_set,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "excluded_leakage_sensitive_features": excluded,
        "fold_metrics": pd.DataFrame(metric_rows),
        "predictions": predictions,
        "skipped": pd.DataFrame(skipped),
    }


def summarise_cv_comparison(cv_results: Mapping[str, Any]) -> pd.DataFrame:
    """Summarise CV metrics by strategy and model."""

    metrics = cv_results.get("fold_metrics", pd.DataFrame())
    if metrics is None or metrics.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "No CV metrics available"}])

    summary = (
        metrics.groupby(["strategy", "strategy_type", "model_family"], dropna=False)
        .agg(
            folds=("fold", "nunique"),
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_n_test=("n_test", "mean"),
        )
        .reset_index()
        .sort_values(["model_family", "mean_rmse", "strategy"])
    )
    return summary


def diagnose_prediction_residuals(
    cv_results: Mapping[str, Any],
    features: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame | None,
    diagnostic_spec: Mapping[str, Any],
    random_state: int = config.RANDOM_STATE,
    metric_crs: str = config.METRIC_CRS,
) -> pd.DataFrame:
    """Diagnose spatial autocorrelation in cross-validated prediction residuals."""

    preds = cv_results.get("predictions", pd.DataFrame())
    if preds is None or preds.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "No cross-validated predictions available"}])

    specs = list(diagnostic_spec.get("weights", [{"name": "listing_knn_8", "type": "knn", "k": 8, "unit": "listing"}]))
    permutations = int(diagnostic_spec.get("permutations", 199))
    rows: list[dict[str, Any]] = []

    for (strategy, model_family), part in preds.groupby(["strategy", "model_family"]):
        for spec in specs:
            try:
                unit = spec.get("unit")
                if unit == "listing":
                    idx = part["row_index"].to_numpy()
                    g = features.loc[idx].copy()
                    w = knn_weights(g, k=int(spec.get("k", 8)), metric_crs=metric_crs, name=str(spec.get("name", "listing_knn")))
                    residuals = part["residual"].to_numpy()
                    mi = moran_i(residuals, w, permutations=permutations, random_state=random_state)
                elif unit == "zone":
                    if zones is None or "zone_id" not in part.columns:
                        raise LabDataError("Zone residual diagnostic requires zones and zone_id predictions.")
                    agg = part.groupby("zone_id", dropna=True)["residual"].mean().rename("mean_residual").reset_index()
                    z = zones.merge(agg, on="zone_id", how="inner")
                    if len(z) < 3:
                        raise LabDataError("Too few zones with residuals for zone-level Moran's I.")
                    w = queen_weights(z, metric_crs=metric_crs, id_col="zone_id", name=str(spec.get("name", "zone_queen")))
                    mi = moran_i(z["mean_residual"].to_numpy(), w, permutations=permutations, random_state=random_state)
                else:
                    raise LabDataError(f"Unsupported residual diagnostic unit: {unit}")

                rows.append(
                    {
                        "strategy": strategy,
                        "model_family": model_family,
                        "diagnostic": spec.get("name"),
                        "unit": unit,
                        "status": "computed",
                        **mi,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "strategy": strategy,
                        "model_family": model_family,
                        "diagnostic": spec.get("name"),
                        "unit": spec.get("unit"),
                        "status": "skipped",
                        "reason": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def compare_municipal_spatial_blocks(
    municipal_table: gpd.GeoDataFrame,
    municipal_spec: Mapping[str, Any],
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Run a compact municipal Track B validation comparison."""

    outcomes = list(municipal_spec.get("outcomes", municipal_table.attrs.get("outcomes", config.MUNICIPAL_TARGETS[:2])))
    covariates = [c for c in municipal_spec.get("candidate_covariates", municipal_table.attrs.get("candidate_covariates", config.MUNICIPAL_COVARIATES)) if c in municipal_table.columns]

    if not covariates:
        return pd.DataFrame([{"status": "skipped", "reason": "No candidate municipal covariates available"}])

    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome not in municipal_table.columns:
            rows.append({"outcome": outcome, "status": "skipped", "reason": "outcome missing"})
            continue

        data = municipal_table.dropna(subset=[outcome, *covariates]).copy()
        if len(data) < 20:
            rows.append({"outcome": outcome, "status": "skipped", "reason": "too few complete municipalities"})
            continue

        # Add spatial clusters from centroids.
        coords = np.column_stack([data.geometry.centroid.x.to_numpy(), data.geometry.centroid.y.to_numpy()])
        n_clusters = min(5, len(data))
        data["_spatial_cluster"] = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(coords)
        y = data[outcome].to_numpy(dtype=float)

        strategies = {
            "random_cv": list(KFold(n_splits=5, shuffle=True, random_state=random_state).split(data)),
        }
        if "nuts3" in data.columns and data["nuts3"].nunique() >= 2:
            groups = data["nuts3"].astype("string").to_numpy()
            strategies["nuts3_block_cv"] = list(GroupKFold(n_splits=min(5, data["nuts3"].nunique())).split(data, groups=groups))
        strategies["spatial_cluster_cv"] = list(GroupKFold(n_splits=n_clusters).split(data, groups=data["_spatial_cluster"]))

        for strategy, splits in strategies.items():
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                pipe = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
                    ]
                )
                X_train = data.iloc[train_idx][covariates]
                X_test = data.iloc[test_idx][covariates]
                y_train = y[train_idx]
                y_test = y[test_idx]
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                rows.append(
                    {
                        "status": "computed",
                        "outcome": outcome,
                        "strategy": strategy,
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        **_metric_record(y_test, y_pred),
                    }
                )

    return pd.DataFrame(rows)


def run_exploratory_diagnostics(
    features: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame | None = None,
    outcome: str = config.DEFAULT_AVEIRO_OUTCOME,
    metric_crs: str = config.METRIC_CRS,
) -> dict[str, Any]:
    """Convenience wrapper for initial spatial diagnostics."""

    out: dict[str, Any] = {"outcome": outcome}
    if outcome in features.columns:
        w = knn_weights(features.dropna(subset=[outcome]), k=8, metric_crs=metric_crs, name="listing_knn_8")
        out["listing_moran"] = moran_i(features.dropna(subset=[outcome])[outcome].to_numpy(), w, permutations=199)
    if zones is not None and "zone_id" in features.columns and outcome in features.columns:
        agg = features.groupby("zone_id")[outcome].mean().rename("zone_mean").reset_index()
        z = zones.merge(agg, on="zone_id", how="inner")
        if len(z) >= 3:
            w = queen_weights(z, metric_crs=metric_crs, id_col="zone_id", name="zone_queen")
            out["zone_moran"] = moran_i(z["zone_mean"].to_numpy(), w, permutations=199)
    return out


def fit_baseline_models(features: gpd.GeoDataFrame, outcome: str = config.DEFAULT_AVEIRO_OUTCOME, random_state: int = config.RANDOM_STATE) -> dict[str, Any]:
    """Notebook-facing alias for a simple global reference fit."""

    return fit_global_reference_model(
        features,
        {
            "outcome": outcome,
            "feature_sets": ["hedonic_controls_only", "hedonic_plus_spatial_features"],
            "model_family": "regularized_linear",
            "standardize_numeric": True,
        },
        random_state=random_state,
    )


def fit_spatial_models(
    features: gpd.GeoDataFrame,
    gwr_spec: Mapping[str, Any] | None = None,
    random_state: int = config.RANDOM_STATE,
    metric_crs: str = config.METRIC_CRS,
) -> dict[str, Any]:
    """Notebook-facing wrapper for optional spatial/local models."""

    return {"gwr": fit_gwr_if_available(features, gwr_spec or {"enabled": True}, random_state=random_state, metric_crs=metric_crs)}
