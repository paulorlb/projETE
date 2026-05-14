"""
Statistical models: OLS, SLX, SAR/SLM, SEM, SDM, SDEM, Moran's I, LM diagnostics,
spatial impact decomposition, and robustness matrices.
"""
from __future__ import annotations

import importlib

from .config import N_PERMUTATIONS
from .io import import_analysis_stack, import_lisa_stack, import_model_stack
from .prep import existing_columns, valid_numeric_predictors, prepare_model_frame


# ---------------------------------------------------------------------------
# Weight subsetting helper
# ---------------------------------------------------------------------------

def subset_weight_to_index(w, index_values):
    """Return a row-standardised sub-matrix of *w* restricted to *index_values*."""
    _, _, _, w_subset = import_model_stack()
    ids = list(index_values)
    try:
        subset = w_subset(w, ids)
    except Exception:
        subset = w
    subset.transform = "R"
    return subset


# ---------------------------------------------------------------------------
# Moran's I on a vector (outcome or residuals)
# ---------------------------------------------------------------------------

def residual_moran_table(residuals, weights_dict: dict, label_prefix: str):
    _, pd, _, _ = import_analysis_stack()
    _, Moran, _, _ = import_model_stack()
    rows = []
    residuals = residuals.dropna()
    for key, w in weights_dict.items():
        try:
            w_sub = subset_weight_to_index(w, residuals.index)
            values = residuals.loc[w_sub.id_order].astype(float).values
            moran = Moran(values, w_sub, permutations=N_PERMUTATIONS)
            rows.append({
                "track": label_prefix,
                "weights": key,
                "N": len(values),
                "moran_i": moran.I,
                "expected_i": moran.EI,
                "p_norm": getattr(moran, "p_norm", None),
                "p_sim": getattr(moran, "p_sim", None),
                "permutations": N_PERMUTATIONS,
            })
        except Exception as exc:
            rows.append({
                "track": label_prefix,
                "weights": key,
                "N": len(residuals),
                "moran_i": None,
                "expected_i": None,
                "p_norm": None,
                "p_sim": None,
                "permutations": N_PERMUTATIONS,
                "error": str(exc),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LISA — Local Moran's I
# ---------------------------------------------------------------------------

def compute_lisa(series, w, permutations: int = N_PERMUTATIONS):
    """Compute Local Moran's I (LISA) for a Series aligned to a weight matrix.

    Returns a DataFrame indexed like *series* with columns:
      local_i, z_score, p_sim, quadrant (HH/LH/LL/HL), cluster (quadrant or 'ns').
    PySAL quadrant codes: 1=HH, 2=LH, 3=LL, 4=HL.
    """
    np, pd, _, _ = import_analysis_stack()
    Moran_Local = import_lisa_stack()

    series = series.dropna()
    w_sub = subset_weight_to_index(w, series.index)
    values = series.loc[w_sub.id_order].astype(float).values
    lisa = Moran_Local(values, w_sub, permutations=permutations)

    quad_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    cluster = [
        quad_labels.get(int(q), "ns") if p < 0.05 else "ns"
        for q, p in zip(lisa.q, lisa.p_sim)
    ]
    return pd.DataFrame(
        {
            "local_i": lisa.Is,
            "z_score": lisa.z_sim,
            "p_sim": lisa.p_sim,
            "quadrant": [quad_labels.get(int(q), "ns") for q in lisa.q],
            "cluster": cluster,
        },
        index=pd.Index(w_sub.id_order, name=series.index.name),
    )


# ---------------------------------------------------------------------------
# OLS (HC3) and coefficient tidying
# ---------------------------------------------------------------------------

def fit_hc3_ols(model_df, y_col: str, x_cols: list[str]):
    """Fit HC3-robust OLS; return (result, residuals_series)."""
    sm, _, _, _ = import_model_stack()
    y = model_df[y_col].astype(float)
    x = sm.add_constant(model_df[x_cols].astype(float), has_constant="add")
    model = sm.OLS(y, x).fit(cov_type="HC3")
    residuals = model.resid.rename("ols_residual")
    return model, residuals


def tidy_statsmodels_coefficients(result, model_name: str):
    _, pd, _, _ = import_analysis_stack()
    return pd.DataFrame({
        "model": model_name,
        "term": result.params.index,
        "estimate": result.params.values,
        "std_error": result.bse.values,
        "p_value": result.pvalues.values,
    })


def ols_comparison_row(result, model_name: str, residual_moran=None) -> dict:
    return {
        "model": model_name,
        "n": int(result.nobs),
        "k": int(result.df_model),
        "status": "estimated",
        "r2": getattr(result, "rsquared", None),
        "adj_r2": getattr(result, "rsquared_adj", None),
        "aic": getattr(result, "aic", None),
        "bic": getattr(result, "bic", None),
        "residual_moran_queen": residual_moran,
    }


# ---------------------------------------------------------------------------
# LM diagnostics (spreg OLS)
# ---------------------------------------------------------------------------

def run_spreg_ols_diagnostics(model_df, y_col: str, x_cols: list[str], w, model_label: str):
    _, pd, _, _ = import_analysis_stack()
    if importlib.util.find_spec("spreg") is None:
        return pd.DataFrame([{
            "diagnostic": "spreg", "value": None, "p_value": None, "note": "spreg not installed"
        }])
    import spreg

    w_sub = subset_weight_to_index(w, model_df.index)
    y = model_df.loc[w_sub.id_order, y_col].astype(float).values.reshape((-1, 1))
    x = model_df.loc[w_sub.id_order, x_cols].astype(float).values

    try:
        model = spreg.OLS(
            y, x, w=w_sub,
            name_y=y_col, name_x=x_cols, name_ds=model_label,
            spat_diag=True, moran=True,
        )
    except TypeError:
        model = spreg.OLS(y, x, w=w_sub, name_y=y_col, name_x=x_cols, spat_diag=True)
    except Exception as exc:
        return pd.DataFrame([{"diagnostic": "spreg.OLS", "value": None, "p_value": None, "note": str(exc)}])

    diagnostic_names = {
        "lm_lag": "LM-Lag",
        "rlm_lag": "Robust LM-Lag",
        "lm_error": "LM-Error",
        "rlm_error": "Robust LM-Error",
        "lm_sarma": "LM-SARMA",
        "moran_res": "Moran residual diagnostic",
    }
    rows = []
    for attr, label in diagnostic_names.items():
        value = getattr(model, attr, None)
        if value is None:
            rows.append({"diagnostic": label, "statistic": None, "p_value": None, "note": f"{attr} not exposed"})
            continue
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            rows.append({"diagnostic": label, "statistic": value[0], "p_value": value[1], "note": None})
        else:
            rows.append({"diagnostic": label, "statistic": None, "p_value": None, "note": str(value)})
    return pd.DataFrame(rows)


def build_lm_decision_table(lm_df, residual_moran_df, track_label: str):
    _, pd, _, _ = import_analysis_stack()

    def p_for(name):
        if lm_df is None or lm_df.empty or "diagnostic" not in lm_df.columns:
            return None
        rows = lm_df.loc[lm_df["diagnostic"].eq(name)]
        if rows.empty or "p_value" not in rows.columns:
            return None
        return rows["p_value"].iloc[0]

    p_lag = p_for("LM-Lag")
    p_rlag = p_for("Robust LM-Lag")
    p_err = p_for("LM-Error")
    p_rerr = p_for("Robust LM-Error")

    queen_moran = None
    if residual_moran_df is not None and not residual_moran_df.empty:
        queen_rows = residual_moran_df[
            residual_moran_df["weights"].astype(str).str.contains("queen", case=False)
        ]
        if not queen_rows.empty:
            queen_moran = queen_rows["moran_i"].iloc[0]

    if p_rlag is not None and p_rerr is not None and p_rlag < 0.05 and p_rerr >= 0.05:
        suggested = "SAR/SLM signal"
    elif p_rlag is not None and p_rerr is not None and p_rerr < 0.05 and p_rlag >= 0.05:
        suggested = "SEM signal"
    elif p_rlag is not None and p_rerr is not None and p_rlag < 0.05 and p_rerr < 0.05:
        suggested = "Ambiguous robust LM signals"
    elif p_lag is not None and p_err is not None and p_lag < 0.05 and p_err < 0.05:
        suggested = "Both simple LM tests reject; use robust tests and theory"
    else:
        suggested = "No strong SAR/SEM signal; consider OLS/SLX and robustness checks"

    return pd.DataFrame([{
        "track": track_label,
        "residual_moran_queen": queen_moran,
        "LM-Lag p": p_lag,
        "Robust LM-Lag p": p_rlag,
        "LM-Error p": p_err,
        "Robust LM-Error p": p_rerr,
        "diagnostic_reading": suggested,
        "caution": "LM tests guide model comparison; they are not automatic selectors.",
    }])


# ---------------------------------------------------------------------------
# SLX (spatial lag of X, estimated by OLS)
# ---------------------------------------------------------------------------

def fit_slx_statsmodels(model_df, y_col: str, x_cols: list[str], w, model_label: str):
    """Add WX columns and estimate SLX via HC3 OLS; return (result, residuals, wx_cols)."""
    _, pd, _, _ = import_analysis_stack()
    _, _, lag_spatial, _ = import_model_stack()

    w_sub = subset_weight_to_index(w, model_df.index)
    ordered = model_df.loc[w_sub.id_order].copy()
    wx_cols = []
    for col in x_cols:
        wx_col = f"W_{col}"
        ordered[wx_col] = lag_spatial(w_sub, ordered[col].astype(float).values)
        wx_cols.append(wx_col)
    result, residuals = fit_hc3_ols(ordered, y_col, x_cols + wx_cols)
    return result, residuals, wx_cols


# ---------------------------------------------------------------------------
# SAR / SEM (ML via spreg)
# ---------------------------------------------------------------------------

def fit_spreg_model(model_df, y_col: str, x_cols: list[str], w, model_type: str, model_label: str):
    """Fit SAR (ML_Lag) or SEM (ML_Error) via spreg; return (model, residuals, summary_dict)."""
    _, pd, _, _ = import_analysis_stack()
    if importlib.util.find_spec("spreg") is None:
        return None, pd.Series(dtype=float), {"model": model_label, "status": "spreg not installed"}
    import spreg

    w_sub = subset_weight_to_index(w, model_df.index)
    ordered = model_df.loc[w_sub.id_order].copy()
    y = ordered[y_col].astype(float).values.reshape((-1, 1))
    x = ordered[x_cols].astype(float).values

    try:
        if model_type.lower() == "sar":
            model = spreg.ML_Lag(y, x, w=w_sub, name_y=y_col, name_x=x_cols, name_ds=model_label)
        elif model_type.lower() == "sem":
            model = spreg.ML_Error(y, x, w=w_sub, name_y=y_col, name_x=x_cols, name_ds=model_label)
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")
    except Exception as exc:
        return None, pd.Series(dtype=float), {"model": model_label, "status": str(exc)}

    residual_values = getattr(model, "u", None)
    residuals = pd.Series(
        residual_values.flatten() if residual_values is not None else [],
        index=ordered.index if residual_values is not None else [],
        name=f"{model_label}_residual",
    )
    return model, residuals, spreg_model_summary_row(model, model_label, len(ordered), len(x_cols))


def spreg_model_summary_row(model, model_label: str, n: int, k: int) -> dict:
    return {
        "model": model_label,
        "n": n,
        "k": k,
        "status": "estimated",
        "pseudo_r2": getattr(model, "pr2", None),
        "log_likelihood": getattr(model, "logll", None),
        "aic": getattr(model, "aic", None),
        "schwarz": getattr(model, "schwarz", None),
        "rho": getattr(model, "rho", None),
        "lambda": getattr(model, "lam", None),
    }


def tidy_spreg_coefficients(model, model_label: str):
    _, pd, _, _ = import_analysis_stack()
    if model is None:
        return pd.DataFrame()
    names = getattr(model, "name_x", None)
    betas = getattr(model, "betas", None)
    z_stats = getattr(model, "z_stat", None)
    if betas is None:
        return pd.DataFrame()
    estimates = betas.flatten()
    if not names or len(names) != len(estimates):
        names = [f"beta_{i}" for i in range(len(estimates))]
    rows = []
    for i, term in enumerate(names):
        row = {"model": model_label, "term": term, "estimate": estimates[i]}
        if z_stats and i < len(z_stats):
            row["z_or_t"] = z_stats[i][0]
            row["p_value"] = z_stats[i][1]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SDM and SDEM helpers
# ---------------------------------------------------------------------------

def selected_interpretable_predictor(x_cols: list[str], priorities: list[str]) -> str | None:
    for priority in priorities:
        for col in x_cols:
            if priority.lower() in col.lower():
                return col
    return x_cols[0] if x_cols else None


def add_wx_columns(model_df, x_cols: list[str], w, selected_x_cols: list[str] | None = None):
    """Add spatially-lagged WX columns; return (ordered_df, wx_col_names, w_sub)."""
    _, pd, _, _ = import_analysis_stack()
    _, _, lag_spatial, _ = import_model_stack()
    selected_x_cols = selected_x_cols or x_cols
    w_sub = subset_weight_to_index(w, model_df.index)
    ordered = model_df.loc[w_sub.id_order].copy()
    wx_cols = []
    for col in selected_x_cols:
        if col not in ordered.columns:
            continue
        wx_col = f"W_{col}"
        ordered[wx_col] = lag_spatial(w_sub, ordered[col].astype(float).values)
        wx_cols.append(wx_col)
    return ordered, wx_cols, w_sub


def fit_sdm_model(model_df, y_col: str, x_cols: list[str], wx_source_cols: list[str], w, model_label: str):
    """Fit SDM (SAR + WX) via spreg ML_Lag; return (model, residuals, summary, wx_cols, w_sub)."""
    _, pd, _, _ = import_analysis_stack()
    ordered, wx_cols, w_sub = add_wx_columns(model_df, x_cols, w, selected_x_cols=wx_source_cols)
    sdm_x_cols = x_cols + wx_cols
    ordered, sdm_x_cols = prepare_model_frame(ordered, y_col, sdm_x_cols)
    if len(sdm_x_cols) > max(3, len(ordered) // 4):
        return None, pd.Series(dtype=float), {
            "model": model_label,
            "status": f"skipped: over-parameterised SDM with n={len(ordered)}, k={len(sdm_x_cols)}",
        }, [], w_sub
    model, residuals, summary = fit_spreg_model(ordered, y_col, sdm_x_cols, w_sub, "sar", model_label)
    summary["sdm_wx_terms"] = ", ".join(wx_cols)
    return model, residuals, summary, wx_cols, w_sub


def fit_sdem_model(model_df, y_col: str, x_cols: list[str], wx_source_cols: list[str], w, model_label: str):
    """Fit SDEM (SEM + WX) via spreg ML_Error; return (model, residuals, summary, wx_cols, w_sub)."""
    _, pd, _, _ = import_analysis_stack()
    ordered, wx_cols, w_sub = add_wx_columns(model_df, x_cols, w, selected_x_cols=wx_source_cols)
    sdem_x_cols = x_cols + wx_cols
    ordered, sdem_x_cols = prepare_model_frame(ordered, y_col, sdem_x_cols)
    if len(sdem_x_cols) > max(3, len(ordered) // 4):
        return None, pd.Series(dtype=float), {
            "model": model_label,
            "status": f"skipped: over-parameterised SDEM with n={len(ordered)}, k={len(sdem_x_cols)}",
        }, [], w_sub
    model, residuals, summary = fit_spreg_model(ordered, y_col, sdem_x_cols, w_sub, "sem", model_label)
    summary["sdem_wx_terms"] = ", ".join(wx_cols)
    return model, residuals, summary, wx_cols, w_sub


# ---------------------------------------------------------------------------
# Spatial impact decomposition (LeSage–Pace framework)
# ---------------------------------------------------------------------------

def dense_w_matrix(w, ids) -> "np.ndarray":
    np, _, _, _ = import_analysis_stack()
    ids = list(ids)
    position = {item: idx for idx, item in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    for i_id in ids:
        if i_id not in w.neighbors:
            continue
        i = position[i_id]
        for j_id, wij in zip(w.neighbors.get(i_id, []), w.weights.get(i_id, [])):
            if j_id in position:
                matrix[i, position[j_id]] = wij
    return matrix


def coefficient_lookup_spreg(model) -> dict:
    if model is None:
        return {}
    names = getattr(model, "name_x", None)
    betas = getattr(model, "betas", None)
    if betas is None:
        return {}
    values = betas.flatten()
    if not names or len(names) != len(values):
        names = [f"beta_{i}" for i in range(len(values))]
    return {name: values[i] for i, name in enumerate(names)}


def spatial_parameter(model) -> float | None:
    if model is None:
        return None
    rho = getattr(model, "rho", None)
    if rho is None:
        return None
    try:
        return float(rho)
    except Exception:
        try:
            return float(rho[0])
        except Exception:
            return None


def compute_spatial_impacts(
    model,
    w,
    ids,
    variables: list[str],
    wx_variables: list[str] | None = None,
    model_label: str = "model",
    y_is_log: bool = True,
):
    """
    Compute LeSage–Pace direct, indirect, and total impacts for SAR/SDM models.

    Parameters
    ----------
    model       : fitted spreg ML_Lag or ML_Error model
    w           : row-standardised spatial weights object
    ids         : index values matching the model observations
    variables   : list of explanatory variable names to decompose
    wx_variables: names of variables for which a WX term was included (SDM only)
    model_label : label for the output table
    y_is_log    : if True, impacts are approximately proportional (log-price scale)
    """
    np, pd, _, _ = import_analysis_stack()
    rho = spatial_parameter(model)
    if rho is None:
        return pd.DataFrame([{"model": model_label, "variable": None,
                               "note": "No spatial autoregressive parameter available."}])

    coef = coefficient_lookup_spreg(model)
    w_matrix = dense_w_matrix(w, ids)
    identity = np.eye(w_matrix.shape[0])
    try:
        multiplier = np.linalg.inv(identity - rho * w_matrix)
    except np.linalg.LinAlgError:
        return pd.DataFrame([{"model": model_label, "variable": None,
                               "note": "Could not invert spatial multiplier."}])

    rows = []
    wx_variables_set = set(wx_variables or [])
    for variable in variables:
        beta = coef.get(variable)
        theta = coef.get(f"W_{variable}", 0.0 if variable not in wx_variables_set else None)
        if beta is None or theta is None:
            rows.append({"model": model_label, "variable": variable,
                         "note": "Variable coefficient not found in fitted model."})
            continue

        impact_matrix = multiplier @ (beta * identity + theta * w_matrix)
        direct = float(np.diag(impact_matrix).mean())
        total = float(impact_matrix.sum(axis=1).mean())
        indirect = total - direct

        if y_is_log and "share" in variable.lower():
            scale_note = (
                "For a 10-pp change multiply impacts by 0.10; read as approx. % effects on price."
            )
        elif y_is_log:
            scale_note = "One-unit X change; impacts are approx. proportional effects on price."
        else:
            scale_note = "Impact is in outcome units."

        rows.append({
            "model": model_label,
            "variable": variable,
            "rho": rho,
            "beta": beta,
            "theta": theta,
            "direct_impact": direct,
            "indirect_impact": indirect,
            "total_impact": total,
            "y_is_log": y_is_log,
            "interpretation_note": scale_note,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Robustness matrix
# ---------------------------------------------------------------------------

def run_ols_robustness_matrix(
    base_gdf,
    y_options: list[str],
    x_cols: list[str],
    weights_dict: dict,
    min_n_values: list | None = None,
    outlier_col: str | None = None,
    fallback_col: str | None = None,
    label: str = "",
):
    """Run OLS + residual Moran across outcome variants, W specifications, and sample filters."""
    _, pd, _, _ = import_analysis_stack()
    rows = []
    min_n_values = min_n_values or [None]

    for y_col in y_options:
        if y_col not in base_gdf.columns:
            continue
        for min_n in min_n_values:
            df = base_gdf.copy()
            if min_n is not None and "n_listings" in df.columns:
                df = df.loc[df["n_listings"] >= min_n].copy()

            scenarios = [("base", df)]
            if outlier_col and outlier_col in df.columns:
                scenarios.append((
                    "exclude_high_outlier_share",
                    df.loc[df[outlier_col].fillna(0) <= 0.25].copy(),
                ))
            if fallback_col and fallback_col in df.columns:
                scenarios.append((
                    "exclude_high_fallback_share",
                    df.loc[df[fallback_col].fillna(0) <= 0.25].copy(),
                ))

            for scenario, scenario_df in scenarios:
                try:
                    model_df, valid_x = prepare_model_frame(scenario_df, y_col, x_cols)
                    result, residuals = fit_hc3_ols(model_df, y_col, valid_x)
                    moran_df = residual_moran_table(residuals, weights_dict, f"{label} robustness")
                    for _, moran_row in moran_df.iterrows():
                        rows.append({
                            "track": label,
                            "scenario": scenario,
                            "dependent_variable": y_col,
                            "min_n": min_n,
                            "weights": moran_row["weights"],
                            "n": int(result.nobs),
                            "k": len(valid_x),
                            "r2": result.rsquared,
                            "residual_moran_i": moran_row.get("moran_i"),
                            "residual_moran_p_sim": moran_row.get("p_sim"),
                            "status": "estimated",
                        })
                except Exception as exc:
                    rows.append({
                        "track": label,
                        "scenario": scenario,
                        "dependent_variable": y_col,
                        "min_n": min_n,
                        "weights": None,
                        "n": len(scenario_df),
                        "k": None,
                        "r2": None,
                        "residual_moran_i": None,
                        "residual_moran_p_sim": None,
                        "status": str(exc),
                    })
    return pd.DataFrame(rows)
