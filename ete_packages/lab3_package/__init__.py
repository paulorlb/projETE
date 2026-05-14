"""
lab3_package — helper library for Sessions 3-4: Spatial Econometric Models.

Re-exports every public function and constant so notebooks can do either:
    from lab3_package import *
or targeted imports:
    from lab3_package.models import fit_spreg_model
"""
from .config import (
    RANDOM_SEED, N_PERMUTATIONS,
    REQUIRED_PACKAGES, OPTIONAL_PACKAGES,
    PROJECTED_CRS, GEOGRAPHIC_CRS,
    GPKG_FILENAME,
    LISTINGS_LAYER_EXPECTED, ZONES_LAYER_EXPECTED, FREGUESIA_LAYER_EXPECTED,
    MUNICIPAL_LAYER_EXPECTED, MUNICIPAL_BOUNDARY_FALLBACK_EXPECTED,
    TARGET_PRICE, TARGET_UNIT_PRICE, AREA_VAR, ZONE_ID,
    MUNICIPAL_PRICE_TARGETS,
    MUNICIPAL_SOCIO_ECONOMIC_INDICATORS,
    MUNICIPAL_GEOGRAPHIC_INDICATORS,
)

from .utils import package_status, display_table, show_object

from .paths import candidate_roots, find_project_root, find_file

from .io import (
    import_analysis_stack,
    import_model_stack,
    import_lisa_stack,
    quote_sql_identifier,
    sqlite_scalar,
    table_columns,
    inspect_gpkg_layers,
    detect_layer,
    summarize_layer_columns,
    layer_summary_rows,
    require_projected_crs,
    read_gpkg_layer,
    repair_invalid_geometries,
    to_metric_crs,
)

from .prep import (
    existing_columns,
    missingness_summary,
    value_counts_table,
    duplicate_summary,
    positive_value_summary,
    numeric_summary,
    add_log_if_positive,
    add_price_logs,
    add_iqr_outlier_flag,
    add_listing_outlier_flags,
    infer_condition_score,
    slugify_label,
    category_share_table,
    add_category_shares,
    aggregate_listing_support,
    fill_support_count_flags,
    spatial_join_with_fallback,
    available_share_columns,
    add_dummy_columns,
    select_share_predictors,
    valid_numeric_predictors,
    prepare_model_frame,
)

from .explore import (
    plot_layer_map,
    plot_point_map,
    plot_lisa_map,
    high_price_support_table,
    high_price_municipal_table,
)

from .weights import (
    import_weights_stack,
    weights_are_symmetric,
    weights_components,
    diagnose_weights,
    build_contiguity_weights,
    build_knn_weights,
    plot_neighbor_graph,
)

from .models import (
    subset_weight_to_index,
    residual_moran_table,
    compute_lisa,
    fit_hc3_ols,
    tidy_statsmodels_coefficients,
    ols_comparison_row,
    run_spreg_ols_diagnostics,
    build_lm_decision_table,
    fit_slx_statsmodels,
    fit_spreg_model,
    spreg_model_summary_row,
    tidy_spreg_coefficients,
    selected_interpretable_predictor,
    add_wx_columns,
    fit_sdm_model,
    fit_sdem_model,
    dense_w_matrix,
    coefficient_lookup_spreg,
    spatial_parameter,
    compute_spatial_impacts,
    run_ols_robustness_matrix,
)

__all__ = [
    # config
    "RANDOM_SEED", "N_PERMUTATIONS",
    "REQUIRED_PACKAGES", "OPTIONAL_PACKAGES",
    "PROJECTED_CRS", "GEOGRAPHIC_CRS", "GPKG_FILENAME",
    "LISTINGS_LAYER_EXPECTED", "ZONES_LAYER_EXPECTED",
    "FREGUESIA_LAYER_EXPECTED", "MUNICIPAL_LAYER_EXPECTED",
    "MUNICIPAL_BOUNDARY_FALLBACK_EXPECTED",
    "TARGET_PRICE", "TARGET_UNIT_PRICE", "AREA_VAR", "ZONE_ID",
    "MUNICIPAL_PRICE_TARGETS", "MUNICIPAL_SOCIO_ECONOMIC_INDICATORS",
    "MUNICIPAL_GEOGRAPHIC_INDICATORS",
    # utils
    "package_status", "display_table", "show_object",
    # paths
    "candidate_roots", "find_project_root", "find_file",
    # io
    "import_analysis_stack", "import_model_stack", "import_lisa_stack",
    "quote_sql_identifier", "sqlite_scalar", "table_columns",
    "inspect_gpkg_layers", "detect_layer", "summarize_layer_columns",
    "layer_summary_rows", "require_projected_crs",
    "read_gpkg_layer", "repair_invalid_geometries", "to_metric_crs",
    # prep
    "existing_columns", "missingness_summary", "value_counts_table",
    "duplicate_summary", "positive_value_summary", "numeric_summary",
    "add_log_if_positive", "add_price_logs",
    "add_iqr_outlier_flag", "add_listing_outlier_flags",
    "infer_condition_score", "slugify_label",
    "category_share_table", "add_category_shares",
    "aggregate_listing_support", "fill_support_count_flags",
    "spatial_join_with_fallback", "available_share_columns",
    "add_dummy_columns", "select_share_predictors",
    "valid_numeric_predictors", "prepare_model_frame",
    # explore
    "plot_layer_map", "plot_point_map", "plot_lisa_map",
    "high_price_support_table", "high_price_municipal_table",
    # weights
    "import_weights_stack", "weights_are_symmetric", "weights_components",
    "diagnose_weights", "build_contiguity_weights", "build_knn_weights",
    "plot_neighbor_graph",
    # models
    "subset_weight_to_index", "residual_moran_table", "compute_lisa",
    "fit_hc3_ols", "tidy_statsmodels_coefficients", "ols_comparison_row",
    "run_spreg_ols_diagnostics", "build_lm_decision_table",
    "fit_slx_statsmodels", "fit_spreg_model",
    "spreg_model_summary_row", "tidy_spreg_coefficients",
    "selected_interpretable_predictor", "add_wx_columns",
    "fit_sdm_model", "fit_sdem_model",
    "dense_w_matrix", "coefficient_lookup_spreg", "spatial_parameter",
    "compute_spatial_impacts", "run_ols_robustness_matrix",
]
