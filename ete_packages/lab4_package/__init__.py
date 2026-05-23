"""Public API for the Topic 4 Spatial Econometrics lab package.

The notebook should import this module as:

    from ete_packages import lab3_package as lab3

and then call the high-level functions exported here.
"""

from .config import *
from .paths import check_file_availability, discover_data_file, ensure_output_dir, find_project_root
from .io import (
    compare_inventory_to_schema_notes,
    inspect_geopackage,
    load_layers,
    read_csv,
    read_geopackage_layer,
)
from .prep import (
    build_aveiro_spatial_features,
    prepare_aveiro_model_table,
    prepare_municipal_market_table,
    summarise_aveiro_model_table,
    summarise_coordinate_quality,
    summarise_spatial_features,
)
from .figures import (
    plot_aveiro_orientation,
    plot_cv_metric_summary,
    plot_gwr_coefficient_map,
    plot_gwr_surface_if_available,
    plot_zone_dummy_coefficients,
)
from .weights import (
    SimpleWeights,
    build_weights_suite,
    knn_weights,
    moran_i,
    neighbour_graph_summary,
    queen_weights,
    weights_summary,
)
from .models import (
    compare_municipal_spatial_blocks,
    compare_random_and_spatial_cv,
    diagnose_global_reference_model,
    diagnose_prediction_residuals,
    fit_baseline_models,
    fit_global_reference_model,
    fit_gwr_if_available,
    fit_ml_models_with_importance,
    fit_spatial_models,
    run_exploratory_diagnostics,
    summarise_cv_comparison,
    summarise_gwr_results,
)
from .reporting import export_lab_outputs, package_versions, summarise_outputs, summarise_qa


def load_context(
    gpkg_path,
    layer_hints=None,
    metric_crs=METRIC_CRS,
    raw_geographic_crs=RAW_GEOGRAPHIC_CRS,
):
    """Notebook-facing convenience wrapper around ``load_layers``."""

    return load_layers(
        gpkg_path=gpkg_path,
        layer_hints=layer_hints or EXPECTED_LAYERS,
        metric_crs=metric_crs,
        raw_geographic_crs=raw_geographic_crs,
    )


def inspect_inputs(paths, gpkg_key="gpkg", layer_hints=None):
    """Return both file availability and GeoPackage inventory."""

    file_status = check_file_availability(paths)
    gpkg_path = paths.get(gpkg_key)
    inventory = inspect_geopackage(gpkg_path, layer_hints=layer_hints or EXPECTED_LAYERS) if gpkg_path else None
    return {"file_status": file_status, "inventory": inventory}


def prepare_track_a(listings, zones, parishes=None, model_spec=None, metric_crs=METRIC_CRS):
    """Prepare Track A and add default spatial features."""

    table = prepare_aveiro_model_table(listings, zones, parishes, model_spec=model_spec, metric_crs=metric_crs)
    features = build_aveiro_spatial_features(table, zones=zones, parishes=parishes, metric_crs=metric_crs)
    return {"model_table": table, "features": features}


def prepare_track_b(municipalities, municipal_spec=None, metric_crs=METRIC_CRS):
    """Prepare Track B municipal data."""

    return prepare_municipal_market_table(municipalities, municipal_spec=municipal_spec, metric_crs=metric_crs)


__all__ = [name for name in globals() if not name.startswith("_")]
