"""
Spatial weights construction, diagnostics, and neighbour-graph visualisation.
"""
from __future__ import annotations

import importlib

from .config import PROJECTED_CRS
from .io import import_analysis_stack, require_projected_crs


# ---------------------------------------------------------------------------
# Lazy import for libpysal weights
# ---------------------------------------------------------------------------

def import_weights_stack():
    """Import and return (Queen, Rook, KNN) from libpysal.weights."""
    if importlib.util.find_spec("libpysal") is None:
        raise ImportError(
            "Spatial weights sections require libpysal. Install it in the notebook kernel."
        )
    from libpysal.weights import KNN, Queen, Rook
    return Queen, Rook, KNN


# ---------------------------------------------------------------------------
# Internal diagnostics helpers
# ---------------------------------------------------------------------------

def weights_are_symmetric(w) -> bool:
    neighbors = w.neighbors
    for node, node_neighbors in neighbors.items():
        for neighbor in node_neighbors:
            if node not in neighbors.get(neighbor, []):
                return False
    return True


def weights_components(w) -> list[int]:
    nodes = list(w.neighbors.keys())
    adjacency = {node: set(w.neighbors.get(node, [])) for node in nodes}
    for node, node_neighbors in list(adjacency.items()):
        for neighbor in node_neighbors:
            adjacency.setdefault(neighbor, set()).add(node)

    seen: set = set()
    component_sizes = []
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency.get(current, []):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)
    return component_sizes


def diagnose_weights(w, name: str, symmetric_before_standardization: bool) -> dict:
    _, pd, _, _ = import_analysis_stack()
    cardinalities = list(w.cardinalities.values())
    component_sizes = weights_components(w)
    undirected_edges = {
        frozenset((node, neighbor))
        for node, neighbors in w.neighbors.items()
        for neighbor in neighbors
        if node != neighbor
    }
    return {
        "weights_name": name,
        "N": int(w.n),
        "n_links_directed": int(sum(cardinalities)),
        "n_links_undirected": int(len(undirected_edges)),
        "islands": list(w.islands),
        "n_islands": len(w.islands),
        "disconnected_components": len(component_sizes),
        "component_sizes": component_sizes,
        "min_cardinality": int(min(cardinalities)) if cardinalities else None,
        "mean_cardinality": float(pd.Series(cardinalities).mean()) if cardinalities else None,
        "median_cardinality": float(pd.Series(cardinalities).median()) if cardinalities else None,
        "max_cardinality": int(max(cardinalities)) if cardinalities else None,
        "symmetric_before_standardization": bool(symmetric_before_standardization),
        "row_standardization_status": w.transform,
    }


# ---------------------------------------------------------------------------
# Public weight builders
# ---------------------------------------------------------------------------

def build_contiguity_weights(gdf, kind: str, name: str) -> tuple:
    """Build Queen or Rook contiguity weights for *gdf*. Returns (name, w, diagnostic_dict)."""
    Queen, Rook, _ = import_weights_stack()
    require_projected_crs(gdf, PROJECTED_CRS, name)
    cls = Queen if kind.lower() == "queen" else Rook
    try:
        w = cls.from_dataframe(gdf, use_index=True)
    except TypeError:
        w = cls.from_dataframe(gdf)
    symmetric_pre = weights_are_symmetric(w)
    w.transform = "R"
    return name, w, diagnose_weights(w, name, symmetric_pre)


def build_knn_weights(gdf, k: int, name: str) -> tuple:
    """Build k-nearest-neighbours weights for *gdf* centroids. Returns (name, w, diagnostic_dict)."""
    _, _, KNN = import_weights_stack()
    require_projected_crs(gdf, PROJECTED_CRS, name)
    if len(gdf) <= k:
        raise ValueError(f"{name}: k={k} is not valid for N={len(gdf)}.")
    centroids = gdf.geometry.centroid
    coords = list(zip(centroids.x, centroids.y))
    ids = list(gdf.index)
    w = KNN.from_array(coords, k=k, ids=ids)
    symmetric_pre = weights_are_symmetric(w)
    w.transform = "R"
    return name, w, diagnose_weights(w, name, symmetric_pre)


# ---------------------------------------------------------------------------
# Neighbour-graph visualisation
# ---------------------------------------------------------------------------

def plot_neighbor_graph(gdf, w, title: str) -> None:
    _, _, _, plt = import_analysis_stack()
    centroids = gdf.geometry.centroid
    centroid_lookup = {idx: geom for idx, geom in zip(gdf.index, centroids)}
    fig, ax = plt.subplots(figsize=(8, 7))
    gdf.boundary.plot(ax=ax, linewidth=0.4, color="lightgrey")
    for node, neighbors in w.neighbors.items():
        p1 = centroid_lookup.get(node)
        if p1 is None:
            continue
        for neighbor in neighbors:
            if str(node) > str(neighbor):
                continue
            p2 = centroid_lookup.get(neighbor)
            if p2 is None:
                continue
            ax.plot([p1.x, p2.x], [p1.y, p2.y], color="tab:blue", linewidth=0.5, alpha=0.45)
    gdf.geometry.centroid.plot(ax=ax, color="black", markersize=8)
    ax.set_title(title)
    ax.set_axis_off()
    plt.show()
