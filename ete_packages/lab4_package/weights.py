"""Spatial weights and residual-autocorrelation diagnostics.

The package uses optional PySAL objects when available, but implements small
fallback routines for the introductory lab so that inventory and QA sections can
run in lean environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config
from .utils import LabDataError, finite_mask, require_columns


@dataclass
class SimpleWeights:
    """Minimal row-standardised neighbour representation.

    Attributes
    ----------
    neighbors:
        Mapping from positional integer index to a list of neighbouring
        positional integer indices.
    name:
        Human-readable weights label.
    transform:
        Currently ``"r"`` for row-standardised or ``"b"`` for binary.
    ids:
        Optional observation identifiers aligned with positional indices.
    """

    neighbors: dict[int, list[int]]
    name: str
    transform: str = "r"
    ids: list[Any] | None = None

    @property
    def n(self) -> int:
        return len(self.neighbors)

    @property
    def pct_islands(self) -> float:
        if not self.neighbors:
            return np.nan
        return 100.0 * sum(len(v) == 0 for v in self.neighbors.values()) / len(self.neighbors)

    @property
    def mean_neighbors(self) -> float:
        if not self.neighbors:
            return np.nan
        return float(np.mean([len(v) for v in self.neighbors.values()]))


def _ensure_metric(gdf: gpd.GeoDataFrame, metric_crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise LabDataError("Spatial weights require a defined CRS.")
    return gdf if str(gdf.crs) == str(metric_crs) else gdf.to_crs(metric_crs)


def knn_weights(
    gdf: gpd.GeoDataFrame,
    k: int = config.DEFAULT_KNN_K,
    metric_crs: str = config.METRIC_CRS,
    id_col: str | None = None,
    name: str | None = None,
) -> SimpleWeights:
    """Build k-nearest-neighbour weights from geometry centroids/points."""

    gdf = _ensure_metric(gdf, metric_crs)
    if len(gdf) < 3:
        raise LabDataError("At least three observations are required for KNN weights.")
    k_eff = max(1, min(int(k), len(gdf) - 1))
    coords = np.column_stack([gdf.geometry.centroid.x.to_numpy(), gdf.geometry.centroid.y.to_numpy()])

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1)
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)

    neighbors = {i: [int(j) for j in row[1:] if int(j) != i] for i, row in enumerate(indices)}
    ids = gdf[id_col].tolist() if id_col and id_col in gdf.columns else list(gdf.index)
    return SimpleWeights(neighbors=neighbors, name=name or f"knn_{k_eff}", ids=ids)


def queen_weights(
    polygons: gpd.GeoDataFrame,
    metric_crs: str = config.METRIC_CRS,
    id_col: str | None = None,
    name: str = "queen",
) -> SimpleWeights:
    """Build queen contiguity weights using PySAL if available, otherwise GeoPandas."""

    polygons = _ensure_metric(polygons, metric_crs).reset_index(drop=True)
    if len(polygons) < 3:
        raise LabDataError("At least three polygons are required for contiguity weights.")

    try:
        from libpysal.weights import Queen

        w = Queen.from_dataframe(polygons, ids=polygons[id_col].tolist() if id_col in polygons.columns else None)
        id_order = list(w.id_order)
        id_to_pos = {id_: pos for pos, id_ in enumerate(id_order)}
        neighbors = {id_to_pos[i]: [id_to_pos[j] for j in js] for i, js in w.neighbors.items()}
        return SimpleWeights(neighbors=neighbors, name=name, ids=id_order)
    except Exception:
        # Fallback: spatial-index candidate pairs and touches/intersects test.
        sindex = polygons.sindex
        neighbors: dict[int, list[int]] = {i: [] for i in range(len(polygons))}
        geoms = polygons.geometry
        for i, geom in enumerate(geoms):
            candidates = list(sindex.query(geom, predicate="intersects"))
            for j in candidates:
                j = int(j)
                if i == j:
                    continue
                if geom.touches(geoms.iloc[j]) or geom.intersects(geoms.iloc[j]):
                    neighbors[i].append(j)
        ids = polygons[id_col].tolist() if id_col and id_col in polygons.columns else list(polygons.index)
        return SimpleWeights(neighbors=neighbors, name=name, ids=ids)


def weights_summary(w: SimpleWeights) -> pd.DataFrame:
    """Return a compact summary of a weights object."""

    degrees = pd.Series({i: len(js) for i, js in w.neighbors.items()}, dtype=float)
    return pd.DataFrame(
        [
            {"metric": "name", "value": w.name},
            {"metric": "n", "value": w.n},
            {"metric": "mean_neighbors", "value": float(degrees.mean())},
            {"metric": "min_neighbors", "value": int(degrees.min()) if len(degrees) else np.nan},
            {"metric": "max_neighbors", "value": int(degrees.max()) if len(degrees) else np.nan},
            {"metric": "islands", "value": int((degrees == 0).sum())},
            {"metric": "pct_islands", "value": w.pct_islands},
        ]
    )


def build_weights_suite(
    listings: gpd.GeoDataFrame | None = None,
    zones: gpd.GeoDataFrame | None = None,
    municipalities: gpd.GeoDataFrame | None = None,
    specs: Iterable[Mapping[str, Any]] | None = None,
    metric_crs: str = config.METRIC_CRS,
) -> dict[str, SimpleWeights]:
    """Build a named set of spatial weights from simple specifications."""

    specs = list(
        specs
        or [
            {"name": "listing_knn_8", "type": "knn", "unit": "listing", "k": 8},
            {"name": "zone_queen", "type": "queen", "unit": "zone"},
            {"name": "municipality_queen", "type": "queen", "unit": "municipality"},
        ]
    )
    data = {"listing": listings, "zone": zones, "municipality": municipalities}
    out: dict[str, SimpleWeights] = {}

    for spec in specs:
        unit = spec.get("unit")
        gdf = data.get(unit)
        if gdf is None:
            continue
        name = str(spec.get("name", f"{unit}_{spec.get('type')}"))
        if spec.get("type") == "knn":
            out[name] = knn_weights(gdf, k=int(spec.get("k", config.DEFAULT_KNN_K)), metric_crs=metric_crs, name=name)
        elif spec.get("type") in {"queen", "rook"}:
            # Rook is not separately implemented in fallback mode; name the requested diagnostic.
            id_col = "zone_id" if unit == "zone" and "zone_id" in gdf.columns else "dtmn" if unit == "municipality" and "dtmn" in gdf.columns else None
            out[name] = queen_weights(gdf, metric_crs=metric_crs, id_col=id_col, name=name)
        else:
            raise LabDataError(f"Unsupported weights type: {spec.get('type')}")
    return out


def moran_i(
    values: Iterable[float],
    weights: SimpleWeights,
    permutations: int = config.DEFAULT_PERMUTATIONS,
    random_state: int = config.RANDOM_STATE,
) -> dict[str, float | int | str]:
    """Compute a simple global Moran's I with optional permutation p-value."""

    y = np.asarray(list(values), dtype=float)
    if len(y) != weights.n:
        raise LabDataError(f"Value length ({len(y)}) does not match weights length ({weights.n}).")
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return {"moran_i": np.nan, "p_sim": np.nan, "n": int(mask.sum()), "note": "Too few finite values"}

    # Subset weights if missing values exist.
    if not mask.all():
        old_to_new = {old: new for new, old in enumerate(np.where(mask)[0])}
        neighbors = {
            old_to_new[i]: [old_to_new[j] for j in js if j in old_to_new]
            for i, js in weights.neighbors.items()
            if i in old_to_new
        }
        y = y[mask]
        weights = SimpleWeights(neighbors=neighbors, name=weights.name, ids=None)

    z = y - y.mean()
    denom = float(np.sum(z**2))
    if denom == 0:
        return {"moran_i": np.nan, "p_sim": np.nan, "n": len(y), "note": "Constant values"}

    s0 = sum(len(js) for js in weights.neighbors.values())
    if s0 == 0:
        return {"moran_i": np.nan, "p_sim": np.nan, "n": len(y), "note": "No neighbours"}

    num = 0.0
    for i, js in weights.neighbors.items():
        if not js:
            continue
        # Row-standardised weights: each neighbour receives 1/k_i.
        wij = 1.0 / len(js) if weights.transform == "r" else 1.0
        num += sum(wij * z[i] * z[j] for j in js)
    s0_effective = weights.n if weights.transform == "r" else s0
    observed = (weights.n / s0_effective) * (num / denom)

    if permutations <= 0:
        return {"moran_i": float(observed), "p_sim": np.nan, "n": len(y), "permutations": 0, "note": "No permutations"}

    rng = np.random.default_rng(random_state)
    sims = np.empty(int(permutations), dtype=float)
    for p in range(int(permutations)):
        zp = rng.permutation(z)
        num_p = 0.0
        for i, js in weights.neighbors.items():
            if not js:
                continue
            wij = 1.0 / len(js) if weights.transform == "r" else 1.0
            num_p += sum(wij * zp[i] * zp[j] for j in js)
        sims[p] = (weights.n / s0_effective) * (num_p / denom)

    # Two-sided permutation p-value.
    p_sim = (np.sum(np.abs(sims) >= abs(observed)) + 1.0) / (len(sims) + 1.0)
    return {
        "moran_i": float(observed),
        "p_sim": float(p_sim),
        "n": int(len(y)),
        "permutations": int(permutations),
        "note": "fallback Moran implementation",
    }


def neighbour_graph_summary(w: SimpleWeights) -> pd.DataFrame:
    """Return one row per observation with neighbour counts."""

    return pd.DataFrame(
        {
            "position": list(w.neighbors.keys()),
            "id": w.ids if w.ids is not None and len(w.ids) == len(w.neighbors) else list(w.neighbors.keys()),
            "n_neighbours": [len(v) for v in w.neighbors.values()],
        }
    )
