# ETE 2025/26 — Topic 1 practical lab

## Geographic data are not just data with coordinates

**Course session:** Spatial Econometrics — Topic 1  
**Lab duration:** approximately 60 minutes  
**Case study:** Aveiro–Ílhavo housing listings, zones/neighbourhoods and parishes  
**Main dataset:** `ETE_Lab.gpkg` plus `dbPrimeYield_AVRILH_HousingListingsDataClean.csv`

This notebook is an illustrative practical companion to the theoretical session on geographic data, representation, support, coordinate reference systems, spatial joins, aggregation and the first conceptual bridge toward spatial weights.

The notebook deliberately stops before formal ESDA, Moran's I, LISA, LM tests or spatial regression. Those belong to later topics. Here the target is **methodological discipline before modelling**.

---

### Data provenance and coordinate QA

The dataset used in this lab is derived from **PrimeYield housing listings** covering Aveiro and Ílhavo municipalities. Before the data arrived here, a coordinate quality-assurance (QA) pipeline was applied to the raw listing records. Only listings that passed the QA — `coordinate_quality_flag == 'ok'` — are included in `PrimeYield_HousingListingsDataClean`.

Records rejected during QA had one of the following problems:

| QA flag | Description | n rejected |
|---|---|---|
| `null_island` | Coordinates at (0, 0) or clearly invalid | 361 |
| `outside_study_area` | Point outside Aveiro–Ílhavo polygon | 127 |
| `duplicate_coordinate_cluster` | Exact same coordinate shared by ≥ 10 listings (artefact) | 1,533 |

This pre-filtering means the listings you work with here are already restricted to records with **reliable point coordinates**. The lab will leverage the pre-computed QA flags to explain why each rejected category matters for spatial analysis.

### GeoPackage layers used

| Layer | Type | CRS | Rows | Role |
|---|---|---|---|---|
| `PrimeYield_HousingListingsDataClean` | Point | EPSG:4326 | 1,184 | Housing listing observations |
| `M0105_M0110_C2021_e_casasapo_ZonesPlaces_clean` | MultiPolygon | EPSG:3763 | 131 | Zone/neighbourhood polygons |
| `BGRI21_CONT_FREG_0105_0110` | Polygon | EPSG:3763 | 14 | Census parish polygons |

The CRS contrast between the listings (EPSG:4326, geographic degrees) and the polygon layers (EPSG:3763, projected metres) is intentional. It forces us to treat **projection as an analytical design choice**, not as GIS housekeeping.

### Learning outcomes

By the end of the lab, students should be able to:

1. distinguish the **object of observation** from the **spatial support** used for analysis;
2. audit CRS and understand why CRS choices are measurement choices;
3. create point geometries from tabular listing data and compare them with the GeoPackage version;
4. interpret coordinate QA flags and understand why they matter before spatial analysis;
5. verify and understand zone assignment strategies (polygon intersection vs. nearest-zone fallback);
6. spatially join listing points to parish polygons;
7. aggregate point observations to polygon supports without forgetting the ecological fallacy risk;
8. build a simple adjacency structure as a conceptual preview of the spatial-weights matrix \(W\).

### References

Anselin, L. (1988). *Spatial econometrics: Methods and models*. Kluwer Academic Publishers.

Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). *Geographically weighted regression: The analysis of spatially varying relationships*. Wiley.

Goodchild, M. F. (1992). Geographical information science. *International Journal of Geographical Information Systems, 6*(1), 31–45.

Longley, P. A., Goodchild, M. F., Maguire, D. J., & Rhind, D. W. (2015). *Geographic information science and systems* (4th ed.). Wiley.

Openshaw, S. (1984). *The modifiable areal unit problem*. Geo Books.

Rey, S. J., Arribas-Bel, D., & Wolf, L. J. (2023). *Geographic data science with Python*. CRC Press.

# 1. Setup

This lab uses a minimal geospatial Python stack: `pandas`, `geopandas`, `shapely` and `matplotlib`.

The notebook searches for the data files by locating the project root (identified by a `README.md` marker) so it can run either from inside the course project folder or from any sub-directory.


```python
import sys
import os

from pathlib import Path
import warnings

# Resolve GDAL_DATA so pyogrio/fiona find their support files on Windows conda environments.
gdal_data = Path(sys.prefix) / "Library" / "share" / "gdal"
if "GDAL_DATA" not in os.environ and gdal_data.exists():
    os.environ["GDAL_DATA"] = str(gdal_data)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyogrio import list_layers, read_info

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 140)

print("Python  :", sys.version.split()[0])
print("pandas  :", pd.__version__)
print("geopandas:", gpd.__version__)
print("numpy   :", np.__version__)
```


```python
def find_project_root(marker="README.md"):
    current = Path.cwd()
    while current != current.parent:
        if (current / marker).exists():
            return current
        current = current.parent
    raise FileNotFoundError(f"Project marker '{marker}' not found in any parent directory.")

project_root = find_project_root()
sys.path.append(str(project_root))
os.chdir(project_root)
print("Working directory:", project_root)
```


```python
# Optional local env file (ignored by git) to keep machine-specific paths out of the repository.
env_file = project_root / ".env.local"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
```


```python
DATA_DIR = project_root / "data"
print("Data directory:", DATA_DIR)
print("Exists:", DATA_DIR.exists())
```


```python
GPKG_PATH   = DATA_DIR / "ETE_Lab.gpkg"
CSV_PATH    = DATA_DIR / "dbPrimeYield_AVRILH_HousingListingsDataClean.csv"
SCHEMA_PATH = DATA_DIR / "dbPrimeYield_AVRILH_ETE_schema.md"

# Layer name constants — change here if layer names are updated in the GeoPackage.
LAYER_LISTINGS = "PrimeYield_HousingListingsDataClean"
LAYER_ZONES    = "M0105_M0110_C2021_e_casasapo_ZonesPlaces_clean"

# Target CRS for all metric operations (distances, areas, spatial joins).
PROJECTED_CRS  = "EPSG:3763"  # PT-TM06-ETRS89, Portugal mainland
GEOGRAPHIC_CRS = "EPSG:4326"  # WGS84 geographic (latitude/longitude)

# Variable name constants — change once here if the schema is updated.
PRICE_VAR      = "price_eur"
AREA_VAR       = "area_living_m2"
UNIT_PRICE_VAR = "unit_price_eur_m2"
ZONE_ID_VAR    = "zone_id"
ZONE_NAME_VAR  = "zone_name"
LAT_VAR        = "listing_latitude"
LON_VAR        = "listing_longitude"

for p in [GPKG_PATH, CSV_PATH, SCHEMA_PATH]:
    status = "OK" if p.exists() else "NOT FOUND"
    print(f"  {status}  {p.name}")
```

# 2. Inspect the GeoPackage before reading everything

A GeoPackage is not just a file with shapes. It is a small **spatial database** that can store multiple layers, each with its own geometry type, schema and CRS.

This is the first practical point of the session: geographic data arrive as a combination of **attributes, geometries, identifiers and metadata**. Understanding the container before loading everything prevents silent CRS mismatches and unnecessary memory use.

We also use a utility function to detect parish-type layers by name, because later versions of the GeoPackage may rename or extend the layer set.


```python
layer_info = list_layers(GPKG_PATH)
layers = layer_info[:, 0].tolist()

print("Layers in GeoPackage:")
pd.DataFrame(layer_info, columns=["layer", "geometry_type"])
```


```python
layer_report = []
for layer in layers:
    info = read_info(GPKG_PATH, layer=layer)
    layer_report.append({
        "layer"         : info["layer_name"],
        "rows"          : info["features"],
        "attribute_cols": len(info["fields"]),
        "geometry_type" : info["geometry_type"],
        "crs"           : str(info["crs"]),
        "bounds"        : tuple(np.round(info["total_bounds"], 3)),
    })

pd.DataFrame(layer_report)
```


```python
# Programmatic detection of parish/census layer — robust to layer renaming across dataset versions.
PARISH_TOKENS = ["FREG", "FREGUESIA", "PARISH", "BGRI"]

def detect_layer(layer_names, required_tokens):
    """Return layer names containing any of the required tokens (case-insensitive)."""
    matches = []
    for name in layer_names:
        if any(tok.upper() in name.upper() for tok in required_tokens):
            matches.append(name)
    return matches

parish_layers = detect_layer(layers, PARISH_TOKENS)
print("Detected parish/census candidate layers:", parish_layers)

if len(parish_layers) == 1:
    LAYER_PARISHES = parish_layers[0]
    print(f"Using parish layer: {LAYER_PARISHES}")
elif len(parish_layers) > 1:
    LAYER_PARISHES = parish_layers[0]
    print(f"Multiple candidates found; using: {LAYER_PARISHES}")
    print(f"Other candidates: {parish_layers[1:]}")
else:
    LAYER_PARISHES = None
    print("No parish layer detected. Parish comparisons will be skipped.")
```

**Interpretation.** Notice that the listing points are in EPSG:4326 (angular degrees), while the two polygon layers are in EPSG:3763 (projected metres). Distance and area operations must be done in a projected CRS — EPSG:4326 coordinates in degrees are not a valid basis for measuring distance in metres.

Also note the **bounding box of the listing layer** in EPSG:4326: it extends well outside Portugal. The negative longitude bound near −73 and the strongly negative latitude are clear indicators of coordinates that fell outside the Aveiro–Ílhavo study area and were already excluded by the coordinate QA pipeline (those with flag `outside_study_area` and `null_island`). The remaining spread in the bounding box will be explained in Section 5.

# 3. Load the layers and inspect the schema

The important methodological distinction that anchors this entire session:

- the **listing** is the object of observation — a housing advertisement;
- the **point geometry** represents its reported location;
- the **zone or parish polygon** is an analytical support — the spatial unit chosen for aggregation.

The listing layer already contains pre-assigned zone identifiers (`zone_id`, `zone_name`, `municipality_name`) placed there by the coordinate QA and zone-assignment pipeline. Section 8 explains this assignment strategy and how to verify it.


```python
zones    = gpd.read_file(GPKG_PATH, layer=LAYER_ZONES)
listings = gpd.read_file(GPKG_PATH, layer=LAYER_LISTINGS)

if LAYER_PARISHES is not None:
    parishes = gpd.read_file(GPKG_PATH, layer=LAYER_PARISHES)
else:
    parishes = None

print("zones    :", zones.shape,    "-", zones.crs)
print("listings :", listings.shape, "-", listings.crs)
if parishes is not None:
    print("parishes :", parishes.shape, "-", parishes.crs)
```


```python
# Inspect the first rows of each layer without the geometry column (for readability).
print("=== Zone layer columns ===")
display(zones.drop(columns="geometry").head(3))

if parishes is not None:
    print("\n=== Parish layer columns ===")
    display(parishes.drop(columns="geometry").head(3))

print("\n=== Listing layer columns (first 3 rows, non-geometry) ===")
display(listings.drop(columns="geometry").head(3))
```


```python
# Confirm that zone_id exists in both layers — this is the authoritative join key.
assert ZONE_ID_VAR in listings.columns, f"'{ZONE_ID_VAR}' column missing from listings."
assert ZONE_ID_VAR in zones.columns,    f"'{ZONE_ID_VAR}' column missing from zones."
print(f"Join key '{ZONE_ID_VAR}' present in both layers.")

# Check for missing zone assignments.
n_no_zone = listings[ZONE_ID_VAR].isna().sum()
print(f"Listings without zone assignment: {n_no_zone} of {len(listings)}")

# Summarise listing years covered.
print("\nListing years covered:")
print(listings["listing_year"].value_counts().sort_index().to_string())
```


```python
# Summarise coordinate QA flags — this is the upstream filter already applied to the data.
print("Coordinate quality flag distribution (only 'ok' records are in this layer):")
display(listings["coordinate_quality_flag"].value_counts().to_frame("count"))

print("\nZone match method distribution (how each listing was assigned to a zone):")
display(listings["zone_match_method"].value_counts().to_frame("count"))
```

**Interpretation of zone match methods.**

| Method | Logic | Pedagogical note |
|---|---|---|
| `intersects` | Listing point falls inside a zone polygon | Geometrically unambiguous assignment |
| `nearest_zone` | Closest zone polygon centroid within 100 m | Used for points just outside a polygon boundary — typically geocoding imprecision at the edge |

The 100-metre cap on the nearest-zone fallback prevents zone assignments to points that genuinely lie outside the study area. Listings assigned via the `nearest_zone` method carry slightly more spatial uncertainty than direct intersect matches. This is worth checking when interpreting high-price outliers in peripheral zones.

# 4. Rebuild point geometries from the CSV

This section connects tabular data and geographic representation. A CSV with latitude and longitude is **not yet a spatial dataset**. It becomes one only after we explicitly construct geometries and assign the correct CRS.

Common mistakes to avoid:
1. **Swapping x and y**: in `points_from_xy`, the first argument is **longitude** (x-axis), the second is **latitude** (y-axis). Reversing them places every point in the wrong hemisphere.
2. **Assigning a projected CRS to raw lon/lat**: do not `set_crs(EPSG:3763)` on degree coordinates. First create the points in EPSG:4326, then reproject if metric operations are needed.
3. **Assuming the CSV and GeoPackage are always identical**: always run a consistency check.


```python
listings_csv_raw = pd.read_csv(CSV_PATH)

required_columns = [LON_VAR, LAT_VAR, PRICE_VAR, AREA_VAR]
missing = [c for c in required_columns if c not in listings_csv_raw.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

listings_from_csv = gpd.GeoDataFrame(
    listings_csv_raw.copy(),
    geometry=gpd.points_from_xy(
        listings_csv_raw[LON_VAR],   # x = longitude
        listings_csv_raw[LAT_VAR]    # y = latitude
    ),
    crs=GEOGRAPHIC_CRS
).reset_index(drop=True)

print("CRS assigned:", listings_from_csv.crs)
print("Rows:", len(listings_from_csv))
listings_from_csv[[LON_VAR, LAT_VAR, PRICE_VAR, AREA_VAR, "geometry"]].head(3)
```


```python
# Consistency check between the CSV-built points and the GeoPackage points.
# Row order must match for this comparison to be meaningful.
same_n   = len(listings_from_csv) == len(listings)
same_crs = listings_from_csv.crs == listings.crs
max_abs_lon_diff = np.nanmax(np.abs(listings_from_csv.geometry.x.values - listings.geometry.x.values)) if same_n else float("nan")
max_abs_lat_diff = np.nanmax(np.abs(listings_from_csv.geometry.y.values - listings.geometry.y.values)) if same_n else float("nan")

pd.DataFrame([{
    "same_number_of_rows"        : same_n,
    "same_crs"                   : same_crs,
    "max_abs_longitude_difference": max_abs_lon_diff,
    "max_abs_latitude_difference" : max_abs_lat_diff,
}])
```

**Interpretation.** The CSV version is useful pedagogically because it shows the conversion from tabular coordinates to point geometry. For the rest of the lab we use the GeoPackage layer because it is already spatially typed and carries all the pre-computed QA and zone-assignment fields.

# 5. Attribute audit: prices, area and coordinate quality flags

Before any spatial operation, inspect the non-spatial variables. Spatial analysis does not rescue weak attribute data — it can amplify errors if invalid values are spatially clustered.

**Note on `unit_price_eur_m2`.** This variable is already computed in the dataset as `price_eur / area_living_m2`. We verify the derivation rather than recomputing it, to show that the schema should always be checked.

**Note on the coordinate QA flags.** Because the dataset was pre-filtered to `coordinate_quality_flag == 'ok'`, there should be no null-island or out-of-study-area coordinates in this layer. The flags `coordinate_is_valid_for_point_analysis` and `coordinate_within_study_area` carry additional nuance — for instance, a listing could be inside the study area but still have a suspicious duplicate cluster flag.


```python
# Verify that the precomputed unit_price_eur_m2 matches manual derivation.
unit_price_check = listings[PRICE_VAR] / listings[AREA_VAR]
max_discrepancy = np.abs(unit_price_check - listings[UNIT_PRICE_VAR]).max()
print(f"Max discrepancy in unit_price_eur_m2 (precomputed vs derived): {max_discrepancy:.6f}")

attribute_summary = listings[[PRICE_VAR, AREA_VAR, UNIT_PRICE_VAR]].describe().T
attribute_summary
```


```python
# Residual anomaly check — even after upstream QA, inspect the analytical subset.
quality_checks = pd.DataFrame({
    "condition": [
        "price_eur <= 0",
        "area_living_m2 <= 0",
        "unit_price_eur_m2 < 250",
        "unit_price_eur_m2 > 10,000",
        "coordinate_is_valid_for_point_analysis == False",
        "coordinate_within_study_area == False",
        "coordinate_duplicate_is_suspicious == True",
        "coordinate_matches_zone_centroid == True",
    ],
    "count": [
        int((listings[PRICE_VAR] <= 0).sum()),
        int((listings[AREA_VAR] <= 0).sum()),
        int((listings[UNIT_PRICE_VAR] < 250).sum()),
        int((listings[UNIT_PRICE_VAR] > 10_000).sum()),
        int((~listings["coordinate_is_valid_for_point_analysis"].astype(bool)).sum()),
        int((~listings["coordinate_within_study_area"].astype(bool)).sum()),
        int((listings["coordinate_duplicate_is_suspicious"].astype(bool)).sum()),
        int((listings["coordinate_matches_zone_centroid"].astype(bool)).sum()),
    ]
})

quality_checks
```


```python
# Create an analytical quality flag combining attribute validity and spatial validity.
# These thresholds are a transparent classroom filter, not a market model.
listings = listings.copy()

listings["valid_attributes"] = (
    (listings[PRICE_VAR] > 10_000) &
    (listings[AREA_VAR] > 10) &
    (listings[UNIT_PRICE_VAR].between(250, 10_000))
)

listings["has_zone_assignment"] = listings[ZONE_ID_VAR].notna()

summary = pd.crosstab(
    listings["valid_attributes"],
    listings["has_zone_assignment"],
    rownames=["valid attributes"],
    colnames=["has zone assignment"]
)

print(f"Total listings: {len(listings)}")
print(f"With valid attributes AND zone assignment: "
      f"{int((listings['valid_attributes'] & listings['has_zone_assignment']).sum())}")
summary
```


```python
# Optional: compare attributes of listings assigned via intersects vs. nearest_zone fallback.
# This matters because fallback assignments carry additional spatial uncertainty.
method_comparison = (
    listings
    .dropna(subset=["zone_match_method"])
    .groupby("zone_match_method")[[PRICE_VAR, AREA_VAR, UNIT_PRICE_VAR]]
    .agg(["median", "count"])
)

print("Attribute comparison by zone match method:")
method_comparison
```

**Interpretation.** The coordinate QA flags tell a story about why the original raw dataset (3,000+ listings) reduced to roughly 1,184 records:

- **`null_island`**: 361 records with coordinates at (0,0) or clearly erroneous values — the geocoding workflow had no location to assign, so a placeholder was used.
- **`outside_study_area`**: 127 records outside the Aveiro–Ílhavo polygon boundary.
- **`duplicate_coordinate_cluster`**: 1,533 records sharing the exact same coordinate with 9 or more other listings. This is a geocoding artefact — when street-level geocoding fails, listings are often snapped to a postal-code centroid, creating a pile-up of many listings at the same point. These look like data, but they carry no real spatial information at the listing level.

**The `coordinate_matches_zone_centroid` flag** identifies listings whose coordinates match a known zone centroid after rounding. This is another manifestation of centroid snapping — the geocoder returned the centre of the administrative zone rather than a specific address.

# 6. CRS audit and why projection matters

For storage and web display, EPSG:4326 (geographic, degrees) is common. For **distance and area computations**, it is inappropriate because its units are angular degrees, not metres.

The polygon layers are already in EPSG:3763. The listing points are in EPSG:4326 and must be transformed before any spatial overlay, distance calculation or same-axis map with the polygon layers.

**Rule applied throughout this notebook:** all metric operations are performed in EPSG:3763.


```python
crs_report = pd.DataFrame([
    {"object": "listings",  "crs": str(listings.crs),  "is_projected": listings.crs.is_projected},
    {"object": "zones",     "crs": str(zones.crs),     "is_projected": zones.crs.is_projected},
])
if parishes is not None:
    crs_report = pd.concat([
        crs_report,
        pd.DataFrame([{"object": "parishes", "crs": str(parishes.crs), "is_projected": parishes.crs.is_projected}])
    ], ignore_index=True)

crs_report
```


```python
# All metric operations use zones.crs as the common CRS.
TARGET_CRS = zones.crs

if TARGET_CRS is None:
    raise ValueError("Zone layer has no CRS. Investigate before continuing.")
if listings.crs is None:
    raise ValueError("Listing layer has no CRS. Longitude/latitude data must be EPSG:4326.")

zones_3763    = zones.to_crs(TARGET_CRS)
listings_3763 = listings.to_crs(TARGET_CRS)

if parishes is not None:
    parishes_3763 = parishes.to_crs(TARGET_CRS)
else:
    parishes_3763 = None

if not (listings_3763.crs == zones_3763.crs):
    raise ValueError("CRS mismatch after reprojection.")

bounds_rows = [
    {"object": "listings (EPSG:4326 original)", "crs": str(listings.crs),     "bounds": tuple(np.round(listings.total_bounds, 5))},
    {"object": "listings (EPSG:3763 projected)", "crs": str(listings_3763.crs), "bounds": tuple(np.round(listings_3763.total_bounds, 1))},
    {"object": "zones (EPSG:3763)",              "crs": str(zones_3763.crs),    "bounds": tuple(np.round(zones_3763.total_bounds, 1))},
]
if parishes_3763 is not None:
    bounds_rows.append({"object": "parishes (EPSG:3763)", "crs": str(parishes_3763.crs), "bounds": tuple(np.round(parishes_3763.total_bounds, 1))})

pd.DataFrame(bounds_rows)
```

**Interpretation.** After reprojection, the listing bounding box shows a far wider extent than the zone bounding box. This is expected: even after the upstream QA removed null-island and outside-study-area records, some points with acceptable coordinate-QA-flags can still lie near the study area boundary or slightly beyond. The zone layer is the authoritative definition of the study area, and the visual comparison in Section 7 will clarify this.

# 7. First visual audit: polygons and points

A plot is not a statistical test, but it is an efficient diagnostic. We map the intended study polygon supports and overlay the listing points transformed to EPSG:3763.

The plot should immediately raise two questions:

1. Which listings are inside the zone polygons?
2. Which listings are outside the intended support, and should they be silently aggregated?


```python
xmin, ymin, xmax, ymax = zones_3763.total_bounds
pad = 1_500  # metres; EPSG:3763 is metric

# Clip listings to a window around the zones for the zoomed view.
local_listings_3763 = listings_3763.cx[xmin - pad : xmax + pad, ymin - pad : ymax + pad]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left panel: zoomed to study area
ax = axes[0]
zones_3763.boundary.plot(ax=ax, linewidth=0.5, color="black")
if parishes_3763 is not None:
    parishes_3763.boundary.plot(ax=ax, linewidth=1.8, color="tab:blue")
local_listings_3763.plot(ax=ax, markersize=4, color="tab:red", alpha=0.35)
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_title("Zoomed to study area (EPSG:3763)")
ax.set_axis_off()
# Add a legend
legend_handles = [
    mpatches.Patch(edgecolor="black",    facecolor="white",   label="Zone boundaries"),
    mpatches.Patch(edgecolor="tab:blue", facecolor="white",   label="Parish boundaries"),
    mpatches.Patch(facecolor="tab:red",  edgecolor="tab:red", label="Housing listings"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

# Right panel: all listing coordinates in EPSG:4326 — shows extent of remaining spread
ax2 = axes[1]
listings.plot(ax=ax2, markersize=3, alpha=0.4, color="tab:red")
ax2.set_title("All listings in EPSG:4326\n(residual spread after upstream QA)")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.grid(True, linewidth=0.3)

plt.suptitle("Housing listings and polygon supports", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
```

# 8. Zone assignment: verification and parish spatial join

## 8a. Verifying the pre-assigned zone identifiers

The listing layer already contains `zone_id`, `zone_name`, `municipality_name` — placed there by the upstream QA pipeline. This is different from the older approach of doing a spatial join fresh each time.

**Why pre-assign rather than re-join each time?**
- The upstream workflow applied zone assignment with a **capped nearest-zone fallback** (within 100 m). A plain `sjoin(..., predicate='within')` on the clean point layer cannot reproduce this fallback and would leave some records without a zone.
- Pre-assignment ensures consistency across all downstream analyses.

We still verify the pre-assigned zones by running a spatial join and comparing. This check teaches two lessons at once: how to do a spatial join, and why data provenance documentation matters.


```python
if not (listings_3763.crs == zones_3763.crs):
    raise ValueError("CRS mismatch: listings and zones must share the same CRS before sjoin.")

# Spatial join using 'within' (strict containment — no fallback).
zones_for_join = zones_3763[[ZONE_ID_VAR, ZONE_NAME_VAR, "geometry"]].copy()

listings_sjoin = gpd.sjoin(
    listings_3763,
    zones_for_join,
    how="left",
    predicate="within",
    lsuffix="pre",
    rsuffix="sjoin"
)
# Drop any index column added by sjoin (name varies across geopandas versions).
drop_cols = [c for c in listings_sjoin.columns if c.startswith("index_right")]
if drop_cols:
    listings_sjoin = listings_sjoin.drop(columns=drop_cols)

col_sjoin = f"{ZONE_ID_VAR}_sjoin" if f"{ZONE_ID_VAR}_sjoin" in listings_sjoin.columns else "zone_id_right"

# Count unique listings that received at least one sjoin zone match.
# (Points that fall exactly on shared zone boundaries may produce duplicate rows.)
if col_sjoin in listings_sjoin.columns:
    n_sjoin_rows      = len(listings_sjoin)
    n_sjoin_unique    = listings_sjoin.dropna(subset=[col_sjoin])["listing_id"].nunique()
    n_boundary_extras = n_sjoin_rows - len(listings_3763)
else:
    n_sjoin_unique, n_boundary_extras = "N/A", "N/A"

n_preassigned = listings_3763[ZONE_ID_VAR].notna().sum()

print(f"Listings with pre-assigned zone_id (includes nearest-zone fallback): {n_preassigned} / {len(listings_3763)}")
print(f"Unique listings matched by fresh 'within' sjoin                     : {n_sjoin_unique} / {len(listings_3763)}")
print(f"Extra sjoin rows from boundary-topology duplicates                  : {n_boundary_extras}")
print()
print("The gap between pre-assigned and sjoin-matched is the nearest-zone fallback group.")
print("The extra sjoin rows arise because points on shared polygon boundaries can match multiple zones.")
```


```python
# Show the listings that needed the nearest-zone fallback.
fallback_mask = listings_3763["zone_match_method"] == "nearest_zone"
fallback_listings = listings_3763[fallback_mask][[ZONE_ID_VAR, ZONE_NAME_VAR, "zone_match_distance_m",
                                                   PRICE_VAR, UNIT_PRICE_VAR, LAT_VAR, LON_VAR]]

print(f"Listings assigned via nearest-zone fallback: {fallback_mask.sum()}")
print(f"  Max fallback distance: {listings_3763.loc[fallback_mask, 'zone_match_distance_m'].max():.1f} m")
print(f"  Mean fallback distance: {listings_3763.loc[fallback_mask, 'zone_match_distance_m'].mean():.1f} m")
fallback_listings.head(5)
```

## 8b. Parish spatial join

The parish layer (`BGRI21_CONT_FREG_0105_0110`) is not pre-assigned in the listing data, so here we do a proper spatial join. The layer covers 14 parishes across Aveiro and Ílhavo — much richer than the old 2-parish layer — but may still not cover the full extent of the zone layer. Coverage analysis below reveals the support mismatch.

**Two things to notice from the zone verification above:**

1. **Pre-assigned vs. fresh sjoin gap.** The pre-assigned zone pipeline used a 100-metre nearest-zone fallback for the ~35 listings that fell just outside any polygon boundary. A plain `sjoin(predicate='within')` misses those, so it matches fewer unique listings.

2. **Boundary-topology duplicates.** Some listing points fall exactly on the shared boundary between two zone polygons. The `sjoin` returns a separate row for each matched polygon, so the result has more rows than input listings. The pre-assignment workflow resolved these ties deterministically, which is why it should be preferred for consistent analysis.


```python
if parishes_3763 is not None:
    freg_col = "freguesia" if "freguesia" in parishes_3763.columns else parishes_3763.columns[0]
    freg_key = parishes_3763[[freg_col, "geometry"]].copy()

    listings_with_parish = gpd.sjoin(
        listings_3763,
        freg_key,
        how="left",
        predicate="within"
    )
    drop_cols = [c for c in listings_with_parish.columns if c.startswith("index_right")]
    if drop_cols:
        listings_with_parish = listings_with_parish.drop(columns=drop_cols)

    n_matched_parish = listings_with_parish[freg_col].notna().sum()

    # Support overlap analysis
    zones_union  = zones_3763.geometry.union_all()
    parish_union = parishes_3763.geometry.union_all()

    overlap_report = pd.DataFrame([{
        "zones_area_km2"                        : zones_union.area / 1_000_000,
        "parishes_area_km2"                     : parish_union.area / 1_000_000,
        "shared_area_km2"                       : zones_union.intersection(parish_union).area / 1_000_000,
        "share_of_zone_area_covered_by_parishes": zones_union.intersection(parish_union).area / zones_union.area,
    }])

    coverage = pd.DataFrame({
        "support"           : ["zone layer", "parish layer"],
        "matched_listings"  : [int(listings_3763[ZONE_ID_VAR].notna().sum()), int(n_matched_parish)],
        "unmatched_listings": [int(listings_3763[ZONE_ID_VAR].isna().sum()),  int(len(listings_3763) - n_matched_parish)],
        "total_listings"    : [len(listings_3763), len(listings_3763)],
    })
    coverage["match_share"] = coverage["matched_listings"] / coverage["total_listings"]

    display(coverage)
    display(overlap_report)
else:
    print("No parish layer loaded — skipping parish spatial join.")
    listings_with_parish = listings_3763.copy()
    freg_col = None
```

**Interpretation.** The zone layer and the parish layer do not cover the same territorial extent. Unmatched listings are not necessarily data errors — they may simply fall outside the parishes included in this parish layer. This is a **support mismatch**, and it must be explicitly documented before any claim about parish-level market patterns.

More broadly: the fact that the same market can be observed at three different supports (points, zones, parishes) without any one being "wrong" is the central lesson of this session. Each support answers a different question and carries a different aggregation assumption.

# 9. Listing-level view: points preserve micro-variation

At the point level, each observation is an individual listing. This is the right support if the question concerns dwelling-level variation, micro-location effects or geocoding quality.

We work with the **analytical subset**: listings with valid attributes, a pre-assigned zone, and records that passed the attribute quality filter. This is not a permanent deletion — it is a documented analytical choice.


```python
analysis_points = listings_3763[
    listings_3763["valid_attributes"] &
    listings_3763[ZONE_ID_VAR].notna()
].copy()

print(f"Analytical subset: {len(analysis_points)} listings " 
      f"(of {len(listings_3763)} total after upstream QA)")

analysis_points[[PRICE_VAR, AREA_VAR, UNIT_PRICE_VAR]].describe().T
```


```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scatter: price vs area (log scale helps with skew)
ax = axes[0]
analysis_points.plot.scatter(x=AREA_VAR, y=PRICE_VAR, alpha=0.3, ax=ax, s=8)
ax.set_title("Asking price vs. living area (point-level listings)")
ax.set_xlabel("Living area (m²)")
ax.set_ylabel("Asking price (EUR)")

# Boxplot: unit price by property type
ax2 = axes[1]
analysis_points.boxplot(column=UNIT_PRICE_VAR, by="property_type_std", ax=ax2)
ax2.set_title("Unit price by property type")
ax2.set_xlabel("Property type")
ax2.set_ylabel("EUR/m²")
plt.suptitle("")

plt.tight_layout()
plt.show()
```


```python
# Unit-price distribution by municipality.
fig, ax = plt.subplots(figsize=(8, 5))
analysis_points.boxplot(column=UNIT_PRICE_VAR, by="municipality_name", ax=ax)
ax.set_title("Unit price distribution by municipality")
ax.set_xlabel("Municipality")
ax.set_ylabel("EUR/m²")
plt.suptitle("")
plt.tight_layout()
plt.show()

print("\nMedian unit price by municipality:")
analysis_points.groupby("municipality_name")[UNIT_PRICE_VAR].agg(["median", "mean", "count"])
```

**Interpretation.** At point support, heterogeneity is visible. There is positive correlation between price and area, but with substantial spread. Unit prices vary by property type and by municipality. Aggregation will compress this structure — which can be analytically useful, but it also changes the object of analysis.

# 10. From points to zones: aggregation creates a new dataset

When listings are aggregated to zones, the **unit of analysis changes**. We are no longer studying listings; we are studying zones summarised by listing-derived indicators.

This is where MAUP (Modifiable Areal Unit Problem) and ecological fallacy risks first appear:

- A **zone median** is not the price of a dwelling.
- Relationships observed between zone summaries **cannot be automatically transferred** to individual listings.
- The indicators depend on **which listings happen to be in scope**: zones with few listings produce unstable estimates.

The join key here is `zone_id` — the same numeric key shared between the listing layer and the zone polygon layer.


```python
zone_summary = (
    analysis_points
    .groupby(ZONE_ID_VAR, as_index=False)
    .agg(
        zone_name_label  = (ZONE_NAME_VAR, "first"),
        municipality     = ("municipality_name", "first"),
        n_listings       = ("listing_id", "count"),
        mean_price       = (PRICE_VAR, "mean"),
        median_price     = (PRICE_VAR, "median"),
        mean_unit_price  = (UNIT_PRICE_VAR, "mean"),
        median_unit_price= (UNIT_PRICE_VAR, "median"),
        mean_area        = (AREA_VAR, "mean"),
        share_apartment  = ("property_type_std", lambda s: (s == "Apartment").mean()),
        share_new        = ("condition_std", lambda s: (s == "new").mean()),
    )
)

# Flag zones with few listings — estimates are unstable.
zone_summary["few_listings_flag"] = zone_summary["n_listings"] < 5

zones_lab = zones_3763.merge(zone_summary, on=ZONE_ID_VAR, how="left")

print(f"Zones with at least one listing: {zone_summary['n_listings'].gt(0).sum()} of {len(zones_3763)}")
print(f"Zones with fewer than 5 listings: {zone_summary['few_listings_flag'].sum()}")
zone_summary.sort_values("n_listings", ascending=False).head(10)[[
    ZONE_ID_VAR, "zone_name_label", "municipality", "n_listings",
    "median_price", "median_unit_price", "share_apartment", "share_new"
]]
```


```python
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Map 1: Median unit price by zone
zones_lab.plot(
    column="median_unit_price",
    ax=axes[0],
    legend=True,
    cmap="YlOrRd",
    missing_kwds={"color": "lightgrey", "label": "No valid listings"},
    edgecolor="black",
    linewidth=0.2
)
axes[0].set_title("Zone support: median unit price (EUR/m²)")
axes[0].set_axis_off()

# Map 2: Number of listings per zone
zones_lab.plot(
    column="n_listings",
    ax=axes[1],
    legend=True,
    cmap="Blues",
    missing_kwds={"color": "lightgrey", "label": "No valid listings"},
    edgecolor="black",
    linewidth=0.2
)
axes[1].set_title("Zone support: number of valid listings")
axes[1].set_axis_off()

plt.suptitle("Zone-level aggregation — same listings, different summaries", fontsize=12)
plt.tight_layout()
plt.show()
```

**Class discussion.** Compare the two maps. A zone with a high median unit price but very few listings (map 2 shows light blue) should **not** be interpreted with the same confidence as a zone with many observations. Sample size and spatial representation are inseparable when interpreting choropleth maps.

# 11. Compare point support, zone support and parish support

The same housing market can be described at multiple supports. These descriptions are **not contradictory** — they answer different questions:

- Point support: what is the price of this listing?
- Zone support: what is the typical price in this neighbourhood?
- Parish support: what is the typical price in this administrative area?

Here we compare three supports using the same indicator: **median unit price per m²**.


```python
# Parish aggregation (only if parish layer is available and listings were joined).
if parishes_3763 is not None and freg_col is not None:
    matched_parish = listings_with_parish.dropna(subset=[freg_col])
    # Re-apply attribute filter to the parish-joined dataset.
    matched_parish = matched_parish[matched_parish["valid_attributes"]]

    parish_summary = (
        matched_parish
        .groupby(freg_col, as_index=False)
        .agg(
            n_listings        = ("listing_id", "count"),
            median_unit_price = (UNIT_PRICE_VAR, "median"),
            mean_unit_price   = (UNIT_PRICE_VAR, "mean"),
            median_price      = (PRICE_VAR, "median"),
            share_apartment   = ("property_type_std", lambda s: (s == "Apartment").mean()),
        )
    )

    parishes_lab = parishes_3763.merge(parish_summary, on=freg_col, how="left")
    n_parish_units = int(parish_summary["median_unit_price"].notna().sum())
    median_parish  = parishes_lab["median_unit_price"].median()
    mean_parish    = parishes_lab["mean_unit_price"].mean()
else:
    parishes_lab = None
    n_parish_units, median_parish, mean_parish = 0, float("nan"), float("nan")

support_comparison = pd.DataFrame([
    {
        "support"                          : "listing points",
        "n_units_with_data"                : len(analysis_points),
        "median_unit_price_indicator_EUR_m2": analysis_points[UNIT_PRICE_VAR].median(),
        "mean_unit_price_indicator_EUR_m2" : analysis_points[UNIT_PRICE_VAR].mean(),
    },
    {
        "support"                          : "zones/neighbourhoods",
        "n_units_with_data"                : int(zones_lab["median_unit_price"].notna().sum()),
        "median_unit_price_indicator_EUR_m2": zones_lab["median_unit_price"].median(),
        "mean_unit_price_indicator_EUR_m2" : zones_lab["mean_unit_price"].mean(),
    },
    {
        "support"                          : "parishes",
        "n_units_with_data"                : n_parish_units,
        "median_unit_price_indicator_EUR_m2": median_parish,
        "mean_unit_price_indicator_EUR_m2" : mean_parish,
    },
])

support_comparison
```


```python
fig, ax = plt.subplots(figsize=(7, 5))

support_comparison.plot.bar(
    x="support",
    y="median_unit_price_indicator_EUR_m2",
    ax=ax,
    legend=False,
    color=["steelblue", "darkorange", "forestgreen"]
)

ax.set_title("The indicator changes when the support changes\n(same market, different aggregation levels)")
ax.set_xlabel("Analytical support")
ax.set_ylabel("Median unit-price indicator (EUR/m²)")
ax.tick_params(axis="x", rotation=20)

for bar, val in zip(ax.patches, support_comparison["median_unit_price_indicator_EUR_m2"]):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()
```

**Interpretation.** This table should not be read as 'one value is correct and the others are wrong.' The correct value depends on the **object of analysis** and the **inferential claim**. A listing-level claim (what did this apartment sell for?) and a parish-level claim (how does the typical price compare across parishes?) are simply different claims about different analytical units.

# 12. Absolute space: Euclidean distance in projected coordinates

The theoretical session distinguishes **absolute**, **relative** and **relational** space. Here we illustrate absolute space directly: metric distance in a common projected coordinate frame (EPSG:3763).

As a simple example, we compute each listing's Euclidean distance to a selected central zone. This is not a full accessibility model — it is only a geometric metric that operationalises the idea of *nearness in absolute space*.

**Note:** this computation requires the projected layer (`listings_3763`), not the geographic layer in EPSG:4326. A distance computed in degrees would be meaningless.


```python
# Select a central/reference zone by name pattern.
# Using zone_name to search (human-readable label in the updated zone layer).
candidate_mask = zones_lab[ZONE_NAME_VAR].str.contains(
    "baixa|centro|congressos|alboi", case=False, na=False
)

if candidate_mask.any():
    reference_row  = zones_lab.loc[candidate_mask].sort_values(ZONE_NAME_VAR).iloc[0]
else:
    # Fall back to the zone with the most listings.
    reference_row  = zones_lab.dropna(subset=["n_listings"]).sort_values("n_listings", ascending=False).iloc[0]

reference_zone_id   = reference_row[ZONE_ID_VAR]
reference_zone_name = reference_row[ZONE_NAME_VAR]
reference_centroid  = reference_row.geometry.centroid

print("Reference zone id  :", reference_zone_id)
print("Reference zone name:", reference_zone_name)
print("Centroid (EPSG:3763):", f"({reference_centroid.x:.1f}, {reference_centroid.y:.1f}) m")

analysis_points = analysis_points.copy()
analysis_points["distance_to_reference_m"] = analysis_points.geometry.distance(reference_centroid)

analysis_points[["listing_id", ZONE_NAME_VAR, PRICE_VAR, UNIT_PRICE_VAR, "distance_to_reference_m"]].head(5)
```


```python
fig, ax = plt.subplots(figsize=(7, 5))

analysis_points.plot.scatter(
    x="distance_to_reference_m",
    y=UNIT_PRICE_VAR,
    alpha=0.3,
    s=6,
    ax=ax
)

ax.set_title(
    f"Absolute-space illustration: unit price vs. Euclidean distance\n"
    f"Reference zone: '{reference_zone_name}'"
)
ax.set_xlabel("Distance to reference zone centroid (metres)")
ax.set_ylabel("Unit price (EUR/m²)")
plt.tight_layout()
plt.show()
```

**Interpretation.** A weak or noisy relationship is not a failure. It reminds us that **metric Euclidean distance is only one possible spatial logic**:

- *Relative space* would require travel time, road network distance or accessibility indices.
- *Relational space* would require a theory of interaction, substitution or dependence between places — for example, substitutability between dwellings in the same commuting zone.

Any of these operationalisations would change the observed relationship. The choice of spatial logic is a methodological decision, not a data discovery.

# 13. Geometry relations as a bridge to the spatial-weights matrix W

The theoretical session introduces \(W\) as a formal representation of spatial relations. Here we do not estimate a spatial model. We only build the **simplest possible topological relation between polygons**: whether two zones share a boundary (Queen contiguity).

This is enough to show that \(W\) is **not discovered automatically** by the software. It **operationalises a theoretical choice** about which units are considered neighbours. Two zones can share a boundary without sharing a housing submarket — the contiguity criterion is a modelling assumption, not a geographic fact.

We use PySAL's `libpysal.weights.Queen` if available, falling back to a direct loop for transparency.


```python
# Try PySAL Queen contiguity first (efficient); fall back to explicit loop for teaching transparency.
try:
    from libpysal.weights import Queen as QueenW
    w_queen = QueenW.from_dataframe(zones_3763, idVariable=ZONE_ID_VAR)
    w_queen.transform = "B"  # Binary
    
    n_neighbours_series = pd.Series(
        {zone_id: len(nbrs) for zone_id, nbrs in w_queen.neighbors.items()},
        name="n_neighbours"
    ).reset_index().rename(columns={"index": ZONE_ID_VAR})
    print("Used libpysal Queen contiguity.")

except ImportError:
    print("libpysal not available — falling back to explicit boundary-touch loop.")
    zone_ids = zones_3763[ZONE_ID_VAR].tolist()
    neighbour_counts = []
    for i, row_i in zones_3763.iterrows():
        n_nbrs = sum(
            1 for j, row_j in zones_3763.iterrows()
            if i != j and row_i.geometry.touches(row_j.geometry)
        )
        neighbour_counts.append({ZONE_ID_VAR: row_i[ZONE_ID_VAR], "n_neighbours": n_nbrs})
    n_neighbours_series = pd.DataFrame(neighbour_counts)

zones_w = zones_lab.merge(n_neighbours_series, on=ZONE_ID_VAR, how="left")

print(f"\nZone neighbour count summary:")
print(zones_w["n_neighbours"].describe())
print(f"Islands (0 neighbours): {(zones_w['n_neighbours'] == 0).sum()}")
```


```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Map: number of contiguous neighbours
zones_w.plot(
    column="n_neighbours",
    ax=axes[0],
    legend=True,
    cmap="YlGnBu",
    missing_kwds={"color": "lightgrey"},
    edgecolor="black",
    linewidth=0.2
)
axes[0].set_title("Preview of W: contiguous neighbours by zone")
axes[0].set_axis_off()

# Histogram of neighbour counts
zones_w["n_neighbours"].dropna().plot.hist(bins=20, ax=axes[1], edgecolor="white")
axes[1].set_title("Distribution of zone neighbour counts (Queen contiguity)")
axes[1].set_xlabel("Number of contiguous neighbours")
axes[1].set_ylabel("Number of zones")

plt.tight_layout()
plt.show()
```


```python
# Row-standardised W preview: compute a spatial lag of median unit price.
# This is only a descriptive illustration of the lag concept, not a regression.

try:
    # Use PySAL row-standardised W if available.
    from libpysal.weights import Queen as QueenW
    w_rs = QueenW.from_dataframe(zones_3763, idVariable=ZONE_ID_VAR)
    w_rs.transform = "R"  # Row-standardise

    zone_values = zones_w.set_index(ZONE_ID_VAR)["median_unit_price"]
    spatial_lag = {}
    for zone_id, nbrs in w_rs.neighbors.items():
        weights = w_rs.weights[zone_id]
        vals = [zone_values.get(n, np.nan) for n in nbrs]
        if nbrs and not all(np.isnan(vals)):
            spatial_lag[zone_id] = np.nansum([w * v for w, v in zip(weights, vals)])
        else:
            spatial_lag[zone_id] = np.nan

    zones_w["spatial_lag_median_unit_price"] = zones_w[ZONE_ID_VAR].map(spatial_lag)

except ImportError:
    print("libpysal not available — skipping spatial-lag computation.")

if "spatial_lag_median_unit_price" in zones_w.columns:
    zones_w[[
        ZONE_ID_VAR, ZONE_NAME_VAR, "median_unit_price",
        "n_neighbours", "spatial_lag_median_unit_price"
    ]].dropna(subset=["median_unit_price"]).sort_values(
        "median_unit_price", ascending=False
    ).head(10)
```

**Methodological warning.** Contiguity-based neighbours are convenient for polygon layers, but they are **not automatically correct** for the housing process. A housing submarket may be better represented by:

- **Distance bands** (all zones within 1 km)
- **k-nearest neighbours** (the k closest zone centroids)
- **Travel-time catchments** (zones reachable within 10 minutes by road)
- **Substitution relations** (zones that compete for the same buyers)

Each of these produces a different \(W\), and each encodes a different theory about what spatial proximity means. The formal study of \(W\) — including islands, disconnected components, row standardisation and their effects on Moran's I — is the subject of Topic 2.

# 14. Save clean outputs for later sessions

The outputs below are modelling-ready in a minimal sense: they have consistent identifiers (`zone_id`), explicit CRS and documented joins. They are **not** a final econometric dataset — they still require decisions about outliers, transformations and spatial weight specifications before modelling.

Later topics will reuse these files for ESDA, Moran's I, LISA, spatial regression and local models.


```python
OUTPUT_DIR = Path.cwd() / "outputs" / "topic1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

points_path   = OUTPUT_DIR / "topic1_listings_enriched.gpkg"
zones_path    = OUTPUT_DIR / "topic1_zones_summary.gpkg"
csv_path      = OUTPUT_DIR / "topic1_zone_summary.csv"

analysis_points.to_file(points_path, layer="listings_enriched", driver="GPKG")
zones_w.to_file(zones_path, layer="zones_summary", driver="GPKG")
zones_w.drop(columns="geometry").to_csv(csv_path, index=False)

print("Saved:")
print(" ", points_path)
print(" ", zones_path)
print(" ", csv_path)
```

# 15. Closing synthesis

This lab illustrated the practical consequences of the first theoretical topic.

### Key lessons

1. **Representation.** Listings are points; zones and parishes are polygons. The same real-world phenomenon can be represented at multiple spatial supports, each carrying different assumptions.

2. **Coordinate QA before spatial analysis.** The raw PrimeYield data contained null-island coordinates, out-of-area records and duplicate coordinate clusters. Each failure mode corrupts spatial analysis in a different way. Pre-filtering by `coordinate_quality_flag` is not optional — it is part of the measurement design.

3. **Zone assignment methods matter.** Most listings are assigned to zones by direct polygon intersection (`intersects`). A small number required a nearest-zone fallback. Understanding which records belong to each category is important when interpreting spatial patterns at zone borders.

4. **CRS: projection is a measurement choice.** Distance and area require a projected CRS. Computing distances in degrees (EPSG:4326) is dimensionally incorrect. All metric operations in this notebook use EPSG:3763.

5. **Support changes the indicator.** Median unit price at the listing level, zone level and parish level produces three numerically different values describing the same market. None is wrong — each answers a different question.

6. **Aggregation introduces MAUP risk and ecological fallacy risk.** Zone medians are not dwelling prices. Relationships observed at the zone level cannot be automatically attributed to individual listings.

7. **W is a methodological choice.** The Queen contiguity matrix previewed in Section 13 operationalises one theory of neighbourhood relations. Different theories produce different \(W\) matrices, different Moran statistics, and different regression results.

---

### Short in-class discussion questions

1. Why does a listing outside the zone layer not automatically constitute a data error?
2. What inferential claim changes when we move from listing-level prices to zone-level median prices?
3. Why was it necessary to reproject the listing points to EPSG:3763 before computing distances?
4. A zone has only 2 listings but a high median unit price. How many alternative explanations can you identify before concluding that the location is genuinely expensive?
5. Would Queen contiguity be a good spatial relation for housing listings? Under what market theory might it be defensible, and when would it be misleading?
6. What additional information would you need to move from *absolute distance* (Euclidean) to *relative distance* (travel time) in this case study?
