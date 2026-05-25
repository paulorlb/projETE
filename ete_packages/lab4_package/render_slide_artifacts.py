"""
Render four styled PNG table images for Topic 4 slide deck.
Output: notebooks/outputs/topic4/slide_art10_leakage_flag.png
        notebooks/outputs/topic4/slide_art11_pipeline_table.png
        notebooks/outputs/topic4/slide_art12_cv_rf_table.png
        notebooks/outputs/topic4/slide_art14_residual_moran_table.png
"""

import os
import textwrap
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── Global style constants ──────────────────────────────────────────────────
FONT_FAMILY        = "DejaVu Sans"
FONT_SIZE_HEADER   = 11
FONT_SIZE_BODY     = 10.5
FONT_SIZE_CAPTION  = 9
FONT_SIZE_TITLE    = 12

COL_HEADER_BG      = "#2C3E50"
COL_HEADER_FG      = "#FFFFFF"
ROW_ALT_A          = "#F7F9FC"
ROW_ALT_B          = "#FFFFFF"
ROW_HIGHLIGHT      = "#FFF3CD"
ROW_HIGHLIGHT_PASS = "#D4EDDA"
GRID_COLOR         = "#BDC3C7"
CAPTION_COLOR      = "#7F8C8D"

DPI                = 150
TIGHT_LAYOUT       = True

OUTPUT_DIR = "notebooks/outputs/topic4"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _hide_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── ART-10: Feature Leakage Flag ────────────────────────────────────────────
def render_art10():
    data = pd.DataFrame({
        "column": [
            "zone_mean_unit_price_eur_m2__leakage_sensitive",
            "zone_id_cat",
        ],
        "feature_type": [
            "leakage_sensitive_outcome_summary",
            "spatial_categorical",
        ],
        "dtype":   ["float64", "string"],
        "missing": [0, 0],
        "unique":  [87, 87],
        "used_by_default": ["False ⚠", "True ✓"],
    })

    col_widths = [0.38, 0.25, 0.09, 0.07, 0.07, 0.14]
    col_labels = list(data.columns)
    nrows, ncols = len(data), len(col_labels)

    fig, ax = plt.subplots(figsize=(12, 2.8))
    fig.patch.set_facecolor("white")
    _hide_axes(ax)

    # Title
    ax.set_title(
        "Feature leakage audit – selected rows (lab Section 11)",
        fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=14, loc="left",
    )

    cell_text = data.values.tolist()
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT_SIZE_BODY)

    # Header row styling
    for j in range(ncols):
        cell = tbl[0, j]
        cell.set_facecolor(COL_HEADER_BG)
        cell.set_text_props(color=COL_HEADER_FG, fontweight="bold",
                            fontsize=FONT_SIZE_HEADER)
        cell.set_edgecolor(GRID_COLOR)

    # Row 0 – leakage (amber)
    for j in range(ncols):
        cell = tbl[1, j]
        cell.set_facecolor(ROW_HIGHLIGHT)
        cell.set_edgecolor(GRID_COLOR)
    tbl[1, 5].set_text_props(color="#C0392B", fontweight="bold")

    # Row 1 – safe (green)
    for j in range(ncols):
        cell = tbl[2, j]
        cell.set_facecolor(ROW_HIGHLIGHT_PASS)
        cell.set_edgecolor(GRID_COLOR)
    tbl[2, 5].set_text_props(color="#27AE60", fontweight="bold")

    tbl.scale(1, 1.6)

    # Annotation
    fig.text(
        0.01, 0.12,
        'used_by_default = False  →  excluded from all CV runs',
        ha="left", va="bottom",
        fontsize=FONT_SIZE_CAPTION, fontstyle="italic", color=CAPTION_COLOR,
    )

    # Caption
    fig.text(
        0.01, 0.01,
        "Feature engineering leakage audit (selected rows) · Source: lab Section 11 · "
        "zone_mean_unit_price_eur_m2 excluded from cross-validation by design.",
        ha="left", va="bottom",
        fontsize=FONT_SIZE_CAPTION, color=CAPTION_COLOR,
    )

    out = os.path.join(OUTPUT_DIR, "slide_art10_leakage_flag.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ART-10] Saved: {out}  ({os.path.getsize(out) / 1024:.1f} KB)")


# ── ART-11: Leakage-Safe Feature Pipeline ───────────────────────────────────
def render_art11():
    col1_safe = [
        "area_living_m2",
        "condition_std",
        "listing_year",
        "preservation_class_std",
        "property_type_std",
        "typology_bucket_std",
        "zone_id_cat",
        "municipality_name_cat",
        "distance_to_aveiro_centre_km",
        "distance_to_ilhavo_centre_km",
        "x_pttm06_m",
        "y_pttm06_m",
        "x_centered_km",
        "y_centered_km",
        "zone_area_km2",
        "zone_listing_count",
        "zone_listing_density_km2",
    ]

    col2_risk = [
        "zone_mean_unit_price_eur_m2",
        "__leakage_sensitive",
        "",
        "(outcome summary,",
        "computed globally",
        "before split)",
    ]

    col3_handling = [
        "Recompute inside",
        "each training fold",
        "only.",
        "",
        "Never on the full",
        "dataset pre-split.",
        "",
        "Or: exclude entirely",
        "(package default).",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Leakage-safe feature pipeline – Aveiro–Ílhavo feature set",
        fontsize=FONT_SIZE_TITLE, fontweight="bold", x=0.02, ha="left", y=1.0,
    )

    headers = [
        "✓  Safe features  (used_by_default = True)",
        "⚠   Leakage-risk  (used_by_default = False)",
        "→  Correct handling",
    ]
    data_cols = [col1_safe, col2_risk, col3_handling]
    bg_colors = [ROW_HIGHLIGHT_PASS, ROW_HIGHLIGHT, ROW_ALT_A]

    for idx, (ax, header, data_col, bg) in enumerate(
            zip(axes, headers, data_cols, bg_colors)):
        _hide_axes(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        n = len(data_col)
        row_h = 0.80 / max(n, 1)   # rows fill 80% of axes height
        top_y = 0.95

        # Header cell
        hdr_rect = mpatches.FancyBboxPatch(
            (0.0, top_y - 0.10), 1.0, 0.10,
            boxstyle="square,pad=0",
            facecolor=COL_HEADER_BG, edgecolor=GRID_COLOR, linewidth=0.8,
            transform=ax.transAxes,
        )
        ax.add_patch(hdr_rect)
        ax.text(
            0.5, top_y - 0.05, header,
            ha="center", va="center",
            fontsize=FONT_SIZE_HEADER - 1, fontweight="bold",
            color=COL_HEADER_FG, transform=ax.transAxes,
            wrap=True,
        )

        # Data rows
        for r, txt in enumerate(data_col):
            y_bottom = top_y - 0.10 - (r + 1) * row_h
            rect = mpatches.FancyBboxPatch(
                (0.0, y_bottom), 1.0, row_h,
                boxstyle="square,pad=0",
                facecolor=bg, edgecolor=GRID_COLOR, linewidth=0.5,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)
            ax.text(
                0.05, y_bottom + row_h / 2, txt,
                ha="left", va="center",
                fontsize=FONT_SIZE_BODY,
                color="#2C3E50", transform=ax.transAxes,
            )

    fig.text(
        0.01, 0.01,
        "Source: lab Sections 11 and 19b · Column 3 = design principle; implementation deferred to lab.",
        ha="left", va="bottom",
        fontsize=FONT_SIZE_CAPTION, color=CAPTION_COLOR,
    )

    out = os.path.join(OUTPUT_DIR, "slide_art11_pipeline_table.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ART-11] Saved: {out}  ({os.path.getsize(out) / 1024:.1f} KB)")


# ── ART-12: CV Performance Table (Random Forest) ────────────────────────────
def render_art12():
    data = pd.DataFrame({
        "Strategy": [
            "Random CV",
            "Zone-block CV",
            "Parish-block CV",
            "Spatial-cluster CV",
        ],
        "Fold unit withheld": [
            "Random listing",
            "Entire zone (87 zones)",
            "Entire parish (14 parishes)",
            "Contiguous spatial cluster",
        ],
        "Mean R²": ["0.647", "0.509", "0.285", "0.129"],
        "Mean RMSE (€/m²)": ["614", "714", "823", "887"],
        "What is tested": [
            "Interpolation from nearby training listings",
            "Predict listings in an unseen zone",
            "Predict listings in an unseen parish",
            "Predict listings in an unseen region",
        ],
    })

    col_widths = [0.14, 0.22, 0.09, 0.13, 0.42]
    col_labels = list(data.columns)
    nrows, ncols = len(data), len(col_labels)

    row_colors = [ROW_ALT_B, ROW_ALT_A, "#FCE4D6", "#F4CCCC"]

    fig, ax = plt.subplots(figsize=(14, 3.4))
    fig.patch.set_facecolor("white")
    _hide_axes(ax)

    ax.set_title(
        "Random Forest – CV performance by strategy  "
        "(unit_price_eur_m2 · n = 1 095 · RANDOM_STATE = 20260521)",
        fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=14, loc="left",
    )

    cell_text = data.values.tolist()
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT_SIZE_BODY)

    # Header
    for j in range(ncols):
        cell = tbl[0, j]
        cell.set_facecolor(COL_HEADER_BG)
        cell.set_text_props(color=COL_HEADER_FG, fontweight="bold",
                            fontsize=FONT_SIZE_HEADER)
        cell.set_edgecolor(GRID_COLOR)

    # Data rows
    for i, bg in enumerate(row_colors):
        for j in range(ncols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor(GRID_COLOR)
        # R² column bold + right-aligned
        r2_cell = tbl[i + 1, 2]
        r2_cell.set_text_props(fontweight="bold", ha="right")

    tbl.scale(1, 1.8)

    # Leakage signature arrow annotation
    fig.text(
        0.97, 0.55,
        "▼ spatial leakage\n    signature",
        ha="center", va="center",
        fontsize=9, color="#C0392B", fontstyle="italic",
    )

    fig.text(
        0.01, 0.01,
        "Random Forest CV performance by strategy · unit_price_eur_m2 · n = 1 095 · "
        "RANDOM_STATE = 20260521 · Source: lab Section 119c.",
        ha="left", va="bottom",
        fontsize=FONT_SIZE_CAPTION, color=CAPTION_COLOR,
    )

    out = os.path.join(OUTPUT_DIR, "slide_art12_cv_rf_table.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ART-12] Saved: {out}  ({os.path.getsize(out) / 1024:.1f} KB)")


# ── ART-14: Residual Moran's I Diagnostic Table ─────────────────────────────
def render_art14():
    data = pd.DataFrame({
        "Strategy": [
            "random_cv",
            "random_cv",
            "zone_block_cv",
            "spatial_cluster_cv",
            "zone_block_cv",
            "spatial_cluster_cv",
        ],
        "Model": [
            "Random Forest",
            "Random Forest",
            "Random Forest",
            "Random Forest",
            "LightGBM",
            "LightGBM",
        ],
        "W matrix": [
            "kNN-8 listing",
            "Queen zone",
            "kNN-8 listing",
            "kNN-8 listing",
            "Queen zone",
            "Queen zone",
        ],
        "Unit": ["listing", "zone", "listing", "listing", "zone", "zone"],
        "Moran's I": ["0.036", "0.014", "0.226", "0.409", "-0.006", "0.442"],
        "p-value":   ["0.010", "0.840", "0.005", "0.005", "0.955", "0.005"],
        "Interpretation": [
            "Near zero – interpolation artefact ⚠",
            "Non-significant",
            "Significant – errors cluster in unseen zones",
            "Strong – geographic structure unmodelled",
            "Non-significant at zone level",
            "Strong clustering at zone level",
        ],
    })

    col_widths = [0.13, 0.12, 0.11, 0.06, 0.08, 0.07, 0.43]
    col_labels = list(data.columns)
    nrows, ncols = len(data), len(col_labels)

    row_bgs = [ROW_ALT_B, ROW_ALT_B, ROW_ALT_A, ROW_ALT_A, ROW_ALT_A, ROW_ALT_A]

    fig, ax = plt.subplots(figsize=(16, 4.2))
    fig.patch.set_facecolor("white")
    _hide_axes(ax)

    ax.set_title(
        "Residual Moran's I on cross-validated residuals  "
        "(199 permutations · RANDOM_STATE = 20260521)",
        fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=14, loc="left",
    )

    # Wrap long interpretation text
    wrapped_interp = [
        textwrap.fill(v, width=42)
        for v in data["Interpretation"].tolist()
    ]
    cell_data = data.copy()
    cell_data["Interpretation"] = wrapped_interp
    cell_text = cell_data.values.tolist()

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT_SIZE_BODY)

    # Header
    for j in range(ncols):
        cell = tbl[0, j]
        cell.set_facecolor(COL_HEADER_BG)
        cell.set_text_props(color=COL_HEADER_FG, fontweight="bold",
                            fontsize=FONT_SIZE_HEADER)
        cell.set_edgecolor(GRID_COLOR)

    # Data rows
    for i, bg in enumerate(row_bgs):
        for j in range(ncols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor(GRID_COLOR)

    # Special Moran's I colouring
    moran_col = 4  # index of "Moran's I" column
    tbl[1, moran_col].set_text_props(color="#E67E22", fontweight="bold")   # row 0 amber
    tbl[4, moran_col].set_text_props(color="#C0392B", fontweight="bold")   # row 3 red
    tbl[6, moran_col].set_text_props(color="#C0392B", fontweight="bold")   # row 5 red

    tbl.scale(1, 2.2)

    # Insight annotation box
    fig.text(
        0.5, 0.04,
        "Key insight: near-zero I under random CV is a leakage artefact, "
        "not evidence of a good model.\nCheck spatial-CV rows for the honest diagnostic.",
        ha="center", va="bottom",
        fontsize=11, color="#555555", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA",
                  edgecolor=GRID_COLOR, alpha=0.9),
    )

    fig.text(
        0.01, 0.01,
        "Residual Moran's I on CV residuals · kNN-8 (listing) and Queen contiguity (zone) · "
        "199 permutations · RANDOM_STATE = 20260521 · Source: lab Section 20.",
        ha="left", va="bottom",
        fontsize=FONT_SIZE_CAPTION, color=CAPTION_COLOR,
    )

    out = os.path.join(OUTPUT_DIR, "slide_art14_residual_moran_table.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ART-14] Saved: {out}  ({os.path.getsize(out) / 1024:.1f} KB)")


# ── Run all ──────────────────────────────────────────────────────────────────
render_art10()
render_art11()
render_art12()
render_art14()

# Verification
print()
checks = {
    "slide_art10_leakage_flag.png":          (10,  600),
    "slide_art11_pipeline_table.png":        (15, 1500),
    "slide_art12_cv_rf_table.png":           (10,  800),
    "slide_art14_residual_moran_table.png":  (15, 1000),
}
for fname, (min_kb, max_kb) in checks.items():
    fpath = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(fpath):
        print(f"FAIL – file not found: {fname}")
    else:
        kb = os.path.getsize(fpath) / 1024
        if kb < min_kb:
            print(f"FAIL – {fname}: {kb:.1f} KB is suspiciously small")
        elif kb > max_kb:
            print(f"WARN – {fname}: {kb:.1f} KB is large; check figure dimensions")
        else:
            print(f"PASS – {fname}: {kb:.1f} KB")
