# =============================================================================
# kyanite_spot_analysis.py
#
# Batch analysis/visualization of per-spot geochemistry CSVs produced by
# xrf_h5_extract_spots.py (<grain_id>_spot_geochemistry.csv — spot
# coordinates, per-zone element/CL means, and XANES pre-edge class).
#
# Produces:
#   - a combined figure: a grid of pie charts, one per grain, showing the
#     Type 1/2/3 XANES class distribution ('Bad data' / unclassified spots,
#     and off-grain spots, are excluded from the pie charts entirely)
#   - CL vs. element scatter plots, one per element, pooling spots from all
#     input grains together and coloring by XANES class ('Bad data' /
#     unclassified spots ARE included here, as grey points; off-grain spots
#     are absent because they have no CL/element mean to plot in the first
#     place — see on_grain note below)
#   - box-and-whisker plots, one per element (same element list as the
#     scatter plots), showing that element's distribution grouped by XANES
#     class, to check for a correlation between class and element amount
#     ('Bad data' / unclassified excluded, same as the pie charts;
#     off-grain spots absent for the same reason as the scatter plots)
#   - a labeled spot-location map per grain: the registered CL image with
#     each spot plotted at its pixel location, colored by XANES class and
#     labeled with its spot number — off-grain spots are shown too, marked
#     with a distinct shape (still colored by class) rather than dropped
#   - a PCA scatter (PC1 vs PC2) over a chosen element list (PCA_ELEMENTS),
#     pooling spots from all input grains, colored by XANES class ('Bad
#     data' / unclassified spots ARE included, as grey points, same as the
#     CL vs. element scatter) — spots missing any PCA_ELEMENTS value are
#     dropped (this also drops off-grain spots, for the same reason as the
#     scatter/box plots)
#
# on_grain (from xrf_h5_extract_spots.py): False means the spot's sampling
# zone didn't overlap the grain mask at all — it sampled some other phase,
# not kyanite. Its CL and every element mean are NaN for that reason (not a
# data-quality problem), which is exactly why the scatter/box/PCA analyses
# above already exclude it via their normal NaN handling — no separate
# filtering needed there. Its category_label (XANES pre-edge class) is left
# untouched, though, since oxidation state is a property of whatever phase
# was actually sampled, not of kyanite specifically, and is still useful
# data on its own — that's why the pie chart explicitly filters on_grain
# (so it doesn't misrepresent kyanite's class distribution) while the spot
# map keeps off-grain spots visible, just flagged, rather than discarding
# that class information outright. A CSV predating this column (or a spot
# whose grain had no mask at extraction time) is treated as on-grain by
# default (see on_grain_mask()).
#
# CSV_INPUT may be a single CSV or a directory; all *_spot_geochemistry.csv
# files in a directory are processed. Per-spot CSVs are reusable data (also
# read by xanes_rf_classifier.py) and live in figs/data/.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import tifffile
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull, QhullError
from kyanite_palette import (BLUE, ORANG, GREY, CATEGORY_ORDER, element_colors as _element_colors,
                              CATEGORY_COLORS as _SHARED_CATEGORY_COLORS)

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

# Anchored to this script's own location (not a hardcoded machine-specific
# path) so the same file runs unmodified on any machine/cluster this repo is
# checked out on.
_REPO_ROOT = Path(__file__).resolve().parent

CSV_INPUT = _REPO_ROOT / 'figs' / 'data'    # file or directory of *_spot_geochemistry.csv
FIGS_DIR  = _REPO_ROOT / 'figs'             # where <grain_id>_CL_registered.tif live
OUT_DIR   = _REPO_ROOT / 'figs' / 'spot_analysis'

ANALYSES = 'all'   # 'pie', 'scatter', 'box', 'map', 'pca', 'all', or a list of these

# Columns to make a pooled "CL vs element" scatter plot for.
# None = auto-detect every element ROI column present in the union of all input files.
SCATTER_ELEMENTS = ['Cr_Ka', 'Fe_Ka', 'V_Ka', 'Mn_Ka', 'Ti_Ka']

# Element columns considered by the PCA scatter (independent of SCATTER_ELEMENTS —
# PCA is sensitive to which variables are included, so this is chosen deliberately
# rather than reusing the scatter/box list).
PCA_ELEMENTS = ['Cr_Ka', 'V_Ka', 'Fe_Ka', 'Ti_Ka', 'Mn_Ka']
PCA_LOG_TRANSFORM = True   # log10-transform elements before z-scoring/PCA, same as kyanite_pca.py
PCA_N_PCS_SCREE = None     # number of PCs shown on the scree plot; None = all (len(PCA_ELEMENTS))
PCA_LOADING_THRESHOLD = 0.3   # |loading| >= this is highlighted on the loadings plot

# Convex-hull outline around each class's points on the PC1-vs-PC2 scatter, so the
# footprint each class occupies in PC space is easy to compare at a glance.
PCA_CLUSTER_OUTLINES = True
PCA_CLUSTER_CLASSES  = None   # which category_label values get an outline; None = CATEGORY_ORDER
                               # (i.e. skip 'Bad data'/unclassified — not a real class to contour)
PCA_CLUSTER_ALPHA    = 0.12   # hull fill alpha (edge is drawn solid at full class color)

SAVE_FIG   = True
SHOW_TITLE = True

# Fixed pie-slice order/coloring, so every grain's pie is comparable at a
# glance. Type 1/2/3 colors come from kyanite_palette (shared with
# xanes_rf_classifier.py's CATEGORY_ORDER and xanes_plot.py's CATEGORY_COLORS,
# which keys its own grey fallback 'Ambiguous' instead of 'Bad data' — see
# CLAUDE.md's "Color conventions" section).
CATEGORY_COLORS = {**_SHARED_CATEGORY_COLORS, 'Bad data': GREY}
# GREY: NaN / unmatched category_label renders identically to 'Bad data'

# Non-element columns from xrf_h5_extract_spots.py's schema — everything else
# in a spot CSV is treated as an element column.
METADATA_COLS = [
    'grain_id', 'spot', 'spot_id', 'area_name', 'category', 'category_label',
    'pixel_count', 'row_px_h5', 'col_px_h5', 'row_px_tiff', 'col_px_tiff',
    'row_matlab', 'col_matlab', 'x_mm', 'y_mm', 'x_rel_um', 'y_rel_um',
    'zone_radius_um', 'zone_pixel_count', 'zone_mask_px_count', 'on_grain', 'CL',
]

SPOT_LABEL_FONTSIZE = 6
SPOT_LABEL_OFFSET   = (4, 4)   # points

# plot_spot_map's legend is placed in whichever image corner has the fewest
# spots within this fraction of the image width/height from that corner —
# so it doesn't sit on top of spot markers, which vary in location grain to
# grain (a single fixed loc, e.g. 'upper right', would hide spots for some
# grains, as seen on NA-CM-G12B7-01's spot 4).
LEGEND_CORNER_MARGIN_FRAC = 0.3

# =============================================================================
# LOAD
# =============================================================================

input_path = Path(CSV_INPUT)
if input_path.is_dir():
    csv_files = sorted(input_path.glob('*_spot_geochemistry.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No *_spot_geochemistry.csv files found in {input_path}')
else:
    csv_files = [input_path]

print(f'Processing {len(csv_files)} CSV(s):')

grain_frames = {}   # grain_id -> its own DataFrame (own element-column subset)
for path in csv_files:
    df = pd.read_csv(path)
    grain_id = df['grain_id'].iloc[0] if len(df) else path.stem.replace('_spot_geochemistry', '')
    grain_frames[grain_id] = df
    print(f'  {grain_id}: {len(df)} spot(s)')

# pd.concat(sort=False) over frames with differing element columns (ROI lists vary
# per grain) produces the column union, NaN-filling rows from grains that lack a
# given column — exactly what the pooled per-element scatter needs.
combined = pd.concat(grain_frames.values(), ignore_index=True, sort=False)

out_dir = Path(OUT_DIR)
if SAVE_FIG:
    out_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def resolved_color(label):
    if pd.isna(label):
        return GREY
    return CATEGORY_COLORS.get(label, GREY)


def on_grain_mask(df):
    """Boolean mask, True for spots on the grain. CSVs without an on_grain
    column (extracted before this column existed) are treated as all
    on-grain; NaN (indeterminate — no grain mask was available at extraction
    time) is also treated as on-grain, the same conservative default."""
    if 'on_grain' not in df.columns:
        return pd.Series(True, index=df.index)
    return df['on_grain'] != False


def detect_elements(df):
    return [c for c in df.columns if c not in METADATA_COLS]


def element_availability(grain_frames, element):
    have = [gid for gid, df in grain_frames.items() if element in df.columns]
    missing = [gid for gid, df in grain_frames.items() if element not in df.columns]
    return have, missing


# =============================================================================
# ANALYSIS 1 — XANES class pie chart grid
# =============================================================================

def pie_counts(df):
    """Restricted to on-grain spots — this pie characterizes the grain's own
    XANES class distribution, and an off-grain spot's classification belongs
    to whatever other phase it actually sampled, not to this grain."""
    on_grain = df[on_grain_mask(df)]
    n_off_grain = len(df) - len(on_grain)
    sub = on_grain[on_grain['category_label'].isin(CATEGORY_ORDER)]
    counts = sub['category_label'].value_counts()
    return [int(counts.get(c, 0)) for c in CATEGORY_ORDER], len(sub), len(on_grain), n_off_grain


def plot_pie_grid(grain_frames):
    grain_ids = sorted(grain_frames)
    n = len(grain_ids)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    colors = [CATEGORY_COLORS[c] for c in CATEGORY_ORDER]

    for ax, grain_id in zip(axes, grain_ids):
        df = grain_frames[grain_id]
        counts, n_classified, n_total, n_off_grain = pie_counts(df)
        if n_classified == 0:
            ax.text(0.5, 0.5, 'no classified\nspots', ha='center', va='center',
                    fontsize=8, color='gray', transform=ax.transAxes)
            ax.axis('off')
        else:
            # Pass all 3 counts, in fixed order/colors, even if some are 0 — ax.pie
            # tolerates zero-value wedges, keeping every grain's slice color/position
            # identical whether or not it has spots of a given type.
            ax.pie(counts, colors=colors, startangle=90,
                   wedgeprops=dict(edgecolor='white', linewidth=0.5))
        title = f'{grain_id}\n(n={n_classified}/{n_total})'
        if n_off_grain:
            title += f'\n{n_off_grain} off-grain excluded'
        ax.set_title(title, fontsize=9)

    for ax in axes[n:]:
        ax.axis('off')

    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c]) for c in CATEGORY_ORDER]
    fig.legend(handles, CATEGORY_ORDER, loc='lower center', ncol=len(CATEGORY_ORDER),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    if SHOW_TITLE:
        fig.suptitle('XANES pre-edge class distribution by grain\n'
                      '(Bad data / unclassified and off-grain spots excluded)', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# =============================================================================
# ANALYSIS 2 — pooled CL vs. element scatter
# =============================================================================

LEGEND_ORDER = CATEGORY_ORDER + ['Bad data']


def plot_cl_scatter(combined, element):
    sub = combined[['grain_id', 'CL', element, 'category_label']].dropna(subset=['CL', element])
    if sub.empty:
        return None, sub

    fig, ax = plt.subplots(figsize=(7, 5))
    for label in LEGEND_ORDER:
        if label == 'Bad data':
            mask = sub['category_label'].isna() | (sub['category_label'] == 'Bad data')
            color, leg_label = GREY, 'Bad data / unclassified'
        else:
            mask = sub['category_label'] == label
            color, leg_label = CATEGORY_COLORS[label], label
        if mask.any():
            ax.scatter(sub.loc[mask, element], sub.loc[mask, 'CL'],
                       s=14, alpha=0.7, color=color, edgecolors='none', label=leg_label)

    if len(sub) >= 2 and sub[element].std() > 0 and sub['CL'].std() > 0:
        r = np.corrcoef(sub[element], sub['CL'])[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}\nn = {len(sub):,}',
                transform=ax.transAxes, va='top', fontsize=9)
    else:
        ax.text(0.05, 0.95, f'n = {len(sub):,}', transform=ax.transAxes, va='top', fontsize=9)

    ax.set_xlabel(element)                       # x = element, y = CL — matches
    ax.set_ylabel('CL intensity (norm.)')         # kyanite_figures.py's scatter convention
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if SHOW_TITLE:
        ax.set_title(f'CL vs {element} — all grains (n={len(sub):,})', fontsize=11)
    plt.tight_layout()
    return fig, sub


# =============================================================================
# ANALYSIS 3 — element distribution by XANES class (box-and-whisker)
# =============================================================================

def plot_element_boxplot(combined, element):
    """One box per XANES class (Type 1/2/3 only — 'Bad data'/unclassified
    excluded, matching the pie chart's convention of only comparing real
    classes) for this element, to check for a class/element correlation."""
    sub = combined[combined['category_label'].isin(CATEGORY_ORDER)][['category_label', element]].dropna()
    if sub.empty:
        return None, sub

    groups = [sub.loc[sub['category_label'] == c, element].values for c in CATEGORY_ORDER]
    labels = [f'{c}\n(n={len(g)})' for c, g in zip(CATEGORY_ORDER, groups)]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.5),
                     flierprops=dict(marker='.', markersize=4, alpha=0.5))
    for patch, c in zip(bp['boxes'], CATEGORY_ORDER):
        patch.set_facecolor(CATEGORY_COLORS[c])
        patch.set_alpha(0.6)

    # Overlay individual points (jittered) for context beyond the box summary.
    rng = np.random.default_rng(0)
    for i, (c, g) in enumerate(zip(CATEGORY_ORDER, groups), start=1):
        if len(g):
            jitter = rng.uniform(-0.08, 0.08, size=len(g))
            ax.scatter(np.full(len(g), i) + jitter, g, s=10, alpha=0.4,
                       color=CATEGORY_COLORS[c], edgecolors='none', zorder=3)

    ax.set_xlabel('XANES pre-edge class')
    ax.set_ylabel(element)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    if SHOW_TITLE:
        ax.set_title(f'{element} by XANES class (n={len(sub):,})', fontsize=11)
    plt.tight_layout()
    return fig, sub


# =============================================================================
# ANALYSIS 4 — PCA (scatter, scree, biplot, loadings) by XANES class
# =============================================================================

def prepare_pca_data(combined, elements, log_transform):
    """Rows with a value for every PCA element, optionally log10-transformed.
    Returns (sub, X) with matching row order/index."""
    cols = list(elements)
    sub = combined[cols + ['category_label']].dropna(subset=cols).copy()
    X = sub[cols].astype(float)
    if log_transform:
        X = np.log10(X)
    valid = np.isfinite(X.values).all(axis=1)
    return sub.loc[valid], X.loc[valid]


def compute_pca(X):
    """Fit PCA on all components (not just PC1/PC2) so the scree plot can show
    the full variance spectrum; scatter/biplot/loadings then just slice PC1/PC2
    out of the same fit instead of re-running PCA per plot."""
    X_scaled = StandardScaler().fit_transform(X.values)
    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100
    loadings = pca.components_.T   # elements x components
    return scores, explained, loadings


def _class_masks(sub):
    """category_label -> (label, boolean mask, color, legend label), in
    LEGEND_ORDER, shared by every PCA plot that colors points/classes."""
    out = []
    for label in LEGEND_ORDER:
        if label == 'Bad data':
            mask = (sub['category_label'].isna() | (sub['category_label'] == 'Bad data')).values
            color, leg_label = GREY, 'Bad data / unclassified'
        else:
            mask = (sub['category_label'] == label).values
            color, leg_label = CATEGORY_COLORS[label], label
        out.append((label, mask, color, leg_label))
    return out


def _draw_hull(ax, points, color, alpha):
    """Convex-hull outline over a class's 2D points; silently skipped if the
    points are too few or degenerate (collinear) to form a hull at all."""
    if len(points) < 3:
        return
    try:
        hull = ConvexHull(points)
    except QhullError:
        return
    ax.add_patch(plt.Polygon(points[hull.vertices], closed=True, facecolor=color,
                              edgecolor=color, alpha=alpha, linewidth=1.5, zorder=1))


def plot_pca_scatter(sub, scores, explained, elements):
    hull_classes = CATEGORY_ORDER if PCA_CLUSTER_CLASSES is None else PCA_CLUSTER_CLASSES

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, mask, color, leg_label in _class_masks(sub):
        if not mask.any():
            continue
        if PCA_CLUSTER_OUTLINES and label in hull_classes:
            _draw_hull(ax, scores[mask][:, :2], color, PCA_CLUSTER_ALPHA)
        ax.scatter(scores[mask, 0], scores[mask, 1], s=18, alpha=0.7,
                   color=color, edgecolors='none', label=leg_label, zorder=2)

    ax.axhline(0, color='0.7', lw=0.5, zorder=0)
    ax.axvline(0, color='0.7', lw=0.5, zorder=0)
    ax.set_xlabel(f'PC1 ({explained[0]:.1f}% var.)')
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}% var.)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if SHOW_TITLE:
        ax.set_title(f'PCA of {", ".join(elements)} (n={len(sub):,})', fontsize=11)
    plt.tight_layout()
    return fig


def plot_pca_scree(explained, n_pcs=None):
    n = len(explained) if n_pcs is None else min(n_pcs, len(explained))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, n + 1), explained[:n], color=BLUE)
    ax.plot(range(1, n + 1), np.cumsum(explained[:n]), 'o-', color=ORANG, lw=1.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if SHOW_TITLE:
        ax.set_title('PCA scree plot', fontsize=11)
    plt.tight_layout()
    return fig


def plot_pca_loadings(loadings, elements, pcs=(1, 2), threshold=0.3):
    fig, axes = plt.subplots(1, len(pcs), figsize=(5 * len(pcs), 4.5), squeeze=False)
    axes = axes.ravel()
    for ax, pc in zip(axes, pcs):
        vals = loadings[:, pc - 1]
        order = np.argsort(vals)[::-1]
        sorted_vals = vals[order]
        sorted_names = [elements[i] for i in order]
        # Fill = fixed element color (so the same element reads the same
        # color across every PC/grain/script); border = significance,
        # independent of element identity — a black outline if |loading|
        # clears the threshold, no border otherwise.
        fill_colors = _element_colors(sorted_names)
        edge_colors = ['black' if abs(v) >= threshold else 'none' for v in sorted_vals]
        edge_widths = [1.5 if abs(v) >= threshold else 0 for v in sorted_vals]

        ax.bar(range(len(sorted_vals)), sorted_vals, color=[fill_colors[n] for n in sorted_names],
               edgecolor=edge_colors, linewidth=edge_widths)
        ax.axhline(0, color='k', lw=1)
        ax.axhline(threshold, color='k', ls='--', lw=0.5)
        ax.axhline(-threshold, color='k', ls='--', lw=0.5)
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel(f'Loading on PC{pc}')
        ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)

    if SHOW_TITLE:
        fig.suptitle('PCA loadings', fontsize=11)
    plt.tight_layout()
    return fig


def plot_pca_biplot(sub, scores, loadings, explained, elements):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for label, mask, color, leg_label in _class_masks(sub):
        if mask.any():
            ax.scatter(scores[mask, 0], scores[mask, 1], s=16, alpha=0.5,
                       color=color, edgecolors='none', label=leg_label, zorder=2)

    # Fix the view to the score cloud's own extent *before* drawing arrows —
    # otherwise matplotlib autoscales to include the arrows/labels too, which
    # (since loadings are unit-norm per component, much larger in relative
    # terms than the scores along one axis) can balloon the axes and shrink
    # the score cloud down to a sliver.
    x_max = np.max(np.abs(scores[:, 0])) if len(scores) else 1.0
    y_max = np.max(np.abs(scores[:, 1])) if len(scores) else 1.0
    xlim, ylim = x_max * 1.15, y_max * 1.15
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    # Scale every loading vector by the same factor (preserves relative loading
    # magnitudes) so the longest arrow reaches ~85% of the tighter of the two
    # fixed axis half-ranges.
    loading_len = np.sqrt((loadings[:, :2] ** 2).sum(axis=1))
    max_loading_len = loading_len.max() if loading_len.size else 1.0
    scale = 0.85 * min(xlim, ylim) / max_loading_len if max_loading_len > 0 else 1.0

    for i, element in enumerate(elements):
        x, y = loadings[i, 0] * scale, loadings[i, 1] * scale
        ax.annotate('', xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2), zorder=3)
        ax.text(x * 1.1, y * 1.1, element, fontsize=9, color='black',
                ha='center', va='center', zorder=4)

    ax.axhline(0, color='0.7', lw=0.5, zorder=0)
    ax.axvline(0, color='0.7', lw=0.5, zorder=0)
    ax.set_xlabel(f'PC1 ({explained[0]:.1f}% var.)')
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}% var.)')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if SHOW_TITLE:
        ax.set_title(f'PCA biplot (n={len(sub):,})', fontsize=11)
    plt.tight_layout()
    return fig


# =============================================================================
# ANALYSIS 5 — per-grain spot-location map
# =============================================================================

def load_cl_background(grain_id):
    path = Path(FIGS_DIR) / f'{grain_id}_CL_registered.tif'
    if not path.exists():
        print(f'  WARNING: {path.name} not found — skipping spot map for {grain_id}.')
        return None
    return tifffile.imread(str(path))


OFF_GRAIN_MARKER = 'X'   # on-grain spots use 'o' — shape flags location, color still carries XANES class


def best_legend_corner(df, img_shape, margin_frac=LEGEND_CORNER_MARGIN_FRAC):
    """Whichever image corner has the fewest spots within margin_frac of the
    image's width/height from that corner -- i.e. the corner a legend box
    drawn there is least likely to sit on top of a spot marker."""
    n_rows, n_cols = img_shape[:2]
    margin_r, margin_c = margin_frac * n_rows, margin_frac * n_cols
    corners = {
        'upper right': lambda r, c: r <= margin_r and c >= n_cols - margin_c,
        'upper left':  lambda r, c: r <= margin_r and c <= margin_c,
        'lower right': lambda r, c: r >= n_rows - margin_r and c >= n_cols - margin_c,
        'lower left':  lambda r, c: r >= n_rows - margin_r and c <= margin_c,
    }
    counts = {loc: 0 for loc in corners}
    for row in df.itertuples():
        for loc, in_corner in corners.items():
            if in_corner(row.row_px_tiff, row.col_px_tiff):
                counts[loc] += 1
    return min(counts, key=counts.get)


def plot_spot_map(grain_id, df, cl_img):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cl_img, cmap='gray', origin='upper')
    # Deliberately NO ax.invert_yaxis() here. origin='upper' already puts row 0 at
    # the top, matching row_px_tiff/col_px_tiff's "row 0 = top" convention (same as
    # MATLAB's imagesc default used in xrf_display.m). Adding invert_yaxis() would
    # silently flip every spot vertically relative to the image.

    on_grain = on_grain_mask(df)
    for row, is_on_grain in zip(df.itertuples(), on_grain):
        color = resolved_color(row.category_label)
        marker = 'o' if is_on_grain else OFF_GRAIN_MARKER
        ax.scatter(row.col_px_tiff, row.row_px_tiff, s=28, color=color, marker=marker,
                   edgecolors='black', linewidths=0.5, zorder=3)
        ax.annotate(str(int(row.spot)), (row.col_px_tiff, row.row_px_tiff),
                    xytext=SPOT_LABEL_OFFSET, textcoords='offset points',
                    fontsize=SPOT_LABEL_FONTSIZE, color='white',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')],
                    zorder=4)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markerfacecolor=CATEGORY_COLORS[c],
                          markeredgecolor='black', label=c) for c in CATEGORY_ORDER]
    handles.append(plt.Line2D([0], [0], marker='o', linestyle='', markerfacecolor=GREY,
                              markeredgecolor='black', label='Bad data / unclassified'))
    handles.append(plt.Line2D([0], [0], marker=OFF_GRAIN_MARKER, linestyle='', markerfacecolor='0.5',
                              markeredgecolor='black', label='Off-grain (other phase) —\ncolor = XANES class still shown'))
    legend_loc = best_legend_corner(df, cl_img.shape)
    ax.legend(handles=handles, loc=legend_loc, fontsize=7, framealpha=0.7)
    if SHOW_TITLE:
        ax.set_title(f'{grain_id} — spot locations ({len(df)} spots)', fontsize=11)
    plt.tight_layout()
    return fig


# =============================================================================
# RUN
# =============================================================================

ALL_ANALYSES = ['pie', 'scatter', 'box', 'map', 'pca']
if ANALYSES == 'all':
    analyses = ALL_ANALYSES
elif isinstance(ANALYSES, (list, tuple)):
    analyses = list(ANALYSES)
else:
    analyses = [ANALYSES]
unknown = [a for a in analyses if a not in ALL_ANALYSES]
if unknown:
    raise ValueError(f"Unknown ANALYSES {unknown}; choose from {ALL_ANALYSES}, 'all', or a list of these.")

# Shared by 'scatter' and 'box' — same element list for both.
scatter_elements = list(SCATTER_ELEMENTS) if SCATTER_ELEMENTS is not None else detect_elements(combined)

if 'pie' in analyses:
    print('\n--- XANES class pie grid ---')
    fig = plot_pie_grid(grain_frames)
    if SAVE_FIG:
        out = out_dir / 'xanes_class_pie_grid.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out.name}')

if 'scatter' in analyses:
    print(f'\n--- CL vs element scatter ({len(scatter_elements)} element(s)) ---')
    for element in scatter_elements:
        if element not in combined.columns:
            print(f"  WARNING: '{element}' not found in any input file — skipping.")
            continue
        have, missing = element_availability(grain_frames, element)
        if missing:
            print(f"  WARNING: '{element}' not present in {len(missing)} grain CSV(s), "
                  f"excluded from this plot: {missing}")
        fig, sub = plot_cl_scatter(combined, element)
        if fig is None:
            print(f'  WARNING: no rows with both CL and {element} present — skipping.')
            continue
        print(f'  {element}: n={len(sub):,} pooled spot(s) from {sub["grain_id"].nunique()} grain(s)')
        if SAVE_FIG:
            out = out_dir / f'CL_vs_{element}_scatter.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            print(f'  Saved: {out.name}')

if 'box' in analyses:
    print(f'\n--- element distribution by XANES class ({len(scatter_elements)} element(s)) ---')
    for element in scatter_elements:
        if element not in combined.columns:
            print(f"  WARNING: '{element}' not found in any input file — skipping.")
            continue
        have, missing = element_availability(grain_frames, element)
        if missing:
            print(f"  WARNING: '{element}' not present in {len(missing)} grain CSV(s), "
                  f"excluded from this plot: {missing}")
        fig, sub = plot_element_boxplot(combined, element)
        if fig is None:
            print(f'  WARNING: no classified rows with {element} present — skipping.')
            continue
        print(f'  {element}: n={len(sub):,} classified spot(s)')
        if SAVE_FIG:
            out = out_dir / f'{element}_by_class_boxplot.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            print(f'  Saved: {out.name}')

if 'pca' in analyses:
    print(f'\n--- PCA: {PCA_ELEMENTS} ---')
    pca_elements = [e for e in PCA_ELEMENTS if e in combined.columns]
    missing_entirely = [e for e in PCA_ELEMENTS if e not in combined.columns]
    if missing_entirely:
        print(f"  WARNING: {missing_entirely} not found in any input file — excluded from PCA.")
    for element in pca_elements:
        have, missing = element_availability(grain_frames, element)
        if missing:
            print(f"  WARNING: '{element}' not present in {len(missing)} grain CSV(s) — "
                  f"spots from those grains are dropped from the PCA: {missing}")
    if len(pca_elements) < 2:
        print('  WARNING: fewer than 2 elements available — skipping PCA.')
    else:
        sub, X = prepare_pca_data(combined, pca_elements, PCA_LOG_TRANSFORM)
        if len(sub) < 2:
            print('  WARNING: fewer than 2 spots with complete data across all PCA elements — skipping.')
        else:
            scores, explained, loadings = compute_pca(X)
            print(f'  n={len(sub):,} spot(s) with complete data across {len(pca_elements)} element(s); '
                  f'PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}% var.')

            fig = plot_pca_scatter(sub, scores, explained, pca_elements)
            if SAVE_FIG:
                out = out_dir / 'pca_pc1_pc2_scatter.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'  Saved: {out.name}')

            fig = plot_pca_scree(explained, PCA_N_PCS_SCREE)
            if SAVE_FIG:
                out = out_dir / 'pca_scree.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'  Saved: {out.name}')

            fig = plot_pca_loadings(loadings, pca_elements, pcs=(1, 2), threshold=PCA_LOADING_THRESHOLD)
            if SAVE_FIG:
                out = out_dir / 'pca_loadings_pc1_pc2.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'  Saved: {out.name}')

            fig = plot_pca_biplot(sub, scores, loadings, explained, pca_elements)
            if SAVE_FIG:
                out = out_dir / 'pca_biplot.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'  Saved: {out.name}')

if 'map' in analyses:
    print(f'\n--- spot location maps ({len(grain_frames)} grain(s)) ---')
    for grain_id, df in grain_frames.items():
        print(f'  --- {grain_id} ({len(df)} spot(s)) ---')
        cl_img = load_cl_background(grain_id)
        if cl_img is None:
            continue
        fig = plot_spot_map(grain_id, df, cl_img)
        if SAVE_FIG:
            out = out_dir / f'{grain_id}_spot_map.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            print(f'  Saved: {out.name}')

plt.show()
