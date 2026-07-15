# =============================================================================
# kyanite_figures.py
#
# Figure generation for CL-EPMA pixel data.
#
# Loads one or more CSVs and produces scatter, contour (density contour lines
# over a scatter), heatmap (the same 2-D KDE, filled with a colorbar instead
# of drawn as lines), violin, binned box plots, or a corrmatrix grid of CL
# intensity vs. chosen elements. scatter, contour, and heatmap all include a
# linear fit line and Pearson r. corrmatrix is different from the rest: it
# doesn't loop per element — instead, for every ordered pair of elements in
# ELEMENTS, it forms the ratio (row element / column element) and shows the
# Pearson r of that ratio vs. CL as one annotated, color-mapped grid cell
# (self-ratio cells on the diagonal are masked out as meaningless).
#
# Two input formats are auto-detected by column name:
#   - Whole-grain CSVs from CL_EPMA_registration.m (*_pixel_data.csv):
#     one figure per element/plot-type for the whole grain.
#   - Per-region CSVs from CL_region_extraction.m (*_region_pixel_data.csv,
#     identified by a 'Region' column): one figure per element/plot-type,
#     with one subplot per region (small multiples) so regions can be
#     compared side by side.
#
# CSV_INPUT may be a single CSV file or a directory; all *_pixel_data.csv
# files found in a directory are processed automatically. Whole-grain and
# region CSVs both live in figs/data/ — the 'Region' column (checked after
# loading, not the filename) decides which code path a given file takes, so
# pointing CSV_INPUT at figs/data/ processes both kinds in one run. Figures
# are saved to WHOLE_GRAIN_OUTPUT_DIR / REGION_OUTPUT_DIR below, independent
# of wherever CSV_INPUT pointed — pointing CSV_INPUT elsewhere (e.g. a
# one-off copy of a CSV) does not change where figures land.
#
# When processing a region CSV with 'scatter' among PLOT_TYPE, an additional
# per-element figure overlays the region pixels (colored by region) on top
# of the whole grain's own CL-vs-element scatter (gray) — see
# REGION_HIGHLIGHT_ON_WHOLE_GRAIN below. This looks up the companion
# whole-grain *_pixel_data.csv by grain_id rather than joining on pixel
# identity (neither CSV carries pixel coordinates), so it draws the full
# grain population underneath and the region's own points on top — visually
# equivalent to excluding region pixels from the gray layer, without needing
# an exact per-pixel join.
#
# 'summary' is different again: it only fires when CSV_INPUT is a directory,
# and it's the one plot type that isn't per-grain — it pools every
# whole-grain (non-region) CSV found into a single grains x ELEMENTS heatmap
# of CL-vs-element Pearson r (annotated with n per cell), so correlation
# strength/consistency can be compared across grains at a glance. Skipped
# (with a warning) if CSV_INPUT isn't a directory or fewer than 2 whole-grain
# CSVs are found.
#
# Outlier spatial QC (OUTLIER_SPATIAL_QC, whole-grain CSVs only): every time
# the outlier logic below (SATURATION_FILTER + OUTLIER_METHOD) is applied to
# an element, also renders where it actually excluded pixels, directly on
# the masked 2-D element map — pixel_data.csv carries no row/col, so this
# reloads the raw element TIFF (MAPS_DIR) and grain mask TIFF (MASK_DIR)
# instead of using the CSV. Skipped per element (with a warning) if either
# file isn't found. This is independent of PLOT_TYPE/OUTLIER_METHOD choice —
# it always shows whatever exclusion the currently-configured method (mad or
# percentile) actually produced, not a comparison between methods.
#
# 'distributions' is 'summary'-shaped: only fires when CSV_INPUT is a
# directory, and pools every whole-grain CSV found rather than looping
# per-grain. For each element, renders a grain x grain small-multiples grid
# of its raw value histogram and a second grid of its log10 histogram (each
# with a fitted normal curve + skew annotation), on each grain's full
# unfiltered masked population — this is what OUTLIER_METHOD='mad' actually
# assumes is log-normal-ish, so the grids are a direct sanity check of that
# assumption per grain/element, not just in aggregate. Also writes
# <ALL_GRAINS_LABEL>_element_distribution_stats.csv (skew/kurtosis, raw and
# log, per grain x element). Saved to DISTRIBUTION_QC_DIR alongside the other
# diagnostics. Skipped (with a warning) if CSV_INPUT isn't a directory or no
# whole-grain CSVs are found.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import tifffile
from pathlib import Path
from scipy.stats import gaussian_kde, skew, kurtosis, norm
from kyanite_palette import BLUE, ORANG, DIVERGING_CMAP, SEQUENTIAL_CMAP, region_colors

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

# Anchored to this script's own location (not a hardcoded machine-specific
# path) so the same file runs unmodified on any machine/cluster this repo is
# checked out on.
_REPO_ROOT = Path(__file__).resolve().parent

CSV_INPUT = _REPO_ROOT / 'figs' / 'data'   # file or directory
ELEMENTS  = ['Cr_Ka', 'V_Ka', 'Fe_Ka', 'Mn_Ka', 'Ti_Ka']          # CSV column names
PLOT_TYPE = 'all'      # 'scatter', 'violin', 'boxplot', 'contour', 'heatmap', 'corrmatrix', 'summary', 'distributions', 'all', or a list of these

# Where figures are saved — independent of CSV_INPUT, so pointing CSV_INPUT
# at figs/data/ (where the pixel-data CSVs actually live) never dumps PNGs
# in among the reusable data files. Whole-grain CSVs' figures go to
# WHOLE_GRAIN_OUTPUT_DIR; region CSVs' (has a 'Region' column) to
# REGION_OUTPUT_DIR. Both are created if missing.
WHOLE_GRAIN_OUTPUT_DIR = _REPO_ROOT / 'figs' / 'whole_grain'
REGION_OUTPUT_DIR      = _REPO_ROOT / 'figs' / 'regions'

# 'summary' output — None (default) saves alongside WHOLE_GRAIN_OUTPUT_DIR,
# since it's still a whole-grain-CSV-derived figure, just pooled across
# grains rather than per-grain. ALL_GRAINS_LABEL is the filename prefix
# (mirrors xanes_rf_classifier.py's OUTPUT_LABEL convention for pooled,
# not-per-grain outputs).
SUMMARY_OUTPUT_DIR = None
ALL_GRAINS_LABEL   = 'all_grains'

# 'corrmatrix' ignores the per-element looping above and instead builds one
# grid per grain (or per region) from every ordered pair of elements in
# ELEMENTS — set ELEMENTS to the full list of columns to compare (needs >=2).
CORRMATRIX_CMAP = DIVERGING_CMAP   # diverging colormap, centered at r = 0 (see kyanite_palette.py)

# A ratio's highlight figure (see plot_element_highlights) requires beating
# both raw component elements' |r| AND clearing this absolute floor, so
# noise-level "wins" (e.g. 0.05 beating 0.03) don't generate a figure.
CORRMATRIX_MIN_R = 0.5

# Highlight-figure panels are heavily overplotted (whole-grain pixel counts),
# same as the main scatter/contour/heatmap plots — so render each _beats
# figure once per style below, rather than plain scatter alone.
CORRMATRIX_BEATS_PLOT_TYPES = ['scatter', 'contour', 'heatmap']

# 'contour' and 'heatmap' both estimate the same 2-D KDE (contour draws it as
# lines over a scatter; heatmap draws it filled with a colorbar, no scatter).
CONTOUR_LEVELS = 8      # number of contour lines for 'contour'
HEATMAP_LEVELS = 30     # number of fill levels for 'heatmap' — more = smoother gradation
KDE_GRIDSIZE   = 150    # resolution of the density grid each is evaluated on (per axis)

# Binning — used by 'violin' and 'boxplot'.
# N_BINS splits the (filtered) element range into equal-width bins.
# Override with BIN_EDGES for explicit control (e.g. np.arange(0, 5000, 200)).
# In region mode, bins are computed independently per region (each region
# bins over its own filtered element range), unless BIN_EDGES is set, in
# which case the same absolute edges are shared across all regions.
N_BINS    = 10
BIN_EDGES = None

# Outlier removal on the element axis, applied per region in region mode.
# Two independent stages, in order:
#
# 1. Saturation/clipping detection (SATURATION_FILTER): flags and excludes
#    pixels piled up near the very top of an element's own value range — the
#    signature of a saturated/clipped detector channel, not ordinary
#    geological variation (a real continuous distribution rarely has many
#    pixels crowded into a razor-thin slice at its extreme). Max side only:
#    a pileup near the *min* is ordinary near-zero/below-detection-limit
#    data, not saturation, and is common/legitimate — flagging it would gut
#    real low-concentration pixels. Runs first and independently of
#    OUTLIER_METHOD below, since it targets a specific instrument artifact
#    rather than general statistical spread, and only ever fires (with a
#    printed WARNING) when a genuine pileup is detected — clean elements are
#    untouched.
# 2. Statistical spread trim (OUTLIER_METHOD): trims whatever's left.
#      'mad' (default) — robust modified z-score, 0.6745*(x - median)/MAD,
#        computed in log-space (element concentrations are right-skewed,
#        same assumption this project already makes before PCA elsewhere —
#        see LOG_TRANSFORM in kyanite_pca_rf.py), excluded where it exceeds
#        MAD_K_LO/MAD_K_HI. Adapts to how spread out each element's own
#        distribution actually is, instead of always chopping a fixed
#        top/bottom fraction — a clean element loses ~0 pixels, a
#        contaminated one loses whatever doesn't fit its own typical spread.
#        None for either K disables that side (default: no low-side trim,
#        matching the old PCT_LO=0 default).
#      'percentile' — legacy fixed-percentile behavior; PCT_LO=0/PCT_HI=100
#        disables it.
OUTLIER_METHOD = 'percentile'   # 'mad' or 'percentile'
MAD_K_LO = None
MAD_K_HI = 3.5
PCT_LO   = 0
PCT_HI   = 99

SATURATION_FILTER    = True     # detect+exclude near-max pileups
SATURATION_BAND_FRAC = 0.001    # width of the "near max" band, as a fraction of the element's own [min, max] range
SATURATION_MIN_FRAC  = 0.005    # minimum fraction of all pixels in that band to call it a pileup (0.5%)
SATURATION_MIN_COUNT = 5        # ...and at least this many pixels, so tiny grains/regions don't trigger on noise

# Spatial QC (whole-grain CSVs only): see header comment. Reloads the raw
# element TIFF + grain mask TIFF (pixel_data.csv has no row/col of its own),
# so it needs its own paths — MAPS_DIR mirrors CL_EPMA_registration.m's
# epma_dir convention (<grain_id>/ subfolder per grain), MASK_DIR mirrors
# where the mask TIFF is always written (figs/data/).
OUTLIER_SPATIAL_QC   = True
MAPS_DIR             = _REPO_ROOT / 'inputs' / 'maps'
MASK_DIR             = _REPO_ROOT / 'figs' / 'data'
OUTLIER_QC_DIR       = _REPO_ROOT / 'figs' / 'diagnostics'
SATURATION_QC_COLOR  = '#8B0000'   # dark red — saturation/clipping exclusion (ORANG below is reused for the statistical-trim exclusion, matching its existing "above-threshold" role elsewhere in this file)

# 'distributions' (see header comment) — None (default) saves alongside
# OUTLIER_QC_DIR, since it's the same kind of not-for-publishing sanity
# check. DIST_GRID_NCOLS controls the small-multiples grid width (rows =
# ceil(n_grains / DIST_GRID_NCOLS)).
DISTRIBUTION_QC_DIR = None
DIST_GRID_NCOLS     = 4

SAVE_FIG   = True      # False to display only
SHOW_TITLE = True      # True to add a grain/element/plot-type title

# Region CSVs only: also draw each region's points, colored by region, on
# top of the whole grain's gray CL-vs-element scatter (one figure per
# element). Requires 'scatter' in PLOT_TYPE and the companion whole-grain
# *_pixel_data.csv to be found; skipped (with a warning) otherwise.
REGION_HIGHLIGHT_ON_WHOLE_GRAIN = True

# Where to look for the companion whole-grain *_pixel_data.csv for a given
# region CSV. None = same directory as the region CSV itself (i.e.
# figs/data/, since whole-grain and region CSVs are colocated there by
# default).
WHOLE_GRAIN_DATA_DIR = None

# Region CSVs only: for the plot types listed in AXIS_MATCH_PLOT_TYPES, give
# every region's subplot in the *_<pt>_by_region.png small-multiples figure
# the same x/y axis limits as the companion whole-grain plot (rather than
# each autoscaling to its own region's data), so the panels can be
# overlaid/compared 1:1. For 'contour'/'heatmap' this only clips the
# displayed view to match — the KDE itself is still estimated from each
# region's own (local) data, so density fields aren't recomputed over the
# wider shared window. Falls back to independent per-region autoscaling
# (with a warning) if the companion whole-grain CSV or the element column
# can't be found.
MATCH_REGION_AXES_TO_WHOLE_GRAIN = True
AXIS_MATCH_PLOT_TYPES = ['scatter', 'contour', 'heatmap']

# =============================================================================
# RESOLVE INPUT → list of CSV paths
# =============================================================================

input_path = Path(CSV_INPUT)
if input_path.is_dir():
    csv_files = sorted(input_path.glob('*_pixel_data.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No *_pixel_data.csv files found in {input_path}')
else:
    csv_files = [input_path]

print(f'Processing {len(csv_files)} CSV(s):')
for p in csv_files:
    print(f'  {p.name}')

ALL_PLOT_TYPES = ['scatter', 'violin', 'boxplot', 'contour', 'heatmap', 'corrmatrix', 'summary', 'distributions']

if PLOT_TYPE == 'all':
    plot_types = ALL_PLOT_TYPES
elif isinstance(PLOT_TYPE, (list, tuple)):
    plot_types = list(PLOT_TYPE)
else:
    plot_types = [PLOT_TYPE]

unknown = [pt for pt in plot_types if pt not in ALL_PLOT_TYPES]
if unknown:
    raise ValueError(f"Unknown PLOT_TYPE(s) {unknown}; choose from {ALL_PLOT_TYPES}, 'all', or a list of these.")

# =============================================================================
# PER-AXES PLOT PRIMITIVES — each draws into a given Axes, so the same code
# renders either a single whole-grain figure or one subplot per region.
# =============================================================================

def compute_bins(x):
    if BIN_EDGES is not None:
        edges = np.asarray(BIN_EDGES, dtype=float)
    else:
        edges = np.linspace(x.min(), x.max(), N_BINS + 1)

    bw  = edges[1] - edges[0]
    dec = max(0, int(np.ceil(-np.log10(bw)))) if bw < 1 else 0
    fmt = f'.{dec}f'
    bin_labels = [f'[{edges[i]:{fmt}}, {edges[i+1]:{fmt}})'
                  for i in range(len(edges) - 1)]
    return edges, bin_labels


def build_plot_df(x, y, edges, bin_labels):
    bins = pd.cut(x, bins=edges, labels=bin_labels, include_lowest=True)
    plot_df = pd.DataFrame({'x': x, 'CL': y, 'bin': bins}).dropna()

    occupied = [lbl for lbl in bin_labels if (plot_df['bin'] == lbl).any()]
    plot_df  = plot_df[plot_df['bin'].isin(occupied)]
    plot_df['bin'] = plot_df['bin'].cat.remove_unused_categories()
    plot_df['bin'] = plot_df['bin'].cat.reorder_categories(occupied)
    counts = plot_df.groupby('bin', observed=True).size()
    return plot_df, occupied, counts


def plot_scatter(ax, x, y, element):
    ax.scatter(x, y, s=4, alpha=0.06, color=BLUE, linewidths=0)
    m, b = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 300)
    ax.plot(xfit, m * xfit + b, 'k-', lw=1.5)
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}\nslope = {m:.3g}\nn = {len(x):,}',
            transform=ax.transAxes, va='top', fontsize=9)
    ax.set_xlabel(element)
    ax.set_ylabel('CL intensity (norm.)')


def plot_violin(ax, x, y, element):
    edges, bin_labels = compute_bins(x)
    plot_df, occupied, counts = build_plot_df(x, y, edges, bin_labels)

    sns.violinplot(data=plot_df, x='bin', y='CL', ax=ax,
                   density_norm='count', inner='box', cut=0,
                   color=BLUE, linewidth=0.8)
    for i, lbl in enumerate(occupied):
        n = counts[lbl]
        label = f'n={n // 1000}k' if n >= 1000 else f'n={n}'
        ax.text(i, 0.97, label, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, color='gray',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
    step = max(1, len(occupied) // 10)
    ax.set_xticks(range(0, len(occupied), step))
    ax.set_xticklabels(occupied[::step], rotation=30, ha='right', fontsize=8)
    ax.set_xlabel(element)
    ax.set_ylabel('CL intensity (norm.)')


def plot_boxplot(ax, x, y, element):
    edges, bin_labels = compute_bins(x)
    plot_df, occupied, counts = build_plot_df(x, y, edges, bin_labels)

    sqrt_counts = np.sqrt(counts.values)
    widths = 0.6 * sqrt_counts / sqrt_counts.max()
    groups = [plot_df.loc[plot_df['bin'] == lbl, 'CL'].values
              for lbl in occupied]
    ax.boxplot(
        groups,
        positions=range(len(occupied)),
        widths=widths,
        patch_artist=True,
        medianprops=dict(color=ORANG, linewidth=2),
        boxprops=dict(facecolor=BLUE, alpha=0.35, linewidth=1.1),
        whiskerprops=dict(color=BLUE, linewidth=0.9),
        capprops=dict(color=BLUE, linewidth=0.9),
        flierprops=dict(marker='.', markersize=2, alpha=0.25, color=BLUE),
    )
    step = max(1, len(occupied) // 10)
    ax.set_xticks(range(0, len(occupied), step))
    ax.set_xticklabels(occupied[::step], rotation=30, ha='right', fontsize=8)
    ax.set_xlabel(element)
    ax.set_ylabel('CL intensity (norm.)')


def kde_grid(x, y, gridsize=KDE_GRIDSIZE):
    # Shared density estimate for 'contour' and 'heatmap', so the filled and
    # line versions show exactly the same underlying field.
    kde = gaussian_kde(np.vstack([x, y]))
    xx, yy = np.mgrid[x.min():x.max():gridsize * 1j, y.min():y.max():gridsize * 1j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def add_fit_and_r(ax, x, y, line_color, text_color, text_bg):
    m, b = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 300)
    ax.plot(xfit, m * xfit + b, color=line_color, lw=1.5, zorder=3)
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'r = {r:.3f}\nn = {len(x):,}',
            transform=ax.transAxes, va='top', fontsize=9, color=text_color,
            bbox=dict(facecolor=text_bg, alpha=0.55, edgecolor='none', pad=2))


def plot_contour(ax, x, y, element):
    # Light scatter for context, with density contour lines on top — clearer
    # than a plain scatter when points are heavily overplotted.
    ax.scatter(x, y, s=4, alpha=0.04, color='0.5', linewidths=0, zorder=1)
    xx, yy, zz = kde_grid(x, y)
    ax.contour(xx, yy, zz, levels=CONTOUR_LEVELS, colors=ORANG, linewidths=1.0, zorder=2)
    add_fit_and_r(ax, x, y, line_color='k', text_color='black', text_bg='white')
    ax.set_xlabel(element)
    ax.set_ylabel('CL intensity (norm.)')


def plot_heatmap(ax, x, y, element):
    # Same KDE field as 'contour', filled with many levels instead of drawn
    # as lines over a scatter — a smooth density heatmap with a colorbar.
    xx, yy, zz = kde_grid(x, y)
    cs = ax.contourf(xx, yy, zz, levels=HEATMAP_LEVELS, cmap=SEQUENTIAL_CMAP)
    ax.figure.colorbar(cs, ax=ax, label='density')
    add_fit_and_r(ax, x, y, line_color='c', text_color='white', text_bg='black')
    ax.set_xlabel(element)
    ax.set_ylabel('CL intensity (norm.)')


def plot_region_highlight(ax, x_all, y_all, region_xy, element):
    # Whole-grain population underneath, in gray; each region's own points
    # on top, colored — the region points are already a subset of the gray
    # cloud, so no exact pixel-level join is needed to get the right picture.
    ax.scatter(x_all, y_all, s=4, alpha=0.05, color='0.6', linewidths=0, zorder=1)
    colors = region_colors(region_xy.keys())
    for region, (x_r, y_r) in region_xy.items():
        ax.scatter(x_r, y_r, s=6, alpha=0.35, color=colors[region], linewidths=0,
                   zorder=2, label=f'{region} (n={len(x_r):,})')
    ax.legend(loc='best', fontsize=7, markerscale=2, framealpha=0.8)
    ax.set_xlabel(element)
    ax.set_ylabel('CL intensity (norm.)')


PLOT_FUNCS = {'scatter': plot_scatter, 'violin': plot_violin, 'boxplot': plot_boxplot,
              'contour': plot_contour, 'heatmap': plot_heatmap}


def render_plot(ax, plot_type, element, x, y):
    PLOT_FUNCS[plot_type](ax, x, y, element)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    sns.despine(ax=ax)


def saturation_mask(x, label, verbose=True):
    # Flags pixels piled up in a thin band near x's own max — the signature
    # of a clipped/saturated detector channel (many pixels tied at or near a
    # hard ceiling), not the single natural extreme point a smooth
    # continuous distribution would have. Max side only: a pileup near the
    # *min* is just ordinary near-zero/below-detection-limit concentration
    # data, not saturation, and is extremely common/legitimate in trace
    # element maps — flagging it there would gut real (low-concentration)
    # data. Band-based (not exact-value ties) so it still catches this after
    # per-pixel normalization (NORMALIZE_BY_CLOCK/I0 upstream) has nudged
    # what was originally an identical raw ceiling into slightly different
    # float values. verbose=False lets a second caller (the spatial QC
    # figure, which re-derives the same mask from the raw TIFF) skip
    # reprinting a warning filtered_xy already issued for this element.
    mask = np.zeros(len(x), dtype=bool)
    if not SATURATION_FILTER or len(x) == 0:
        return mask
    n = len(x)
    xmin, xmax = x.min(), x.max()
    band = SATURATION_BAND_FRAC * (xmax - xmin)
    if band == 0:
        return mask
    near_max = x >= xmax - band
    count = int(near_max.sum())
    if count >= SATURATION_MIN_COUNT and count / n >= SATURATION_MIN_FRAC:
        if verbose:
            print(f'  WARNING: {label}: {count:,} px ({count / n:.2%}) piled up near the max '
                  f'of its range — likely detector saturation/clipping, excluded')
        mask |= near_max
    return mask


def mad_keep_mask(x, k_lo, k_hi):
    # Robust modified z-score (Iglewicz & Hoya): 0.6745*(x - median)/MAD,
    # computed in log-space — element concentrations are right-skewed
    # (lognormal-ish), same assumption kyanite_pca_rf.py/kyanite_spot_analysis.py
    # already make before z-scoring for PCA. On raw values, MAD reads the
    # natural long high-concentration tail as "outliers" and would strip out
    # exactly the scientifically important pixels (e.g. high-Fe quenching
    # zones). Non-positive values (zero/negative, e.g. a background-
    # subtracted floor) can't be log-transformed and aren't high-tail
    # outliers anyway, so they pass through unevaluated (kept).
    # MAD == 0 means over half the (positive) values are identical — nothing
    # is statistically distinguishable as an outlier, so keep everything
    # rather than divide by zero.
    keep = np.ones(len(x), dtype=bool)
    positive = x > 0
    xs = x[positive]
    if len(xs) < 2:
        return keep

    lx = np.log(xs)
    med = np.median(lx)
    mad = np.median(np.abs(lx - med))
    if mad == 0:
        return keep

    z = 0.6745 * (lx - med) / mad
    sub_keep = np.ones(len(xs), dtype=bool)
    if k_lo is not None:
        sub_keep &= z >= -k_lo
    if k_hi is not None:
        sub_keep &= z <= k_hi
    keep[positive] = sub_keep
    return keep


def outlier_keep_mask(x):
    if OUTLIER_METHOD == 'mad':
        return mad_keep_mask(x, MAD_K_LO, MAD_K_HI)
    elif OUTLIER_METHOD == 'percentile':
        lo, hi = np.percentile(x, [PCT_LO, PCT_HI])
        return (x >= lo) & (x <= hi)
    else:
        raise ValueError(f"Unknown OUTLIER_METHOD {OUTLIER_METHOD!r}; choose 'mad' or 'percentile'")


def filtered_xy(df, element):
    x_all = df[element].values
    y_all = df['CL'].values

    sat = saturation_mask(x_all, element)
    x_s, y_s = x_all[~sat], y_all[~sat]
    if len(x_s) < 2:
        return x_s, y_s, int(sat.sum())

    keep = outlier_keep_mask(x_s)
    n_removed = int(sat.sum()) + int((~keep).sum())
    return x_s[keep], y_s[keep], n_removed


def outlier_spatial_qc_figure(grain_id, element, out_dir):
    # Renders where SATURATION_FILTER/OUTLIER_METHOD actually excluded
    # pixels, directly on the masked 2-D element map. pixel_data.csv has no
    # row/col of its own, so this reloads the raw element TIFF + grain mask
    # TIFF instead — exclusion decisions are scale-invariant (percentile and
    # log-space MAD are both unaffected by a positive scalar, e.g. whatever
    # normalize_epma applied), so which pixels get excluded here exactly
    # matches what filtered_xy just computed from the CSV, even though the
    # displayed concentration values may be in different (e.g. unnormalized
    # raw) units. Whole-grain only — region masks aren't saved as a TIFF.
    map_path = Path(MAPS_DIR) / grain_id / f'{grain_id}_{element}.tif'
    mask_path = Path(MASK_DIR) / f'{grain_id}_mask.tif'
    if not map_path.exists() or not mask_path.exists():
        missing = map_path if not map_path.exists() else mask_path
        print(f'  WARNING: outlier spatial QC skipped for {element} — file not found: {missing}')
        return

    arr = tifffile.imread(map_path).astype(float)
    mask = tifffile.imread(mask_path) > 128
    vals = arr[mask]
    if len(vals) < 2:
        return

    # Same two-stage logic as filtered_xy, but keeping the boolean masks
    # (rather than just the filtered arrays) so each stage can be colored
    # separately. verbose=False: filtered_xy already printed this warning
    # for this element earlier in the same run.
    sat = saturation_mask(vals, element, verbose=False)
    stat_excluded = np.zeros(len(vals), dtype=bool)
    rest_idx = np.where(~sat)[0]
    if len(rest_idx) >= 2:
        keep_rest = outlier_keep_mask(vals[rest_idx])
        stat_excluded[rest_idx[~keep_rest]] = True

    cat = np.zeros(len(vals), dtype=int)   # 0 = kept
    cat[sat] = 1                           # saturation-excluded
    cat[stat_excluded] = 2                 # statistical-trim-excluded
    cat2d = np.full(arr.shape, -1, dtype=int)
    cat2d[mask] = cat

    n = len(vals)
    n_sat, n_stat = int(sat.sum()), int(stat_excluded.sum())
    color_map = {
        -1: (1, 1, 1, 1),
        0:  (0.85, 0.85, 0.85, 1),
        1:  (*mcolors.to_rgb(SATURATION_QC_COLOR), 1.0),
        2:  (*mcolors.to_rgb(ORANG), 1.0),
    }
    rgba = np.zeros((*cat2d.shape, 4))
    for k, c in color_map.items():
        rgba[cat2d == k] = c

    masked_arr = np.where(mask, arr, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axes[0].imshow(masked_arr, cmap=SEQUENTIAL_CMAP)
    axes[0].set_title(f'{element} concentration (masked)', fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(rgba)
    axes[1].set_title(f'gray=kept, dark red=saturation ({n_sat / n:.1%}),\n'
                       f'orange=statistical trim, {OUTLIER_METHOD} ({n_stat / n:.1%})', fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if SHOW_TITLE:
        fig.suptitle(f'{grain_id} — {element}: outlier exclusion QC', fontsize=12)
    plt.tight_layout()

    if SAVE_FIG:
        out = out_dir / f'{grain_id}_{element}_outlier_exclusion_QC.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out.name}')
    plt.close(fig)


def filtered_ratio_xy(df, num, den):
    # Same statistical trim as filtered_xy (OUTLIER_METHOD), applied to the
    # ratio itself (the quantity actually being correlated against CL), plus
    # a finite-value mask to drop divide-by-zero/NaN ratios first. No
    # saturation check here — clipping is a property of a raw detector
    # channel, not a derived ratio.
    ratio_all = df[num].values / df[den].values
    y_all = df['CL'].values
    finite = np.isfinite(ratio_all)
    ratio_all, y_all = ratio_all[finite], y_all[finite]
    n_removed = int((~finite).sum())
    if len(ratio_all) == 0:
        return ratio_all, y_all, n_removed
    keep = outlier_keep_mask(ratio_all)
    n_removed += int((~keep).sum())
    return ratio_all[keep], y_all[keep], n_removed


def compute_corr_matrix(elements, df):
    # Off-diagonal cells: r(row element / col element) vs. CL. Diagonal
    # cells: r(element) vs. CL directly (the raw-element baseline), so
    # ratios can be compared against the plain-element correlation they'd
    # need to beat.
    n = len(elements)
    rmat = np.full((n, n), np.nan)
    for i, num in enumerate(elements):
        for j, den in enumerate(elements):
            if i == j:
                x, y, _ = filtered_xy(df, num)
            else:
                x, y, _ = filtered_ratio_xy(df, num, den)
            if len(x) >= 2 and np.std(x) > 0:
                rmat[i, j] = np.corrcoef(x, y)[0, 1]
    return pd.DataFrame(rmat, index=elements, columns=elements)


def plot_corr_matrix(ax, elements, df, rdf=None):
    if rdf is None:
        rdf = compute_corr_matrix(elements, df)
    n = len(elements)
    sns.heatmap(rdf, ax=ax, cmap=CORRMATRIX_CMAP, vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', annot_kws={'fontsize': 8},
                cbar_kws={'label': 'Pearson r'},
                square=True, linewidths=0.5, linecolor='white')
    # Diagonal (raw element vs. CL) outlined — it's the baseline the ratio
    # cells around it are being compared against, not a ratio itself.
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', linewidth=2))
    ax.set_xlabel('denominator (diagonal = raw element vs. CL)')
    ax.set_ylabel('numerator')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')
    return rdf


def compute_summary_matrix(elements, grain_dfs):
    # One row per grain (sorted), one column per element. A cell is left NaN
    # if that grain's CSV doesn't have the element column, or has too few
    # valid points — annotated as 'n/a' rather than a misleading 0.
    grains = sorted(grain_dfs.keys())
    rmat = np.full((len(grains), len(elements)), np.nan)
    nmat = np.full((len(grains), len(elements)), np.nan)
    for i, grain_id in enumerate(grains):
        df = grain_dfs[grain_id]
        for j, element in enumerate(elements):
            if element not in df.columns:
                continue
            x, y, _ = filtered_xy(df, element)
            if len(x) >= 2 and np.std(x) > 0:
                rmat[i, j] = np.corrcoef(x, y)[0, 1]
                nmat[i, j] = len(x)
    rdf = pd.DataFrame(rmat, index=grains, columns=elements)
    ndf = pd.DataFrame(nmat, index=grains, columns=elements)
    return grains, rdf, ndf


def plot_summary_heatmap(ax, elements, grains, rdf, ndf):
    sns.heatmap(rdf, ax=ax, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, center=0,
                annot=False, cbar_kws={'label': 'Pearson r'},
                linewidths=0.5, linecolor='white')
    for i, grain in enumerate(grains):
        for j, element in enumerate(elements):
            r, n = rdf.iloc[i, j], ndf.iloc[i, j]
            if pd.isna(r):
                label, color = 'n/a', 'gray'
            else:
                n_label = f'{int(n) // 1000}k' if n >= 1000 else f'{int(n)}'
                label = f'{r:.2f}\nn={n_label}'
                color = 'white' if abs(r) >= 0.6 else 'black'
            ax.text(j + 0.5, i + 0.5, label, ha='center', va='center',
                    fontsize=7, color=color)
    ax.set_xlabel('element')
    ax.set_ylabel('grain')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')


def element_distribution_stats(elements, grain_dfs):
    # Skew/kurtosis on each grain's full unfiltered masked population (not
    # the outlier-trimmed one — testing normality on already-trimmed data
    # would be circular), raw and log10 (log10 of the positive-valued subset
    # only, matching mad_keep_mask's own restriction). A true normal has
    # skew=0, excess kurtosis=0; the log columns are the direct check of the
    # log-normal assumption OUTLIER_METHOD='mad' relies on.
    rows = []
    for element in elements:
        for grain_id in sorted(grain_dfs.keys()):
            df = grain_dfs[grain_id]
            if element not in df.columns:
                continue
            x = df[element].values
            x = x[np.isfinite(x)]
            pos = x[x > 0]
            frac_nonpos = 1 - len(pos) / len(x) if len(x) else np.nan
            skew_raw, kurt_raw = skew(x), kurtosis(x)
            if len(pos) >= 2:
                lx = np.log10(pos)
                skew_log, kurt_log = skew(lx), kurtosis(lx)
            else:
                skew_log, kurt_log = np.nan, np.nan
            rows.append(dict(element=element, grain=grain_id, n=len(x),
                              frac_nonpositive=frac_nonpos,
                              skew_raw=skew_raw, kurt_raw=kurt_raw,
                              skew_log=skew_log, kurt_log=kurt_log))
    return pd.DataFrame(rows)


def plot_distribution_grid(element, grain_dfs, transform, out_dir):
    grains = sorted(g for g, df in grain_dfs.items() if element in df.columns)
    if not grains:
        return
    ncols = DIST_GRID_NCOLS
    nrows = int(np.ceil(len(grains) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, grain_id in zip(axes, grains):
        x = grain_dfs[grain_id][element].values
        x = x[np.isfinite(x)]
        if transform == 'log':
            x = x[x > 0]
            xlabel = f'log10({element})'
            x = np.log10(x)
        else:
            xlabel = element
        if len(x) < 2:
            ax.axis('off')
            continue

        ax.hist(x, bins=80, color=BLUE, alpha=0.8, density=True)
        mu, sigma = np.mean(x), np.std(x)
        if sigma > 0:
            xx = np.linspace(x.min(), x.max(), 300)
            ax.plot(xx, norm.pdf(xx, mu, sigma), color=ORANG, lw=1.5)
        ax.text(0.97, 0.95, f'skew={skew(x):.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8)
        ax.set_title(grain_id, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_yticks([])

    for ax in axes[len(grains):]:
        ax.axis('off')

    if SHOW_TITLE:
        fig.suptitle(f'{element}: {"log10" if transform == "log" else "raw"} distribution per grain '
                     f'(orange = fitted normal)', fontsize=13)
    plt.tight_layout()

    if SAVE_FIG:
        out = out_dir / f'{ALL_GRAINS_LABEL}_{element}_{transform}_distribution_grid.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved: {out.name}')
    plt.close(fig)


def plot_element_highlights(elements, df, rdf, grain_id, out_dir):
    # For each element E, find every ratio with E as the NUMERATOR whose |r|
    # beats BOTH raw elements it's built from AND clears CORRMATRIX_MIN_R —
    # i.e. a genuine, non-trivial improvement over either component alone —
    # and, if any exist, save one figure per E: E's raw scatter (the
    # baseline being beaten) followed by a scatter for each winning ratio.
    # Only checking the numerator role (not also denominator) means each
    # ratio is only ever considered for one figure, so a given scatter never
    # gets duplicated across two different elements' _beats figures.
    diag = np.diag(rdf.values)

    for i, elem in enumerate(elements):
        winners = []  # (label, num, den, r)
        for j, other in enumerate(elements):
            if j == i:
                continue
            threshold = max(abs(diag[i]), abs(diag[j]), CORRMATRIX_MIN_R)

            r_num = rdf.loc[elem, other]   # ratio elem/other — elem is numerator
            if pd.notna(r_num) and abs(r_num) > threshold:
                winners.append((f'{elem}/{other}', elem, other, r_num))

        if not winners:
            continue
        winners.sort(key=lambda w: abs(w[3]), reverse=True)
        print(f'  {elem}: {len(winners)} ratio(s) beat both raw elements')

        n_panels = 1 + len(winners)
        x0, y0, _ = filtered_xy(df, elem)
        ratio_xy = [(label, r) + filtered_ratio_xy(df, num, den)[:2]
                    for label, num, den, r in winners]

        for pt in CORRMATRIX_BEATS_PLOT_TYPES:
            fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
            axes = np.atleast_1d(axes)

            render_plot(axes[0], pt, elem, x0, y0)
            axes[0].set_title(f'{elem} (raw, r={diag[i]:.3f})', fontsize=10)

            for ax, (label, r, x, y) in zip(axes[1:], ratio_xy):
                render_plot(ax, pt, label, x, y)
                ax.set_title(f'{label} (r={r:.3f})', fontsize=10)

            if SHOW_TITLE:
                fig.suptitle(f'{grain_id} — ratios beating {elem} vs. CL ({pt})', fontsize=12)
            plt.tight_layout()

            if SAVE_FIG:
                out = out_dir / f'{grain_id}_corrmatrix_{elem}_beats_{pt}.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'  Saved: {out.name}')

# =============================================================================
# RUN
# =============================================================================

# Populated with each whole-grain (non-region) CSV as the main loop below
# reads it, so the 'summary' step at the end can reuse them without
# re-reading from disk.
whole_grain_summary_dfs = {}

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    region_mode = 'Region' in df.columns
    out_dir = Path(REGION_OUTPUT_DIR) if region_mode else Path(WHOLE_GRAIN_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if region_mode:
        grain_id = csv_path.stem.replace('_region_pixel_data', '')
        regions  = list(df['Region'].drop_duplicates())
        print(f'\n--- {grain_id} ({len(df):,} px, {len(regions)} region(s): {", ".join(regions)}) ---')

        whole_grain_df = None
        needs_whole_grain = ((REGION_HIGHLIGHT_ON_WHOLE_GRAIN and 'scatter' in plot_types)
                              or (MATCH_REGION_AXES_TO_WHOLE_GRAIN
                                  and any(pt in plot_types for pt in AXIS_MATCH_PLOT_TYPES)))
        if needs_whole_grain:
            wg_dir  = Path(WHOLE_GRAIN_DATA_DIR) if WHOLE_GRAIN_DATA_DIR else csv_path.parent
            wg_path = wg_dir / f'{grain_id}_pixel_data.csv'
            if wg_path.exists():
                whole_grain_df = pd.read_csv(wg_path)
            else:
                print(f'  WARNING: companion whole-grain CSV not found at {wg_path} — '
                      f'skipping region-highlight figure(s) and region-axis matching')
    else:
        grain_id = csv_path.stem.replace('_pixel_data', '')
        print(f'\n--- {grain_id} ({len(df):,} px) ---')
        # Excludes non-grain lookalikes the glob also picks up (e.g.
        # *_local_regression_pixel_data.csv, which has no 'CL' column) from
        # the summary heatmap as phantom all-n/a rows.
        if 'CL' in df.columns:
            whole_grain_summary_dfs[grain_id] = df

    available = [e for e in ELEMENTS if e in df.columns]
    missing   = [e for e in ELEMENTS if e not in df.columns]
    if missing:
        print(f'  WARNING: columns not found, skipping: {missing}')

    element_plot_types = [pt for pt in plot_types if pt not in ('corrmatrix', 'summary', 'distributions')]

    for element in available:

        if region_mode:
            n_r = len(regions)
            for pt in element_plot_types:
                fig, axes = plt.subplots(1, n_r, figsize=(5 * n_r, 5), sharey=True)
                axes = np.atleast_1d(axes)

                # Whole-grain filtered range for this element, computed once and
                # reused both to pin every region subplot's axes to it (so the
                # panels overlay 1:1) and as the background layer for the
                # region-highlight figure below.
                x_all = y_all = None
                needs_x_all = whole_grain_df is not None and (
                    pt == 'scatter' or (pt in AXIS_MATCH_PLOT_TYPES and MATCH_REGION_AXES_TO_WHOLE_GRAIN))
                if needs_x_all:
                    if element in whole_grain_df.columns:
                        x_all, y_all, _ = filtered_xy(whole_grain_df, element)
                    else:
                        print(f'  WARNING: {element} not in companion whole-grain CSV — falling back to '
                              f'independent per-region axis scaling / skipping region-highlight figure')
                match_axes = pt in AXIS_MATCH_PLOT_TYPES and MATCH_REGION_AXES_TO_WHOLE_GRAIN and x_all is not None

                for ax, region in zip(axes, regions):
                    sub = df[df['Region'] == region]
                    x, y, n_removed = filtered_xy(sub, element)
                    if len(x) < 2:
                        ax.text(0.5, 0.5, 'insufficient data', ha='center', va='center',
                                transform=ax.transAxes, fontsize=9, color='gray')
                        ax.set_title(region, fontsize=10)
                    else:
                        render_plot(ax, pt, element, x, y)
                        ax.set_title(region, fontsize=10)
                        print(f'  [{region}] {element} ({pt}): {len(x):,} px ({n_removed:,} removed)')
                    if match_axes:
                        ax.set_xlim(x_all.min(), x_all.max())
                        ax.set_ylim(y_all.min(), y_all.max())

                if SHOW_TITLE:
                    fig.suptitle(f'{grain_id} — {element} ({pt}) by region', fontsize=12)
                plt.tight_layout()

                if SAVE_FIG:
                    out = out_dir / f'{grain_id}_{element}_{pt}_by_region.png'
                    fig.savefig(out, dpi=200, bbox_inches='tight')
                    print(f'  Saved: {out.name}')

                if pt == 'scatter' and REGION_HIGHLIGHT_ON_WHOLE_GRAIN and x_all is not None:
                    region_xy = {}
                    for region in regions:
                        sub = df[df['Region'] == region]
                        x_r, y_r, _ = filtered_xy(sub, element)
                        if len(x_r) == 0:
                            continue
                        region_xy[region] = (x_r, y_r)

                    if region_xy:
                        fig_h, ax_h = plt.subplots(figsize=(7, 6))
                        plot_region_highlight(ax_h, x_all, y_all, region_xy, element)
                        ax_h.grid(True, alpha=0.25, linewidth=0.5)
                        sns.despine(ax=ax_h)
                        if SHOW_TITLE:
                            ax_h.set_title(f'{grain_id} — {element}: regions on whole-grain scatter', fontsize=11)
                        plt.tight_layout()

                        if SAVE_FIG:
                            out_h = out_dir / f'{grain_id}_{element}_scatter_regions_highlight.png'
                            fig_h.savefig(out_h, dpi=200, bbox_inches='tight')
                            print(f'  Saved: {out_h.name}')

        else:
            x, y, n_removed = filtered_xy(df, element)
            print(f'  {element}: {len(x):,} px after filter ({n_removed:,} removed)')

            if OUTLIER_SPATIAL_QC:
                qc_dir = Path(OUTLIER_QC_DIR)
                qc_dir.mkdir(parents=True, exist_ok=True)
                outlier_spatial_qc_figure(grain_id, element, qc_dir)

            for pt in element_plot_types:
                fig, ax = plt.subplots(figsize=(10, 5))
                render_plot(ax, pt, element, x, y)
                if SHOW_TITLE:
                    ax.set_title(grain_id, fontsize=11)
                plt.tight_layout()

                if SAVE_FIG:
                    out = out_dir / f'{grain_id}_{element}_{pt}.png'
                    fig.savefig(out, dpi=200, bbox_inches='tight')
                    print(f'  Saved: {out.name}')

    # 'corrmatrix' isn't per-element — one grid per grain (or per region)
    # built from every ordered pair within `available`.
    if 'corrmatrix' in plot_types:
        if len(available) < 2:
            print(f'  WARNING: corrmatrix needs >=2 ELEMENTS columns, got {len(available)}; skipping')
        elif region_mode:
            n_r = len(regions)
            fig, axes = plt.subplots(1, n_r, figsize=(5 * len(available) * n_r / 3 + 2, 5 * len(available) / 3 + 1))
            axes = np.atleast_1d(axes)
            for ax, region in zip(axes, regions):
                sub = df[df['Region'] == region]
                plot_corr_matrix(ax, available, sub)
                ax.set_title(region, fontsize=10)
            if SHOW_TITLE:
                fig.suptitle(f'{grain_id} — CL vs. element-ratio correlation by region', fontsize=12)
            plt.tight_layout()

            if SAVE_FIG:
                out = out_dir / f'{grain_id}_corrmatrix_by_region.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'  Saved: {out.name}')

        else:
            rdf = compute_corr_matrix(available, df)

            fig, ax = plt.subplots(figsize=(len(available) * 0.9 + 2, len(available) * 0.8 + 2))
            plot_corr_matrix(ax, available, df, rdf=rdf)
            if SHOW_TITLE:
                ax.set_title(f'{grain_id} — CL vs. element-ratio correlation', fontsize=11)
            plt.tight_layout()

            if SAVE_FIG:
                out = out_dir / f'{grain_id}_corrmatrix.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'  Saved: {out.name}')

            # Ratios that beat both of their raw component elements get
            # their own highlight figure (whole-grain mode only).
            plot_element_highlights(available, df, rdf, grain_id, out_dir)

# 'summary' isn't per-CSV — one heatmap pooling every whole-grain (non-region)
# CSV found across the whole CSV_INPUT directory, so it's handled once here
# rather than inside the per-csv_path loop above.
if 'summary' in plot_types:
    if not input_path.is_dir():
        print("\nWARNING: 'summary' requires CSV_INPUT to be a directory; skipping")
    elif len(whole_grain_summary_dfs) < 2:
        print(f'\nWARNING: summary needs >=2 whole-grain CSVs, found '
              f'{len(whole_grain_summary_dfs)}; skipping')
    else:
        summary_elements = [e for e in ELEMENTS
                             if any(e in gdf.columns for gdf in whole_grain_summary_dfs.values())]
        for e in summary_elements:
            missing_grains = [g for g, gdf in whole_grain_summary_dfs.items() if e not in gdf.columns]
            if missing_grains:
                print(f'  WARNING: {e} missing from {len(missing_grains)} grain(s), '
                      f'shown as n/a: {missing_grains}')
        missing_entirely = [e for e in ELEMENTS if e not in summary_elements]
        if missing_entirely:
            print(f'  WARNING: columns not found in any grain, skipping: {missing_entirely}')

        if len(summary_elements) == 0:
            print('  WARNING: no ELEMENTS columns found in any whole-grain CSV; skipping summary')
        else:
            grains, rdf, ndf = compute_summary_matrix(summary_elements, whole_grain_summary_dfs)
            print(f'\n--- {ALL_GRAINS_LABEL} summary ({len(grains)} grains) ---')

            fig, ax = plt.subplots(figsize=(len(summary_elements) * 0.9 + 2, len(grains) * 0.7 + 2))
            plot_summary_heatmap(ax, summary_elements, grains, rdf, ndf)
            if SHOW_TITLE:
                ax.set_title(f'{ALL_GRAINS_LABEL} — CL vs. element correlation by grain', fontsize=12)
            plt.tight_layout()

            if SAVE_FIG:
                summary_out_dir = Path(SUMMARY_OUTPUT_DIR) if SUMMARY_OUTPUT_DIR else Path(WHOLE_GRAIN_OUTPUT_DIR)
                summary_out_dir.mkdir(parents=True, exist_ok=True)
                out = summary_out_dir / f'{ALL_GRAINS_LABEL}_summary_r_heatmap.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'  Saved: {out.name}')

# 'distributions' — same shape as 'summary': pools every whole-grain CSV
# found rather than looping per-grain, so it's handled once here too.
if 'distributions' in plot_types:
    if not input_path.is_dir():
        print("\nWARNING: 'distributions' requires CSV_INPUT to be a directory; skipping")
    elif len(whole_grain_summary_dfs) < 1:
        print(f'\nWARNING: distributions needs >=1 whole-grain CSV, found '
              f'{len(whole_grain_summary_dfs)}; skipping')
    else:
        dist_elements = [e for e in ELEMENTS
                          if any(e in gdf.columns for gdf in whole_grain_summary_dfs.values())]
        missing_entirely = [e for e in ELEMENTS if e not in dist_elements]
        if missing_entirely:
            print(f'  WARNING: columns not found in any grain, skipping: {missing_entirely}')

        if len(dist_elements) == 0:
            print('  WARNING: no ELEMENTS columns found in any whole-grain CSV; skipping distributions')
        else:
            dist_dir = Path(DISTRIBUTION_QC_DIR) if DISTRIBUTION_QC_DIR else Path(OUTLIER_QC_DIR)
            dist_dir.mkdir(parents=True, exist_ok=True)
            print(f'\n--- {ALL_GRAINS_LABEL} distributions ({len(whole_grain_summary_dfs)} grains) ---')

            stats_df = element_distribution_stats(dist_elements, whole_grain_summary_dfs)
            if SAVE_FIG:
                stats_out = dist_dir / f'{ALL_GRAINS_LABEL}_element_distribution_stats.csv'
                stats_df.to_csv(stats_out, index=False)
                print(f'  Saved: {stats_out.name}')

            for element in dist_elements:
                med_skew_log = stats_df.loc[stats_df.element == element, 'skew_log'].median()
                med_kurt_log = stats_df.loc[stats_df.element == element, 'kurt_log'].median()
                print(f'  {element}: median log10 skew={med_skew_log:.2f}, kurtosis={med_kurt_log:.2f} '
                      f'across {len(whole_grain_summary_dfs)} grain(s)')
                plot_distribution_grid(element, whole_grain_summary_dfs, 'raw', dist_dir)
                plot_distribution_grid(element, whole_grain_summary_dfs, 'log', dist_dir)

plt.show()
