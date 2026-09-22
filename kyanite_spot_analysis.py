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
#   - the same per-grain map again, but with each spot colored on a
#     continuous scale by its Lorentzian pre-edge fit centroid (eV) — the
#     quantitative version of the Type 1/2/3 class — read from
#     inputs/prepeak_fits/<grain_id>_prepeak_fits.csv and joined on spot
#     number. Only grains that have a fit report get one. The color scale is
#     pooled across every input grain so the maps are comparable; a spot
#     whose fit failed is drawn grey, like a 'Bad data' spot on the class map
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

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import tifffile
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull, QhullError
from kyanite_palette import (BLUE, ORANG, GREY, CATEGORY_ORDER, OVERLAY_CMAP,
                              element_colors as _element_colors,
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
# Spot-numbering reference figures — a QC/lookup aid, not an analysis result, so
# they follow the project's "diagnostics live in figs/diagnostics/" rule rather
# than sitting alongside the analysis figures in OUT_DIR.
DIAGNOSTICS_DIR = _REPO_ROOT / 'figs' / 'diagnostics'

# Raw Lorentzian pre-edge fit reports, <grain_id>_prepeak_fits.csv, one per grain
# (Athena/Larch "Pre-edge Peak Fit Report" export). Only grains that have one get a
# centroid map; the rest are skipped silently.
PREPEAK_DIR = _REPO_ROOT / 'inputs' / 'prepeak_fits'

ANALYSES = 'all'   # 'pie', 'scatter', 'box', 'map', 'centroid_map', 'centroid_hist',
                   # 'spot_index', 'pca', 'all', or a list of these

# Marker fill for the 'spot_index' diagnostic. A single neutral color on purpose:
# that figure is a numbering key, so nothing in it should read as encoded data.
SPOT_INDEX_COLOR = 'white'

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

# --- Pre-edge fit centroid map (the continuous analog of the Type 1/2/3 spot map) ---
# The centroid is an absolute energy, not a signed/zero-centered quantity, so it takes
# a sequential colormap (see CLAUDE.md "Color conventions"), not the diverging one —
# specifically OVERLAY_CMAP ('viridis'), the project's sequential colormap for markers
# drawn over a dark CL image, whose low end stays legible on that background.
CENTROID_CMAP = OVERLAY_CMAP
# Color-scale limits, in eV. None = derive from CENTROID_RANGE_PCT percentiles of every
# valid centroid POOLED ACROSS ALL INPUT GRAINS, so one eV value is one color in every
# grain's map and the maps are directly comparable (the same reason the XANES class
# colors are fixed). Set explicitly to lock a scale across separate runs.
CENTROID_VMIN = None
CENTROID_VMAX = None
CENTROID_RANGE_PCT = (2, 98)   # robust limits; values outside are clamped, not dropped
                                # (the colorbar grows arrows to show clamping happened)

# Fit-quality screen. A fit flagged here is drawn in GREY at its real location, exactly
# like a 'Bad data'/unclassified spot on the XANES class map, and is excluded from the
# color-scale percentiles above.
#   - a NaN fit_centroid_stderr (lmfit could not estimate the uncertainty) is ALWAYS
#     treated as a failed fit
#   - CENTROID_MAX_STDERR: additionally reject any fit whose centroid stderr (eV)
#     exceeds this; None disables that extra check
# 0.20 eV: NA-CM-G12B7-02 carries two fits that are physically impossible for an Fe
# pre-edge centroid (spot 9 at 7118.28 eV, ~5 eV above the pre-edge region, and spot
# 17 at 7113.83 eV), and both report a far larger stderr (0.35, 0.37) than any other
# fit in the project. 0.20 sits in the clean gap above the next largest (0.19) and
# drops exactly those two. Every RH-XA-57081P-05/-07 fit is well under it (max
# stderr 0.019 and 0.098), so this screen changes nothing for those grains.
CENTROID_MAX_STDERR = 0.20

# --- Multi-grain centroid panel figure ---
# Grains cut from the same thin section are most easily compared side by side under
# one color scale, so 'centroid_map' can additionally emit a combined multipanel
# figure. The scale is already pooled across grains (above); this just puts the
# panels in one figure with a single shared colorbar.
#   None            — per-grain maps only, no combined figure
#   'all'           — one figure holding every grain that has a fit report
#   ['g1', 'g2']    — one figure holding just those grains
#   [['g1','g2'], ['g3','g4']]  — one figure per sub-list
#   {'section_A': ['g1','g2']}  — same, but naming the output file
# Output: <label>_centroid_map_panel.png
CENTROID_PANEL_GROUPS = 'all'
CENTROID_PANEL_NCOLS = None    # panel columns; None = one row, wrapping past 4 grains
CENTROID_PANEL_SIZE_IN = 5.0   # display size (inches) of the largest panel's long edge

# True: every panel is drawn at the same µm per inch, using each grain's own
# µm/px, so grains appear at their true relative size — the point of comparing
# grains from one section. Panels are padded (not stretched) to a common physical
# window, so a smaller grain simply gets more empty margin. False: each panel fills
# its own box instead, ignoring relative size — better when grains differ wildly.
CENTROID_PANEL_TRUE_SCALE = False
CENTROID_PANEL_SCALEBAR_UM = 200    # scale bar length in µm; None to omit
CENTROID_PANEL_PAD_COLOR = 'black'  # fill behind a grain smaller than the common window;
                                    # black to blend with the CL image's own dark background

# µm/px for CENTROID_PANEL_TRUE_SCALE. Read per grain from xrf_h5_to_tiff.py's
# metadata sidecar (<grain_id>_<el>_Ka.txt's step_size_pos1_um), the same mechanism
# xrf_display.py and CL_local_regression_map.py use — grains in this project are
# imaged at different resolutions (1.0 vs 2.0 µm/px), so a single hardcoded constant
# would silently draw them at the wrong relative size. CENTROID_PANEL_PIXEL_UM is
# only the fallback, used with a warning, if no sidecar is found/parseable.
MAPS_DIR = _REPO_ROOT / 'inputs' / 'maps'
CENTROID_PANEL_PIXEL_UM_FROM_SIDECAR = True
CENTROID_PANEL_PIXEL_UM = 2.0

# --- Centroid distribution summary ---
# The continuous counterpart to the XANES class pie grid: one histogram row per
# grain, stacked on a single shared energy axis so a shift between grains reads as
# one vertical scan. Every grain uses identical bin edges and identical axis
# limits — a histogram comparison is meaningless otherwise.
CENTROID_HIST_BINS = 20
CENTROID_HIST_RANGE = None        # (lo, hi) in eV; None = pooled valid-centroid range.
                                   # Deliberately independent of the maps' color limits:
                                   # those clamp outliers for display, but a histogram
                                   # should show every value where it actually falls.
# Color each bar by its own bin centre, through the same colormap/limits the maps
# use, so a bar's color matches the spots at that energy. False = flat house BLUE.
CENTROID_HIST_COLOR_BY_VALUE = True
CENTROID_HIST_SHOW_MEDIAN = True   # per-grain median line, for comparing central tendency
CENTROID_HIST_WIDTH_IN = 7.0
CENTROID_HIST_ROW_HEIGHT_IN = 1.1

# Filename prefix for a combined figure over every grain, matching
# kyanite_figures.py's ALL_GRAINS_LABEL convention.
ALL_GRAINS_LABEL = 'all_grains'

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
    # merged in from <grain_id>_prepeak_fits.csv (see load_prepeak_fits) — listed here
    # so detect_elements() doesn't mistake them for element ROI columns
    'fit_centroid', 'fit_centroid_stderr', 'fit_r2', 'centroid_ok',
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


# --- Lorentzian pre-edge fit centroids ------------------------------------------
# A separate, later-arriving measurement per spot: the quantitative/continuous
# version of the hand-assigned Type 1/2/3 pre-edge class. Joined onto the spot
# CSVs on spot number, the same key everything else in this project joins on.

def _prepeak_spot_number(dataset_name):
    """Spot number from a fit report's 'Data Set' name, e.g.
    'FeXAFS_57081P-Ky5-spot01.001' -> 1. The trailing '.NNN' is the scan repeat
    suffix, not the spot, so it's stripped before taking the trailing digits —
    after which this is the same 'spot number = trailing digits, whatever tag
    word precedes it' rule xrf_h5_extract_spots.py uses for h5 area names."""
    name = re.sub(r'\.\d+\s*$', '', str(dataset_name).strip())
    match = re.search(r'(\d+)\s*$', name)
    return int(match.group(1)) if match else None


def load_prepeak_fits(grain_id):
    """Parse <grain_id>_prepeak_fits.csv into a spot-indexed frame of centroid,
    centroid stderr, R^2 and a centroid_ok fit-quality flag. Returns None if the
    grain has no fit report."""
    path = Path(PREPEAK_DIR) / f'{grain_id}_prepeak_fits.csv'
    if not path.exists():
        return None

    # The report opens with a few '#' comment lines, the last of which is the real
    # (also '#'-prefixed) header. Find it rather than assuming a fixed line count.
    with open(path) as fh:
        lines = fh.readlines()
    header_idx = next((i for i, line in enumerate(lines) if re.match(r'^#\s*Data Set', line)), None)
    if header_idx is None:
        print(f'  WARNING: {path.name} has no "# Data Set" header line — skipping.')
        return None

    fits = pd.read_csv(path, skiprows=header_idx, skipinitialspace=True)
    fits.columns = [c.lstrip('#').strip() for c in fits.columns]
    missing = [c for c in ('Data Set', 'fit_centroid') if c not in fits.columns]
    if missing:
        print(f'  WARNING: {path.name} is missing column(s) {missing} — skipping.')
        return None

    fits['spot'] = fits['Data Set'].map(_prepeak_spot_number)
    unparsed = int(fits['spot'].isna().sum())
    if unparsed:
        print(f'  WARNING: {path.name}: {unparsed} row(s) with no parseable spot number — dropped.')
        fits = fits[fits['spot'].notna()]
    fits['spot'] = fits['spot'].astype(int)

    # Re-fitting a spot in a later session appends a second row for it rather than
    # replacing the first; keep the last (most recent) fit of each spot.
    n_dup = int(fits['spot'].duplicated().sum())
    if n_dup:
        print(f'  {path.name}: {n_dup} duplicate spot row(s) (re-fits) — keeping the last of each.')
        fits = fits.drop_duplicates('spot', keep='last')

    stderr = pd.to_numeric(fits.get('fit_centroid_stderr'), errors='coerce')
    ok = pd.to_numeric(fits['fit_centroid'], errors='coerce').notna() & stderr.notna()
    if CENTROID_MAX_STDERR is not None:
        ok &= stderr <= CENTROID_MAX_STDERR

    out = pd.DataFrame({
        'spot': fits['spot'].values,
        'fit_centroid': pd.to_numeric(fits['fit_centroid'], errors='coerce').values,
        'fit_centroid_stderr': stderr.values,
        'fit_r2': pd.to_numeric(fits.get('R^2'), errors='coerce').values
                  if 'R^2' in fits.columns else np.nan,
        'centroid_ok': ok.values,
    })
    return out.sort_values('spot').reset_index(drop=True)


_stray = sorted(Path(PREPEAK_DIR).parent.glob('*_prepeak_fits.csv'))
if _stray:
    print(f'\nWARNING: {len(_stray)} *_prepeak_fits.csv file(s) sit directly in '
          f'{Path(PREPEAK_DIR).parent}/ and will be ignored — move them into '
          f'{Path(PREPEAK_DIR).name}/: {[p.name for p in _stray]}')

_with_fits = []
for grain_id, df in grain_frames.items():
    fits = load_prepeak_fits(grain_id)
    if fits is None:
        continue
    merged = df.merge(fits, on='spot', how='left', validate='one_to_one')
    # A spot with no row in the fit report merges in as NaN, which makes
    # centroid_ok object dtype — and `~` on an object Series of Python bools is
    # integer bitwise NOT (~True == -2), not logical negation, so downstream
    # masks silently go wrong. Coerce to a real bool dtype once, here.
    merged['centroid_ok'] = merged['centroid_ok'].fillna(False).astype(bool)
    unmatched = sorted(set(fits['spot']) - set(df['spot']))
    if unmatched:
        print(f'  WARNING: {grain_id}: {len(unmatched)} fitted spot(s) have no row in the '
              f'spot geochemistry CSV and are not mapped: {unmatched}')
    grain_frames[grain_id] = merged
    n_ok = int(merged['centroid_ok'].fillna(False).sum())
    _with_fits.append(f'{grain_id} ({n_ok}/{len(merged)} spots fitted OK)')
if _with_fits:
    print(f'Pre-edge fit centroids loaded for: {", ".join(_with_fits)}')
else:
    print(f'No *_prepeak_fits.csv found in {PREPEAK_DIR} — centroid maps will be skipped.')

# pd.concat(sort=False) over frames with differing element columns (ROI lists vary
# per grain) produces the column union, NaN-filling rows from grains that lack a
# given column — exactly what the pooled per-element scatter needs.
combined = pd.concat(grain_frames.values(), ignore_index=True, sort=False)

out_dir = Path(OUT_DIR)
diagnostics_dir = Path(DIAGNOSTICS_DIR)
if SAVE_FIG:
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

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


def image_extent(cl_img, scale):
    """imshow extent for an image whose data coordinates are multiplied by `scale`
    (1.0 = pixels, µm/px = microns). Matplotlib's default extent puts pixel CENTRES
    on integer coordinates, which is where draw_spot plots row/col_px_tiff — so the
    half-pixel offsets are kept here rather than using a plain (0, w, h, 0) box,
    which would shift every spot half a pixel off its real location."""
    nrows, ncols = cl_img.shape[:2]
    return (-0.5 * scale, (ncols - 0.5) * scale,
            (nrows - 0.5) * scale, -0.5 * scale)


def draw_cl_background(ax, cl_img, scale=1.0):
    ax.imshow(cl_img, cmap='gray', origin='upper', extent=image_extent(cl_img, scale))
    # Deliberately NO ax.invert_yaxis() here. origin='upper' already puts row 0 at
    # the top, matching row_px_tiff/col_px_tiff's "row 0 = top" convention (same as
    # MATLAB's imagesc default used in xrf_display.m). Adding invert_yaxis() would
    # silently flip every spot vertically relative to the image.
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def draw_spot(ax, row, color, marker, scale=1.0, label=True):
    """One spot marker at its registered pixel location, optionally labeled with
    its spot number."""
    x, y = row.col_px_tiff * scale, row.row_px_tiff * scale
    ax.scatter(x, y, s=28, color=color, marker=marker,
               edgecolors='black', linewidths=0.5, zorder=3)
    if label:
        ax.annotate(str(int(row.spot)), (x, y),
                    xytext=SPOT_LABEL_OFFSET, textcoords='offset points',
                    fontsize=SPOT_LABEL_FONTSIZE, color='white',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')],
                    zorder=4)


def plot_spot_map(grain_id, df, cl_img):
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_cl_background(ax, cl_img)

    for row, is_on_grain in zip(df.itertuples(), on_grain_mask(df)):
        draw_spot(ax, row, resolved_color(row.category_label),
                  'o' if is_on_grain else OFF_GRAIN_MARKER)

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
# ANALYSIS 6 — per-grain pre-edge fit centroid map
#
# Same overlay as ANALYSIS 5, but each spot is colored by its Lorentzian
# pre-edge fit centroid (eV) on a continuous scale instead of by its
# hand-assigned Type 1/2/3 class — a quantitative measure of Fe speciation
# rather than a qualitative one. Everything else is deliberately identical:
# same pixel coordinates, same spot-number labels, same off-grain 'X' marker,
# same grey for a spot with no usable value.
# =============================================================================

def centroid_color_limits(grain_frames):
    """(vmin, vmax) in eV, from valid centroids pooled across every input grain,
    so one energy is one color in every grain's map. Returns None if no grain has
    a usable centroid."""
    if CENTROID_VMIN is not None and CENTROID_VMAX is not None:
        return CENTROID_VMIN, CENTROID_VMAX
    pooled = pd.concat(
        [df.loc[df['centroid_ok'].fillna(False), 'fit_centroid']
         for df in grain_frames.values() if 'centroid_ok' in df.columns],
        ignore_index=True) if any('centroid_ok' in df.columns for df in grain_frames.values()) \
        else pd.Series(dtype=float)
    pooled = pooled.dropna()
    if pooled.empty:
        return None
    lo_pct, hi_pct = CENTROID_RANGE_PCT
    vmin = CENTROID_VMIN if CENTROID_VMIN is not None else float(np.percentile(pooled, lo_pct))
    vmax = CENTROID_VMAX if CENTROID_VMAX is not None else float(np.percentile(pooled, hi_pct))
    if vmin == vmax:      # degenerate (one spot, or all identical) — give the bar width
        vmin, vmax = vmin - 0.05, vmax + 0.05
    return vmin, vmax


# Figure text stays publication-clean: the quantity and the grain, nothing else.
# Run metadata — how many spots were fitted, how many failed, whether the scale is
# shared — is reported on the console (and, for the numbering, in the spot_index
# diagnostic), not printed onto a figure that could go into a paper.
CENTROID_CBAR_LABEL = 'Fe pre-edge fit centroid (eV)'
CENTROID_TITLE = 'Fe pre-edge fit centroid'


def draw_centroid_spots(ax, df, mappable, vmin, vmax, scale=1.0):
    """Every spot of one grain, colored by fit centroid. Spots are deliberately
    NOT labeled — the numbers crowd each other badly wherever spots cluster, and
    they carry no information about the quantity being mapped. The 'spot_index'
    analysis renders a separate diagnostic figure that is just the numbering, to
    be read alongside these. Returns
    (n_ok, n_bad, n_below, n_above) — the last two count values clamped to the
    ends of the shared scale, so the colorbar can grow the matching arrows."""
    ok = df['centroid_ok'].fillna(False)
    n_bad = n_below = n_above = 0
    for row, is_on_grain, is_ok in zip(df.itertuples(), on_grain_mask(df), ok):
        if is_ok:
            # Normalize clamps out-of-range values to the end colors rather than
            # dropping them; count them so the clamping is visible on the colorbar.
            if row.fit_centroid < vmin:
                n_below += 1
            elif row.fit_centroid > vmax:
                n_above += 1
            color = mappable.to_rgba(row.fit_centroid)
        else:
            color = GREY   # failed/absent fit — same grey the class map uses for 'Bad data'
            n_bad += 1
        draw_spot(ax, row, color, 'o' if is_on_grain else OFF_GRAIN_MARKER,
                  scale=scale, label=False)
    return int(ok.sum()), n_bad, n_below, n_above


def centroid_extend(n_below, n_above):
    """Which colorbar end(s) need an arrow, given how many values clamped where."""
    if n_below and n_above:
        return 'both'
    if n_below:
        return 'min'
    if n_above:
        return 'max'
    return 'neither'


def format_ev_axis(axis):
    """Absolute photon energies, so show the full value on every tick. Matplotlib's
    default would factor out the shared ~7113 eV as a '+7.113e3' offset label,
    leaving ticks reading '0.25', '0.30', ... which is unreadable as an energy.
    Needed on any axis carrying centroid values — colorbar or x-axis alike."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}'))


def style_centroid_colorbar(cbar):
    cbar.set_label(CENTROID_CBAR_LABEL, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    format_ev_axis(cbar.ax.yaxis)


def centroid_legend_handles(df, n_bad):
    handles = []
    if n_bad:
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='', markerfacecolor=GREY,
                                  markeredgecolor='black', label='No usable fit'))
    if (~on_grain_mask(df)).any():
        handles.append(plt.Line2D([0], [0], marker=OFF_GRAIN_MARKER, linestyle='',
                                  markerfacecolor='0.5', markeredgecolor='black',
                                  label='Off-grain (other phase) —\ncolor = centroid still shown'))
    return handles


def plot_centroid_map(grain_id, df, cl_img, vmin, vmax):
    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=CENTROID_CMAP)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_cl_background(ax, cl_img)
    n_ok, n_bad, n_below, n_above = draw_centroid_spots(ax, df, mappable, vmin, vmax)

    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02,
                        extend=centroid_extend(n_below, n_above))
    style_centroid_colorbar(cbar)

    handles = centroid_legend_handles(df, n_bad)
    if handles:
        ax.legend(handles=handles, loc=best_legend_corner(df, cl_img.shape),
                  fontsize=7, framealpha=0.7)

    if SHOW_TITLE:
        ax.set_title(f'{grain_id} — {CENTROID_TITLE}', fontsize=11)
    plt.tight_layout()
    return fig, n_ok, n_bad, n_below + n_above


# =============================================================================
# ANALYSIS 6b — multi-grain centroid panel figure
#
# The same maps as ANALYSIS 6, tiled into one figure under a single shared
# colorbar, for comparing grains cut from the same thin section. The color
# scale is already pooled across grains, so this adds no new normalization —
# it just puts the panels side by side and draws the scale once.
# =============================================================================

# Same field and same sidecar convention xrf_display.py reads pixel size from.
_SIDECAR_PIXEL_UM_RE = re.compile(r'step_size_pos1_um\s*:\s*([-\d.eE]+)')


def read_pixel_um_from_sidecar(tif_path):
    """Fast-axis (X, pos1) pixel size in microns from xrf_h5_to_tiff.py's metadata
    sidecar for this TIFF (same base name, .txt extension). None if the sidecar is
    missing or the field can't be parsed."""
    sidecar = tif_path.with_suffix('.txt')
    if not sidecar.exists():
        return None
    m = _SIDECAR_PIXEL_UM_RE.search(sidecar.read_text())
    return float(m.group(1)) if m else None


def grain_pixel_um(grain_id):
    """(µm/px, from_sidecar) for one grain, from the first of its element maps that
    carries a parseable sidecar. Falls back to CENTROID_PANEL_PIXEL_UM."""
    if CENTROID_PANEL_PIXEL_UM_FROM_SIDECAR:
        folder = Path(MAPS_DIR) / grain_id
        if folder.is_dir():
            for tif in sorted(folder.glob('*.tif')):
                found = read_pixel_um_from_sidecar(tif)
                if found:
                    return found, True
    return CENTROID_PANEL_PIXEL_UM, False


def centroid_panel_groups(available):
    """Normalize CENTROID_PANEL_GROUPS into an ordered {label: [grain_id]} dict,
    dropping (with a warning) any grain that has no pre-edge fits."""
    spec = CENTROID_PANEL_GROUPS
    if spec is None:
        return {}
    if isinstance(spec, str):
        if spec != 'all':
            raise ValueError(f"CENTROID_PANEL_GROUPS={spec!r}; the only string accepted is 'all'.")
        groups = {ALL_GRAINS_LABEL: list(available)}
    elif isinstance(spec, dict):
        groups = {str(k): list(v) for k, v in spec.items()}
    else:
        seq = list(spec)
        if seq and all(isinstance(x, str) for x in seq):
            seq = [seq]           # a flat list of grain ids is a single group
        groups = {}
        for grains in seq:
            grains = list(grains)
            # Join short groups for a self-describing filename; fall back to a count
            # rather than generating an unusable 300-character name for a long one.
            label = '_'.join(grains) if len(grains) <= 3 else f'{len(grains)}_grains'
            groups[label] = grains

    resolved = {}
    for label, grains in groups.items():
        keep = [g for g in grains if g in available]
        dropped = [g for g in grains if g not in available]
        if dropped:
            print(f"  WARNING: panel group '{label}': no pre-edge fits for {dropped} — omitted.")
        if len(keep) < 2:
            print(f"  WARNING: panel group '{label}' has {len(keep)} grain(s) with fits "
                  f"— skipping (a panel figure needs at least 2).")
            continue
        resolved[label] = keep
    return resolved


def draw_scalebar(ax, length_um, scale, cl_img, px_um):
    """Horizontal scale bar in the lower-left of a panel. `scale` is the panel's
    data units per pixel (µm/px in true-scale mode, 1.0 in pixel mode), so the bar
    is length_um/px_um pixels long either way."""
    x0, x1, y1, y0 = image_extent(cl_img, scale)
    bar = (length_um / px_um) * scale
    mx, my = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    bx, by = x0 + mx, y1 - my
    ax.plot([bx, bx + bar], [by, by], color='white', lw=2.5, solid_capstyle='butt',
            zorder=5, path_effects=[pe.withStroke(linewidth=4.5, foreground='black')])
    ax.text(bx + bar / 2, by - 0.012 * (y1 - y0), f'{length_um:g} µm', color='white',
            ha='center', va='bottom', fontsize=7, zorder=5,
            path_effects=[pe.withStroke(linewidth=1.8, foreground='black')])


def plot_centroid_panel(label, panels, vmin, vmax):
    """panels: list of (grain_id, df, cl_img, px_um, px_from_sidecar)."""
    n = len(panels)
    ncols = CENTROID_PANEL_NCOLS or min(n, 4)
    ncols = max(1, min(int(ncols), n))
    nrows = int(np.ceil(n / ncols))
    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=CENTROID_CMAP)

    # In true-scale mode every panel shows the same physical window, sized to the
    # largest grain, so grains are drawn at their real relative size rather than
    # each being stretched to fill its own box.
    true_scale = CENTROID_PANEL_TRUE_SCALE
    if true_scale:
        win_w = max(img.shape[1] * px for _, _, img, px, _ in panels)
        win_h = max(img.shape[0] * px for _, _, img, px, _ in panels)
        long_edge = max(win_w, win_h)
        panel_w = CENTROID_PANEL_SIZE_IN * win_w / long_edge
        panel_h = CENTROID_PANEL_SIZE_IN * win_h / long_edge
    else:
        panel_w = panel_h = CENTROID_PANEL_SIZE_IN

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout='constrained',
                             figsize=(ncols * panel_w + 1.6, nrows * panel_h + 0.9))
    flat = axes.ravel()

    totals = {'ok': 0, 'bad': 0, 'below': 0, 'above': 0, 'spots': 0}
    legend_handles, seen_labels = [], set()
    for ax, (grain_id, df, cl_img, px_um, _) in zip(flat, panels):
        scale = px_um if true_scale else 1.0
        ax.set_facecolor(CENTROID_PANEL_PAD_COLOR)
        draw_cl_background(ax, cl_img, scale=scale)
        n_ok, n_bad, n_below, n_above = draw_centroid_spots(
            ax, df, mappable, vmin, vmax, scale=scale)
        totals['ok'] += n_ok
        totals['bad'] += n_bad
        totals['below'] += n_below
        totals['above'] += n_above
        totals['spots'] += len(df)

        if true_scale:
            # Centre each grain in the shared window. Y limits stay inverted
            # (row 0 at top), matching draw_cl_background's origin='upper'.
            cx = (cl_img.shape[1] - 1) / 2 * scale
            cy = (cl_img.shape[0] - 1) / 2 * scale
            ax.set_xlim(cx - win_w / 2, cx + win_w / 2)
            ax.set_ylim(cy + win_h / 2, cy - win_h / 2)

        ax.set_title(grain_id, fontsize=9)
        for handle in centroid_legend_handles(df, n_bad):
            if handle.get_label() not in seen_labels:
                seen_labels.add(handle.get_label())
                legend_handles.append(handle)

    if CENTROID_PANEL_SCALEBAR_UM:
        # One bar is enough when every panel shares a scale; otherwise each panel
        # has its own and needs its own bar.
        bar_axes = [(flat[0], panels[0])] if true_scale else list(zip(flat, panels))
        for ax, (_, _, cl_img, px_um, _) in bar_axes:
            draw_scalebar(ax, CENTROID_PANEL_SCALEBAR_UM,
                          px_um if true_scale else 1.0, cl_img, px_um)

    for ax in flat[n:]:
        ax.axis('off')

    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.03, pad=0.015,
                        extend=centroid_extend(totals['below'], totals['above']))
    style_centroid_colorbar(cbar)

    if legend_handles:
        fig.legend(handles=legend_handles, loc='lower left', fontsize=7,
                   framealpha=0.7, ncols=len(legend_handles))
    if SHOW_TITLE:
        fig.suptitle(CENTROID_TITLE, fontsize=12)
    return fig, totals


# =============================================================================
# ANALYSIS 6c — centroid distribution summary across grains
#
# The continuous counterpart to ANALYSIS 1's XANES class pie grid: where that
# summarizes each grain's qualitative Type 1/2/3 split, this summarizes the same
# grains' quantitative centroid distributions. One histogram row per grain on a
# single shared energy axis, identical bin edges and axis limits throughout.
#
# Off-grain spots are excluded, for the same reason the pie chart excludes them:
# this characterizes THIS grain's own Fe speciation, and an off-grain spot
# measured some other phase. Note that this needs an explicit filter here —
# unlike CL and the element means, a centroid comes from the XANES fit rather
# than the XRF zone, so an off-grain spot has a perfectly real centroid and
# would otherwise sail through into the histogram.
# =============================================================================

def centroid_hist_values(df):
    """(on-grain valid centroids, n_off_grain_dropped, n_failed_fits) for one grain."""
    ok = df['centroid_ok'].fillna(False)
    on_grain = on_grain_mask(df)
    values = df.loc[ok & on_grain, 'fit_centroid'].dropna().to_numpy()
    return values, int((ok & ~on_grain).sum()), int((~ok).sum())


def plot_centroid_histograms(grain_values, vmin, vmax):
    """grain_values: ordered {grain_id: array of that grain's valid centroids}."""
    grain_ids = list(grain_values)
    n = len(grain_ids)

    pooled = np.concatenate([v for v in grain_values.values() if len(v)])
    if CENTROID_HIST_RANGE is not None:
        lo, hi = CENTROID_HIST_RANGE
    else:
        lo, hi = float(pooled.min()), float(pooled.max())
    if lo == hi:
        lo, hi = lo - 0.05, hi + 0.05
    edges = np.linspace(lo, hi, int(CENTROID_HIST_BINS) + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    if CENTROID_HIST_COLOR_BY_VALUE:
        cmap = plt.get_cmap(CENTROID_CMAP)
        norm = Normalize(vmin=vmin, vmax=vmax)
        bar_color = [cmap(norm(c)) for c in centers]
    else:
        bar_color = BLUE

    fig, axes = plt.subplots(
        n, 1, sharex=True, sharey=True, squeeze=False, layout='constrained',
        figsize=(CENTROID_HIST_WIDTH_IN, CENTROID_HIST_ROW_HEIGHT_IN * n + 1.0))
    axes = axes.ravel()

    for ax, grain_id in zip(axes, grain_ids):
        values = grain_values[grain_id]
        if len(values):
            counts = np.histogram(values, bins=edges)[0]
            ax.bar(centers, counts / counts.sum(), width=width * 0.92,
                   color=bar_color, edgecolor='black', linewidth=0.3)
            if CENTROID_HIST_SHOW_MEDIAN:
                ax.axvline(float(np.median(values)), color=ORANG, lw=1.4, zorder=4)
        else:
            ax.text(0.5, 0.5, 'no usable fits', ha='center', va='center',
                    fontsize=8, color='gray', transform=ax.transAxes)
        # Grain id as a row label rather than a per-panel title — keeps the rows
        # tight against each other so the shared energy axis stays easy to read down.
        ax.set_ylabel(grain_id, rotation=0, ha='right', va='center', fontsize=9)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlim(lo - width / 2, hi + width / 2)
    axes[-1].set_xlabel(CENTROID_CBAR_LABEL, fontsize=10)
    format_ev_axis(axes[-1].xaxis)
    fig.supylabel('Fraction of spots', fontsize=10)

    if CENTROID_HIST_SHOW_MEDIAN:
        # 'outside upper right' keeps the key clear of the suptitle; it needs the
        # constrained layout engine this figure is built with.
        fig.legend(handles=[plt.Line2D([0], [0], color=ORANG, lw=1.4, label='Median')],
                   loc='outside upper right', fontsize=8, frameon=False)
    if SHOW_TITLE:
        fig.suptitle(f'{CENTROID_TITLE} distribution by grain', fontsize=12)
    return fig, (lo, hi), width


# =============================================================================
# ANALYSIS 7 — per-grain spot-numbering diagnostic
#
# Just the numbering: the registered CL image with every spot plotted in one
# neutral color and labeled with its spot number. The centroid maps deliberately
# leave their spots unlabeled (the numbers crowd each other wherever spots
# cluster, and say nothing about the mapped quantity), so this is the key you
# read alongside them to find a given spot. A QC/lookup aid rather than an
# analysis result, so it goes to figs/diagnostics/.
# =============================================================================

def plot_spot_index_map(grain_id, df, cl_img):
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_cl_background(ax, cl_img)

    on_grain = on_grain_mask(df)
    for row, is_on_grain in zip(df.itertuples(), on_grain):
        draw_spot(ax, row, SPOT_INDEX_COLOR,
                  'o' if is_on_grain else OFF_GRAIN_MARKER, label=True)

    n_off = int((~on_grain).sum())
    if n_off:
        ax.legend(handles=[plt.Line2D([0], [0], marker=OFF_GRAIN_MARKER, linestyle='',
                                      markerfacecolor=SPOT_INDEX_COLOR, markeredgecolor='black',
                                      label='Off-grain (other phase)')],
                  loc=best_legend_corner(df, cl_img.shape), fontsize=7, framealpha=0.7)
    if SHOW_TITLE:
        ax.set_title(f'{grain_id} — spot numbering ({len(df)} spots)', fontsize=11)
    plt.tight_layout()
    return fig, n_off


# =============================================================================
# RUN
# =============================================================================

ALL_ANALYSES = ['pie', 'scatter', 'box', 'map', 'centroid_map', 'centroid_hist',
                'spot_index', 'pca']
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

# Shared by 'centroid_map' and 'centroid_hist' — both key their colors to the same
# pooled scale, so it's derived once here rather than per analysis.
centroid_grains = {g: df for g, df in grain_frames.items() if 'centroid_ok' in df.columns}
centroid_limits = centroid_color_limits(centroid_grains) if centroid_grains else None
NO_CENTROIDS_MSG = (f'  No grain has a usable pre-edge fit centroid '
                    f'(looked in {PREPEAK_DIR}) — skipping.')

if 'centroid_map' in analyses:
    print('\n--- pre-edge fit centroid maps ---')
    if centroid_limits is None:
        print(NO_CENTROIDS_MSG)
    else:
        vmin, vmax = centroid_limits
        how = ('explicit CENTROID_VMIN/VMAX'
               if CENTROID_VMIN is not None and CENTROID_VMAX is not None
               else f'pooled {CENTROID_RANGE_PCT[0]}/{CENTROID_RANGE_PCT[1]} percentiles')
        print(f'  Shared color scale across {len(centroid_grains)} grain(s): '
              f'{vmin:.3f}-{vmax:.3f} eV ({how})')
        for grain_id, df in centroid_grains.items():
            cl_img = load_cl_background(grain_id)
            if cl_img is None:
                continue
            fig, n_ok, n_bad, n_clamped = plot_centroid_map(grain_id, df, cl_img, vmin, vmax)
            notes = []
            if n_bad:
                notes.append(f'{n_bad} without a usable fit (grey)')
            if n_clamped:
                notes.append(f'{n_clamped} clamped to the scale ends')
            print(f'  {grain_id}: {n_ok}/{len(df)} spot(s) colored by centroid'
                  + (f'; {", ".join(notes)}' if notes else ''))
            if SAVE_FIG:
                out = out_dir / f'{grain_id}_centroid_map.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'  Saved: {out.name}')

        groups = centroid_panel_groups(list(centroid_grains))
        for label, grain_ids in groups.items():
            panels = []
            for grain_id in grain_ids:
                cl_img = load_cl_background(grain_id)
                if cl_img is None:
                    continue
                px_um, from_sidecar = grain_pixel_um(grain_id)
                panels.append((grain_id, centroid_grains[grain_id], cl_img, px_um, from_sidecar))
            if len(panels) < 2:
                print(f"  WARNING: panel group '{label}': fewer than 2 grains have a "
                      f'registered CL image — skipping.')
                continue

            if CENTROID_PANEL_TRUE_SCALE:
                no_sidecar = [g for g, _, _, _, ok in panels if not ok]
                if no_sidecar:
                    print(f'  WARNING: no metadata sidecar found/parseable for {no_sidecar} — '
                          f'falling back to CENTROID_PANEL_PIXEL_UM='
                          f'{CENTROID_PANEL_PIXEL_UM:.4g} µm/px for those, so relative panel '
                          f'sizes may be wrong.')
                sizes = {g: px for g, _, _, px, _ in panels}
                print(f'  Panel group \'{label}\': {len(panels)} grain(s) at true relative '
                      f'scale, µm/px = ' + ', '.join(f'{g} {px:.4g}' for g, px in sizes.items()))
            else:
                print(f"  Panel group '{label}': {len(panels)} grain(s), each panel scaled "
                      f'to its own box (not comparable in size)')

            fig, totals = plot_centroid_panel(label, panels, vmin, vmax)
            print(f"    {totals['ok']}/{totals['spots']} spot(s) colored by centroid"
                  + (f"; {totals['bad']} without a usable fit (grey)" if totals['bad'] else '')
                  + (f"; {totals['below'] + totals['above']} clamped to the scale ends"
                     if totals['below'] + totals['above'] else ''))
            if SAVE_FIG:
                out = out_dir / f'{label}_centroid_map_panel.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'    Saved: {out.name}')

if 'centroid_hist' in analyses:
    print('\n--- pre-edge fit centroid distribution by grain ---')
    if centroid_limits is None:
        print(NO_CENTROIDS_MSG)
    else:
        vmin, vmax = centroid_limits
        grain_values, skipped = {}, []
        for grain_id in sorted(centroid_grains):
            values, n_off, n_failed = centroid_hist_values(centroid_grains[grain_id])
            if not len(values):
                skipped.append(grain_id)
                continue
            grain_values[grain_id] = values
            notes = []
            if n_off:
                notes.append(f'{n_off} off-grain excluded')
            if n_failed:
                notes.append(f'{n_failed} without a usable fit')
            print(f'  {grain_id}: n={len(values)}, median {np.median(values):.3f} eV, '
                  f'range {values.min():.3f}-{values.max():.3f} eV'
                  + (f' ({"; ".join(notes)})' if notes else ''))
        if skipped:
            print(f'  WARNING: no on-grain spot with a usable fit for {skipped} — omitted.')
        if not grain_values:
            print('  No grain has an on-grain spot with a usable fit — skipping.')
        else:
            fig, (lo, hi), width = plot_centroid_histograms(grain_values, vmin, vmax)
            print(f'  {len(grain_values)} grain(s), {CENTROID_HIST_BINS} shared bins over '
                  f'{lo:.3f}-{hi:.3f} eV ({width * 1000:.1f} meV/bin)')
            if SAVE_FIG:
                out = out_dir / 'centroid_histogram_by_grain.png'
                fig.savefig(out, dpi=200, bbox_inches='tight')
                print(f'  Saved: {out.name}')

if 'spot_index' in analyses:
    print(f'\n--- spot numbering diagnostics ({len(grain_frames)} grain(s)) ---')
    for grain_id, df in grain_frames.items():
        cl_img = load_cl_background(grain_id)
        if cl_img is None:
            continue
        fig, n_off = plot_spot_index_map(grain_id, df, cl_img)
        print(f'  {grain_id}: {len(df)} spot(s) labeled'
              + (f' ({n_off} off-grain)' if n_off else ''))
        if SAVE_FIG:
            out = diagnostics_dir / f'{grain_id}_spot_index_map.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            print(f'  Saved: {out.name}')

plt.show()
