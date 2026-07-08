# =============================================================================
# kyanite_spot_analysis.py
#
# Batch analysis/visualization of per-spot geochemistry CSVs produced by
# xrf_h5_extract_spots.py (<grain_id>_spot_geochemistry.csv — spot
# coordinates, per-zone element/CL means, and XANES pre-edge class).
#
# Produces:
#   - a combined figure: a grid of pie charts, one per grain, showing the
#     Type 1/2/3 XANES class distribution ('Bad data' / unclassified spots
#     are excluded from the pie charts entirely)
#   - CL vs. element scatter plots, one per element, pooling spots from all
#     input grains together and coloring by XANES class ('Bad data' /
#     unclassified spots ARE included here, as grey points)
#   - a labeled spot-location map per grain: the registered CL image with
#     each spot plotted at its pixel location, colored by XANES class and
#     labeled with its spot number
#
# CSV_INPUT may be a single CSV or a directory; all *_spot_geochemistry.csv
# files in a directory are processed. Note: as of this writing, the real
# per-spot CSVs live in figs/xanes/, not figs/ directly.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import tifffile
from pathlib import Path

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

CSV_INPUT = '/Users/mstein/bin/kyanite/figs/xanes'   # file or directory of *_spot_geochemistry.csv
FIGS_DIR  = '/Users/mstein/bin/kyanite/figs'         # where <grain_id>_CL_registered.tif live
OUT_DIR   = '/Users/mstein/bin/kyanite/figs/spot_analysis'

ANALYSES = ['pie', 'scatter', 'map']   # 'pie', 'scatter', 'map', 'all', or a list of these

# Columns to make a pooled "CL vs element" scatter plot for.
# None = auto-detect every element ROI column present in the union of all input files.
SCATTER_ELEMENTS = ['Cr_Ka', 'Fe_Ka', 'V_Ka', 'Mn_Ka', 'Ti_Ka']

SAVE_FIG   = True
SHOW_TITLE = True

# Fixed pie-slice order/coloring, so every grain's pie is comparable at a glance.
CATEGORY_ORDER = ['Type 1', 'Type 2', 'Type 3']
CATEGORY_COLORS = {
    'Type 1': '#D85B30',
    'Type 2': '#4C9F70',
    'Type 3': '#7A5195',
    'Bad data': '#999999',
}
GREY = '#999999'   # NaN / unmatched category_label renders identically to 'Bad data'

# Non-element columns from xrf_h5_extract_spots.py's schema — everything else
# in a spot CSV is treated as an element column.
METADATA_COLS = [
    'grain_id', 'spot', 'spot_id', 'area_name', 'category', 'category_label',
    'pixel_count', 'row_px_h5', 'col_px_h5', 'row_px_tiff', 'col_px_tiff',
    'row_matlab', 'col_matlab', 'x_mm', 'y_mm', 'x_rel_um', 'y_rel_um',
    'zone_radius_um', 'zone_pixel_count', 'zone_mask_px_count', 'CL',
]

SPOT_LABEL_FONTSIZE = 6
SPOT_LABEL_OFFSET   = (4, 4)   # points

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
    sub = df[df['category_label'].isin(CATEGORY_ORDER)]
    counts = sub['category_label'].value_counts()
    return [int(counts.get(c, 0)) for c in CATEGORY_ORDER], len(sub), len(df)


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
        counts, n_classified, n_total = pie_counts(df)
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
        ax.set_title(f'{grain_id}\n(n={n_classified}/{n_total})', fontsize=9)

    for ax in axes[n:]:
        ax.axis('off')

    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c]) for c in CATEGORY_ORDER]
    fig.legend(handles, CATEGORY_ORDER, loc='lower center', ncol=len(CATEGORY_ORDER),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    if SHOW_TITLE:
        fig.suptitle('XANES pre-edge class distribution by grain\n'
                      '(Bad data / unclassified spots excluded)', fontsize=12)
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
# ANALYSIS 3 — per-grain spot-location map
# =============================================================================

def load_cl_background(grain_id):
    path = Path(FIGS_DIR) / f'{grain_id}_CL_registered.tif'
    if not path.exists():
        print(f'  WARNING: {path.name} not found — skipping spot map for {grain_id}.')
        return None
    return tifffile.imread(str(path))


def plot_spot_map(grain_id, df, cl_img):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cl_img, cmap='gray', origin='upper')
    # Deliberately NO ax.invert_yaxis() here. origin='upper' already puts row 0 at
    # the top, matching row_px_tiff/col_px_tiff's "row 0 = top" convention (same as
    # MATLAB's imagesc default used in xrf_display.m). Adding invert_yaxis() would
    # silently flip every spot vertically relative to the image.

    for row in df.itertuples():
        color = resolved_color(row.category_label)
        ax.scatter(row.col_px_tiff, row.row_px_tiff, s=28, color=color,
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
    ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.7)
    if SHOW_TITLE:
        ax.set_title(f'{grain_id} — spot locations ({len(df)} spots)', fontsize=11)
    plt.tight_layout()
    return fig


# =============================================================================
# RUN
# =============================================================================

ALL_ANALYSES = ['pie', 'scatter', 'map']
if ANALYSES == 'all':
    analyses = ALL_ANALYSES
elif isinstance(ANALYSES, (list, tuple)):
    analyses = list(ANALYSES)
else:
    analyses = [ANALYSES]
unknown = [a for a in analyses if a not in ALL_ANALYSES]
if unknown:
    raise ValueError(f"Unknown ANALYSES {unknown}; choose from {ALL_ANALYSES}, 'all', or a list of these.")

if 'pie' in analyses:
    print('\n--- XANES class pie grid ---')
    fig = plot_pie_grid(grain_frames)
    if SAVE_FIG:
        out = out_dir / 'xanes_class_pie_grid.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out.name}')

if 'scatter' in analyses:
    scatter_elements = list(SCATTER_ELEMENTS) if SCATTER_ELEMENTS is not None else detect_elements(combined)
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
