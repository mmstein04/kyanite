# =============================================================================
# kyanite_figures.py
#
# Figure generation for CL-EPMA pixel data.
#
# Loads one or more CSVs and produces scatter, contour (density contour lines
# over a scatter), heatmap (the same 2-D KDE, filled with a colorbar instead
# of drawn as lines), violin, or binned box plots of CL intensity vs. chosen
# elements. scatter, contour, and heatmap all include a linear fit line and
# Pearson r.
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
# files found in a directory are processed automatically (this also matches
# *_region_pixel_data.csv, since it shares the same suffix).
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

CSV_INPUT = '/Users/mstein/bin/kyanite/region_figs/MW609-01_region_pixel_data.csv'   # file or directory
ELEMENTS  = ['Cr_Ka']          # CSV column names
PLOT_TYPE = 'all'      # 'scatter', 'violin', 'boxplot', 'contour', 'heatmap', or 'all'

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

# Outlier removal on the element axis (percentile cutoffs), applied per
# region in region mode. Set PCT_LO = 0 and PCT_HI = 100 to disable.
PCT_LO = 0
PCT_HI = 99

SAVE_FIG   = True      # False to display only
SHOW_TITLE = True      # True to add a grain/element/plot-type title

BLUE  = '#3B9BDD'
ORANG = '#D85B30'

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

plot_types = ['scatter', 'violin', 'boxplot', 'contour', 'heatmap'] if PLOT_TYPE == 'all' else [PLOT_TYPE]

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
    ax.text(0.05, 0.95, f'r = {r:.3f}\nn = {len(x):,}',
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
    cs = ax.contourf(xx, yy, zz, levels=HEATMAP_LEVELS, cmap='inferno')
    ax.figure.colorbar(cs, ax=ax, label='density')
    add_fit_and_r(ax, x, y, line_color='c', text_color='white', text_bg='black')
    ax.set_xlabel(element)
    ax.set_ylabel('CL intensity (norm.)')


PLOT_FUNCS = {'scatter': plot_scatter, 'violin': plot_violin, 'boxplot': plot_boxplot,
              'contour': plot_contour, 'heatmap': plot_heatmap}


def render_plot(ax, plot_type, element, x, y):
    PLOT_FUNCS[plot_type](ax, x, y, element)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    sns.despine(ax=ax)


def filtered_xy(df, element):
    x_all = df[element].values
    y_all = df['CL'].values
    lo, hi = np.percentile(x_all, [PCT_LO, PCT_HI])
    keep = (x_all >= lo) & (x_all <= hi)
    return x_all[keep], y_all[keep], int((~keep).sum())

# =============================================================================
# RUN
# =============================================================================

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    region_mode = 'Region' in df.columns
    out_dir = csv_path.parent

    if region_mode:
        grain_id = csv_path.stem.replace('_region_pixel_data', '')
        regions  = list(df['Region'].drop_duplicates())
        print(f'\n--- {grain_id} ({len(df):,} px, {len(regions)} region(s): {", ".join(regions)}) ---')
    else:
        grain_id = csv_path.stem.replace('_pixel_data', '')
        print(f'\n--- {grain_id} ({len(df):,} px) ---')

    available = [e for e in ELEMENTS if e in df.columns]
    missing   = [e for e in ELEMENTS if e not in df.columns]
    if missing:
        print(f'  WARNING: columns not found, skipping: {missing}')

    for element in available:

        if region_mode:
            n_r = len(regions)
            for pt in plot_types:
                fig, axes = plt.subplots(1, n_r, figsize=(5 * n_r, 5), sharey=True)
                axes = np.atleast_1d(axes)
                for ax, region in zip(axes, regions):
                    sub = df[df['Region'] == region]
                    x, y, n_removed = filtered_xy(sub, element)
                    if len(x) < 2:
                        ax.text(0.5, 0.5, 'insufficient data', ha='center', va='center',
                                transform=ax.transAxes, fontsize=9, color='gray')
                        ax.set_title(region, fontsize=10)
                        continue
                    render_plot(ax, pt, element, x, y)
                    ax.set_title(region, fontsize=10)
                    print(f'  [{region}] {element} ({pt}): {len(x):,} px ({n_removed:,} removed)')

                if SHOW_TITLE:
                    fig.suptitle(f'{grain_id} — {element} ({pt}) by region', fontsize=12)
                plt.tight_layout()

                if SAVE_FIG:
                    out = out_dir / f'{grain_id}_{element}_{pt}_by_region.png'
                    fig.savefig(out, dpi=200, bbox_inches='tight')
                    print(f'  Saved: {out.name}')

        else:
            x, y, n_removed = filtered_xy(df, element)
            print(f'  {element}: {len(x):,} px after filter ({n_removed:,} removed)')

            for pt in plot_types:
                fig, ax = plt.subplots(figsize=(10, 5))
                render_plot(ax, pt, element, x, y)
                if SHOW_TITLE:
                    ax.set_title(grain_id, fontsize=11)
                plt.tight_layout()

                if SAVE_FIG:
                    out = out_dir / f'{grain_id}_{element}_{pt}.png'
                    fig.savefig(out, dpi=200, bbox_inches='tight')
                    print(f'  Saved: {out.name}')

plt.show()
