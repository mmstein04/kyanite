# =============================================================================
# kyanite_figures.py
#
# Figure generation for CL-EPMA pixel data.
# Loads one or more CSVs exported by CL_EPMA_registration.m and produces
# scatter, violin, or binned box plots of CL intensity vs. chosen elements.
#
# CSV_INPUT may be a single CSV file or a directory; all *_pixel_data.csv
# files found in a directory are processed automatically.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

CSV_INPUT = '/Users/mstein/bin/kyanite/figs/'   # file or directory
ELEMENTS  = ['Ti_Ka', 'Fe_Ka', 'Cr_Ka', 'Mn_Ka', 'V_Ka']          # CSV column names
PLOT_TYPE = 'all'      # 'scatter', 'violin', 'boxplot', or 'all'

# Binning — used by 'violin' and 'boxplot'.
# N_BINS splits the (filtered) element range into equal-width bins.
# Override with BIN_EDGES for explicit control (e.g. np.arange(0, 5000, 200)).
N_BINS    = 10
BIN_EDGES = None

# Outlier removal on the element axis (percentile cutoffs).
# Set PCT_LO = 0 and PCT_HI = 100 to disable.
PCT_LO = 0
PCT_HI = 99

SAVE_FIG   = True      # False to display only
SHOW_TITLE = False     # True to add a grain/element/plot-type title

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

plot_types = ['scatter', 'violin', 'boxplot'] if PLOT_TYPE == 'all' else [PLOT_TYPE]

BLUE  = '#3B9BDD'
ORANG = '#D85B30'

# =============================================================================
# PLOT FUNCTION
# =============================================================================

def make_plot(plot_type, element, grain_id, out_dir, x, y, plot_df, occupied, counts):
    fig, ax = plt.subplots(figsize=(10, 5))

    # ---- Scatter ------------------------------------------------------------
    if plot_type == 'scatter':
        ax.scatter(x, y, s=4, alpha=0.06, color=BLUE, linewidths=0)
        m, b = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 300)
        ax.plot(xfit, m * xfit + b, 'k-', lw=1.5)
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}\nn = {len(x):,}',
                transform=ax.transAxes, va='top', fontsize=9)
        ax.set_xlabel(element)
        ax.set_ylabel('CL intensity (norm.)')

    # ---- Violin -------------------------------------------------------------
    elif plot_type == 'violin':
        sns.violinplot(data=plot_df, x='bin', y='CL', ax=ax,
                       density_norm='count', inner='box', cut=0,
                       color=BLUE, linewidth=0.8)
        for i, lbl in enumerate(occupied):
            n = counts[lbl]
            label = f'n={n // 1000}k' if n >= 1000 else f'n={n}'
            ax.text(i, 1.01, label, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7, color='gray')
        step = max(1, len(occupied) // 10)
        ax.set_xticks(range(0, len(occupied), step))
        ax.set_xticklabels(occupied[::step], rotation=30, ha='right', fontsize=8)
        ax.set_xlabel(element)
        ax.set_ylabel('CL intensity (norm.)')

    # ---- Boxplot ------------------------------------------------------------
    elif plot_type == 'boxplot':
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

    # ---- Common styling -----------------------------------------------------
    if SHOW_TITLE:
        ax.set_title(f'{grain_id} — CL vs. {element}  ({plot_type})', fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    sns.despine(ax=ax)
    plt.tight_layout()

    if SAVE_FIG:
        out = out_dir / f'{grain_id}_{element}_{plot_type}.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out.name}')

    return fig

# =============================================================================
# RUN
# =============================================================================

for csv_path in csv_files:
    grain_id = csv_path.stem.replace('_pixel_data', '')
    out_dir  = csv_path.parent
    df       = pd.read_csv(csv_path)
    y_all    = df['CL'].values
    print(f'\n--- {grain_id} ({len(df):,} pixels) ---')

    # skip columns missing from this CSV, warn once per grain
    available = [e for e in ELEMENTS if e in df.columns]
    missing   = [e for e in ELEMENTS if e not in df.columns]
    if missing:
        print(f'  WARNING: columns not found, skipping: {missing}')

    for element in available:
        x_all = df[element].values
        lo, hi = np.percentile(x_all, [PCT_LO, PCT_HI])
        keep   = (x_all >= lo) & (x_all <= hi)
        x = x_all[keep]
        y = y_all[keep]
        print(f'  {element}: {keep.sum():,} px after filter '
              f'({(~keep).sum():,} removed)')

        if BIN_EDGES is not None:
            edges = np.asarray(BIN_EDGES, dtype=float)
        else:
            edges = np.linspace(x.min(), x.max(), N_BINS + 1)

        bw  = edges[1] - edges[0]
        dec = max(0, int(np.ceil(-np.log10(bw)))) if bw < 1 else 0
        fmt = f'.{dec}f'
        bin_labels = [f'[{edges[i]:{fmt}}, {edges[i+1]:{fmt}})'
                      for i in range(len(edges) - 1)]
        bins = pd.cut(x, bins=edges, labels=bin_labels, include_lowest=True)
        plot_df = pd.DataFrame({'x': x, 'CL': y, 'bin': bins}).dropna()

        occupied = [lbl for lbl in bin_labels if (plot_df['bin'] == lbl).any()]
        plot_df  = plot_df[plot_df['bin'].isin(occupied)]
        plot_df['bin'] = plot_df['bin'].cat.remove_unused_categories()
        plot_df['bin'] = plot_df['bin'].cat.reorder_categories(occupied)
        counts = plot_df.groupby('bin', observed=True).size()

        for pt in plot_types:
            make_plot(pt, element, grain_id, out_dir, x, y, plot_df, occupied, counts)

plt.show()
