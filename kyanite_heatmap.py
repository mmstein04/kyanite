# =============================================================================
# kyanite_heatmap.py
#
# Shared annotated-heatmap drawing for the all-grains summary figures
# (kyanite_figures.py's 'summary', kyanite_rf_shap_plots.py's 'summary',
# kyanite_pca.py's SUMMARY) — one implementation so their cell borders, value
# labels, text-color cutoff, and colorbar style can't drift apart. Callers
# set their own tick labels/axis labels; this only draws the cells, values,
# and colorbar.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

CELL_FONTSIZE = 11
MISSING_COLOR = '0.85'   # NaN cells (no data for that grain/column), labeled 'n/a'


def annotated_heatmap(ax, values, cmap, vmin, vmax, cbar_label, fmt='.2f', ticks=None):
    """Draw `values` (2-D array, rows top to bottom) as a heatmap on `ax`,
    each cell labeled with its value, colorbar below. Cell (i, j) is
    centered at x=j, y=i. Label text is white on the darkest 40% of the
    scale either side of zero, black elsewhere. Returns the colorbar."""
    values = np.asarray(values, dtype=float)
    cmap = (plt.colormaps[cmap] if isinstance(cmap, str) else cmap).copy()
    cmap.set_bad(MISSING_COLOR)
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    span = max(abs(vmin), abs(vmax))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                label, color = 'n/a', 'gray'
            else:
                # Tiny negatives would otherwise print as '-0.00'
                label = format(v, fmt)
                if float(label) == 0:
                    label = format(0.0, fmt)
                color = 'white' if abs(v) >= 0.6 * span else 'black'
            ax.text(j, i, label, ha='center', va='center', fontsize=CELL_FONTSIZE, color=color)

    # White cell borders
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = ax.figure.colorbar(im, ax=ax, location='bottom', pad=0.08, shrink=0.8, aspect=30)
    cbar.set_label(cbar_label)
    if ticks is not None:
        cbar.set_ticks(ticks)
    return cbar
