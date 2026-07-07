# =============================================================================
# xanes_plot.py
#
# Plots the XANES portion of Fe K-edge mu(E) spectra from spot-analysis CSVs
# (normalized energy vs. norm, one file per spot) so pre-edge peak behavior
# can be visually classified.
#
# Input CSVs are grouped into samples by filename (everything before
# "-spotNN" / "_spotNN"). For each sample this produces:
#   - an overlay of the full XANES window, all spots colored by spot order
#   - an overlay zoomed on the pre-edge window
#   - a small-multiples grid, one pre-edge zoom per spot, for classifying
#     peak shape/position spot by spot
#
# XANES_INPUT may be a single CSV or a directory; all *.csv files in a
# directory are processed.
# =============================================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

XANES_INPUT = '/Users/mstein/bin/kyanite/xanes'   # file or directory
OUT_DIR     = '/Users/mstein/bin/kyanite/figs/xanes'

EDGE_ENERGY      = 7125          # Fe K edge (eV), reference line; None to disable
XANES_WINDOW     = (7000, 7250)    # eV, full-window overlay plot
PRE_EDGE_WINDOW  = (7112, 7116)    # eV, zoom window for pre-edge peak

# expected pre-edge peak positions (eV, label, color) — drawn as reference lines
PRE_EDGE_REFS = [
    (7113.0, 'Fe²⁺', '#D85B30'),
    (7114.5, 'Fe³⁺', '#7A5195'),
]

SMALL_MULTIPLES = True   # per-spot pre-edge grid, for classifying peak shape spot by spot
GRID_SHARE_Y    = False  # False lets each spot autoscale, so peak shape isn't squashed
                          # flat by spots with a much larger edge jump; True compares
                          # absolute pre-edge intensity across spots instead
SAVE_FIG        = True
SHOW_TITLE      = True

SPOT_NAME_RE = re.compile(r'^(.*?)[-_]spot\d+', re.IGNORECASE)

# =============================================================================
# LOAD
# =============================================================================

input_path = Path(XANES_INPUT)
if input_path.is_dir():
    csv_files = sorted(input_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {input_path}')
else:
    csv_files = [input_path]

print(f'Processing {len(csv_files)} CSV(s):')

samples = {}   # sample_id -> list of (spot_id, energy, norm)
for path in csv_files:
    stem = path.stem
    m = SPOT_NAME_RE.match(stem)
    sample_id = m.group(1) if m else stem

    df = pd.read_csv(path, comment='#', header=None, names=['energy', 'norm'],
                      skipinitialspace=True)
    samples.setdefault(sample_id, []).append((stem, df['energy'].values, df['norm'].values))

for sample_id, spots in samples.items():
    print(f'  {sample_id}: {len(spots)} spot(s)')

out_dir = Path(OUT_DIR)
if SAVE_FIG:
    out_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PLOTTING
# =============================================================================

def window_mask(energy, window):
    lo, hi = window
    return (energy >= lo) & (energy <= hi)


def plot_pre_edge_refs(ax, label=True):
    for energy, name, color in PRE_EDGE_REFS:
        ax.axvline(energy, color=color, ls=':', lw=1.2, zorder=1,
                   label=f'{name} ({energy:g} eV)' if label else None)


def plot_overlay(ax, spots, window, cmap):
    n = len(spots)
    colors = cmap(np.linspace(0, 1, n))
    for (spot_id, energy, norm), color in zip(spots, colors):
        keep = window_mask(energy, window)
        ax.plot(energy[keep], norm[keep], color=color, lw=1, alpha=0.7)

    if EDGE_ENERGY is not None:
        ax.axvline(EDGE_ENERGY, color='0.4', ls='--', lw=1, zorder=0)
    plot_pre_edge_refs(ax)
    ax.legend(fontsize=7, loc='upper left')

    ax.set_xlim(window)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('normalized μ(E)')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('spot order')


def plot_small_multiples(sample_id, spots, window):
    n = len(spots)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.8 * rows),
                              sharex=True, sharey=GRID_SHARE_Y)
    axes = np.atleast_1d(axes).ravel()

    for ax, (spot_id, energy, norm) in zip(axes, spots):
        keep = window_mask(energy, window)
        ax.plot(energy[keep], norm[keep], color='#3B9BDD', lw=1)
        if EDGE_ENERGY is not None:
            ax.axvline(EDGE_ENERGY, color='0.6', ls='--', lw=0.6, zorder=0)
        plot_pre_edge_refs(ax, label=False)
        spot_label = spot_id.replace(sample_id, '').lstrip('-_')
        ax.set_title(spot_label, fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes[n:]:
        ax.axis('off')

    fig.supxlabel('Energy (eV)', fontsize=9)
    fig.supylabel('normalized μ(E)', fontsize=9)
    if SHOW_TITLE:
        fig.suptitle(f'{sample_id} — pre-edge, per spot', fontsize=11)
    plt.tight_layout()
    return fig


# =============================================================================
# RUN
# =============================================================================

for sample_id, spots in samples.items():
    print(f'\n--- {sample_id} ({len(spots)} spot(s)) ---')
    cmap = plt.cm.rainbow

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_overlay(ax, spots, XANES_WINDOW, cmap)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if SHOW_TITLE:
        ax.set_title(f'{sample_id} — XANES', fontsize=11)
    plt.tight_layout()
    if SAVE_FIG:
        out = out_dir / f'{sample_id}_xanes_overlay.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out.name}')

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_overlay(ax, spots, PRE_EDGE_WINDOW, cmap)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if SHOW_TITLE:
        ax.set_title(f'{sample_id} — pre-edge', fontsize=11)
    plt.tight_layout()
    if SAVE_FIG:
        out = out_dir / f'{sample_id}_pre_edge_overlay.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out.name}')

    if SMALL_MULTIPLES:
        fig = plot_small_multiples(sample_id, spots, PRE_EDGE_WINDOW)
        if SAVE_FIG:
            out = out_dir / f'{sample_id}_pre_edge_grid.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            print(f'  Saved: {out.name}')

plt.show()
