# =============================================================================
# xanes_plot.py
#
# Plots the XANES portion of Fe K-edge mu(E) spectra from spot-analysis CSVs
# (normalized energy vs. norm, one file per spot) so pre-edge peak behavior
# can be visually classified.
#
# Can also automatically classify each spot's pre-edge doublet by the relative
# height of the Fe2+ peak (~7113 eV) vs. the Fe3+ peak (~7114.5 eV):
#   Type 1 — Fe2+ peak clearly taller
#   Type 2 — Fe2+ and Fe3+ peaks about the same height
#   Type 3 — Fe3+ peak clearly taller
#   Ambiguous — spectrum failed a QC check (see classify_spot / FLAGS below)
# Method: find genuine local maxima (any real inflection in a lightly
# smoothed curve, however faint — not filtered by prominence) on each side
# of a soft Fe2+/Fe3+ divider (SPLIT_ENERGY); the taller one on each side is
# that side's peak. Falls back to a windowed max only when a side has no
# local maximum at all (fully monotonic — common in this dataset), since a
# fixed-window search would otherwise grab part of the other peak's rising
# flank when the two peaks sit close together. The trough is the minimum of
# the smoothed curve between the two chosen peak energies.
#     diff  = height(Fe2+) - height(Fe3+)
#     scale = max(height(Fe2+), height(Fe3+)) - trough   ("peak-to-trough length")
#     ratio = diff / scale   (bounded to [-1, +1])
#     Type 1 if ratio > RATIO_THRESHOLD, Type 3 if ratio < -RATIO_THRESHOLD,
#     else Type 2.
# QC flags route a spectrum to "Ambiguous" instead of forcing Type 1/2/3:
#   zero_scale        — no resolvable pre-edge feature (peaks ~= trough)
#   noisy             — smoothing residual large relative to overall pre-edge amplitude
#   irregular_shape   — more (prominence-filtered) local extrema than a clean
#                       doublet should have
#   intensity_outlier — this spot's overall pre-edge intensity is a robust
#                       outlier vs. the rest of its sample (e.g. hit a fluid
#                       inclusion / different phase / bad normalization)
# CLASSIFY defaults to False: even with the QC flags above, automatic
# classification doesn't reliably match expert-by-eye judgment across every
# grain in this dataset (peak shapes and separations vary a lot between
# grains) — treat it as a rough starting point, not a substitute for
# classifying by hand from the pre-edge grid figures.
#
# Input CSVs are grouped into samples by filename (everything before
# "-spotNN" / "_spotNN"). For each sample this produces:
#   - an overlay of the full XANES window, all spots colored by spot order
#   - an overlay zoomed on the pre-edge window
#   - a small-multiples grid, one pre-edge zoom per spot, for classifying
#     peak shape/position spot by spot (labeled with the automatic
#     classification and marked with the detected Fe2+/Fe3+/trough points,
#     when CLASSIFY is on)
#   - a classification CSV (one row per spot), when CLASSIFY is on
#
# XANES_INPUT may be a single CSV or a directory; all *.csv files in a
# directory are processed.
# =============================================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from pathlib import Path

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

XANES_INPUT = '/Users/mstein/bin/kyanite/xanes'   # file or directory
OUT_DIR     = '/Users/mstein/bin/kyanite/figs/xanes'

EDGE_ENERGY      = None          # Fe K edge (eV), reference line; None to disable
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

# --- Pre-edge classification ------------------------------------------------
CLASSIFY = False   # compute Type 1/2/3/Ambiguous per spot, label the grid, write a CSV
                    # (off by default -- automatic classification isn't reliable enough
                    # yet across all grains; classify by hand instead)

# Sub-windows searched for each peak's maximum (eV), chosen from
# PRE_EDGE_REFS with tolerance for peak positions drifting a few tenths of
# an eV between spots.
FE2_WINDOW = (7112.2, 7113.8)
FE3_WINDOW = (7113.8, 7115.3)

# Smoothing applied before peak-finding (Savitzky-Golay). Data are ~0.1 eV
# spaced; a 7-point window is ~0.7 eV, well under the ~1-1.5 eV peak FWHM.
SMOOTH_WINDOW    = 7
SMOOTH_POLYORDER = 3

RATIO_THRESHOLD = 0.5   # classify by ratio = diff / (peak-to-trough scale)

# Minimum peak/trough prominence to count as real structure, as a fraction of
# this spectrum's own pre-edge amplitude (max-min within the Fe2+/Fe3+
# region) — scales automatically across spots/samples with very different
# absolute intensity, unlike a fixed absolute prominence.
PROMINENCE_FRAC = 0.08

# QC thresholds
NOISE_FRAC    = 0.30   # flag 'noisy' if smoothing-residual RMS > this fraction of scale
MIN_SCALE_ABS = 1e-6   # flag 'zero_scale' if peak-to-trough scale below this
OUTLIER_MAD_K = 6.0     # flag 'intensity_outlier' if robust z-score of mean pre-edge
                        # intensity (vs. other spots in the sample) exceeds this many MADs

CATEGORY_COLORS = {
    'Type 1': '#D85B30',
    'Type 2': '#4C9F70',
    'Type 3': '#7A5195',
    'Ambiguous': '#999999',
}

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
if SAVE_FIG or CLASSIFY:
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


# =============================================================================
# CLASSIFICATION
# =============================================================================

REGION = (FE2_WINDOW[0], FE3_WINDOW[1])          # region a real pre-edge peak must fall within
SPLIT_ENERGY = (FE2_WINDOW[1] + FE3_WINDOW[0]) / 2  # soft Fe2+/Fe3+ divider, used only when
                                                      # a peak can't be paired with a second one


def _fallback_side_height(e_w, y_s, side_window, exclude_energy):
    """Highest point of y_s in side_window, for the side with no resolvable peak of
    its own — i.e. a shoulder height, not a true local max."""
    m = window_mask(e_w, side_window)
    if exclude_energy is not None:
        m = m & (np.abs(e_w - exclude_energy) > 1e-9)
    if not m.any():
        return np.nan, np.nan
    i = np.argmax(y_s[m])
    return e_w[m][i], y_s[m][i]


def classify_spot(energy, norm):
    """Compute pre-edge peak metrics and a Type 1/2/3/undetermined call for one spot.

    Finds genuine local maxima/minima (scipy.signal.find_peaks, with a
    prominence floor scaled to this spectrum's own pre-edge amplitude so it
    adapts to very different intensity scales) rather than slicing the data
    at a fixed energy — doublets in this dataset are not all the same width
    (e.g. ~0.9 eV apart in some grains vs. ~1.5 eV in others), so a fixed
    split point can land on the wrong side of a real peak. Since the Fe2+
    pre-edge energy is always lower than Fe3+'s, once real peaks are found
    they're assigned purely by energy order — no reference-position
    matching needed. A fixed split is only used as a fallback to decide
    which side a *lone* peak belongs to when the other side has no
    resolvable peak at all (common in this dataset).
    """
    order = np.argsort(energy)
    e = np.asarray(energy)[order]
    y = np.asarray(norm)[order]

    pad = 0.5
    keep = window_mask(e, (REGION[0] - pad, REGION[1] + pad))
    e_w, y_w = e[keep], y[keep]

    flags = []
    if len(e_w) < SMOOTH_WINDOW:
        y_s = y_w.copy()
    else:
        wl = SMOOTH_WINDOW if SMOOTH_WINDOW % 2 == 1 else SMOOTH_WINDOW + 1
        y_s = savgol_filter(y_w, wl, SMOOTH_POLYORDER)

    in_region = window_mask(e_w, REGION)
    if in_region.sum() < 3:
        return dict(peak2_energy=np.nan, peak2_height=np.nan,
                    peak3_energy=np.nan, peak3_height=np.nan,
                    trough_energy=np.nan, trough_height=np.nan,
                    diff=np.nan, scale=np.nan, ratio=np.nan,
                    type_by_ratio='undetermined', flags=['zero_scale'],
                    mean_intensity=float(np.mean(y_w)) if len(y_w) else np.nan)

    y_region = y_s[in_region]
    e_region = e_w[in_region]
    amplitude = y_region.max() - y_region.min()
    prominence = max(PROMINENCE_FRAC * amplitude, 1e-12)

    # Raw (unfiltered) local maxima — genuine inflections in the smoothed curve,
    # however faint — used to locate each side's peak. Restricting candidates by
    # energy side (not slicing the data there) avoids two failure modes: (a) a
    # fixed-window search grabbing a point on the *other* peak's rising flank when
    # the two peaks sit close together, and (b) requiring a peak to be "prominent"
    # just to be found at all, which would blind us to a real but subtle shoulder.
    raw_pk_idx, _ = find_peaks(y_region)
    fe2_idx = raw_pk_idx[e_region[raw_pk_idx] < SPLIT_ENERGY]
    fe3_idx = raw_pk_idx[e_region[raw_pk_idx] >= SPLIT_ENERGY]

    if len(fe2_idx):
        i2 = fe2_idx[np.argmax(y_region[fe2_idx])]
        peak2_energy, peak2_height = e_region[i2], y_region[i2]
    else:
        # no local max at all on the Fe2+ side — monotonic; the shoulder height
        # (if any) is just the highest point in that side's window
        peak2_energy, peak2_height = _fallback_side_height(
            e_w, y_s, (FE2_WINDOW[0], SPLIT_ENERGY), None)

    if len(fe3_idx):
        i3 = fe3_idx[np.argmax(y_region[fe3_idx])]
        peak3_energy, peak3_height = e_region[i3], y_region[i3]
    else:
        peak3_energy, peak3_height = _fallback_side_height(
            e_w, y_s, (SPLIT_ENERGY, FE3_WINDOW[1]), None)

    if np.isnan(peak2_height) or np.isnan(peak3_height):
        flags.append('zero_scale')
        return dict(peak2_energy=peak2_energy, peak2_height=peak2_height,
                    peak3_energy=peak3_energy, peak3_height=peak3_height,
                    trough_energy=np.nan, trough_height=np.nan,
                    diff=np.nan, scale=np.nan, ratio=np.nan,
                    type_by_ratio='undetermined', flags=flags,
                    mean_intensity=float(np.mean(y_w)))

    lo_e, hi_e = sorted((peak2_energy, peak3_energy))
    between = window_mask(e_w, (lo_e, hi_e))
    if between.sum() >= 2:
        it = np.argmin(y_s[between])
        trough_energy, trough_height = e_w[between][it], y_s[between][it]
    else:
        trough_height = min(peak2_height, peak3_height)
        trough_energy = peak2_energy if peak2_height < peak3_height else peak3_energy

    # Significant (prominence-filtered) extrema, for QC only — flags spectra with
    # more real structure than a clean doublet (rise, peak, [trough, peak], fall)
    # should have.
    sig_pk_idx, _ = find_peaks(y_region, prominence=prominence)
    sig_tr_idx, _ = find_peaks(-y_region, prominence=prominence)
    if len(sig_pk_idx) > 2 or len(sig_tr_idx) > 1:
        flags.append('irregular_shape')

    diff = peak2_height - peak3_height
    scale = max(peak2_height, peak3_height) - trough_height

    if not np.isfinite(scale) or scale <= MIN_SCALE_ABS:
        if 'zero_scale' not in flags:
            flags.append('zero_scale')
        ratio = np.nan
        type_by_ratio = 'undetermined'
    else:
        ratio = diff / scale
        if ratio > RATIO_THRESHOLD:
            type_by_ratio = 'Type 1'
        elif ratio < -RATIO_THRESHOLD:
            type_by_ratio = 'Type 3'
        else:
            type_by_ratio = 'Type 2'

        # Compared to the overall pre-edge amplitude, not the peak-to-trough scale:
        # a shallow-but-real dip can make scale tiny even for a clean spectrum, which
        # would make ordinary smoothing residual look disproportionately large.
        residual_rms = float(np.std(y_w - y_s))
        if residual_rms > NOISE_FRAC * amplitude:
            flags.append('noisy')

    return dict(peak2_energy=float(peak2_energy), peak2_height=float(peak2_height),
                peak3_energy=float(peak3_energy), peak3_height=float(peak3_height),
                trough_energy=float(trough_energy), trough_height=float(trough_height),
                diff=float(diff), scale=float(scale),
                ratio=float(ratio) if np.isfinite(ratio) else np.nan,
                type_by_ratio=type_by_ratio, flags=flags,
                mean_intensity=float(np.mean(y_w)))


def robust_z(values):
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad


def classify_sample(spots):
    """Classify every spot in a sample, adding a per-sample intensity-outlier check."""
    results = [classify_spot(energy, norm) for _, energy, norm in spots]

    z = robust_z([r['mean_intensity'] for r in results])
    for r, zi in zip(results, z):
        if np.isfinite(zi) and abs(zi) > OUTLIER_MAD_K:
            r['flags'].append('intensity_outlier')

    for r in results:
        r['category'] = 'Ambiguous' if r['flags'] else r['type_by_ratio']

    return results


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


def plot_small_multiples(sample_id, spots, window, results=None):
    n = len(spots)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.8 * rows),
                              sharex=True, sharey=GRID_SHARE_Y)
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, (spot_id, energy, norm)) in enumerate(zip(axes, spots)):
        keep = window_mask(energy, window)
        ax.plot(energy[keep], norm[keep], color='#3B9BDD', lw=1, zorder=1)
        if EDGE_ENERGY is not None:
            ax.axvline(EDGE_ENERGY, color='0.6', ls='--', lw=0.6, zorder=0)
        plot_pre_edge_refs(ax, label=False)
        spot_label = spot_id.replace(sample_id, '').lstrip('-_')

        title = spot_label
        color = 'black'
        if results is not None:
            r = results[i]
            color = CATEGORY_COLORS.get(r['category'], 'black')
            title = f"{spot_label}: {r['category']}"
            if np.isfinite(r.get('peak2_height', np.nan)):
                ax.plot(r['peak2_energy'], r['peak2_height'], 'o', color=color, ms=4, zorder=3)
                ax.plot(r['peak3_energy'], r['peak3_height'], 'o', color=color, ms=4, zorder=3)
                ax.plot(r['trough_energy'], r['trough_height'], 'x', color='0.3', ms=4, zorder=3)

        ax.set_title(title, fontsize=7, color=color)
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

all_classifications = []
for sample_id, spots in samples.items():
    print(f'\n--- {sample_id} ({len(spots)} spot(s)) ---')
    cmap = plt.cm.rainbow

    results = None
    if CLASSIFY:
        results = classify_sample(spots)
        for (spot_id, _, _), r in zip(spots, results):
            flags = ';'.join(r['flags'])
            ratio_str = f"{r['ratio']:+.2f}" if np.isfinite(r['ratio']) else 'n/a'
            print(f"  {spot_id:30s} ratio={ratio_str:>6s} -> {r['category']}"
                  + (f"  [{flags}]" if flags else ''))
        n_by_cat = pd.Series([r['category'] for r in results]).value_counts()
        print(f'  Summary: {dict(n_by_cat)}')

        rows = [{'sample_id': sample_id, 'spot_id': spot_id, **r,
                 'flags': ';'.join(r['flags'])}
                for (spot_id, _, _), r in zip(spots, results)]
        df_class = pd.DataFrame(rows)
        all_classifications.append(df_class)
        out = out_dir / f'{sample_id}_pre_edge_classification.csv'
        df_class.to_csv(out, index=False)
        print(f'  Saved: {out.name}')

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
        fig = plot_small_multiples(sample_id, spots, PRE_EDGE_WINDOW, results=results)
        if SAVE_FIG:
            out = out_dir / f'{sample_id}_pre_edge_grid.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            print(f'  Saved: {out.name}')

if CLASSIFY and all_classifications:
    combined = pd.concat(all_classifications, ignore_index=True)
    out = out_dir / 'pre_edge_classification_all.csv'
    combined.to_csv(out, index=False)
    print(f'\nSaved combined: {out}')

plt.show()
