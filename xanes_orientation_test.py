# =============================================================================
# xanes_orientation_test.py
#
# Crystallographic-orientation test on the Fe pre-edge fit centroid: one point
# on a grain, re-measured after rotating the sample in successive steps in the
# beamline chamber. Plots rotation angle vs. Lorentzian pre-edge fit centroid
# (with fit-stderr error bars), to check whether the measured centroid (and
# so the inferred Fe speciation) depends on how the crystal is oriented.
#
# Input is the raw Larch/Athena "Pre-edge Peak Fit Report" export, same format
# kyanite_spot_analysis.py reads from inputs/prepeak_fits/. The difference is
# that each 'Data Set' here is one ORIENTATION of a single spot rather than a
# different spot, with the angle encoded in the name (e.g. 'MW609-2-45deg.001'
# -> 45). No CL image, mask or spot geochemistry CSV is needed.
#
# Re-fits of one orientation are resolved the same way kyanite_spot_analysis.py
# resolves re-fits of one spot: the most recent 'Fit Label' timestamp wins
# ('_N' suffix = a later same-minute repeat), not row position, since Larch
# writes each data set's newest fit first.
#
# Console reports the error-weighted mean centroid, reduced chi-square against
# "no orientation effect" (a constant centroid), and the 0 vs 180 deg
# difference. Linear dichroism in a pre-edge goes as cos^2 of the angle between
# the polarization and the crystal axes, so it repeats every 180 deg — 0 and 180
# should agree if the rotation returned the crystal to an equivalent position.
# =============================================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from kyanite_palette import BLUE, ORANG

# =============================================================================
# PARAMETERS
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parent

GRAIN_ID = 'MW609-02'   # used for the title and output filename
PREPEAK_FILE = _REPO_ROOT / 'inputs' / 'prepeak_fits' / 'MW609-2_prepeak_fits.csv'
OUT_DIR = _REPO_ROOT / 'figs' / 'xanes'

ERR_SIGMA = 1            # error bar half-width, as a multiple of fit_centroid_stderr
SHOW_WEIGHTED_MEAN = True   # horizontal line at the error-weighted mean centroid

# Same Wilke et al. (2001) Fe2+/Fe3+ reference centroids as kyanite_spot_analysis.py.
# Only lines that fall inside the data's axis range are drawn: extending the axis
# down to Fe2+ (7112.1 eV) would squash a ~0.2 eV orientation spread flat.
REFERENCE_LINES = {'Fe²⁺': 7112.1, 'Fe³⁺': 7113.5}
REFERENCE_EXTEND_AXIS = False
REFERENCE_COLOR = '0.2'

FIGSIZE = (5.0, 3.6)
FIG_DPI = 300
SHOW_TITLE = True
SAVE_FIG = True

Y_LABEL = 'Fe pre-edge fit centroid (eV)'
X_LABEL = 'Sample rotation (°)'


# =============================================================================
# LOAD
# =============================================================================

def _orientation_deg(dataset_name):
    """Rotation angle from a 'Data Set' name, e.g. 'MW609-2-135deg.001' -> 135.0.
    No sign is parsed: the '-' before the angle is the name's separator."""
    match = re.search(r'(\d+(?:\.\d+)?)\s*deg', str(dataset_name), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def load_orientation_fits(path):
    path = Path(path)
    with open(path) as fh:
        lines = fh.readlines()
    header_idx = next((i for i, line in enumerate(lines) if re.match(r'^#\s*Data Set', line)), None)
    if header_idx is None:
        raise ValueError(f'{path.name} has no "# Data Set" header line')

    fits = pd.read_csv(path, skiprows=header_idx, skipinitialspace=True)
    fits.columns = [c.lstrip('#').strip() for c in fits.columns]
    fits['Data Set'] = fits['Data Set'].astype(str).str.strip()
    fits['angle'] = fits['Data Set'].map(_orientation_deg)
    unparsed = fits.loc[fits['angle'].isna(), 'Data Set'].tolist()
    if unparsed:
        print(f'  WARNING: no "<N>deg" angle in data set name(s) {unparsed} — dropped.')
        fits = fits[fits['angle'].notna()]

    # Most recent fit per orientation, by Fit Label timestamp (see header).
    label = fits['Fit Label'].astype(str).str.strip()
    stamp = pd.to_datetime('2000-' + label.str.replace(r'_\d+$', '', regex=True),
                           format='%Y-%b-%d %H:%M', errors='coerce')
    repeat = pd.to_numeric(label.str.extract(r'_(\d+)$')[0], errors='coerce').fillna(0)
    if stamp.isna().any():
        print('  WARNING: unparseable Fit Label(s) — assuming newest-first file order.')
        stamp = pd.Series(pd.Timestamp(0), index=fits.index)
    fits = fits.assign(_stamp=stamp, _repeat=repeat, _pos=-np.arange(len(fits)))
    for angle, grp in fits.groupby('angle'):
        if len(grp) > 1:
            print(f'  {angle:g}°: {len(grp)} fits — keeping the most recent. All fits:')
            for _, r in grp.sort_values(['_stamp', '_repeat', '_pos']).iterrows():
                print(f'      {r["Fit Label"]:>16s}  centroid {r["fit_centroid"]:.4f} '
                      f'± {r["fit_centroid_stderr"]:.4f} eV')
    fits = (fits.sort_values(['angle', '_stamp', '_repeat', '_pos'])
                .drop_duplicates('angle', keep='last'))

    return pd.DataFrame({
        'angle': fits['angle'].values,
        'data_set': fits['Data Set'].values,
        'fit_label': fits['Fit Label'].astype(str).str.strip().values,
        'fit_centroid': pd.to_numeric(fits['fit_centroid'], errors='coerce').values,
        'fit_centroid_stderr': pd.to_numeric(fits['fit_centroid_stderr'], errors='coerce').values,
        'fit_r2': pd.to_numeric(fits['R^2'], errors='coerce').values,
    }).sort_values('angle').reset_index(drop=True)


# =============================================================================
# STATS (console only — keeps the figure text publishable)
# =============================================================================

def report(df):
    print(f'\n{GRAIN_ID} — pre-edge fit centroid by orientation:')
    for _, r in df.iterrows():
        print(f'  {r["angle"]:6g}°  {r["fit_centroid"]:.4f} ± {r["fit_centroid_stderr"]:.4f} eV'
              f'  (R² {r["fit_r2"]:.4f}, fit {r["fit_label"]})')

    y, s = df['fit_centroid'].to_numpy(), df['fit_centroid_stderr'].to_numpy()
    w = 1 / s ** 2
    mean = float(np.sum(w * y) / np.sum(w))
    mean_err = float(np.sqrt(1 / np.sum(w)))
    chi2 = float(np.sum(((y - mean) / s) ** 2))
    dof = len(y) - 1
    print(f'  Range (max − min): {y.max() - y.min():.4f} eV')
    print(f'  Error-weighted mean: {mean:.4f} ± {mean_err:.4f} eV')
    if dof > 0:
        print(f'  χ² vs. constant centroid: {chi2:.2f} on {dof} dof '
              f'(reduced χ² = {chi2 / dof:.2f}; ≫1 suggests scatter beyond fit error)')

    ends = df.set_index('angle')
    if {0.0, 180.0} <= set(ends.index):
        d = ends.loc[180.0, 'fit_centroid'] - ends.loc[0.0, 'fit_centroid']
        sd = np.hypot(ends.loc[180.0, 'fit_centroid_stderr'], ends.loc[0.0, 'fit_centroid_stderr'])
        print(f'  180° − 0°: {d:+.4f} ± {sd:.4f} eV ({abs(d) / sd:.1f}σ) — '
              f'expected ~0 for 180°-periodic dichroism')
    return mean


# =============================================================================
# PLOT
# =============================================================================

def plot_orientation(df, mean):
    err = ERR_SIGMA * df['fit_centroid_stderr']
    fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')

    if SHOW_WEIGHTED_MEAN:
        ax.axhline(mean, color=ORANG, lw=1.2, zorder=1, label='Weighted mean')
    ax.errorbar(df['angle'], df['fit_centroid'], yerr=err, fmt='none',
                ecolor='0.35', elinewidth=0.9, capsize=3, zorder=2)
    ax.scatter(df['angle'], df['fit_centroid'], s=45, color=BLUE,
               edgecolors='black', linewidths=0.5, zorder=3)

    ax.set_xticks(df['angle'])
    pad = 0.08 * (df['angle'].max() - df['angle'].min() or 1)
    ax.set_xlim(df['angle'].min() - pad, df['angle'].max() + pad)

    lo = float((df['fit_centroid'] - err).min())
    hi = float((df['fit_centroid'] + err).max())
    margin = 0.15 * (hi - lo or 0.1)
    lo, hi = lo - margin, hi + margin
    refs = list((REFERENCE_LINES or {}).values())
    if REFERENCE_EXTEND_AXIS and refs:
        lo, hi = min(lo, *refs), max(hi, *refs)
    ax.set_ylim(lo, hi)
    for name, energy in (REFERENCE_LINES or {}).items():
        if lo <= energy <= hi:
            ax.axhline(energy, color=REFERENCE_COLOR, ls='--', lw=1.0, zorder=1)
            # Left end, just below the line — clear of the 0° point's error bar.
            ax.annotate(name, (0, energy), xycoords=('axes fraction', 'data'),
                        xytext=(3, -2), textcoords='offset points', ha='left',
                        va='top', fontsize=8, color=REFERENCE_COLOR)

    # Full absolute energies on every tick (no '+7.113e3' offset label).
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2f}'))
    ax.set_xlabel(X_LABEL, fontsize=10)
    ax.set_ylabel(Y_LABEL, fontsize=10)
    ax.tick_params(labelsize=8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if SHOW_WEIGHTED_MEAN:
        ax.legend(loc='upper left', fontsize=8, frameon=False)
    if SHOW_TITLE:
        ax.set_title(f'{GRAIN_ID} — Fe pre-edge fit centroid vs. orientation', fontsize=11)
    return fig


if __name__ == '__main__':
    print(f'Reading {PREPEAK_FILE}')
    df = load_orientation_fits(PREPEAK_FILE)
    mean = report(df)
    fig = plot_orientation(df, mean)
    if SAVE_FIG:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(OUT_DIR) / f'{GRAIN_ID}_centroid_vs_orientation.png'
        fig.savefig(out, dpi=FIG_DPI)
        print(f'\nSaved {out}')
    plt.close(fig)
