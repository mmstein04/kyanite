# =============================================================================
# kyanite_outlier_method_comparison.py
#
# Diagnostic: kyanite_figures.py's outlier exclusion (SATURATION_FILTER +
# OUTLIER_METHOD) only ever runs one statistical-trim configuration per run,
# and its own outlier_exclusion_QC.png shows that one configuration's result
# without anything to compare it against. This script sweeps a fixed list of
# candidate trim configurations (METHODS below) over every grain's raw
# element maps and renders them side by side, spatially, so you can visually
# judge which one actually catches the edge/inclusion/bad-data pixels you can
# already see by eye in the map, rather than picking thresholds blind.
#
# For every grain with a saved mask TIFF (figs/data/<grain_id>_mask.tif) and
# every ELEMENTS column with a raw map TIFF on disk
# (inputs/maps/<grain_id>/<grain_id>_<element>.tif), one figure is saved:
# the masked concentration map, followed by one panel per METHODS entry
# showing gray=kept / dark red=saturation-excluded / orange=statistically-
# trimmed, exactly like kyanite_figures.py's own spatial QC coloring so the
# two are visually comparable. Saturation detection (SATURATION_FILTER and
# friends) is held fixed across all methods being compared, using the same
# defaults kyanite_figures.py ships with — only the statistical-trim step
# (OUTLIER_METHOD + its thresholds) varies between panels, so what's being
# compared is the trim step alone, on top of one common baseline.
#
# Also writes one summary CSV (grain x element x method -> n_kept /
# n_saturation_excluded / n_stat_excluded / frac_excluded) across the whole
# sweep, so counts can be compared numerically as well as visually.
#
# R-MATRIX COMPARISON: pixel exclusion is only half the question — the point
# of trimming outliers is to get a cleaner CL-vs-element Pearson r, so this
# also builds one grains x ELEMENTS heatmap of that r per METHODS entry
# (mirrors kyanite_figures.py's own 'summary' plot type, just one per method
# instead of one for whichever OUTLIER_METHOD happens to be configured).
# Pairing CL with each element is done by re-deriving both from the same
# in-memory grain mask (arr[mask] for the element map, cl_arr[mask] for
# <grain_id>_CL_registered.tif in figs/) rather than joining on pixel_data.csv
# rows — pixel_data.csv carries no row/col, and MATLAB's cl_reg(mask)/
# epma_raw{e}(mask) indexing is column-major while numpy's arr[mask] is
# row-major, so the two would silently mismatch if paired by row position.
# Indexing both fresh arrays with the identical boolean mask keeps element i
# of one aligned with element i of the other regardless of that convention
# mismatch. Whichever units the element TIFFs happen to be in (raw vs.
# normalize_epma-normalized) doesn't matter either, since Pearson r is
# invariant to any affine rescaling of either axis.
#
# This script only reads inputs already produced by CL_EPMA_registration.m /
# CL_mask_edit.m — it doesn't touch kyanite_figures.py's own parameters or
# outputs, and finding the right configuration here is just a starting point:
# copy the winning METHODS entry's OUTLIER_METHOD/MAD_K_*/PCT_* values into
# kyanite_figures.py's own parameter block once you've picked one.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import tifffile
from pathlib import Path
from kyanite_palette import ORANG, SEQUENTIAL_CMAP, DIVERGING_CMAP

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parent

MASK_DIR = _REPO_ROOT / 'figs' / 'data'      # grain masks; also where grains are discovered from
MAPS_DIR = _REPO_ROOT / 'inputs' / 'maps'    # raw element map TIFFs, <grain_id>/<grain_id>_<element>.tif
CL_DIR   = _REPO_ROOT / 'figs'               # registered CL TIFFs, <grain_id>_CL_registered.tif
OUTPUT_DIR = _REPO_ROOT / 'figs' / 'diagnostics'

GRAINS   = None   # None = every <grain_id>_mask.tif found in MASK_DIR; or a list of grain_id strings
ELEMENTS = ['Cr_Ka', 'V_Ka', 'Fe_Ka', 'Mn_Ka', 'Ti_Ka']   # only elements with a map TIFF for a given grain are used

# Saturation detection — same defaults/meaning as kyanite_figures.py, held
# fixed across every method below so the comparison isolates the
# statistical-trim step. See that script's header comment for the rationale.
SATURATION_FILTER    = True
SATURATION_BAND_FRAC = 0.001
SATURATION_MIN_FRAC  = 0.005
SATURATION_MIN_COUNT = 5

# Candidate statistical-trim configurations to compare, left to right in each
# figure. 'none' is a baseline (saturation only, no statistical trim) so you
# can see what saturation detection alone would have left behind. Add/remove
# entries here — no other code needs to change.
METHODS = [
    {'label': 'pct_hi99',          'method': 'percentile', 'pct_lo': 0, 'pct_hi': 99},
    {'label': 'pct_hi95',          'method': 'percentile', 'pct_lo': 0, 'pct_hi': 95},
    {'label': 'mad_hi3.5',         'method': 'mad', 'k_lo': None, 'k_hi': 3.5},
    {'label': 'mad_hi4',           'method': 'mad', 'k_lo': None, 'k_hi': 4},
    {'label': 'mad_hi5',           'method': 'mad', 'k_lo': None, 'k_hi': 5},
]

SUMMARY_CSV = 'outlier_method_comparison_summary.csv'   # written to OUTPUT_DIR

# R-matrix comparison (see header comment) — one grains x ELEMENTS Pearson-r
# heatmap per METHODS entry, plus one long-format CSV across all of them.
ALL_GRAINS_LABEL = 'all_grains'
R_SUMMARY_CSV    = 'outlier_method_comparison_r_summary.csv'   # written to OUTPUT_DIR

# =============================================================================
# FILTER LOGIC — mirrors kyanite_figures.py's saturation_mask/mad_keep_mask/
# outlier_keep_mask, parameterized per METHODS entry instead of module-level
# globals, so behavior stays identical to what kyanite_figures.py would do
# with the same settings.
# =============================================================================

def saturation_mask(x):
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
        mask |= near_max
    return mask


def mad_keep_mask(x, k_lo, k_hi):
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


def stat_keep_mask(x, cfg):
    if cfg['method'] == 'none':
        return np.ones(len(x), dtype=bool)
    elif cfg['method'] == 'mad':
        return mad_keep_mask(x, cfg['k_lo'], cfg['k_hi'])
    elif cfg['method'] == 'percentile':
        lo, hi = np.percentile(x, [cfg['pct_lo'], cfg['pct_hi']])
        return (x >= lo) & (x <= hi)
    else:
        raise ValueError(f"Unknown method {cfg['method']!r} in METHODS entry {cfg['label']!r}")


def categorize(vals, cfg):
    # 0 = kept, 1 = saturation-excluded, 2 = statistically-trimmed.
    sat = saturation_mask(vals)
    stat_excluded = np.zeros(len(vals), dtype=bool)
    rest_idx = np.where(~sat)[0]
    if len(rest_idx) >= 2:
        keep_rest = stat_keep_mask(vals[rest_idx], cfg)
        stat_excluded[rest_idx[~keep_rest]] = True
    cat = np.zeros(len(vals), dtype=int)
    cat[sat] = 1
    cat[stat_excluded] = 2
    return cat, int(sat.sum()), int(stat_excluded.sum())

# =============================================================================
# RUN
# =============================================================================

mask_dir = Path(MASK_DIR)
maps_dir = Path(MAPS_DIR)
cl_dir   = Path(CL_DIR)
out_dir  = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

if GRAINS is None:
    grain_ids = sorted(p.stem.replace('_mask', '') for p in mask_dir.glob('*_mask.tif'))
else:
    grain_ids = list(GRAINS)

if not grain_ids:
    raise FileNotFoundError(f'No <grain_id>_mask.tif files found in {mask_dir}')

print(f'Comparing {len(METHODS)} method(s) over {len(grain_ids)} grain(s): {", ".join(grain_ids)}')

color_map = {
    -1: (1, 1, 1, 1),
    0:  (0.85, 0.85, 0.85, 1),
    1:  (*mcolors.to_rgb('#8B0000'), 1.0),
    2:  (*mcolors.to_rgb(ORANG), 1.0),
}

summary_rows = []
r_rows = []
ncols = min(4, len(METHODS) + 1)

for grain_id in grain_ids:
    mask_path = mask_dir / f'{grain_id}_mask.tif'
    mask = tifffile.imread(mask_path) > 128

    # Registered CL, re-derived from the same mask (see header comment for
    # why this can't just be pulled from pixel_data.csv by row position).
    # Loaded once per grain and reused across every element/method below.
    cl_path = cl_dir / f'{grain_id}_CL_registered.tif'
    cl_vals = None
    if not cl_path.exists():
        print(f'  WARNING: {grain_id}: {cl_path.name} not found — skipping r-matrix rows for this grain')
    else:
        cl_arr = tifffile.imread(cl_path).astype(float)
        if cl_arr.shape != mask.shape:
            print(f'  WARNING: {grain_id}: {cl_path.name} shape {cl_arr.shape} != mask shape '
                  f'{mask.shape} — skipping r-matrix rows for this grain')
        else:
            cl_vals = cl_arr[mask]

    grain_elements = [e for e in ELEMENTS if (maps_dir / grain_id / f'{grain_id}_{e}.tif').exists()]
    missing = [e for e in ELEMENTS if e not in grain_elements]
    if missing:
        print(f'  WARNING: {grain_id}: no map TIFF for {missing}, skipping those')
    if not grain_elements:
        print(f'  WARNING: {grain_id}: no ELEMENTS map TIFFs found, skipping grain')
        continue

    print(f'\n--- {grain_id} ---')
    for element in grain_elements:
        map_path = maps_dir / grain_id / f'{grain_id}_{element}.tif'
        arr = tifffile.imread(map_path).astype(float)
        vals = arr[mask]
        if len(vals) < 2:
            print(f'  WARNING: {grain_id} {element}: fewer than 2 in-mask pixels, skipping')
            continue
        n = len(vals)
        masked_arr = np.where(mask, arr, np.nan)

        nrows = int(np.ceil((len(METHODS) + 1) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
        axes = np.atleast_1d(axes).ravel()

        im0 = axes[0].imshow(masked_arr, cmap=SEQUENTIAL_CMAP)
        axes[0].set_title(f'{element} concentration\n(masked, n={n:,})', fontsize=9)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        for ax, cfg in zip(axes[1:], METHODS):
            cat, n_sat, n_stat = categorize(vals, cfg)
            cat2d = np.full(arr.shape, -1, dtype=int)
            cat2d[mask] = cat
            rgba = np.zeros((*cat2d.shape, 4))
            for k, c in color_map.items():
                rgba[cat2d == k] = c
            ax.imshow(rgba)
            frac_excluded = (n_sat + n_stat) / n
            ax.set_title(f'{cfg["label"]}\nsat={n_sat/n:.1%}, trim={n_stat/n:.1%}, '
                         f'total={frac_excluded:.1%}', fontsize=9)
            summary_rows.append(dict(grain=grain_id, element=element, method=cfg['label'],
                                      n_total=n, n_kept=n - n_sat - n_stat,
                                      n_saturation_excluded=n_sat, n_stat_excluded=n_stat,
                                      frac_excluded=frac_excluded))
            print(f'  {element} [{cfg["label"]}]: sat={n_sat:,} ({n_sat/n:.2%}), '
                  f'trim={n_stat:,} ({n_stat/n:.2%}), total excluded={frac_excluded:.2%}')

            if cl_vals is not None:
                keep = cat == 0
                x_f, y_f = vals[keep], cl_vals[keep]
                if len(x_f) >= 2 and np.std(x_f) > 0:
                    r = np.corrcoef(x_f, y_f)[0, 1]
                else:
                    r = np.nan
                r_rows.append(dict(grain=grain_id, element=element, method=cfg['label'],
                                    r=r, n=len(x_f)))

        for ax in axes[1 + len(METHODS):]:
            ax.axis('off')
        for ax in axes[:1 + len(METHODS)]:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f'{grain_id} — {element}: outlier method comparison '
                     '(gray=kept, dark red=saturation, orange=statistical trim)', fontsize=11)
        plt.tight_layout()

        out = out_dir / f'{grain_id}_{element}_outlier_method_comparison.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out.name}')
        plt.close(fig)

summary_df = pd.DataFrame(summary_rows)
summary_out = out_dir / SUMMARY_CSV
summary_df.to_csv(summary_out, index=False)
print(f'\nSaved: {summary_out}')

# =============================================================================
# R-MATRIX COMPARISON — one grains x ELEMENTS Pearson-r heatmap per method,
# built from r_rows collected above. Same grain/element ordering across all
# of them so panels are visually comparable method to method.
# =============================================================================

r_df = pd.DataFrame(r_rows)
if r_df.empty:
    print(f'\nWARNING: no CL-vs-element r values computed (no {cl_dir}/*_CL_registered.tif '
          'found, or none matched their grain\'s mask shape) — skipping r-matrix heatmaps')
else:
    r_out = out_dir / R_SUMMARY_CSV
    r_df.to_csv(r_out, index=False)
    print(f'Saved: {r_out}')

    r_grains = sorted(r_df['grain'].unique())
    r_elements = [e for e in ELEMENTS if e in r_df['element'].unique()]

    print(f'\n--- R-matrix comparison ({len(r_grains)} grain(s), {len(METHODS)} method(s)) ---')
    for cfg in METHODS:
        label = cfg['label']
        sub = r_df[r_df['method'] == label]
        rmat = np.full((len(r_grains), len(r_elements)), np.nan)
        nmat = np.full((len(r_grains), len(r_elements)), np.nan)
        for i, grain in enumerate(r_grains):
            for j, element in enumerate(r_elements):
                row = sub[(sub['grain'] == grain) & (sub['element'] == element)]
                if len(row):
                    rmat[i, j] = row['r'].values[0]
                    nmat[i, j] = row['n'].values[0]
        rdf_wide = pd.DataFrame(rmat, index=r_grains, columns=r_elements)
        ndf_wide = pd.DataFrame(nmat, index=r_grains, columns=r_elements)

        fig, ax = plt.subplots(figsize=(len(r_elements) * 0.9 + 2, len(r_grains) * 0.7 + 2))
        sns.heatmap(rdf_wide, ax=ax, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, center=0,
                    annot=False, cbar_kws={'label': 'Pearson r'},
                    linewidths=0.5, linecolor='white')
        for i in range(len(r_grains)):
            for j in range(len(r_elements)):
                r, n = rdf_wide.iloc[i, j], ndf_wide.iloc[i, j]
                if pd.isna(r):
                    txt, color = 'n/a', 'gray'
                else:
                    n_label = f'{int(n) // 1000}k' if n >= 1000 else f'{int(n)}'
                    txt = f'{r:.2f}\nn={n_label}'
                    color = 'white' if abs(r) >= 0.6 else 'black'
                ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center', fontsize=7, color=color)
        ax.set_xlabel('element')
        ax.set_ylabel('grain')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        ax.set_title(f'{ALL_GRAINS_LABEL} — CL vs. element r by grain, method = {label}', fontsize=11)
        plt.tight_layout()

        out = out_dir / f'{ALL_GRAINS_LABEL}_{label}_r_heatmap.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'  Saved: {out.name}')
        plt.close(fig)
