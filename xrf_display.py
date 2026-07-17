"""
xrf_display.py

Visualize XRF element-map TIFFs with grain mask overlay, plus optional
element-ratio maps (e.g. Cr/V) rendered in the exact same style. Replaces
xrf_display.m — nothing here needs MATLAB (no cpselect/imwarp/registration),
so the port lets it reuse this project's shared conventions directly instead
of hand-copying them: kyanite_palette.py's SEQUENTIAL_CMAP (in place of
MATLAB's parula) and the same saturation + MAD outlier logic
kyanite_figures.py uses for OUTLIER_METHOD='mad' (in place of a fixed
percentile clip).

Display range convention: the saturation + MAD-outlier logic below is used
only to pick the imshow color range (vmin/vmax), not to drop pixels — every
in-mask pixel is still drawn, with values beyond vmin/vmax simply clamped to
the end colors. That's what actually fixes washed-out internal zoning: a
handful of extreme pixels no longer pins the whole scale, without punching
holes in the rendered map the way row-wise exclusion would.

Output: figs/map_renders/<grain_id>_<tag>_display.png
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

from kyanite_palette import SEQUENTIAL_CMAP

# =============================================================================
# PARAMETERS — edit this section for each use
# =============================================================================

# Anchored to this script's own location (not a hardcoded machine-specific
# path) so the same file runs unmodified on any machine/cluster this repo is
# checked out on.
_REPO_ROOT = Path(__file__).resolve().parent

MAPS_DIR   = _REPO_ROOT / 'inputs' / 'maps'
MASK_DIR   = _REPO_ROOT / 'figs' / 'data'
OUTPUT_DIR = _REPO_ROOT / 'figs' / 'map_renders'

# Grain and element selection. GRAIN_IDS may be a single string, a list for
# batch processing, or None to auto-discover every grain that has both a
# maps folder (MAPS_DIR/<grain_id>/) and a mask (MASK_DIR/<grain_id>_mask.tif).
GRAIN_IDS = 'NA-GS-P84-06'
ELEMENTS  = ['Cr', 'Fe', 'Mn', 'Ti', 'V']   # bare symbol; maps as <grain_id>_<el>_Ka.tif

# Element-ratio maps (numerator, denominator), e.g. Cr/V and Fe/Mn. Both
# elements need a map file on disk (same _Ka naming as ELEMENTS — they don't
# need to also appear in ELEMENTS itself). Empty list disables ratio maps.
RATIOS = [('Cr', 'V'), ('Cr', 'Fe'), ('Fe', 'Mn')]

# Visualization options
CMAP          = SEQUENTIAL_CMAP   # project convention for continuous-intensity maps (kyanite_palette.py)
BKGD_COLOR    = 'black'
FONT_COLOR    = 'white'
SHOW_COLORBAR = True

# Outlier/contrast-scaling conventions — same two-stage logic as
# kyanite_figures.py's default (OUTLIER_METHOD='mad', MAD_K_HI=4), just
# reused here to set the display range instead of excluding data points (see
# module docstring). Hand-copied rather than imported: kyanite_figures.py is
# a flat script whose top level runs a whole analysis on import, not an
# importable library, so its saturation_mask/mad_keep_mask are mirrored here
# in miniature instead — keep the two in sync if the algorithm changes.
#
# 1. Saturation/clipping: pixels piled up in a thin band near the element's
#    own max value (signature of a saturated detector channel, not a real
#    continuous distribution's natural extreme) are excluded from the
#    vmin/vmax calculation, and flagged with a printed warning. Max side
#    only — a pileup near the min is ordinary near-zero/below-detection-limit
#    data, not saturation.
SATURATION_FILTER    = True
SATURATION_BAND_FRAC = 0.001
SATURATION_MIN_FRAC  = 0.005
SATURATION_MIN_COUNT = 5

# 2. Statistical spread trim: robust MAD z-score computed in log-space
#    (element concentrations are right-skewed), matching this project's
#    MAD_K_HI=4 default (no low-side trim).
MAD_K_LO = None
MAD_K_HI = 4

# Ratio maps often have a more skewed distribution than raw element counts,
# so their MAD trim is independently configurable.
RATIO_MAD_K_LO = None
RATIO_MAD_K_HI = 4

# Figure export
SAVE_FIGS = True   # False to render without writing PNGs
FIG_DPI   = 300

# Scale bar
SHOW_SCALEBAR = True

# Pixel size (µm/px) the scale bar is computed from. By default this is read
# straight from xrf_h5_to_tiff.py's metadata sidecar (<grain_id>_<el>_Ka.txt,
# step_size_pos1_um — fast axis / X, the axis the horizontal scale bar spans)
# for whichever loaded element has one, so it can't drift out of sync with
# the actual scan geometry. PIXEL_UM below is only the fallback used (with a
# warning) if no sidecar is found/parseable for any loaded element.
PIXEL_UM_FROM_SIDECAR = True
PIXEL_UM        = 2.0    # µm/pixel fallback — scalar for all grains, or a list matching GRAIN_IDS
SCALEBAR_UM     = 100    # physical length of scale bar in µm
SCALEBAR_POS    = 'se'   # 'se' | 'sw' | 'ne' | 'nw'
SCALEBAR_MARGIN = 0.04   # margin from edge, as a fraction of image dimensions

# =============================================================================

_SIDECAR_PIXEL_UM_RE = re.compile(r'step_size_pos1_um\s*:\s*([-\d.eE]+)')


def read_pixel_um_from_sidecar(tif_path):
    """Fast-axis (X, pos1) pixel size in microns from xrf_h5_to_tiff.py's
    metadata sidecar for this TIFF (same base name, .txt extension) — the
    axis the horizontal scale bar spans. None if the sidecar is missing or
    the field can't be parsed."""
    sidecar = tif_path.with_suffix('.txt')
    if not sidecar.exists():
        return None
    m = _SIDECAR_PIXEL_UM_RE.search(sidecar.read_text())
    return float(m.group(1)) if m else None


def discover_grain_ids():
    """Every subfolder of MAPS_DIR that also has a grain mask in MASK_DIR,
    sorted — used when GRAIN_IDS is None to run every available grain."""
    candidates = sorted(p.name for p in MAPS_DIR.iterdir() if p.is_dir())
    found = [g for g in candidates if (MASK_DIR / f'{g}_mask.tif').exists()]
    skipped = [g for g in candidates if g not in found]
    if skipped:
        print(f'Skipping {len(skipped)} grain(s) with no mask in {MASK_DIR}: {", ".join(skipped)}')
    return found


def saturation_mask(x, label, verbose=True):
    """Flags pixels piled up near x's own max — see kyanite_figures.py's
    function of the same name/logic (mirrored here, see module docstring)."""
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
                  f'of its range — likely detector saturation/clipping, excluded from display range')
        mask |= near_max
    return mask


def mad_keep_mask(x, k_lo, k_hi):
    """Robust MAD z-score in log-space — see kyanite_figures.py's function
    of the same name/logic (mirrored here, see module docstring)."""
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


def display_range(x, label, mad_k_lo, mad_k_hi):
    """vmin/vmax for imshow: saturation- and MAD-outlier pixels are excluded
    from this calculation only — every in-mask pixel is still drawn, just
    clamped to the returned bounds (see module docstring)."""
    sat = saturation_mask(x, label)
    kept = x[~sat]
    if len(kept) < 2:
        return float(x.min()), float(x.max())

    keep = mad_keep_mask(kept, mad_k_lo, mad_k_hi)
    inliers = kept[keep]
    if len(inliers) == 0:
        return float(kept.min()), float(kept.max())
    return max(0.0, float(inliers.min())), float(inliers.max())


def render_and_save_map(grain_id, label, filename_tag, img, mask, grain_px_um, mad_k_lo, mad_k_hi):
    """Renders one masked, contrast-scaled map (element or ratio) in the
    project's standard style and optionally exports it as a PNG. Shared by
    both the element and ratio display loops so their look never drifts."""
    finite_mask = mask & np.isfinite(img)
    grain_vals = img[finite_mask]
    if grain_vals.size == 0:
        print(f'  WARNING: {grain_id} — {label}: no finite in-mask pixels, skipping.')
        return

    vmin, vmax = display_range(grain_vals, label, mad_k_lo, mad_k_hi)

    nrows, ncols = img.shape
    fig_h = 6.0
    fig, ax = plt.subplots(figsize=(fig_h * ncols / nrows, fig_h))
    fig.patch.set_facecolor(BKGD_COLOR)
    ax.set_facecolor(BKGD_COLOR)

    im = ax.imshow(img, cmap=CMAP, vmin=vmin, vmax=vmax)
    im.set_alpha(finite_mask.astype(float))

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{grain_id}  {label}', color=FONT_COLOR)

    if SHOW_COLORBAR:
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(colors=FONT_COLOR)
        cb.outline.set_edgecolor(FONT_COLOR)

    if SHOW_SCALEBAR:
        sb_px = SCALEBAR_UM / grain_px_um
        mx = SCALEBAR_MARGIN * ncols
        my = SCALEBAR_MARGIN * nrows
        bar_h = max(3, round(nrows * 0.012))

        if SCALEBAR_POS == 'se':
            x1, y1 = ncols - mx - sb_px, nrows - my - bar_h
        elif SCALEBAR_POS == 'sw':
            x1, y1 = mx, nrows - my - bar_h
        elif SCALEBAR_POS == 'ne':
            x1, y1 = ncols - mx - sb_px, my
        elif SCALEBAR_POS == 'nw':
            x1, y1 = mx, my
        else:
            raise ValueError(f"Unknown SCALEBAR_POS {SCALEBAR_POS!r}; choose 'se', 'sw', 'ne', or 'nw'")

        ax.add_patch(plt.Rectangle((x1, y1), sb_px, bar_h, facecolor=FONT_COLOR, edgecolor='none'))
        ax.text(x1 + sb_px / 2, y1 - bar_h, f'{SCALEBAR_UM:g} µm',
                color=FONT_COLOR, fontsize=9, ha='center', va='bottom')

    if SAVE_FIGS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f'{grain_id}_{filename_tag}_display.png'
        fig.savefig(out_path, dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f'Saved: {out_path}')

    plt.close(fig)


def main():
    if GRAIN_IDS is None:
        grain_ids = discover_grain_ids()
        if not grain_ids:
            raise FileNotFoundError(f'No grains with both a maps folder and a mask found under {MAPS_DIR}')
    else:
        grain_ids = [GRAIN_IDS] if isinstance(GRAIN_IDS, str) else list(GRAIN_IDS)
    pixel_um = [PIXEL_UM] if np.isscalar(PIXEL_UM) else list(PIXEL_UM)

    print(f'Processing {len(grain_ids)} grain(s):')
    for g in grain_ids:
        print(f'  {g}')
    if RATIOS:
        print(f'Ratio maps requested ({len(RATIOS)}):')
        for num_el, den_el in RATIOS:
            print(f'  {num_el}/{den_el}')

    for gi, grain_id in enumerate(grain_ids):
        print(f'\n--- {grain_id} ---')

        mask_path = MASK_DIR / f'{grain_id}_mask.tif'
        if not mask_path.exists():
            raise FileNotFoundError(f'Grain mask not found: {mask_path}')
        mask = tifffile.imread(str(mask_path)) > 128

        # Load element maps: union of ELEMENTS and every ratio component.
        ratio_components = [el for pair in RATIOS for el in pair]
        elements_to_load = list(dict.fromkeys(ELEMENTS + ratio_components))

        imgs = {}
        sidecar_px_um, sidecar_source = None, None
        for el in elements_to_load:
            fn = MAPS_DIR / grain_id / f'{grain_id}_{el}_Ka.tif'
            if not fn.exists():
                print(f'  WARNING: {grain_id}: element map not found ({fn.name}) — skipping.')
                continue
            imgs[el] = tifffile.imread(str(fn)).astype(np.float64)
            if PIXEL_UM_FROM_SIDECAR and sidecar_px_um is None:
                found = read_pixel_um_from_sidecar(fn)
                if found is not None:
                    sidecar_px_um, sidecar_source = found, fn.name

        if PIXEL_UM_FROM_SIDECAR and sidecar_px_um is not None:
            grain_px_um = sidecar_px_um
            print(f'  Pixel size: {grain_px_um:.4g} µm/px (from {sidecar_source} metadata sidecar)')
        else:
            grain_px_um = pixel_um[min(gi, len(pixel_um) - 1)]
            if PIXEL_UM_FROM_SIDECAR:
                print(f'  WARNING: no metadata sidecar found/parseable for any loaded element — '
                      f'falling back to PIXEL_UM={grain_px_um:.4g} µm/px')

        # Plot elements
        for el in ELEMENTS:
            if el not in imgs:
                continue
            img = imgs[el].copy()
            img[~mask] = np.nan
            render_and_save_map(grain_id, f'{el}_Ka', f'{el}_Ka', img, mask,
                                 grain_px_um, MAD_K_LO, MAD_K_HI)

        # Plot element ratios
        for num_el, den_el in RATIOS:
            if num_el not in imgs or den_el not in imgs:
                print(f'  WARNING: {grain_id}: skipping {num_el}/{den_el} ratio — missing element map.')
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_img = imgs[num_el] / imgs[den_el]
            ratio_img[~np.isfinite(ratio_img)] = np.nan   # div-by-zero / 0-over-0
            ratio_img[~mask] = np.nan

            label = f'{num_el}/{den_el}'
            filename_tag = f'{num_el}_{den_el}_ratio'
            render_and_save_map(grain_id, label, filename_tag, ratio_img, mask,
                                 grain_px_um, RATIO_MAD_K_LO, RATIO_MAD_K_HI)


if __name__ == '__main__':
    main()
