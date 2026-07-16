"""
CL_local_regression_map.py

Slide a circular window across every pixel of an already-registered grain,
run a per-pixel CL-vs-element linear regression on the pixels inside that
window (intersected with the grain mask), and store the resulting slope and
Pearson r back at the center pixel. Produces continuous "slope map" and
"R map" images showing how the CL-element relationship varies spatially
across the grain, complementing the fixed polygon regions of
CL_region_extraction.m.

Replaces CL_local_regression_map.m — imread of TIFFs and 2-D convolution are
the only imaging operations it actually used, nothing MATLAB-only. The port
also lets it import kyanite_palette's DIVERGING_CMAP/SEQUENTIAL_CMAP directly
instead of a hand-rolled RdBu_r-matching colormap and MATLAB's parula.

Batch mode: GRAIN_IDS may be a single string, a list, or None to
auto-discover and run every grain that has a registered CL image, a mask,
and an EPMA/XRF maps folder. A grain that fails (missing input, size
mismatch, etc.) is skipped with a warning rather than aborting the batch.

Per-grain pixel size: like xrf_display.py, the window radius's physical
size only comes out correct if each grain's actual µm/px is used — grains
in this project are NOT all imaged at the same resolution (e.g. 1.0 vs.
2.0 µm/px), so a single hardcoded EPMA_PIXEL_UM would silently make the
window radius wrong (in physical terms) for whichever grains don't match
it. Pixel size is therefore read from xrf_h5_to_tiff.py's metadata sidecar
per grain, with EPMA_PIXEL_UM as a fallback (with a warning) if no sidecar
is found/parseable.

WORKFLOW (per grain):
  1. Load the registered CL image, grain mask, and EPMA/XRF maps already
     produced by CL_EPMA_registration.m for this grain — no warping,
     control-point picking, or grain-mask generation here.
  2. Build a binary circular kernel of the requested physical radius.
  3. Compute local regression sums (n, Sx, Sy, Sxx, Syy, Sxy) via 2-D
     convolution with that kernel — mathematically equivalent to running an
     explicit per-window linear regression at every pixel, but runs as a
     handful of convolutions instead of a double loop. Uses direct (not
     FFT-based) convolution, matching MATLAB's conv2 exactly — FFT round-off
     could otherwise leak past the degenerate-window check below, which
     relies on an absolute near-zero threshold.
  4. Derive per-pixel slope and Pearson r maps for every element from those
     sums; mask out pixels outside the grain or with too few valid
     neighbors in the window.
  5. Save maps (.npz + long-format .csv, in figs/data/) and QC figures
     (slope map grid, R map grid, window-coverage map, in
     figs/diagnostics/); the one true analysis-result figure (Cr R-vs-CL)
     is saved directly in figs/local_regression/.

Note: RowIdx/ColIdx in the pixel-data CSV are 0-based (numpy convention),
unlike the original MATLAB script's 1-based indices — nothing downstream
joins on them, so there's no MATLAB-side consumer to stay compatible with.

Output (per grain):
  figs/data/<grain_id>_local_regression.npz            — slope/R/n maps + metadata
  figs/data/<grain_id>_local_regression_pixel_data.csv — long-format per-pixel table
  figs/diagnostics/<grain_id>_local_regression_slope_QC.png
  figs/diagnostics/<grain_id>_local_regression_R_QC.png
  figs/diagnostics/<grain_id>_local_regression_n_map.png
  figs/diagnostics/<grain_id>_local_regression_analysis_log.txt
  figs/local_regression/<grain_id>_local_regression_R_Cr_vs_CL.png (skipped
    with a warning if no Cr element map is found)
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from scipy.signal import convolve2d

from kyanite_palette import DIVERGING_CMAP, SEQUENTIAL_CMAP

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parent

# Grain selection. A single string, a list for batch processing, or None to
# auto-discover every grain that has a registered CL image (INPUT_DIR), a
# mask (DATA_DIR), and an EPMA/XRF maps folder (MAPS_DIR/<grain_id>/).
GRAIN_IDS = None

# Directory containing the outputs of CL_EPMA_registration.m (registered CL
# TIFFs and the grain mask TIFF).
INPUT_DIR = _REPO_ROOT / 'figs'

# Reusable data (mask, pixel data, etc.) lives in its own subfolder of
# INPUT_DIR, per CL_EPMA_registration.m's convention. QC-only figures
# (slope/R grids, window-coverage map) and the log go in DIAGNOSTICS_DIR;
# OUTPUT_DIR holds only the true analysis-result figure (Cr R-vs-CL).
DATA_DIR        = INPUT_DIR / 'data'
DIAGNOSTICS_DIR = INPUT_DIR / 'diagnostics'
OUTPUT_DIR      = _REPO_ROOT / 'figs' / 'local_regression'

# Base folder of per-grain EPMA/XRF element map TIFFs (same folder
# CL_EPMA_registration.m used, i.e. MAPS_DIR/<grain_id>/). All *.tif files
# in a grain's subfolder are auto-discovered.
MAPS_DIR = _REPO_ROOT / 'inputs' / 'maps'

# Use the color registered CL as the background for the Cr comparison
# figure. Falls back to the grayscale registered CL (with a warning) if
# False or if the color TIFF isn't found. Never affects the regression
# math, which always uses the grayscale registered CL.
USE_COLOR_DISPLAY = True

# --- Spatial calibration -----------------------------------------------------
# Pixel size is read straight from xrf_h5_to_tiff.py's metadata sidecar
# (<grain_id>_<el>_Ka.txt's step_size_pos1_um, the fast/X axis) for whichever
# loaded element has one. EPMA_PIXEL_UM is only the fallback used (with a
# warning) if no sidecar is found/parseable for any of a grain's maps.
PIXEL_UM_FROM_SIDECAR = True
EPMA_PIXEL_UM = 2.0    # µm/pixel fallback — must match this grain's actual registration value

# --- Moving-window regression parameters -------------------------------------
# Physical radius of the circular regression window, in µm (kept in µm
# rather than px so it stays meaningful across grains imaged at different
# pixel sizes; converted to px per grain using that grain's own µm/px).
WINDOW_RADIUS_UM = 30.0

# --- Element map normalization ------------------------------------------------
# True:  re-normalize each element map to [0, 1] using its min/max within
#        the grain mask (matches CL_EPMA_registration.m's normalize_epma = True).
# False: keep raw pixel values (counts/intensity units from the TIFF).
NORMALIZE_EPMA = False

# =============================================================================

_SIDECAR_PIXEL_UM_RE = re.compile(r'step_size_pos1_um\s*:\s*([-\d.eE]+)')


def read_pixel_um_from_sidecar(tif_path):
    """Fast-axis (X, pos1) pixel size in microns from xrf_h5_to_tiff.py's
    metadata sidecar for this TIFF (same base name, .txt extension) — see
    xrf_display.py's function of the same name/logic (mirrored here)."""
    sidecar = tif_path.with_suffix('.txt')
    if not sidecar.exists():
        return None
    m = _SIDECAR_PIXEL_UM_RE.search(sidecar.read_text())
    return float(m.group(1)) if m else None


def discover_grain_ids():
    """Every grain with a registered CL image, a mask, and a non-empty EPMA
    maps folder — used when GRAIN_IDS is None to run every available grain."""
    candidates = sorted(p.stem[:-len('_CL_registered')]
                         for p in INPUT_DIR.glob('*_CL_registered.tif'))
    found, skipped = [], []
    for g in candidates:
        has_mask = (DATA_DIR / f'{g}_mask.tif').exists()
        maps_dir = MAPS_DIR / g
        has_maps = maps_dir.is_dir() and any(maps_dir.glob('*.tif'))
        (found if has_mask and has_maps else skipped).append(g)
    if skipped:
        print(f'Skipping {len(skipped)} grain(s) missing a mask and/or maps folder: {", ".join(skipped)}')
    return found


def load_gray01(path):
    """Load a TIFF and rescale to [0, 1] by its dtype's natural range —
    integer types by their max representable value (matches MATLAB's
    im2double), float types left as-is (already normalized upstream)."""
    img = tifffile.imread(str(path))
    if np.issubdtype(img.dtype, np.integer):
        return img.astype(np.float64) / np.iinfo(img.dtype).max
    return img.astype(np.float64)


def circular_kernel(radius_px):
    coords = np.arange(-radius_px, radius_px + 1)
    kx, ky = np.meshgrid(coords, coords)
    return (kx**2 + ky**2 <= radius_px**2).astype(np.float64)


def conv_same(img, kernel):
    """Direct (spatial-domain) 2-D convolution, zero-padded — matches
    MATLAB's conv2(..., 'same') exactly, unlike an FFT-based convolution
    whose round-off could leak past the degenerate-window check below."""
    return convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=0)


def process_grain(grain_id):
    log_lines = []

    def log(msg=''):
        print(msg)
        log_lines.append(msg)

    for d in (OUTPUT_DIR, DATA_DIR, DIAGNOSTICS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    cl_filename       = f'{grain_id}_CL_registered.tif'
    cl_color_filename = f'{grain_id}_CL_registered_color.tif'
    mask_filename     = f'{grain_id}_mask.tif'
    epma_dir = MAPS_DIR / grain_id

    # =========================================================================
    # Auto-discover EPMA maps from epma_dir
    # =========================================================================
    output_suffixes = ('_CL_registered.tif', '_CL_registered_color.tif', '_mask.tif')
    tif_files = sorted(p for p in epma_dir.glob('*.tif') if not p.name.endswith(output_suffixes))
    if not tif_files:
        raise FileNotFoundError(f'No EPMA map TIFFs found in: {epma_dir}')

    prefix = f'{grain_id}_'
    epma_labels = []
    for p in tif_files:
        label = p.stem[len(prefix):] if p.stem.startswith(prefix) else p.stem
        label = re.sub(r'_it\d+$', '', label)
        epma_labels.append(label)
    n_elements = len(tif_files)

    # Per-grain pixel size — see module docstring.
    epma_pixel_um, px_source = None, None
    if PIXEL_UM_FROM_SIDECAR:
        for p in tif_files:
            found = read_pixel_um_from_sidecar(p)
            if found is not None:
                epma_pixel_um, px_source = found, p.name
                break
    if epma_pixel_um is None:
        epma_pixel_um = EPMA_PIXEL_UM
        if PIXEL_UM_FROM_SIDECAR:
            print(f'  WARNING: no metadata sidecar found/parseable for any loaded element — '
                  f'falling back to EPMA_PIXEL_UM={epma_pixel_um:.4g} um/px')

    window_radius_px = round(WINDOW_RADIUS_UM / epma_pixel_um)
    min_window_px = int(np.ceil(0.5 * np.pi * window_radius_px ** 2))

    print(f'Auto-discovered {n_elements} EPMA maps in: {epma_dir}')
    print(f'=== CL Local Regression Map: {grain_id} ===\n')
    if px_source is not None:
        print(f'  Pixel size: {epma_pixel_um:.4g} um/px (from {px_source} metadata sidecar)')

    log(f'CL LOCAL REGRESSION MAP ANALYSIS LOG — {grain_id}')
    log()
    log('Parameters:')
    log(f'  Input directory:        {INPUT_DIR}')
    log(f'  EPMA directory:         {epma_dir}')
    log(f'  Data directory:         {DATA_DIR}')
    log(f'  Diagnostics directory:  {DIAGNOSTICS_DIR}')
    log(f'  Output directory:       {OUTPUT_DIR}')
    log(f'  EPMA maps ({n_elements} total):')
    for e, (p, lbl) in enumerate(zip(tif_files, epma_labels)):
        log(f'    [{e}]  {p.name:<30s}  label: {lbl}')
    log(f'  Window radius:           {WINDOW_RADIUS_UM:.4f} um  ({window_radius_px} px)')
    log(f'  Min valid px per window: {min_window_px}')
    log(f'  Normalize EPMA maps:     {NORMALIZE_EPMA}')
    log(f'  Spatial calibration:     {epma_pixel_um:.4f} um/px'
        + (f'  (from {px_source})' if px_source else '  (fallback — no sidecar found)'))
    log()

    # =========================================================================
    # Load registered CL, grain mask, and EPMA maps
    # =========================================================================
    print('Loading registered CL, grain mask, and EPMA maps...')

    cl_path = INPUT_DIR / cl_filename
    if not cl_path.exists():
        raise FileNotFoundError(f'Registered CL image not found: {cl_path}\n'
                                 f'Run CL_EPMA_registration.m for this grain first.')
    cl_reg = load_gray01(cl_path)
    nrows, ncols = cl_reg.shape
    print(f'  Registered CL loaded: {nrows} x {ncols} pixels')

    # Color registered CL — display only (Cr comparison figure); the
    # regression math above always uses the grayscale cl_reg.
    cl_color_path = INPUT_DIR / cl_color_filename
    have_color = cl_color_path.exists()
    if USE_COLOR_DISPLAY and have_color:
        cl_disp = tifffile.imread(str(cl_color_path))
        print('  Using registered color CL for display.')
    else:
        if USE_COLOR_DISPLAY and not have_color:
            print(f'  WARNING: USE_COLOR_DISPLAY=True but {cl_color_filename} not found; '
                  f'falling back to grayscale.')
        cl_disp = cl_reg

    mask_path = DATA_DIR / mask_filename
    if not mask_path.exists():
        raise FileNotFoundError(f'Grain mask not found: {mask_path}\n'
                                 f'Run CL_EPMA_registration.m for this grain first.')
    grain_mask = tifffile.imread(str(mask_path)) > 128
    if grain_mask.shape != (nrows, ncols):
        raise ValueError(f'Grain mask size {grain_mask.shape} does not match '
                          f'registered CL size {(nrows, ncols)}.')
    print(f'  Grain mask loaded: {int(grain_mask.sum())} px in grain.')

    # Warn if the grain touches the image border — outside-image pixels are
    # zero-padded by conv_same, which would otherwise silently bias windows there.
    touches_border = (grain_mask[0, :].any() or grain_mask[-1, :].any()
                       or grain_mask[:, 0].any() or grain_mask[:, -1].any())
    if touches_border:
        print('  WARNING: grain mask touches the image border — windows near the '
              'border may be biased by zero-padding.')
        log()
        log('  ** WARNING: grain mask touches the image border. **')

    epma_raw_abs = []
    for p, lbl in zip(tif_files, epma_labels):
        img = tifffile.imread(str(p)).astype(np.float64)
        epma_raw_abs.append(img)
        print(f'  {lbl} map loaded:  {img.shape[0]} x {img.shape[1]} pixels')

    # Sanity check: all EPMA maps must be the same size; auto-crop to
    # smallest common dimensions if they differ (e.g. colorbar width variation).
    epma_nrows = [img.shape[0] for img in epma_raw_abs]
    epma_ncols = [img.shape[1] for img in epma_raw_abs]
    min_rows, min_cols = min(epma_nrows), min(epma_ncols)
    if len(set(epma_nrows)) > 1 or len(set(epma_ncols)) > 1:
        print(f'  EPMA maps are not all the same size — auto-cropping to {min_rows} x {min_cols} px.')
        log(f'  NOTE: EPMA maps had inconsistent sizes — auto-cropped to {min_rows} x {min_cols}.')
        epma_raw_abs = [img[:min_rows, :min_cols] for img in epma_raw_abs]

    if (min_rows, min_cols) != (nrows, ncols):
        raise ValueError(f'EPMA map grid ({min_rows} x {min_cols}) does not match registered '
                          f'CL size ({nrows} x {ncols}).\nMAPS_DIR/{grain_id} may not be the '
                          f'folder used during registration for this grain.')

    print(f'\nWorking grid: {nrows} rows x {ncols} cols\n')
    log(f'  Working grid: {nrows} rows x {ncols} cols  '
        f'({nrows*epma_pixel_um:.1f} x {ncols*epma_pixel_um:.1f} um at {epma_pixel_um:.4f} um/px)')
    log()

    # Normalization basis for elements (mirrors CL_EPMA_registration.m)
    epma_norm = [None] * n_elements
    if NORMALIZE_EPMA:
        for e, img in enumerate(epma_raw_abs):
            v = img[grain_mask]
            vmin, vmax = v.min(), v.max()
            epma_norm[e] = (img - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(img)

    # =========================================================================
    # Circular kernel and shared window-coverage map
    # =========================================================================
    print('--- BUILDING MOVING-WINDOW KERNEL ---')
    K = circular_kernel(window_radius_px)
    disk_area_px = K.sum()

    m = grain_mask.astype(np.float64)
    n_map = conv_same(m, K)   # # in-mask pixels within the window, per center pixel — shared across elements

    print(f'  Window radius: {WINDOW_RADIUS_UM:.2f} um ({window_radius_px} px)  |  '
          f'disk area: {disk_area_px:.0f} px  |  min valid px: {min_window_px}')
    print(f'  n_map range within grain mask: [{n_map[grain_mask].min():.0f}, {n_map[grain_mask].max():.0f}]')

    log('Moving-window kernel:')
    log(f'  Radius:      {WINDOW_RADIUS_UM:.4f} um  ({window_radius_px} px)')
    log(f'  Disk area:   {disk_area_px:.0f} px')
    log(f'  Min valid px per window: {min_window_px}')
    log(f'  n_map range within grain mask: [{n_map[grain_mask].min():.0f}, {n_map[grain_mask].max():.0f}]')
    log()

    # =========================================================================
    # Per-element moving-window regression
    # =========================================================================
    print('--- COMPUTING PER-PIXEL LOCAL REGRESSIONS ---')

    y = cl_reg.copy()
    y[~grain_mask] = 0
    Sy_map  = conv_same(y, K)
    Syy_map = conv_same(y**2, K)

    valid_window = grain_mask & (n_map >= min_window_px)

    slope_maps = np.full((nrows, ncols, n_elements), np.nan)
    r_maps     = np.full((nrows, ncols, n_elements), np.nan)

    eps64 = np.finfo(np.float64).eps
    summary_rows = []

    for e in range(n_elements):
        x = (epma_norm[e] if NORMALIZE_EPMA else epma_raw_abs[e]).copy()
        x[~grain_mask] = 0

        Sx_map  = conv_same(x, K)
        Sxx_map = conv_same(x**2, K)
        Sxy_map = conv_same(x * y, K)

        denom_slope = n_map * Sxx_map - Sx_map**2
        denom_r_sq  = denom_slope * (n_map * Syy_map - Sy_map**2)

        with np.errstate(divide='ignore', invalid='ignore'):
            slope_e = (n_map * Sxy_map - Sx_map * Sy_map) / denom_slope
            r_e     = (n_map * Sxy_map - Sx_map * Sy_map) / np.sqrt(denom_r_sq)

        # Degenerate windows (near-zero denominator, e.g. constant x within
        # the window) produce Inf/NaN from 0/0 or division by ~0 — treat as invalid.
        degenerate = (denom_slope <= eps64 * disk_area_px**2) | (denom_r_sq <= 0)

        invalid = ~valid_window | degenerate
        slope_e[invalid] = np.nan
        r_e[invalid] = np.nan
        r_e = np.clip(r_e, -1, 1)   # clip tiny numerical overshoot past +-1

        slope_maps[:, :, e] = slope_e
        r_maps[:, :, e] = r_e

        valid_here = ~np.isnan(slope_e) & grain_mask
        n_valid = int(valid_here.sum())
        n_grain = int(grain_mask.sum())
        pct_nan = 100 * (n_grain - n_valid) / n_grain

        lbl = epma_labels[e]
        slope_v, r_v = slope_e[valid_here], r_e[valid_here]
        summary_rows.append(dict(
            Element=lbl, N_valid=n_valid, PctNaN=pct_nan,
            SlopeMean=slope_v.mean(), SlopeStd=slope_v.std(),
            SlopeMin=slope_v.min(), SlopeMax=slope_v.max(),
            RMean=r_v.mean(), RStd=r_v.std(), RMin=r_v.min(), RMax=r_v.max(),
        ))
        print(f'  {lbl:<10s}  valid px: {n_valid:6d} / {n_grain:6d} ({pct_nan:.1f}% NaN)  |  '
              f'slope [{slope_v.min():.4g}, {slope_v.max():.4g}]  |  r [{r_v.min():.3f}, {r_v.max():.3f}]')

    log('Per-element local regression summary (valid pixels only):')
    log(f'  {"Element":<10s} {"N_valid":<10s} {"PctNaN":<8s} {"SlopeMean":<12s} {"SlopeStd":<12s} '
        f'{"SlopeMin":<12s} {"SlopeMax":<12s} {"RMean":<10s} {"RStd":<10s} {"RMin":<10s} {"RMax":<10s}')
    for row in summary_rows:
        log(f'  {row["Element"]:<10s} {row["N_valid"]:<10d} {row["PctNaN"]:<8.1f} '
            f'{row["SlopeMean"]:<12.4g} {row["SlopeStd"]:<12.4g} {row["SlopeMin"]:<12.4g} {row["SlopeMax"]:<12.4g} '
            f'{row["RMean"]:<10.4f} {row["RStd"]:<10.4f} {row["RMin"]:<10.4f} {row["RMax"]:<10.4f}')
    log()

    # =========================================================================
    # Save data outputs
    # =========================================================================
    print('--- SAVING OUTPUTS ---')

    npz_file = DATA_DIR / f'{grain_id}_local_regression.npz'
    np.savez_compressed(
        npz_file, slope_maps=slope_maps, r_maps=r_maps, n_map=n_map,
        epma_labels=np.array(epma_labels), grain_mask=grain_mask,
        window_radius_px=window_radius_px, window_radius_um=WINDOW_RADIUS_UM,
        min_window_px=min_window_px, normalize_epma=NORMALIZE_EPMA,
        epma_pixel_um=epma_pixel_um, grain_id=grain_id,
    )
    print(f'  Maps saved to: {npz_file}')

    # Long-format pixel data CSV: one row per valid (pixel, element).
    rows_idx, cols_idx = np.nonzero(grain_mask)
    n_map_at_px = n_map[rows_idx, cols_idx]
    csv_parts = []
    for e in range(n_elements):
        slope_at_px = slope_maps[rows_idx, cols_idx, e]
        r_at_px = r_maps[rows_idx, cols_idx, e]
        valid_e = ~np.isnan(slope_at_px)
        csv_parts.append(pd.DataFrame({
            'RowIdx':  rows_idx[valid_e],
            'ColIdx':  cols_idx[valid_e],
            'Element': epma_labels[e],
            'N':       n_map_at_px[valid_e],
            'Slope':   slope_at_px[valid_e],
            'R':       r_at_px[valid_e],
        }))
    pixel_df = pd.concat(csv_parts, ignore_index=True)
    csv_file = DATA_DIR / f'{grain_id}_local_regression_pixel_data.csv'
    pixel_df.to_csv(csv_file, index=False)
    print(f'  Pixel data CSV saved to: {csv_file}  ({len(pixel_df)} rows)')

    log('Output data:')
    log(f'  Grain pixels (mask):   {int(grain_mask.sum())}')
    log(f'  Pixel data CSV rows:   {len(pixel_df)}  (valid pixel x element pairs)')
    log(f'  EPMA normalization:    '
        f'{"in-grain-mask min/max" if NORMALIZE_EPMA else "none — raw pixel counts preserved"}')
    log()

    # =========================================================================
    # QC figures
    # =========================================================================
    print('--- SAVING QC FIGURES ---')

    n_cols2 = 3
    n_rows2 = int(np.ceil(n_elements / n_cols2))

    # ---- Slope map grid ------------------------------------------------------
    fig, axes = plt.subplots(n_rows2, n_cols2, figsize=(3.8 * n_cols2, 3.4 * n_rows2), squeeze=False)
    for e in range(n_elements):
        ax = axes.flat[e]
        slope_e = slope_maps[:, :, e]
        valid_here = ~np.isnan(slope_e)
        clim_abs = np.percentile(np.abs(slope_e[valid_here]), 98) if valid_here.any() else 1.0
        if clim_abs <= 0:
            clim_abs = 1.0
        disp_slope = np.where(grain_mask, slope_e, np.nan)
        ax.set_facecolor((0.85, 0.85, 0.85))   # NaN background
        im = ax.imshow(disp_slope, cmap=DIVERGING_CMAP, vmin=-clim_abs, vmax=clim_abs)
        ax.axis('off')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'{epma_labels[e]} (slope)', fontsize=9)
    for ax in axes.flat[n_elements:]:
        ax.axis('off')
    fig.suptitle(f'{grain_id} — local CL-vs-element slope (r = {WINDOW_RADIUS_UM:.1f} um)')
    slope_qc_file = DIAGNOSTICS_DIR / f'{grain_id}_local_regression_slope_QC.png'
    fig.savefig(slope_qc_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Slope QC figure saved to: {slope_qc_file}')

    # ---- R map grid -----------------------------------------------------------
    fig, axes = plt.subplots(n_rows2, n_cols2, figsize=(3.8 * n_cols2, 3.4 * n_rows2), squeeze=False)
    for e in range(n_elements):
        ax = axes.flat[e]
        disp_r = np.where(grain_mask, r_maps[:, :, e], np.nan)
        ax.set_facecolor((0.85, 0.85, 0.85))
        im = ax.imshow(disp_r, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
        ax.axis('off')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'{epma_labels[e]} (R)', fontsize=9)
    for ax in axes.flat[n_elements:]:
        ax.axis('off')
    fig.suptitle(f'{grain_id} — local CL-vs-element Pearson R (r = {WINDOW_RADIUS_UM:.1f} um)')
    r_qc_file = DIAGNOSTICS_DIR / f'{grain_id}_local_regression_R_QC.png'
    fig.savefig(r_qc_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  R QC figure saved to: {r_qc_file}')

    # ---- Chromium-specific figure: local R map next to the CL image ---------
    # Cr3+ is a known CL activator in kyanite, so this pairing is inspected on
    # every run regardless of how many other elements were mapped.
    cr_idx = next((i for i, lbl in enumerate(epma_labels)
                    if re.match(r'^Cr(_|$)', lbl, re.IGNORECASE)), None)
    cr_cl_r_file = None
    if cr_idx is not None:
        disp_r_cr = np.where(grain_mask, r_maps[:, :, cr_idx], np.nan)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.5))
        if cl_disp.ndim == 3:
            disp_cl = (cl_disp.astype(np.float64) / np.iinfo(cl_disp.dtype).max
                       if np.issubdtype(cl_disp.dtype, np.integer) else cl_disp)
            ax1.imshow(disp_cl)
        else:
            ax1.imshow(cl_disp, cmap='gray', vmin=0, vmax=1)
        ax1.axis('off')
        ax1.set_title(f'{grain_id} — registered CL', fontsize=10)

        im2 = ax2.imshow(disp_r_cr, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
        ax2.set_facecolor((0.85, 0.85, 0.85))
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2)
        ax2.set_title(f'{epma_labels[cr_idx]} (local Pearson R, radius = {WINDOW_RADIUS_UM:.1f} um)', fontsize=10)

        fig.suptitle(f'{grain_id} — CL image vs. local CL-{epma_labels[cr_idx]} correlation')
        cr_cl_r_file = OUTPUT_DIR / f'{grain_id}_local_regression_R_Cr_vs_CL.png'
        fig.savefig(cr_cl_r_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Cr R-vs-CL figure saved to: {cr_cl_r_file}')
    else:
        print(f'  WARNING: no element map matching "Cr" (chromium) found among: '
              f'{", ".join(epma_labels)} — skipping Cr-specific figure.')
        log()
        log('  ** WARNING: no Cr element map found — Cr-specific R-vs-CL figure skipped. **')

    # ---- Window-coverage (n) map -----------------------------------------------
    disp_n = np.where(grain_mask, n_map, np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.set_facecolor((0.85, 0.85, 0.85))
    im = ax.imshow(disp_n, cmap=SEQUENTIAL_CMAP)
    ax.axis('off')
    fig.colorbar(im, ax=ax)
    ax.set_title(f'{grain_id} — window coverage (n, radius = {WINDOW_RADIUS_UM:.1f} um) | '
                 f'min valid = {min_window_px}', fontsize=10, pad=10)
    n_map_file = DIAGNOSTICS_DIR / f'{grain_id}_local_regression_n_map.png'
    fig.savefig(n_map_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Window coverage figure saved to: {n_map_file}')

    log_file = DIAGNOSTICS_DIR / f'{grain_id}_local_regression_analysis_log.txt'
    log_file.write_text('\n'.join(log_lines) + '\n')

    print('\n=== COMPLETE ===')
    print(f'All outputs written to: {OUTPUT_DIR} (figures), {DATA_DIR} (data), {DIAGNOSTICS_DIR} (diagnostics)')
    print('Key files:')
    print(f'  diagnostics/{grain_id}_local_regression_analysis_log.txt')
    print(f'  data/{grain_id}_local_regression.npz')
    print(f'  data/{grain_id}_local_regression_pixel_data.csv')
    print(f'  diagnostics/{grain_id}_local_regression_slope_QC.png')
    print(f'  diagnostics/{grain_id}_local_regression_R_QC.png')
    if cr_cl_r_file is not None:
        print(f'  {cr_cl_r_file.name}')
    print(f'  diagnostics/{grain_id}_local_regression_n_map.png')


def main():
    if GRAIN_IDS is None:
        grain_ids = discover_grain_ids()
        if not grain_ids:
            raise FileNotFoundError(
                f'No grains with a registered CL image ({INPUT_DIR}), mask ({DATA_DIR}), '
                f'and maps folder ({MAPS_DIR}) found.')
    else:
        grain_ids = [GRAIN_IDS] if isinstance(GRAIN_IDS, str) else list(GRAIN_IDS)

    print(f'Processing {len(grain_ids)} grain(s):')
    for g in grain_ids:
        print(f'  {g}')

    failed = []
    for grain_id in grain_ids:
        print(f'\n{"=" * 80}')
        try:
            process_grain(grain_id)
        except Exception as exc:
            print(f'  WARNING: {grain_id} failed — {exc}')
            failed.append(grain_id)

    print(f'\n{"=" * 80}')
    print(f'=== BATCH COMPLETE: {len(grain_ids) - len(failed)}/{len(grain_ids)} grain(s) succeeded ===')
    if failed:
        print(f'Failed: {", ".join(failed)}')


if __name__ == '__main__':
    main()
