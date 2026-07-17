"""
xrf_h5_extract_spots.py

Build a per-spot CSV combining, for each XANES spot marked in a Larch/GSECARS
XRF map HDF5 file:
  - pixel and physical stage coordinates
  - mean element concentrations from xrmmap/roimap/sum_cor over a small
    circular zone around the spot, restricted to grain-mask pixels
  - mean CL brightness over that same zone, from the registered CL image
  - the hand-assigned XANES pre-edge classification (Type 1/2/3/Bad data)
  - on_grain: whether the spot's zone overlapped the grain mask at all.
    A spot placed off the grain (e.g. on a neighboring phase) gets on_grain
    = False and NaN CL/element means (there's no kyanite chemistry to
    average there), but keeps its category/category_label untouched — the
    XANES pre-edge classification is a property of whatever phase the spot
    actually sampled, not of kyanite specifically, so it's still meaningful
    data for that other phase. on_grain is NaN (indeterminate) only if no
    grain mask was found at all for this grain.

Data sources:
  xrmmap/areas/<name>        — boolean mask, same shape as the element maps;
                                one (or a few, for drawn regions) True
                                pixel(s) mark the saved spot location
  xrmmap/positions/pos       — (rows, cols, n_addr) physical motor position
                                at every pixel
  xrmmap/positions/name      — axis labels for the last axis
  xrmmap/config/scan         — pos1/pos2 axis identity, range, and per-axis
                                step size (used both for physical coordinates
                                and to build a true-physical-distance
                                circular zone)
  xrmmap/roimap/sum_cor      — (rows, cols, n_rois) element/scaler ROI maps,
                                native h5 orientation (kept as a live h5py
                                Dataset and only ever sliced in small windows
                                — these files are multi-GB)
  xrmmap/roimap/sum_name     — ROI labels, e.g. "Fe Ka"
  xrmmap/scalars/Clock, I0   — per-pixel normalization maps (optional)
  figs/data/<GRAIN_ID>_mask.tif      — binary grain mask (8-bit, 0/255),
                                        written by CL_EPMA_registration.m
  figs/<GRAIN_ID>_CL_registered.tif  — registered CL image (16-bit, scaled
                                        from a [0,1] normalized value),
                                        written by CL_EPMA_registration.m
  inputs/xanes_classification/<GRAIN_ID>_pre_edge_classification.csv
                                — hand classification, written by
                                  xanes_classification_split.py

Registration does not resample the EPMA/XRF grid — the registered CL image,
grain mask, and element maps all live on the exact same pixel grid as the
raw h5 scan (row-flipped to match xrf_h5_to_tiff.py's TIFF export
convention). So element data is read directly from the h5's full ROI list
rather than depending on the (possibly smaller, curated) subset already
exported to maps/<grain_id>/*.tif.

Spot numbering in the HDF5 area names does not necessarily match the sample
prefix used elsewhere in this project (e.g. h5 area "LLF6-Area2-spot01" vs.
grain_id "LLF6-01") — everything here joins on the trailing spot number via
GRAIN_ID + that number, not the h5 area's own name.

Pixel coordinates are reported in two frames:
  - row_px_h5/col_px_h5     — native HDF5 orientation (row 0 = bottom of scan)
  - row_px_tiff/col_px_tiff — row-flipped to match the TIFFs exported by
                              xrf_h5_to_tiff.py (np.flipud), which is also the
                              pixel grid used by CL_EPMA_registration.m and
                              CL_region_extraction.m for the registered CL
                              image, element maps, and mask. The circular
                              zone extraction below operates in this frame.

Output: one row per spot, written to OUTPUT_CSV (if set) and printed to the
console.
"""

import re
import h5py
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path

# =============================================================================
# PARAMETERS — edit this section for each use
# =============================================================================

# Anchored to this script's own location (not a hardcoded machine-specific
# path) so the same file runs unmodified on any machine/cluster this repo is
# checked out on.
_REPO_ROOT = Path(__file__).resolve().parent

H5_FILE  = _REPO_ROOT / 'inputs' / 'xrf' / 'NVD3-01_xrf.h5'
GRAIN_ID = 'NVD3-01'   # drives figs/data/<GRAIN_ID>_mask.tif, figs/<GRAIN_ID>_CL_registered.tif,
                       # inputs/xanes_classification/<GRAIN_ID>_pre_edge_classification.csv,
                       # and spot_id = <GRAIN_ID>_spotNN

FIGS_DIR           = _REPO_ROOT / 'figs'
CLASSIFICATION_DIR = _REPO_ROOT / 'inputs' / 'xanes_classification'


# Reusable data — read back by kyanite_spot_analysis.py and
# xanes_rf_classifier.py, so it lives in figs/data/ alongside the rest of
# the project's reusable per-grain data files, not among any figures.
OUTPUT_CSV = _REPO_ROOT / 'figs' / 'data' / f'{GRAIN_ID}_spot_geochemistry.csv'

# Only include areas whose name matches this regex (case-insensitive).
# Set to None to include every area in xrmmap/areas (drawn regions included).
NAME_FILTER = r'spot'

# --- Point geochemistry / CL zone extraction --------------------------------

# Physical radius (microns) of the circular zone averaged around each spot,
# restricted to grain-mask pixels. Built from the h5's own per-axis pixel
# size (step1/step2), not assumed isotropic.
ZONE_RADIUS_UM = 5.0

# Elements to extract from xrmmap/roimap/sum_cor. None = every ROI in the h5
# except EXCLUDE_ROIS (scaler/normalization channels, not elements).
# Explicit list = use exactly those ROI names, as stored in h5
# (space-separated, e.g. 'Cr Ka'); a name missing from this grain's h5 (ROI
# lists vary slightly between grains) is skipped with a warning.
ELEMENTS = None
EXCLUDE_ROIS = ['Clock', 'I0', 'I1', 'I2', 'I3', 'Clock_raw', 'I0_raw',
                'I1_raw', 'I2_raw', 'I3_raw', 'OutputCounts']

# Normalize element ROI counts by per-pixel clock/I0, matching
# xrf_h5_to_tiff.py's convention (clock first, then I0), so these zone means
# stay comparable to whatever's already been exported to maps/<grain_id>/.
NORMALIZE_BY_CLOCK = False
NORMALIZE_BY_I0    = True

# =============================================================================

SPOT_NUM_RE = re.compile(r'spot0*(\d+)', re.IGNORECASE)


def _scalar(ds):
    v = ds[()]
    return v.decode() if hasattr(v, 'decode') else v


def spot_sort_key(name):
    m = SPOT_NUM_RE.search(name)
    return (int(m.group(1)) if m else float('inf'), name)


def load_grain_mask(figs_dir, grain_id):
    path = Path(figs_dir) / 'data' / f'{grain_id}_mask.tif'
    if not path.exists():
        print(f"WARNING: grain mask not found at {path} — element/CL zone means will be NaN.")
        return None
    return tifffile.imread(str(path)) > 128


def load_cl_image(figs_dir, grain_id):
    path = Path(figs_dir) / f'{grain_id}_CL_registered.tif'
    if not path.exists():
        print(f"WARNING: registered CL image not found at {path} — CL column will be NaN.")
        return None
    return tifffile.imread(str(path)).astype(np.float64) / 65535.0


def load_classification(classification_dir, grain_id):
    path = Path(classification_dir) / f'{grain_id}_pre_edge_classification.csv'
    if not path.exists():
        print(f"WARNING: XANES classification not found at {path} — "
              f"category/category_label will be blank.")
        return {}
    cdf = pd.read_csv(path)
    return {int(row.spot): (row.category, row.category_label) for row in cdf.itertuples()}


def select_element_rois(all_names, elements_param, exclude_rois):
    """Resolve ELEMENTS to a {csv_column_name: h5_roi_index} dict."""
    if elements_param is None:
        names = [n for n in all_names if n not in exclude_rois]
    else:
        names = []
        for n in elements_param:
            if n not in all_names:
                print(f"  WARNING: ROI '{n}' not found in this h5 — skipping.")
                continue
            names.append(n)
    return {n.replace(' ', '_'): all_names.index(n) for n in names}


def window_bounds(center, radius_px, dim_size):
    lo = max(0, center - radius_px)
    hi = min(dim_size - 1, center + radius_px)
    return lo, hi


def h5_row_window_from_tiff(row_lo_tiff, row_hi_tiff, n_rows):
    """A contiguous TIFF-space row range flips to another contiguous h5-native range."""
    return n_rows - 1 - row_hi_tiff, n_rows - 1 - row_lo_tiff


def disk_zone_mask(center_row, center_col, row_lo, col_lo, win_shape,
                    step_row_um, step_col_um, radius_um):
    """Boolean disk over a local window, using true physical distance (handles
    anisotropic pixels — row and column steps need not be equal)."""
    rr, cc = np.mgrid[row_lo:row_lo + win_shape[0], col_lo:col_lo + win_shape[1]]
    dist_um = np.sqrt(((rr - center_row) * step_row_um) ** 2 +
                       ((cc - center_col) * step_col_um) ** 2)
    return dist_um <= radius_um


def extract_zone_geochem(row_px_tiff, col_px_tiff, n_rows, n_cols,
                          step_row_um, step_col_um, radius_um,
                          grain_mask, cl_img, sum_cor_ds, roi_indices,
                          clock_arr, i0_arr):
    """Mean CL and element values over a circular zone (grain-mask pixels
    only) around one spot, plus QC pixel counts."""
    row_radius_px = int(np.ceil(radius_um / step_row_um))
    col_radius_px = int(np.ceil(radius_um / step_col_um))
    row_lo, row_hi = window_bounds(row_px_tiff, row_radius_px, n_rows)
    col_lo, col_hi = window_bounds(col_px_tiff, col_radius_px, n_cols)
    win_shape = (row_hi - row_lo + 1, col_hi - col_lo + 1)

    disk = disk_zone_mask(row_px_tiff, col_px_tiff, row_lo, col_lo,
                           win_shape, step_row_um, step_col_um, radius_um)

    result = {
        'zone_radius_um': radius_um,
        'zone_pixel_count': int(disk.sum()),
        'zone_mask_px_count': 0,
        'on_grain': np.nan,   # indeterminate until a grain mask is available (see below)
        'CL': np.nan,
    }
    for col_name in roi_indices:
        result[col_name] = np.nan

    if grain_mask is None:
        return result

    zone_mask = disk & grain_mask[row_lo:row_hi + 1, col_lo:col_hi + 1]
    n_zone = int(zone_mask.sum())
    result['zone_mask_px_count'] = n_zone
    result['on_grain'] = n_zone > 0
    if n_zone == 0:
        print(f"  WARNING: spot at (row={row_px_tiff}, col={col_px_tiff}) has zero "
              f"overlap with the grain mask — CL/element means will be NaN, "
              f"on_grain=False (spot likely samples a different phase).")
        return result

    if cl_img is not None:
        result['CL'] = float(np.nanmean(cl_img[row_lo:row_hi + 1, col_lo:col_hi + 1][zone_mask]))

    if roi_indices:
        row_lo_h5, row_hi_h5 = h5_row_window_from_tiff(row_lo, row_hi, n_rows)
        window = np.flipud(sum_cor_ds[row_lo_h5:row_hi_h5 + 1, col_lo:col_hi + 1, :])

        clock_w = clock_arr[row_lo:row_hi + 1, col_lo:col_hi + 1] if clock_arr is not None else None
        i0_w = i0_arr[row_lo:row_hi + 1, col_lo:col_hi + 1] if i0_arr is not None else None

        for col_name, idx in roi_indices.items():
            data = window[:, :, idx].astype(np.float64)
            if clock_w is not None:
                data = data / clock_w
            if i0_w is not None:
                data = data / i0_w
            result[col_name] = float(np.nanmean(data[zone_mask]))

    return result


def main():
    if GRAIN_ID not in Path(H5_FILE).stem:
        print(f"WARNING: GRAIN_ID '{GRAIN_ID}' is not a substring of H5_FILE "
              f"'{Path(H5_FILE).name}' — double check these refer to the same grain.")

    with h5py.File(H5_FILE, 'r') as f:
        scan = f['xrmmap/config/scan']
        pos1_addr = _scalar(scan['pos1'])
        pos2_addr = _scalar(scan['pos2'])
        start1, stop1 = _scalar(scan['start1']), _scalar(scan['stop1'])
        start2, stop2 = _scalar(scan['start2']), _scalar(scan['stop2'])
        step1_mm = _scalar(scan['step1'])
        step2_mm = _scalar(scan['step2'])
        step_col_um = step1_mm * 1000   # pos1 / fast axis / X / column direction
        step_row_um = step2_mm * 1000   # pos2 / slow axis / Y / row direction

        pos_names = [n.decode() if hasattr(n, 'decode') else n
                     for n in f['xrmmap/positions/name'][:]]
        pos_addrs = [a.decode() if hasattr(a, 'decode') else a
                     for a in f['xrmmap/positions/address'][:]]
        pos = f['xrmmap/positions/pos']  # (rows, cols, n_addr)

        idx1 = pos_addrs.index(pos1_addr)
        idx2 = pos_addrs.index(pos2_addr)
        print(f"Scan axes: pos1 = {pos1_addr} ({pos_names[idx1]}, {start1} to {stop1} mm, "
              f"step {step_col_um:.3g} um)  |  pos2 = {pos2_addr} ({pos_names[idx2]}, "
              f"{start2} to {stop2} mm, step {step_row_um:.3g} um)")

        roi_names_raw = [n.decode() if hasattr(n, 'decode') else n
                         for n in f['xrmmap/roimap/sum_name'][:]]
        roi_indices = select_element_rois(roi_names_raw, ELEMENTS, EXCLUDE_ROIS)
        print(f"Extracting {len(roi_indices)} element ROI(s): {list(roi_indices.keys())}")
        sum_cor_ds = f['xrmmap/roimap/sum_cor']  # live h5py Dataset — never loaded fully
        n_rows, n_cols = sum_cor_ds.shape[:2]

        clock_arr = None
        if NORMALIZE_BY_CLOCK:
            clock_arr = np.flipud(np.array(f['xrmmap/scalars/Clock'], dtype=np.float64))
            clock_arr[clock_arr == 0] = np.nan

        i0_arr = None
        if NORMALIZE_BY_I0:
            i0_arr = np.flipud(np.array(f['xrmmap/scalars/I0'], dtype=np.float64))
            i0_arr[i0_arr == 0] = np.nan

        grain_mask = load_grain_mask(FIGS_DIR, GRAIN_ID)
        cl_img = load_cl_image(FIGS_DIR, GRAIN_ID)
        classification = load_classification(CLASSIFICATION_DIR, GRAIN_ID)

        areas = f['xrmmap/areas']
        names = list(areas.keys())
        if NAME_FILTER is not None:
            pattern = re.compile(NAME_FILTER, re.IGNORECASE)
            names = [n for n in names if pattern.search(n)]
        names.sort(key=spot_sort_key)

        if not names:
            print("No areas matched NAME_FILTER — nothing to extract.")
            return

        rows = []
        unmatched_classification = []
        for name in names:
            mask = areas[name][()]
            n_rows_map = mask.shape[0]
            yx = np.argwhere(mask)
            if len(yx) == 0:
                print(f"  WARNING: '{name}' has no True pixels — skipping.")
                continue

            # Native HDF5 pixel index (row 0 = bottom of scan, per xrmmap/positions/pos).
            row_px, col_px = yx.mean(axis=0)
            row_px, col_px = int(round(row_px)), int(round(col_px))
            x_mm = float(pos[row_px, col_px, idx1])
            y_mm = float(pos[row_px, col_px, idx2])

            # xrf_h5_to_tiff.py flips rows vertically on export (np.flipud) so the
            # TIFF reads top-to-bottom, and CL_EPMA_registration.m / kyanite
            # region extraction operate on that flipped grid.
            row_px_tiff = n_rows_map - 1 - row_px

            m = SPOT_NUM_RE.search(name)
            spot_num = int(m.group(1)) if m else None
            spot_id = f"{GRAIN_ID}_spot{spot_num:02d}" if spot_num is not None else None

            category, category_label = (np.nan, np.nan)
            if spot_num is not None and spot_num in classification:
                category, category_label = classification[spot_num]
            elif spot_num is not None:
                unmatched_classification.append(spot_num)

            row = {
                'grain_id': GRAIN_ID,
                'spot': spot_num,
                'spot_id': spot_id,
                'area_name': name,
                'category': category,
                'category_label': category_label,
                'pixel_count': len(yx),
                'row_px_h5': row_px,          # native HDF5 orientation (row 0 = bottom of scan)
                'col_px_h5': col_px,
                'row_px_tiff': row_px_tiff,   # 0-based row/col into exported TIFFs / registered CL / element maps
                'col_px_tiff': col_px,
                'row_matlab': row_px_tiff + 1,  # 1-based, for direct MATLAB indexing
                'col_matlab': col_px + 1,
                'x_mm': x_mm,
                'y_mm': y_mm,
                'x_rel_um': (x_mm - start1) * 1000,
                'y_rel_um': (y_mm - start2) * 1000,
            }

            zone = extract_zone_geochem(
                row_px_tiff, col_px, n_rows, n_cols,
                step_row_um, step_col_um, ZONE_RADIUS_UM,
                grain_mask, cl_img, sum_cor_ds, roi_indices,
                clock_arr, i0_arr,
            )
            row.update(zone)
            rows.append(row)

        if classification and unmatched_classification:
            print(f"\nNote: {len(unmatched_classification)} spot(s) had no matching row in "
                  f"the classification CSV: {sorted(set(unmatched_classification))}")

        df = pd.DataFrame(rows)
        element_cols = list(roi_indices.keys())
        col_order = (
            ['grain_id', 'spot', 'spot_id', 'area_name', 'category', 'category_label',
             'pixel_count',
             'row_px_h5', 'col_px_h5', 'row_px_tiff', 'col_px_tiff', 'row_matlab', 'col_matlab',
             'x_mm', 'y_mm', 'x_rel_um', 'y_rel_um',
             'zone_radius_um', 'zone_pixel_count', 'zone_mask_px_count', 'on_grain',
             'CL']
            + element_cols
        )
        df = df[col_order]

        with pd.option_context('display.max_rows', None, 'display.width', 200):
            print(df.to_string(index=False))

        if OUTPUT_CSV is not None:
            Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\nWrote {len(df)} spot(s) to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
