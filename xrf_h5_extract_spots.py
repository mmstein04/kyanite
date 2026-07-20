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
the grain's own grain_id + that number, not the h5 area's own name.

Batch mode: GRAIN_IDS may be a single string, a list, or None to
auto-discover every grain with a raw XRF h5 file in H5_DIR (the only hard
requirement for extraction — a missing mask/CL image/classification CSV for
a given grain degrades gracefully, NaN + a warning, same as always). A grain
that fails partway (e.g. an unreadable/malformed h5) is skipped with a
warning rather than aborting the whole batch.

Pixel coordinates are reported in two frames:
  - row_px_h5/col_px_h5     — native HDF5 orientation (row 0 = bottom of scan)
  - row_px_tiff/col_px_tiff — row-flipped to match the TIFFs exported by
                              xrf_h5_to_tiff.py (np.flipud), which is also the
                              pixel grid used by CL_EPMA_registration.m and
                              CL_region_extraction.m for the registered CL
                              image, element maps, and mask. The circular
                              zone extraction below operates in this frame.

Output: one row per spot, written to OUTPUT_DIR/<grain_id>_spot_geochemistry.csv
(if SAVE_CSV) and printed to the console.
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

# Folder of raw XRF HDF5 files, one per grain: H5_DIR/<grain_id>_xrf.h5.
H5_DIR = _REPO_ROOT / 'inputs' / 'xrf'

# Grain selection. A single string, a list for batch processing, or None to
# auto-discover every grain with an h5 file in H5_DIR — see "Batch mode" above.
# Drives figs/data/<grain_id>_mask.tif, figs/<grain_id>_CL_registered.tif,
# inputs/xanes_classification/<grain_id>_pre_edge_classification.csv, and
# spot_id = <grain_id>_spotNN.
GRAIN_IDS = None

FIGS_DIR           = _REPO_ROOT / 'figs'
CLASSIFICATION_DIR = _REPO_ROOT / 'inputs' / 'xanes_classification'

# Reusable data — read back by kyanite_spot_analysis.py and
# xanes_rf_classifier.py, so it lives in figs/data/ alongside the rest of
# the project's reusable per-grain data files, not among any figures.
OUTPUT_DIR = _REPO_ROOT / 'figs' / 'data'
SAVE_CSV   = True   # False to print to console only, without writing any CSV

# Only include areas whose name matches this regex (case-insensitive).
# Set to None to include every area in xrmmap/areas (drawn regions included).
# Some grains carry more than one family of point under xrmmap/areas — e.g.
# generic '<prefix>_spotNN' points (used for Cr/V XANES) alongside dedicated
# '<prefix>-FeN' points (Fe-only XANES). NAME_FILTER selects which family to
# extract; e.g. r'-Fe\d+$' to extract only the Fe-dedicated points from a
# grain that has both. Spot numbers are always taken from the trailing digits
# of the area name (see SPOT_NUM_RE below), regardless of which family/tag
# word precedes them, so 'Fe7' and 'spot07' both resolve to spot number 7 and
# spot_id '<grain_id>_spot07' — fine as long as you aren't extracting both
# families for the same grain in one run (they'd collide on spot number).
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

# --- Line-scan expansion -----------------------------------------------------

# Some grains mark a XANES line scan in xrmmap/areas as just two single-pixel
# areas, '<prefix>_linestart'/'<prefix>_linestop' — the individual points in
# between (however many were actually measured) have no area entry of their
# own. EXPAND_LINE_SCANS = True looks up those intermediate points' pixel
# locations from their raw per-point files (named 'Fe_XANES_<prefix>_FeLine_
# <n>.<repeat>' in XANES_RAW_DIR) instead, and extracts each as its own spot,
# independent of NAME_FILTER (linestart/linestop never carry per-pixel data
# themselves, so they're excluded from the regular NAME_FILTER path either
# way). Each line's points are numbered continuing on from this grain's
# highest already-extracted spot number (e.g. Fe1..Fe7 -> spot1..spot7, then
# a 15-point line -> spot8..spot22) — there's no reserved numbering scheme yet
# for a grain that also needs Cr/V spots extracted in the same run; don't mix
# them until one exists.
EXPAND_LINE_SCANS = True
XANES_RAW_DIR = _REPO_ROOT / 'inputs' / 'xanes_raw'

# =============================================================================

# Trailing digits of the area name, regardless of tag word ('spot07', 'Fe7',
# etc. all resolve by position, not by requiring a literal 'spot' prefix) —
# matches this project's join convention of "trailing spot number + grain_id",
# not the area's own name/tag.
SPOT_NUM_RE = re.compile(r'(\d+)\s*$')

LINE_ENDPOINT_RE = re.compile(r'^(.+)_line(start|stop)$', re.IGNORECASE)
LINE_POINT_FILE_RE = re.compile(r'FeLine_(\d+)\.(\d+)$', re.IGNORECASE)
STAGE_FINEX_RE = re.compile(r'SampleStage\.FineX:\s*([-\d.]+)')
STAGE_FINEY_RE = re.compile(r'SampleStage\.FineY:\s*([-\d.]+)')


def _scalar(ds):
    v = ds[()]
    return v.decode() if hasattr(v, 'decode') else v


def spot_sort_key(name):
    m = SPOT_NUM_RE.search(name)
    return (int(m.group(1)) if m else float('inf'), name)


def discover_grain_ids():
    """Every grain with a raw XRF h5 file in H5_DIR — the only hard
    requirement for extraction; a missing mask/CL image/classification CSV
    for a given grain degrades gracefully (NaN + warning) rather than
    blocking it. Used when GRAIN_IDS is None to run every available grain."""
    return sorted(p.name[:-len('_xrf.h5')] for p in Path(H5_DIR).glob('*_xrf.h5'))


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


def find_line_prefixes(area_names):
    """Pair up '<prefix>_linestart'/'<prefix>_linestop' area names — these two
    markers never carry per-pixel geochemistry of their own, only the
    endpoints of a line to expand via discover_line_point_files(), so this
    looks at every area name regardless of NAME_FILTER."""
    starts, stops = {}, {}
    for n in area_names:
        m = LINE_ENDPOINT_RE.match(n)
        if not m:
            continue
        prefix, which = m.group(1), m.group(2).lower()
        (starts if which == 'start' else stops)[prefix] = n

    for p in sorted(set(starts) - set(stops)):
        print(f"  WARNING: '{starts[p]}' has no matching linestop — skipping this line.")
    for p in sorted(set(stops) - set(starts)):
        print(f"  WARNING: '{stops[p]}' has no matching linestart — skipping this line.")
    return sorted(set(starts) & set(stops))


def discover_line_point_files(xanes_raw_dir, prefix):
    """{line_point_number: path} for one line, picking the lowest repeat
    suffix ('.001' before '.002', etc.) per point when it was rescanned."""
    candidates = {}
    for p in Path(xanes_raw_dir).glob(f'Fe_XANES_{prefix}_FeLine_*.*'):
        m = LINE_POINT_FILE_RE.search(p.name)
        if not m:
            continue
        num, rep = int(m.group(1)), m.group(2)
        candidates.setdefault(num, []).append((rep, p))

    points = {}
    for num, files in sorted(candidates.items()):
        files.sort(key=lambda rf: rf[0])
        if len(files) > 1:
            reps = ', '.join(r for r, _ in files)
            print(f"  NOTE: line point {num} ('{prefix}') has {len(files)} repeat scan(s) "
                  f"({reps}) — using .{files[0][0]}.")
        points[num] = files[0][1]
    return points


def read_stage_position_mm(path):
    """Parse SampleStage.FineX/FineY (mm) from a raw XANES point file's header."""
    x = y = None
    with open(path, 'r', errors='ignore') as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            if x is None:
                m = STAGE_FINEX_RE.search(line)
                if m:
                    x = float(m.group(1))
            if y is None:
                m = STAGE_FINEY_RE.search(line)
                if m:
                    y = float(m.group(1))
    if x is None or y is None:
        raise ValueError(f"SampleStage.FineX/FineY not found in header of {path}")
    return x, y


def mm_to_pixel_h5(x_mm, y_mm, start1, step1_mm, start2, step2_mm, n_rows, n_cols):
    """Invert the scan's linear stage-position <-> pixel mapping (native HDF5
    orientation, row 0 = bottom). Validated against every area-based spot's
    own true pixel index (exact match, 0 px difference) for this project's
    raster scans, which are perfectly uniform/linear in stage position."""
    col_px = int(round((x_mm - start1) / step1_mm))
    row_px = int(round((y_mm - start2) / step2_mm))
    col_px = min(max(col_px, 0), n_cols - 1)
    row_px = min(max(row_px, 0), n_rows - 1)
    return row_px, col_px


def process_grain(grain_id):
    h5_file = Path(H5_DIR) / f'{grain_id}_xrf.h5'

    with h5py.File(h5_file, 'r') as f:
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

        grain_mask = load_grain_mask(FIGS_DIR, grain_id)
        cl_img = load_cl_image(FIGS_DIR, grain_id)
        classification = load_classification(CLASSIFICATION_DIR, grain_id)

        areas = f['xrmmap/areas']
        all_area_names = list(areas.keys())
        line_prefixes = find_line_prefixes(all_area_names) if EXPAND_LINE_SCANS else []

        names = all_area_names
        if NAME_FILTER is not None:
            pattern = re.compile(NAME_FILTER, re.IGNORECASE)
            names = [n for n in names if pattern.search(n)]
        # linestart/linestop never carry per-pixel data themselves — they're
        # only ever handled via the line-expansion path below, regardless of
        # whether NAME_FILTER happens to also match them.
        names = [n for n in names if not LINE_ENDPOINT_RE.match(n)]
        names.sort(key=spot_sort_key)

        if not names and not line_prefixes:
            print("No areas matched NAME_FILTER and no line scans found — nothing to extract.")
            return None

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
            spot_id = f"{grain_id}_spot{spot_num:02d}" if spot_num is not None else None

            category, category_label = (np.nan, np.nan)
            if spot_num is not None and spot_num in classification:
                category, category_label = classification[spot_num]
            elif spot_num is not None:
                unmatched_classification.append(spot_num)

            row = {
                'grain_id': grain_id,
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

        for prefix in line_prefixes:
            used_nums = [r['spot'] for r in rows if r['spot'] is not None]
            next_num = (max(used_nums) if used_nums else 0) + 1

            line_files = discover_line_point_files(XANES_RAW_DIR, prefix)
            if not line_files:
                print(f"  WARNING: line '{prefix}' found in xrmmap/areas but no "
                      f"'Fe_XANES_{prefix}_FeLine_*' files found in {XANES_RAW_DIR} — skipping.")
                continue

            point_nums = sorted(line_files)
            expected = list(range(point_nums[0], point_nums[-1] + 1))
            if point_nums != expected:
                print(f"  WARNING: line '{prefix}' point numbers aren't contiguous "
                      f"({point_nums}) — extracting whatever was found, in order.")

            for line_num in point_nums:
                path = line_files[line_num]
                x_mm, y_mm = read_stage_position_mm(path)
                row_px, col_px = mm_to_pixel_h5(x_mm, y_mm, start1, step1_mm,
                                                 start2, step2_mm, n_rows, n_cols)
                row_px_tiff = n_rows - 1 - row_px

                spot_num = next_num
                next_num += 1
                spot_id = f"{grain_id}_spot{spot_num:02d}"

                category, category_label = (np.nan, np.nan)
                if spot_num in classification:
                    category, category_label = classification[spot_num]
                else:
                    unmatched_classification.append(spot_num)

                row = {
                    'grain_id': grain_id,
                    'spot': spot_num,
                    'spot_id': spot_id,
                    'area_name': f'{prefix}_FeLine_{line_num}',
                    'category': category,
                    'category_label': category_label,
                    'pixel_count': 1,
                    'row_px_h5': row_px,
                    'col_px_h5': col_px,
                    'row_px_tiff': row_px_tiff,
                    'col_px_tiff': col_px,
                    'row_matlab': row_px_tiff + 1,
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

            first_num = next_num - len(point_nums)
            print(f"  Expanded line '{prefix}' into {len(point_nums)} spot(s): "
                  f"spot{first_num:02d}..spot{next_num - 1:02d}")

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

        if SAVE_CSV:
            output_csv = Path(OUTPUT_DIR) / f'{grain_id}_spot_geochemistry.csv'
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"\nWrote {len(df)} spot(s) to {output_csv}")

        return df


def main():
    if GRAIN_IDS is None:
        grain_ids = discover_grain_ids()
        if not grain_ids:
            raise FileNotFoundError(f'No *_xrf.h5 files found in {H5_DIR}.')
    else:
        grain_ids = [GRAIN_IDS] if isinstance(GRAIN_IDS, str) else list(GRAIN_IDS)

    print(f'Processing {len(grain_ids)} grain(s):')
    for g in grain_ids:
        print(f'  {g}')

    failed = []
    for grain_id in grain_ids:
        print(f'\n{"=" * 80}')
        print(f'=== {grain_id} ===')
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
