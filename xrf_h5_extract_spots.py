"""
xrf_h5_extract_spots.py

Extract the pixel and physical stage coordinates of named point "areas"
(XANES spot locations marked in GSE Mapviewer during the XRF map scan) from
a Larch/GSECARS XRF map HDF5 file.

Data source: xrmmap/areas/<name>        — boolean mask, same shape as the
                                           element maps; one (or a few,
                                           for drawn regions) True pixel(s)
                                           mark the saved location
             xrmmap/positions/pos       — (rows, cols, n_addr) physical
                                           motor position at every pixel
             xrmmap/positions/name      — axis labels for the last axis
             xrmmap/config/scan         — pos1/pos2 axis identity + range

Spot numbering in the HDF5 area names does not necessarily match the sample
prefix used in exported XANES spot CSVs (e.g. h5 area "LLF6-Area2-spot01"
vs. CSV "LLF6-01_spot01.csv") — join on the trailing spot number, not the
full name.

Pixel coordinates are reported in two frames:
  - row_px_h5/col_px_h5     — native HDF5 orientation (row 0 = bottom of scan)
  - row_px_tiff/col_px_tiff — row-flipped to match the TIFFs exported by
                              xrf_h5_to_tiff.py (np.flipud), which is also the
                              pixel grid used by CL_EPMA_registration.m and
                              CL_region_extraction.m for the registered CL
                              image, element maps, and mask. Use
                              row_px_tiff/col_px_tiff (or the 1-based
                              row_matlab/col_matlab) to pull CL/element values
                              at these spots out of that pipeline's outputs.

Output: one row per area, written to OUTPUT_CSV (if set) and printed to
the console.
"""

import re
import h5py
import numpy as np
import pandas as pd

# =============================================================================
# PARAMETERS — edit this section for each use
# =============================================================================

H5_FILE    = '/Users/mstein/bin/kyanite/NVD3-01_xrf.h5'
OUTPUT_CSV = None   # e.g. '/Users/mstein/bin/kyanite/NVD3-01_spot_coordinates.csv'; None to skip writing

# Only include areas whose name matches this regex (case-insensitive).
# Set to None to include every area in xrmmap/areas (drawn regions included).
NAME_FILTER = r'spot'

# =============================================================================

SPOT_NUM_RE = re.compile(r'spot0*(\d+)', re.IGNORECASE)


def _scalar(ds):
    v = ds[()]
    return v.decode() if hasattr(v, 'decode') else v


def spot_sort_key(name):
    m = SPOT_NUM_RE.search(name)
    return (int(m.group(1)) if m else float('inf'), name)


def main():
    with h5py.File(H5_FILE, 'r') as f:
        scan = f['xrmmap/config/scan']
        pos1_addr = _scalar(scan['pos1'])
        pos2_addr = _scalar(scan['pos2'])
        start1, stop1 = _scalar(scan['start1']), _scalar(scan['stop1'])
        start2, stop2 = _scalar(scan['start2']), _scalar(scan['stop2'])

        pos_names = [n.decode() if hasattr(n, 'decode') else n
                     for n in f['xrmmap/positions/name'][:]]
        pos_addrs = [a.decode() if hasattr(a, 'decode') else a
                     for a in f['xrmmap/positions/address'][:]]
        pos = f['xrmmap/positions/pos']  # (rows, cols, n_addr)

        idx1 = pos_addrs.index(pos1_addr)
        idx2 = pos_addrs.index(pos2_addr)
        print(f"Scan axes: pos1 = {pos1_addr} ({pos_names[idx1]}, {start1} to {stop1} mm)"
              f"  |  pos2 = {pos2_addr} ({pos_names[idx2]}, {start2} to {stop2} mm)")

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
            # region extraction operate on that flipped grid. Convert here so
            # these indices index directly into the exported/registered TIFFs.
            row_px_tiff = n_rows_map - 1 - row_px

            rows.append({
                'area_name': name,
                'pixel_count': len(yx),
                'row_px_h5': row_px,          # native HDF5 orientation (row 0 = bottom of scan)
                'col_px_h5': col_px,
                'row_px_tiff': row_px_tiff,   # 0-based row/col into exported TIFFs / registered CL / element maps
                'col_px_tiff': col_px,
                'row_matlab': row_px_tiff + 1,  # 1-based, for direct MATLAB indexing, e.g. cl_img(row_matlab, col_matlab)
                'col_matlab': col_px + 1,
                'x_mm': x_mm,
                'y_mm': y_mm,
                'x_rel_um': (x_mm - start1) * 1000,
                'y_rel_um': (y_mm - start2) * 1000,
            })

        df = pd.DataFrame(rows)
        with pd.option_context('display.max_rows', None, 'display.width', 120):
            print(df.to_string(index=False))

        if OUTPUT_CSV is not None:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\nWrote {len(df)} spot(s) to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
