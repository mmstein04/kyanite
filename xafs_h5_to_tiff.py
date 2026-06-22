"""
xafs_h5_to_tiff.py

Extract element maps from a Larch/GSECARS XRF map HDF5 file and save each
as a 32-bit float TIFF, ready for use in CL_EPMA_registration.m.

Data source: xrmmap/roimap/sum_cor  [rows x cols x n_rois]
             xrmmap/roimap/sum_name [n_rois] — ROI labels, e.g. "Fe Ka"

Output filename convention: <sample>_<El>_<Line>.tif  (e.g. NA-CM-G12B7-02_Fe_Ka.tif)
"""

import h5py
import numpy as np
import tifffile
from pathlib import Path

# =============================================================================
# PARAMETERS — edit this section for each use
# =============================================================================

H5_FILE   = '/Users/mstein/bin/kyanite/NA-GS-P84-06_xafs.h5'
OUTPUT_DIR = '/Users/mstein/bin/kyanite'

# Sample name prefix used in output filenames.
SAMPLE = 'NA-GS-P84-06'

# Elements to export.  Use exact names as stored in the HDF5 file (see the
# "Available ROIs" list printed at startup).  Set to None to export all ROIs.

# ELEMENTS =  None


ELEMENTS = [
    'Fe Ka',
    'Cr Ka',
    'Ti Ka',
    'V Ka',
    'Mn Ka',

]


# =============================================================================

def roi_name_to_filename(sample, roi_name):
    """'Fe Ka' -> 'NA-CM-G12B7-02_Fe_Ka.tif'"""
    return f"{sample}_{roi_name.replace(' ', '_')}.tif"

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5_FILE, 'r') as f:
        names_raw = f['xrmmap/roimap/sum_name'][:]
        names = [n.decode() for n in names_raw]
        sum_cor = f['xrmmap/roimap/sum_cor']  # (rows, cols, n_rois), float32

        print(f"Available ROIs ({len(names)} total):")
        for i, n in enumerate(names):
            print(f"  [{i:2d}] {n}")
        print()

        targets = ELEMENTS if ELEMENTS is not None else names

        for roi in targets:
            if roi not in names:
                print(f"  WARNING: '{roi}' not found in file — skipping.")
                continue

            idx = names.index(roi)
            data = sum_cor[:, :, idx]   # read just this slice; stays float32
            data = np.array(data, dtype=np.float32)
            data = np.flipud(data)      # HDF5 rows run bottom-to-top; flip to match image convention

            out_name = roi_name_to_filename(SAMPLE, roi)
            out_path = out_dir / out_name
            tifffile.imwrite(str(out_path), data, photometric='minisblack')

            print(f"  {roi:12s}  shape={data.shape}  "
                  f"min={data.min():.4f}  max={data.max():.4f}  ->  {out_name}")

    print(f"\nDone. {len(targets)} maps written to {out_dir}/")

if __name__ == '__main__':
    main()
