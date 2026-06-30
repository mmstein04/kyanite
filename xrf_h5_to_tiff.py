"""
xrf_h5_to_tiff.py

Extract element maps from a Larch/GSECARS XRF map HDF5 file and save each
as a 32-bit float TIFF, ready for use in CL_EPMA_registration.m.

Data source: xrmmap/roimap/sum_cor  [rows x cols x n_rois]
             xrmmap/roimap/sum_name [n_rois] — ROI labels, e.g. "Fe Ka"

Output filename convention: <sample>_<El>_<Line>.tif  (e.g. NA-CM-G12B7-02_Fe_Ka.tif)
"""

import datetime
import h5py
import numpy as np
import tifffile
from pathlib import Path

# =============================================================================
# PARAMETERS — edit this section for each use
# =============================================================================

H5_FILE   = '/Users/mstein/bin/kyanite/LLF6-01_xrf.h5'
OUTPUT_DIR = '/Users/mstein/bin/kyanite/maps'

# Sample name prefix used in output filenames.
SAMPLE = 'LLF6-01'
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

# Normalize fluorescence counts by the per-pixel clock (dwell time).
# Produces counts per clock tick, which removes dwell-time variation and
# makes maps from different scans directly comparable across grains.
# Clock map is read from xrmmap/scalars/Clock.
NORMALIZE_BY_CLOCK = False

# Normalize fluorescence counts by the per-pixel I0 (incident beam intensity).
# Removes flux variation across the map. Applied after clock normalization if
# both are enabled. I0 map is read from xrmmap/scalars/I0.
NORMALIZE_BY_I0 = True

# =============================================================================

def roi_name_to_filename(sample, roi_name):
    """'Fe Ka' -> 'NA-CM-G12B7-02_Fe_Ka.tif'"""
    return f"{sample}_{roi_name.replace(' ', '_')}.tif"


def write_metadata(meta_path, *, h5_file, sample, roi, roi_index, data,
                   step1_um, step2_um, start1, stop1, start2, stop2,
                   dwell_s, normalize_by_clock, normalize_by_i0,
                   n_rois_in_file):
    """Write a sidecar metadata text file for one element-map TIFF."""
    valid = data[np.isfinite(data)]
    rows, cols = data.shape
    nan_count = np.sum(~np.isfinite(data))

    norm_parts = []
    if normalize_by_clock:
        norm_parts.append("clock (dwell time)")
    if normalize_by_i0:
        norm_parts.append("I0 (incident beam intensity)")
    normalization = " and ".join(norm_parts) if norm_parts else "none (raw fluorescence counts)"

    lines = [
        "# XRF element-map metadata",
        f"generated_by       : xrf_h5_to_tiff.py",
        f"generated_utc      : {datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "# Source",
        f"h5_file            : {Path(h5_file).resolve()}",
        f"h5_dataset         : xrmmap/roimap/sum_cor[:, :, {roi_index}]",
        f"roi_name           : {roi}",
        f"roi_index          : {roi_index}  (0-based, out of {n_rois_in_file} ROIs in file)",
        f"sample             : {sample}",
        "",
        "# Image properties",
        f"bit_depth          : 32-bit float (float32)",
        f"dimensions_px      : {rows} rows x {cols} cols  (height x width)",
        f"pixel_count_total  : {rows * cols}",
        f"pixel_count_valid  : {len(valid)}  (finite, non-NaN)",
        f"pixel_count_nan    : {nan_count}  (set to NaN during normalization where divisor = 0)",
        "",
        "# Value statistics (valid pixels only)",
        f"value_min          : {valid.min():.6g}" if len(valid) else "value_min          : N/A",
        f"value_max          : {valid.max():.6g}" if len(valid) else "value_max          : N/A",
        f"value_mean         : {valid.mean():.6g}" if len(valid) else "value_mean         : N/A",
        f"value_median       : {np.median(valid):.6g}" if len(valid) else "value_median       : N/A",
        f"value_std          : {valid.std():.6g}" if len(valid) else "value_std          : N/A",
        "",
        "# Normalization",
        f"normalized_by      : {normalization}",
        f"  (clock norm divides raw counts by per-pixel clock ticks to remove dwell-time variation)",
        f"  (I0 norm divides by per-pixel incident beam intensity to remove flux variation)",
        f"  (normalizations are applied in order: clock first, then I0)",
        f"row_flip_applied   : yes  (rows flipped vertically; HDF5 stores bottom-to-top)",
        "",
        "# Scan geometry",
        f"step_size_pos1_um  : {step1_um:.2f} µm  (fast axis / X)",
        f"step_size_pos2_um  : {step2_um:.2f} µm  (slow axis / Y)",
        f"range_pos1_mm      : {start1:.4f} to {stop1:.4f} mm  ({abs(stop1-start1)*1000:.1f} µm)",
        f"range_pos2_mm      : {start2:.4f} to {stop2:.4f} mm  ({abs(stop2-start2)*1000:.1f} µm)",
        f"dwell_time_s       : {dwell_s:.4g} s/pixel",
    ]

    meta_path.write_text("\n".join(lines) + "\n")

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5_FILE, 'r') as f:
        # --- Scan metadata ---------------------------------------------------
        scan = f['xrmmap/config/scan']
        def _scalar(ds):
            v = ds[()]
            return v.decode() if hasattr(v, 'decode') else float(v)

        step1_mm  = _scalar(scan['step1'])
        step2_mm  = _scalar(scan['step2'])
        start1    = _scalar(scan['start1'])
        stop1     = _scalar(scan['stop1'])
        start2    = _scalar(scan['start2'])
        stop2     = _scalar(scan['stop2'])
        dwell_s   = _scalar(scan['time1'])

        print(f"Scan metadata:")
        print(f"  Step size (pos1 / X): {step1_mm*1000:.1f} µm  ({step1_mm} mm)")
        print(f"  Step size (pos2 / Y): {step2_mm*1000:.1f} µm  ({step2_mm} mm)")
        print(f"  Range pos1 (X):  {start1} to {stop1} mm  ({(stop1-start1)*1000:.0f} µm)")
        print(f"  Range pos2 (Y):  {start2} to {stop2} mm  ({(stop2-start2)*1000:.0f} µm)")
        print(f"  Dwell time:      {dwell_s} s/pixel")
        print()

        names_raw = f['xrmmap/roimap/sum_name'][:]
        names = [n.decode() for n in names_raw]
        sum_cor = f['xrmmap/roimap/sum_cor']  # (rows, cols, n_rois), float32

        print(f"Available ROIs ({len(names)} total):")
        for i, n in enumerate(names):
            print(f"  [{i:2d}] {n}")
        print()

        if NORMALIZE_BY_CLOCK:
            clock = np.array(f['xrmmap/scalars/Clock'], dtype=np.float32)
            clock = np.flipud(clock)
            clock[clock == 0] = np.nan   # avoid divide-by-zero; zero-dwell pixels become NaN
            print(f"Clock map loaded: min={np.nanmin(clock):.2f}  max={np.nanmax(clock):.2f}")

        if NORMALIZE_BY_I0:
            i0 = np.array(f['xrmmap/scalars/I0'], dtype=np.float32)
            i0 = np.flipud(i0)
            i0[i0 == 0] = np.nan   # avoid divide-by-zero
            print(f"I0 map loaded:    min={np.nanmin(i0):.2f}  max={np.nanmax(i0):.2f}")

        targets = ELEMENTS if ELEMENTS is not None else names

        for roi in targets:
            if roi not in names:
                print(f"  WARNING: '{roi}' not found in file — skipping.")
                continue

            idx = names.index(roi)
            data = np.array(sum_cor[:, :, idx], dtype=np.float32)
            data = np.flipud(data)      # HDF5 rows run bottom-to-top; flip to match image convention

            if NORMALIZE_BY_CLOCK:
                data = data / clock
            if NORMALIZE_BY_I0:
                data = data / i0

            out_name = roi_name_to_filename(SAMPLE, roi)
            out_path = out_dir / out_name
            tifffile.imwrite(str(out_path), data, photometric='minisblack')

            meta_path = out_path.with_suffix('.txt')
            write_metadata(
                meta_path,
                h5_file=H5_FILE, sample=SAMPLE, roi=roi, roi_index=idx,
                data=data,
                step1_um=step1_mm * 1000, step2_um=step2_mm * 1000,
                start1=start1, stop1=stop1, start2=start2, stop2=stop2,
                dwell_s=dwell_s,
                normalize_by_clock=NORMALIZE_BY_CLOCK,
                normalize_by_i0=NORMALIZE_BY_I0,
                n_rois_in_file=len(names),
            )

            print(f"  {roi:12s}  shape={data.shape}  "
                  f"min={np.nanmin(data):.4f}  max={np.nanmax(data):.4f}  ->  {out_name}")

    print(f"\nDone. {len(targets)} maps written to {out_dir}/")

if __name__ == '__main__':
    main()
