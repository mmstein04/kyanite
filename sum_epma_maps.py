"""
sum_epma_maps.py

Sum two or more EPMA element maps into a single combined map. Typical use:
combining multiple acquisitions of the same element (e.g. Zr_La + Zr_Lb) or
summing compositionally related maps. Replaces sum_epma_maps.m — nothing
here needs MATLAB (plain TIFF I/O and array arithmetic, no Image Processing
Toolbox function actually used beyond imread/imfinfo).

If input images differ in size (e.g. due to colorbar width variation), all
maps are auto-cropped to the smallest common dimensions before summing,
trimming from the right and bottom edges.

Output: a 32-bit float TIFF containing the raw pixel sum. Float32 preserves
exact count values regardless of input bit depth and is fully compatible
with CL_EPMA_registration.m as an EPMA input.
"""

import numpy as np
import tifffile
from pathlib import Path

# =============================================================================
# PARAMETERS — edit this section for each use
# =============================================================================

# Anchored to this script's own location (not a hardcoded machine-specific
# path) so the same file runs unmodified on any machine/cluster this repo is
# checked out on.
_REPO_ROOT = Path(__file__).resolve().parent

INPUT_DIR  = _REPO_ROOT / 'inputs' / 'maps'
OUTPUT_DIR = _REPO_ROOT / 'inputs' / 'maps'

# List all map filenames to be summed (relative to INPUT_DIR).
INPUT_FILES = [
    'NA-CM-G12B4-02_P_Ka_it2.tif',
    'NA-CM-G12B4-02_P_Ka_it3.tif',
    'NA-CM-G12B4-02_P_Ka_it4.tif',
    'NA-CM-G12B4-02_P_Ka_it6.tif',
    'NA-CM-G12B4-02_P_Ka_it7.tif',
]

# Output filename (saved to OUTPUT_DIR).
OUTPUT_FILE = 'NA-CM-G12B4-02_sumP.tif'

# =============================================================================


def main():
    n = len(INPUT_FILES)
    print(f'Loading {n} maps...')

    imgs = []
    for i, fname in enumerate(INPUT_FILES, start=1):
        fpath = INPUT_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f'File not found: {fpath}')
        raw = tifffile.imread(str(fpath))

        # Collapse RGB to grayscale if needed (warn user)
        if raw.ndim == 3:
            print(f'  WARNING: Map [{i}] ({fname}) is RGB — converting to grayscale before summing.')
            raw = (0.2989 * raw[..., 0].astype(np.float64)
                   + 0.5870 * raw[..., 1].astype(np.float64)
                   + 0.1140 * raw[..., 2].astype(np.float64)).astype(np.float32)

        imgs.append(raw)
        bit_depth = raw.dtype.itemsize * 8
        print(f'  [{i}] {fname:<35s}  {raw.shape[0]} x {raw.shape[1]} px, {bit_depth}-bit')

    # =========================================================================
    # AUTO-CROP TO SMALLEST DIMENSIONS
    # =========================================================================

    nrows = [img.shape[0] for img in imgs]
    ncols = [img.shape[1] for img in imgs]
    min_rows, min_cols = min(nrows), min(ncols)

    if len(set(nrows)) > 1 or len(set(ncols)) > 1:
        print(f'\nMaps differ in size — auto-cropping to {min_rows} x {min_cols} px.')
        for i, fname in enumerate(INPUT_FILES, start=1):
            img = imgs[i - 1]
            if img.shape[0] != min_rows or img.shape[1] != min_cols:
                print(f'  [{i}] Cropped {img.shape[1] - min_cols} col(s), {img.shape[0] - min_rows} row(s) '
                      f'from right/bottom of {fname}')
            imgs[i - 1] = img[:min_rows, :min_cols]
    else:
        print(f'\nAll maps are {min_rows} x {min_cols} px — no cropping needed.')

    # =========================================================================
    # SUM
    # =========================================================================

    # Accumulate in float64 to avoid overflow for any input bit depth, then
    # cast to float32 for storage (exact for typical EPMA count ranges).
    sum_img = np.zeros((min_rows, min_cols), dtype=np.float64)
    for img in imgs:
        sum_img += img.astype(np.float64)

    print('\nSum statistics:')
    print(f'  Min:  {sum_img.min():.2f}')
    print(f'  Max:  {sum_img.max():.2f}')
    print(f'  Mean: {sum_img.mean():.2f}')

    # =========================================================================
    # SAVE
    # =========================================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FILE
    tifffile.imwrite(str(out_path), sum_img.astype(np.float32), photometric='minisblack')

    print(f'\nSaved: {out_path}')
    print(f'Output: {min_rows} x {min_cols} px, 32-bit float')


if __name__ == '__main__':
    main()
