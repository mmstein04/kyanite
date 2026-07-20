"""
xanes_spot_csv_rename.py

Rename processed per-spot XANES mu(E) CSVs (normalized absorption spectra,
named by the xrmmap/areas identifier they had when collected on the beamline,
e.g. 'Fe_XANES_G1287_Ky1-Fe1.001.csv') into this project's
'<grain_id>_spotNN.csv' convention under inputs/xanes/ (read by xanes_plot.py).

Spot numbers are taken directly from xrf_h5_extract_spots.py's own
process_grain() -- including its line-scan expansion (a '_linestart'/
'_linestop' pair expands into individual points with no area entry of their
own) -- rather than re-derived here, so this can never drift out of sync with
figs/data/<grain_id>_spot_geochemistry.csv's own numbering for the same spots.
"""

import re
import shutil
from pathlib import Path

import h5py

import xrf_h5_extract_spots as xh5

_REPO_ROOT = Path(__file__).resolve().parent

# =============================================================================
# PARAMETERS — edit this section for each use
# =============================================================================

SPOT_CSV_DIR = _REPO_ROOT / 'spot_csv'
OUTPUT_DIR = _REPO_ROOT / 'inputs' / 'xanes'

# Must match whatever xrf_h5_extract_spots.py actually uses for these grains'
# real geochemistry extraction, or spot numbers here will drift out of sync
# with figs/data/<grain_id>_spot_geochemistry.csv.
NAME_FILTER = r'-Fe\d+$'
EXPAND_LINE_SCANS = True

DRY_RUN = True     # print the plan only; False to actually rename the files
OVERWRITE = False  # False: skip any destination file that already exists

# =============================================================================

SOURCE_NAME_RE = re.compile(r'^Fe_XANES_(.+)\.\d+\.csv$', re.IGNORECASE)


def grain_has_relevant_areas(grain_id):
    """Cheap pre-check (area names only, no scan/mask/CL loading) so
    process_grain() only runs its full extraction for grains that could
    possibly match NAME_FILTER or have a line scan to expand."""
    h5_file = Path(xh5.H5_DIR) / f'{grain_id}_xrf.h5'
    with h5py.File(h5_file, 'r') as f:
        names = list(f['xrmmap/areas'].keys())
    if NAME_FILTER is not None:
        pattern = re.compile(NAME_FILTER, re.IGNORECASE)
        if any(pattern.search(n) for n in names):
            return True
    if EXPAND_LINE_SCANS and xh5.find_line_prefixes(names):
        return True
    return False


def build_area_to_spot_map():
    """area_name -> (grain_id, spot) across every grain with a matching area,
    using xrf_h5_extract_spots.py's own process_grain()."""
    xh5.NAME_FILTER = NAME_FILTER
    xh5.EXPAND_LINE_SCANS = EXPAND_LINE_SCANS
    xh5.SAVE_CSV = False

    mapping = {}
    for grain_id in xh5.discover_grain_ids():
        if not grain_has_relevant_areas(grain_id):
            continue
        df = xh5.process_grain(grain_id)
        if df is None:
            continue
        for _, row in df.iterrows():
            if row['spot'] is not None:
                mapping[row['area_name']] = (grain_id, int(row['spot']))
    return mapping


def main():
    mapping = build_area_to_spot_map()

    csv_files = sorted(Path(SPOT_CSV_DIR).glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSVs found in {SPOT_CSV_DIR}')

    matched, unmatched = [], []
    for path in csv_files:
        m = SOURCE_NAME_RE.match(path.name)
        if not m:
            unmatched.append((path, "filename doesn't match 'Fe_XANES_<area_name>.<n>.csv'"))
            continue
        area_name = m.group(1)
        if area_name not in mapping:
            unmatched.append((path, f"area '{area_name}' not found in any grain's extraction"))
            continue
        grain_id, spot_num = mapping[area_name]
        dest = Path(OUTPUT_DIR) / f'{grain_id}_spot{spot_num:02d}.csv'
        matched.append((path, dest))

    print(f'\n{len(matched)} file(s) matched, {len(unmatched)} unmatched.\n')
    for path, dest in matched:
        print(f'  {path.name}  ->  {dest.relative_to(_REPO_ROOT)}')
    if unmatched:
        print('\nUnmatched (left in place):')
        for path, reason in unmatched:
            print(f'  {path.name}: {reason}')

    if DRY_RUN:
        print('\nDRY_RUN=True — nothing written. Set DRY_RUN=False to execute.')
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for path, dest in matched:
        if dest.exists() and not OVERWRITE:
            print(f'  SKIP (exists): {dest.name}')
            skipped += 1
            continue
        shutil.move(str(path), str(dest))
        written += 1
    print(f'\nMoved {written} file(s) into {OUTPUT_DIR}; skipped {skipped} existing.')


if __name__ == '__main__':
    main()
