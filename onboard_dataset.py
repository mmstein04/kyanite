"""
onboard_dataset.py

Take one grain's raw data — however someone else happened to name and
organize it — and stage it into this project's exact file/folder
conventions (see CLAUDE.md), so it can be dropped straight into
CL_EPMA_registration.m, xrf_h5_to_tiff.py, etc. without hand-renaming files.

All of the source-side knowledge (what a person's files are actually
called, which element/line each element-map file is) lives in a per-grain
YAML manifest — see dataset_manifest.example.yaml for the schema and
inline documentation of every field. This script never guesses a mapping
it wasn't told or couldn't parse from filename_pattern; anything it can't
resolve is skipped with a warning rather than silently renamed wrong.

Conventions staged (see CLAUDE.md "File conventions" for the source of
truth this mirrors):
  - EPMA/XRF element maps -> maps/<grain_id>/<grain_id>_<Element>_<Line>.tif
  - CL image              -> <project_root>/<grain_id>_CL_raw<ext>
                             (cl_filename in CL_EPMA_registration.m is a
                             free parameter, so this is a staging
                             convenience/predictable name, not a hard
                             requirement)
  - XANES classification  -> xanes_classification/<grain_id>_pre_edge_classification.csv
  - Raw XRF HDF5          -> optionally symlinked to <project_root>/<grain_id>_xrf.h5
                             (also analyzes xrmmap/areas spot naming and
                             suggests NAME_FILTER/spot-number-regex overrides
                             for xrf_h5_extract_spots.py if the default
                             'spot<N>' convention doesn't match)

Every file operation is a copy (or, for the multi-GB h5, a symlink) —
originals are never moved or modified. Defaults to a dry run: prints the
full plan and any warnings/conflicts without touching disk. Set
DRY_RUN = False to actually execute it.
"""

import re
import shutil
from pathlib import Path

import yaml

# =============================================================================
# PARAMETERS — edit this section for each use
# =============================================================================

MANIFEST_FILE = 'dataset_manifest.yaml'
PROJECT_ROOT  = '.'

DRY_RUN   = True    # False to actually copy/symlink files
OVERWRITE = False   # False: skip (with a warning) any destination that already exists

# =============================================================================

# Same convention xrf_h5_extract_spots.py uses to pull a trailing spot number
# out of an xrmmap/areas name (e.g. 'LLF6-Area2-spot01' -> 01).
DEFAULT_SPOT_NUM_RE = re.compile(r'spot0*(\d+)', re.IGNORECASE)
FALLBACK_SPOT_NUM_RES = [
    (r'trailing digits', re.compile(r'(\d+)\s*$')),
    (r"'pt<N>'", re.compile(r'pt0*(\d+)', re.IGNORECASE)),
    (r"'point<N>'", re.compile(r'point0*(\d+)', re.IGNORECASE)),
]


def load_manifest(path):
    with open(path) as f:
        manifest = yaml.safe_load(f)
    if 'grain_id' not in manifest:
        raise ValueError(f"{path}: manifest must set 'grain_id'")
    return manifest


def _copy_op(src, dst):
    return {'kind': 'copy', 'src': Path(src), 'dst': Path(dst)}


def _symlink_op(src, dst):
    return {'kind': 'symlink', 'src': Path(src), 'dst': Path(dst)}


def plan_cl_image(manifest, project_root, grain_id):
    cfg = manifest.get('cl_image')
    if not cfg:
        return [], []
    src = Path(cfg['source'])
    dst = Path(cfg['dest']) if cfg.get('dest') else project_root / f"{grain_id}_CL_raw{src.suffix}"
    warnings = [] if src.exists() else [f"cl_image.source not found: {src}"]
    return [_copy_op(src, dst)], warnings


def plan_epma_maps(manifest, project_root, grain_id):
    cfg = manifest.get('epma_maps')
    if not cfg:
        return [], []

    src_dir = Path(cfg['source_dir'])
    out_dir = Path(cfg['output_dir']) if cfg.get('output_dir') else project_root / 'maps' / grain_id
    element_alias = cfg.get('element_alias', {})
    line_alias = cfg.get('line_alias', {})
    explicit_files = cfg.get('files', {})
    include = cfg.get('include', '*.tif*')
    pattern = re.compile(cfg['filename_pattern']) if cfg.get('filename_pattern') else None

    ops, warnings = [], []
    if not src_dir.is_dir():
        return [], [f"epma_maps.source_dir not found: {src_dir}"]

    resolved = {}  # dest element_line -> [source names] (to catch collisions)
    unresolved = []

    for src_path in sorted(src_dir.glob(include)):
        name = src_path.name
        element_line = None

        if name in explicit_files:
            element_line = explicit_files[name]
        elif pattern:
            m = pattern.fullmatch(name)
            if m and 'element' in m.groupdict() and 'line' in m.groupdict():
                el = element_alias.get(m.group('element'), m.group('element'))
                ln = line_alias.get(m.group('line'), m.group('line'))
                element_line = f"{el}_{ln}"

        if element_line is None:
            unresolved.append(name)
            continue

        dst = out_dir / f"{grain_id}_{element_line}{src_path.suffix}"
        resolved.setdefault(element_line, []).append(name)
        ops.append(_copy_op(src_path, dst))

    if unresolved:
        warnings.append(
            f"epma_maps: {len(unresolved)} file(s) in {src_dir} did not match "
            f"any explicit 'files' entry or 'filename_pattern' and were skipped: "
            f"{', '.join(unresolved)}"
        )
    for element_line, names in resolved.items():
        if len(names) > 1:
            warnings.append(
                f"epma_maps: {len(names)} source files all resolved to "
                f"'{element_line}' ({', '.join(names)}) — fix the manifest so "
                f"each element/line is unique, otherwise one will overwrite "
                f"the other on disk."
            )

    return ops, warnings


def plan_xanes_classification(manifest, project_root, grain_id):
    cfg = manifest.get('xanes_classification')
    if not cfg:
        return [], []
    src = Path(cfg['source'])
    dst = (Path(cfg['dest']) if cfg.get('dest')
           else project_root / 'xanes_classification' / f"{grain_id}_pre_edge_classification.csv")
    warnings = [] if src.exists() else [f"xanes_classification.source not found: {src}"]
    return [_copy_op(src, dst)], warnings


def suggest_name_filter(h5_path):
    """Inspect xrmmap/areas key names and report whether the default
    'spot<N>' convention (used by xrf_h5_extract_spots.py's NAME_FILTER /
    SPOT_NUM_RE) will actually match them, suggesting an override if not."""
    import h5py

    with h5py.File(h5_path, 'r') as f:
        if 'xrmmap/areas' not in f:
            return ["xrf_h5: no xrmmap/areas group found — cannot check spot naming"]
        names = list(f['xrmmap/areas'].keys())

    if not names:
        return ["xrf_h5: xrmmap/areas is empty — nothing to check"]

    default_hits = [n for n in names if DEFAULT_SPOT_NUM_RE.search(n)]
    if len(default_hits) == len(names):
        return [f"xrf_h5: all {len(names)} xrmmap/areas names match the default "
                f"'spot<N>' convention — no NAME_FILTER/regex changes needed."]

    msgs = [f"xrf_h5: only {len(default_hits)}/{len(names)} xrmmap/areas names "
            f"matched the default 'spot<N>' convention. Example names: "
            f"{', '.join(names[:5])}"]

    for label, rx in FALLBACK_SPOT_NUM_RES:
        hits = [n for n in names if rx.search(n)]
        if len(hits) == len(names):
            msgs.append(
                f"  -> {label} matches all {len(names)} names. In "
                f"xrf_h5_extract_spots.py, update NAME_FILTER to match your "
                f"area-name prefix and adjust SPOT_NUM_RE accordingly."
            )
            break
    else:
        msgs.append(
            "  -> No built-in fallback pattern matched every name either. "
            "You'll need a custom NAME_FILTER/spot-number regex in "
            "xrf_h5_extract_spots.py — inspect the printed names above."
        )
    return msgs


def plan_xrf_h5(manifest, project_root, grain_id):
    cfg = manifest.get('xrf_h5')
    if not cfg:
        return [], []
    src = Path(cfg['source'])
    warnings = []
    if not src.exists():
        return [], [f"xrf_h5.source not found: {src}"]

    ops = []
    if cfg.get('stage', True):
        dst = Path(cfg['dest']) if cfg.get('dest') else project_root / f"{grain_id}_xrf.h5"
        ops.append(_symlink_op(src, dst))

    if cfg.get('suggest_name_filter', True):
        try:
            warnings.extend(suggest_name_filter(src))
        except Exception as e:
            warnings.append(f"xrf_h5: could not inspect {src} for spot naming: {e}")

    return ops, warnings


def execute(ops, dry_run, overwrite):
    done, skipped = [], []
    for op in ops:
        src, dst = op['src'], op['dst']
        if dst.exists() and not overwrite:
            skipped.append(dst)
            continue
        if dry_run:
            done.append(dst)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if op['kind'] == 'copy':
            shutil.copy2(src, dst)
        elif op['kind'] == 'symlink':
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        done.append(dst)
    return done, skipped


def write_log(log_path, grain_id, ops, warnings, dry_run):
    lines = [
        "# Dataset onboarding log",
        f"grain_id : {grain_id}",
        f"mode     : {'DRY RUN (nothing written)' if dry_run else 'EXECUTED'}",
        "",
        "# Planned operations",
    ]
    for op in ops:
        lines.append(f"  [{op['kind']}] {op['src']}  ->  {op['dst']}")
    lines.append("")
    lines.append("# Warnings")
    if warnings:
        lines.extend(f"  {w}" for w in warnings)
    else:
        lines.append("  (none)")
    log_path.write_text("\n".join(lines) + "\n")


def main():
    manifest = load_manifest(MANIFEST_FILE)
    grain_id = manifest['grain_id']
    project_root = Path(PROJECT_ROOT)

    all_ops, all_warnings = [], []
    for plan_fn in (plan_cl_image, plan_epma_maps, plan_xanes_classification, plan_xrf_h5):
        ops, warnings = plan_fn(manifest, project_root, grain_id)
        all_ops.extend(ops)
        all_warnings.extend(warnings)

    print(f"Onboarding '{grain_id}' from manifest {MANIFEST_FILE}")
    print(f"{'DRY RUN — no files will be written' if DRY_RUN else 'EXECUTING'}\n")

    print(f"Planned operations ({len(all_ops)}):")
    for op in all_ops:
        print(f"  [{op['kind']:7s}] {op['src']}  ->  {op['dst']}")

    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  ! {w}")

    done, skipped = execute(all_ops, DRY_RUN, OVERWRITE)

    if skipped:
        print(f"\nSkipped (destination already exists, OVERWRITE=False): {len(skipped)}")
        for dst in skipped:
            print(f"  {dst}")

    log_path = project_root / f"{grain_id}_onboarding_log.txt"
    if not DRY_RUN:
        write_log(log_path, grain_id, all_ops, all_warnings, DRY_RUN)
        print(f"\nLog written to {log_path}")

    print(f"\n{'Would write' if DRY_RUN else 'Wrote'} {len(done)} file(s), "
          f"skipped {len(skipped)}, {len(all_warnings)} warning(s).")
    if DRY_RUN:
        print("Set DRY_RUN = False at the top of this script to execute this plan.")


if __name__ == '__main__':
    main()
