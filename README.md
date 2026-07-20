# Kyanite CL–EPMA–XRF Workflow — User Guide

This guide describes how to run this codebase. It assumes familiarity with
the underlying petrology/spectroscopy (CL, EPMA, synchrotron XRF, XANES) but
not with these specific scripts. It covers the pipeline in execution order,
the inputs and outputs of each step, and decisions that are not obvious from
the code alone.

For a reference table of every file and parameter, see `CLAUDE.md`. This
document covers the same information in narrative form.

## What the workflow does

For each kyanite grain, the workflow registers a CL image onto an
EPMA/synchrotron-XRF element-map grid, masks out non-grain pixels, and
extracts per-pixel CL intensity alongside per-pixel trace-element
concentration for each mapped element. From the extracted data, the pipeline
supports: correlating CL against chemistry pixel-by-pixel, subdividing the
grain into regions or CL textural domains, and ranking which elements
predict CL via PCA/Random Forest/SHAP. A separate, related track classifies
XANES pre-edge spectra at discrete spot locations and relates the
classification to CL and chemistry.

Every downstream script keys off **`grain_id`**, a string used as a filename
prefix throughout the pipeline (e.g. `LLF6-01`). It is set once, at
registration, and reused unchanged by every subsequent script for that
grain.

## Before you start

**Requirements**
- MATLAB + Image Processing Toolbox (`cpselect`, `imwarp`, `activecontour`, `visboundaries`, etc.)
- Python: `h5py`, `numpy`, `tifffile`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `shap`, `pyyaml`

On a shared HPC/cluster account, install these into a dedicated virtualenv
rather than a shared `--user`/system site-packages install — the latter can
silently upgrade a package (e.g. `numpy`) out from under unrelated tools that
pin an older version:
```
python3 -m venv ~/kyanite_env
source ~/kyanite_env/bin/activate
pip install h5py numpy tifffile pandas matplotlib seaborn scipy scikit-learn shap pyyaml
```
`kyanite.sh` is an example SLURM batch script that activates this venv and
runs a Python step (`kyanite_rf_shap.py` by default, the compute-heavy step)
non-interactively; adapt the script name and `#SBATCH` directives to the
step/cluster you're using.

**Directory structure.** Scripts are located at the repository root.
`inputs/` and `figs/` are created automatically as the pipeline runs:
```
kyanite/
├── README.md                          this file
├── CLAUDE.md                          terse per-script parameter/file reference
├── dataset_manifest.example.yaml      template for onboard_dataset.py
│
├── onboard_dataset.py                 Step 0 (optional) — stage a foreign-named dataset
├── xrf_h5_to_tiff.py                  Step 1 — export XRF element maps from HDF5
├── CL_EPMA_registration.m             Step 2 — register CL, mask grain, extract pixel data
├── CL_mask_edit.m                     Step 3 (optional) — fix a mask after the fact
├── CL_region_extraction.m             Step 4 (optional) — sub-grain regions / texture domains
├── CL_local_regression_map.py         Step 5 (optional) — continuous local CL-vs-element regression
├── kyanite_figures.py                 Step 6 — whole-grain figures from pixel CSV
├── kyanite_pca.py                     Step 7a — PCA (whole-grain + pooled region-PCA)
├── kyanite_rf_shap.py                 Step 7b — Random Forest / SHAP fit, exports CSVs only
├── kyanite_rf_shap_plots.py           Step 7c — figures from kyanite_rf_shap.py's CSVs
├── kyanite_sample_size_convergence.py  diagnostic for Step 7b's subsampling
├── xanes_spot_csv_rename.py            Step 8.0 (optional) — rename processed XANES mu(E)
│                                      CSVs into <grain_id>_spotNN.csv convention
├── xanes_plot.py                      Step 8.1 — plot pre-edge doublets for hand classification
├── xanes_classification_split.py      Step 8.2 — split hand classification CSV per grain
├── xrf_h5_extract_spots.py            Step 8.3 — per-spot geochemistry + CL + XANES class CSV
├── kyanite_spot_analysis.py           Step 8.4 — batch spot analysis (pie/scatter/box/map/PCA)
├── xanes_rf_classifier.py             Step 8.5 — RF classifying XANES class from chemistry
├── sum_epma_maps.py                   utility — sum element-line maps (e.g. Zr_La + Zr_Lb)
├── xrf_display.py                     utility — visualize element/ratio maps with mask overlay
├── kyanite_palette.py                 shared colors (element/region/category/colormap
│                                      conventions) — imported by the Python scripts above;
│                                      the MATLAB scripts keep local copies of the same values
├── kyanite.sh                          example SLURM batch script (HPC) — activates a venv,
│                                      runs one Python step non-interactively
│
├── inputs/                            everything the pipeline reads as input
│   ├── xrf/                             raw synchrotron XRF scans: <grain_id>_xrf.h5
│   ├── cl/                              raw CL micrographs (pre-registration):
│   │                                    <grain_id>_CL_color.<ext>, and any alternative
│   │                                    mask_image_file for CL_EPMA_registration.m
│   ├── xanes/                           raw per-spot XANES spectrum CSVs, named
│   │                                    <grain_id>_spotNN.csv (see xanes_spot_csv_rename.py
│   │                                    if yours instead arrive named by beamline
│   │                                    acquisition-time area identity)
│   ├── xanes_raw/                       raw per-point XANES files (e.g. from a XANES line
│   │                                    scan), used only to recover a line-scanned point's
│   │                                    pixel location when it has no xrmmap/areas entry
│   │                                    of its own (xrf_h5_extract_spots.py's
│   │                                    EXPAND_LINE_SCANS/XANES_RAW_DIR)
│   ├── xanes_classification/            hand (or auto) pre-edge classification CSVs,
│   │                                    one per grain
│   └── maps/<grain_id>/                 element-map TIFFs for one grain — the one exception
│                                        in inputs/: generated by xrf_h5_to_tiff.py from
│                                        inputs/xrf/, but every downstream script only ever
│                                        reads it (epma_dir)
│
└── figs/                              everything the pipeline writes
    ├── data/                            reusable data other scripts read back in: grain mask,
    │                                   pixel/region-pixel data CSV/MAT, control-point MATs,
    │                                   mask-edit history, local-regression NPZ/CSV, per-spot
    │                                   geochemistry CSV, kyanite_pca.py's PCA tables
    │                                   (variance/loadings/scores/centroid distances),
    │                                   kyanite_rf_shap.py's RF/SHAP CSVs, and
    │                                   xanes_rf_classifier.py's importance/predictions CSVs —
    │                                   data files live here regardless of which script/mode
    │                                   produced them
    ├── diagnostics/                     not-for-publishing sanity/alignment-check + run-metadata
    │                                   outputs from CL_EPMA_registration.m / CL_mask_edit.m /
    │                                   CL_region_extraction.m / CL_local_regression_map.py
    │                                   (analysis logs, all-maps QC, registration overlay,
    │                                   mask-image registration check, shift-sensitivity,
    │                                   mask check, mask-edit diff/log, region overlay/QC,
    │                                   local-regression slope/R/n-map QC) plus kyanite_pca.py's,
    │                                   kyanite_rf_shap.py's, and xanes_rf_classifier.py's run logs,
    │                                   and all of kyanite_sample_size_convergence.py's output
    │                                   (raw sweep CSV, convergence figures, log — it's itself a
    │                                   diagnostic, so it has no dedicated folder of its own)
    ├── mask_edit_backups/               pre-edit snapshots written automatically by CL_mask_edit.m
    ├── whole_grain/                     kyanite_figures.py's whole-grain pixel-CSV figures
    │                                   (scatter/violin/boxplot/contour/heatmap/corrmatrix)
    ├── pca/                             kyanite_pca.py's whole-grain PCA figures (scree,
    │                                   loadings, PC-vs-CL scatter)
    ├── rf/                               kyanite_rf_shap_plots.py's RF figures
    │                                   (observed-vs-predicted, permutation importance)
    ├── shap/                             kyanite_rf_shap_plots.py's SHAP figures
    │                                   (importance, interactions, dependence)
    ├── regions/                         CL_region_extraction.m's non-reusable region outputs
    │                                   (region summary CSV, texture class map PNG) plus
    │                                   kyanite_figures.py / kyanite_pca.py figures for
    │                                   region pixel CSVs (RF/SHAP has no region-mode output)
    ├── local_regression/                CL_local_regression_map.py's one true result figure
    │                                   (Cr R-vs-CL); slope/R/n-map QC now live in diagnostics/,
    │                                   and the NPZ/CSV in data/
    ├── map_renders/                     xrf_display.py's rendered element/ratio-map PNGs
    │                                   (visualizations of the inputs/maps/ TIFFs)
    ├── xanes/                           xanes_plot.py's pre-edge classification-support figures
    │                                   (spot geochemistry CSVs live in data/, not here)
    └── spot_analysis/                   kyanite_spot_analysis.py's batch figures
                                        (pie/scatter/box/PCA/map) plus xanes_rf_classifier.py's
                                        two figures (importance, confusion matrix) — shared
                                        since both pool the same per-spot CSVs and fall under
                                        the same "spot analysis" umbrella, even though they're
                                        different analyses (xanes_rf_classifier.py's reusable
                                        CSVs go to data/, its run log to diagnostics/)
```

**If raw data does not already follow these conventions** (for example, a
collaborator's grain with differently-named files), use `onboard_dataset.py`
to convert it rather than renaming files by hand (see below). Every
subsequent step assumes files are already named and placed per the
conventions above.

---

## The pipeline

### Step 0 (optional) — Onboard a foreign-named dataset
**`onboard_dataset.py`**

Copies (or, for the large XRF HDF5, symlinks) a grain's raw files — CL
image, element-map TIFFs, XANES classification CSV, XRF HDF5 — from their
source locations and names into the file/folder conventions this workflow
expects. Driven by a per-grain YAML manifest: copy
`dataset_manifest.example.yaml`, fill in the source paths and the mapping
from element-map filenames to `<Element>_<Line>`, then run with
`DRY_RUN = True` to review the plan before any files are written.

This step is optional. Skip it if the data is already laid out per the
conventions above.

### Step 1 — Export XRF element maps from the raw HDF5
**`xrf_h5_to_tiff.py`**

Required only if element maps come from a Larch/GSECARS synchrotron XRF
scan (`.h5`) rather than EPMA TIFFs from microprobe software. Set `H5_FILE`
(typically `inputs/xrf/<grain_id>_xrf.h5`), `OUTPUT_DIR` (typically
`inputs/maps/<grain_id>`), `GRAIN_ID`, and `ELEMENTS` (or `None` for all
ROIs in the file; the script prints the full ROI list at startup).
`NORMALIZE_BY_CLOCK`/`NORMALIZE_BY_I0` should be decided once per grain and
applied consistently in every downstream script that reads this HDF5
(notably `xrf_h5_extract_spots.py`); otherwise values are not comparable.

Output: `<grain_id>_<Element>_<Line>.tif` (32-bit float) and a `.txt`
metadata sidecar per map, both in `OUTPUT_DIR`.

### Step 2 — Register CL to EPMA/XRF and extract pixel data
**`CL_EPMA_registration.m`**

Every grain must go through this step exactly once. Parameters set at the
top of the script: `grain_id`, `input_dir` (folder holding the raw CL
image; default `inputs/cl`), `cl_filename` (any format readable by
`imread`), `epma_dir` (folder of element-map TIFFs; default
`inputs/maps/<grain_id>`; all `*.tif` files are auto-discovered),
`epma_ref_file` (the highest-contrast map, used for control-point picking),
and `mask_method`. `epma_pixel_um` is read automatically from
`xrf_h5_to_tiff.py`'s metadata sidecar for whichever EPMA map has one
(`epma_pixel_um_from_sidecar = true`, the same mechanism `xrf_display.py`/
`CL_local_regression_map.py` use) — the hardcoded `epma_pixel_um` value is
only a fallback if no sidecar is found.

1. `cpselect` opens interactively — click matching control points on the CL
   image and the reference EPMA/XRF map, then close the window to warp.
2. Registration quality (RMSE in px and µm) is reported. If RMSE is too
   high, repeat control-point selection before using downstream
   correlations.
3. The grain mask is built via `mask_method`:
   - `otsu` — automatic threshold; fastest, works when the grain has good
     CL/background contrast.
   - `manual` — fixed threshold supplied by the user.
   - `interactive` — threshold with a live preview slider.
   - `polygon` — grain boundary drawn by hand; used when background is
     heterogeneous or a neighboring phase interferes with automatic
     thresholding.
   - `activecontour` — snake-fits a starting contour to grain edges.
4. The mask is applied; Pearson r and a linear fit are computed for CL vs.
   each element, printed to console and the analysis log, as an immediate
   registration-quality check. (No figure is drawn here — exploratory
   scatter/violin/contour/etc. figures are generated downstream by
   `kyanite_figures.py` from the exported pixel CSV.) A shift-sensitivity
   analysis quantifies how much correlations degrade under a small
   deliberate mis-registration, to assess how much alignment precision
   affects the reported r values.

Outputs are written to `figs/`: registered CL (grayscale 16-bit and
original color). The grain mask and pixel data
(`data/<grain_id>_mask.tif`, `data/<grain_id>_pixel_data.csv`/`.mat`,
control-point MATs), which are reusable inputs for other scripts, are
written to `figs/data/`. Diagnostic outputs not intended for publication
(analysis log, registration overlay, mask-image registration check,
shift-sensitivity, all-maps QC, mask check) are written to
`figs/diagnostics/`.

### Step 3 (optional) — Fix a mask after the fact
**`CL_mask_edit.m`**

Use this instead of rerunning Step 2 when a mask problem is found later (an
inclusion masked in, or real grain masked out). No re-registration or
control-point selection is required. Set `grain_id`, `input_dir`, `epma_dir`,
and match `normalize_epma`/`pct_lo_cut`/`pct_hi_cut`/`shift_range` to the
values used in Step 2 for this grain, so the before/after comparison is
valid. `epma_pixel_um` is read from the same sidecar mechanism as Step 2
(`epma_pixel_um_from_sidecar = true`), so it stays in sync with Step 2's
value for this grain automatically rather than needing to be hand-copied.

Add and remove polygons are drawn interactively, with a live preview after
each edit; `u` undoes the last edit, `d` finishes. Before overwriting
anything, the script backs up existing outputs to
`figs/mask_edit_backups/<grain_id>_<timestamp>/`. With
`regenerate_downstream = true` (default), it re-extracts pixel data and
recomputes the CL-vs-element Pearson correlations (console + log only, same
as Step 2 — no figure), the shift-sensitivity analysis, and the mask-check/
mask-edit-diff QC figures and run log in `figs/diagnostics/`, keeping `figs/`
internally consistent.

### Step 4 (optional) — Sub-grain regions or full-grain texture domains
**`CL_region_extraction.m`**

Requires a grain that has already been through Step 2. Like `CL_mask_edit.m`,
`epma_pixel_um` is read from the same metadata-sidecar mechanism as Step 2
(`epma_pixel_um_from_sidecar = true`), so region area (µm²) calculations stay
in sync with Step 2's value automatically. Two modes, controlled by
`classification_mode`:

- **`false` (default)** — freeform ROIs. Draw and name any number of
  polygons (e.g. "core", "rim"); overlapping or partial coverage is
  allowed. Used for predefined regions of interest.
- **`true`** — exhaustive texture classification. Subdivides the entire
  grain into non-overlapping domains drawn from a fixed vocabulary
  (`TEXTURE_CLASSES`, e.g. user-defined sector/oscillatory zoning types).
  Each new domain is automatically clipped to the grain-mask area not
  already claimed by a prior domain, so polygons do not need to be drawn
  precisely; any remaining grain-mask area when drawing stops is labeled
  `Unclassified`. Requires `restrict_to_grain_mask = true`.

Both modes produce a long-format per-pixel CSV with a `Region` column (the
texture class, in classification mode). `_region_pixel_data.csv` uses the
same filename in both modes, so running one mode after the other for the
same grain overwrites the file; a warning is printed when this happens.

Outputs are split the same way as Step 2: reusable data (region/texture-domain
polygon MAT, `_region_pixel_data.csv`/`.mat`, texture class map TIFF) go to
`figs/data/`; QC figures and the analysis log go to `figs/diagnostics/`; the
region summary CSV and the texture class map PNG go to `figs/regions/`.

### Step 5 (optional) — Continuous local-window CL-vs-element regression
**`CL_local_regression_map.py`**

Provides a spatially continuous alternative to Step 4's fixed polygons.
Slides a circular window across every grain-mask pixel, regresses CL
against each element over the pixels within that window, and stores the
resulting slope and Pearson r at the window center, producing maps of local
CL–element relationship strength and sign rather than a single value for
the whole grain or region. Requires each grain to already be through Step
2. `GRAIN_IDS` may be a single grain, a list, or `None` (default) to
auto-discover and run every grain with a registered CL image, mask, and
maps folder in one go — a grain that fails partway (missing input, size
mismatch) is skipped with a warning rather than aborting the batch. Each
grain's µm/px is read from `xrf_h5_to_tiff.py`'s metadata sidecar rather
than a single hardcoded value, since grains in this project aren't all
imaged at the same resolution — a fixed value would silently make the
window radius physically wrong for whichever grains don't match it.
Outputs slope/R map grids (one panel per element) and a window-coverage
map, which indicates where low pixel counts make a slope/R estimate
unreliable — these QC figures and the analysis log go to
`figs/diagnostics/`, the slope/R/n NPZ and long-format CSV go to
`figs/data/`, and the one true analysis-result figure (a Cr-specific R map
next to the CL image, since Cr³⁺ is a known CL activator) is saved
directly in `figs/local_regression/`.

### Step 6 — Whole-grain figures from the exported pixel CSV
**`kyanite_figures.py`**

Standalone; requires only `figs/data/<grain_id>_pixel_data.csv` from Step
2 (or `_region_pixel_data.csv` from Step 4 — both live in `figs/data/`,
so pointing `CSV_INPUT` at that directory picks up and auto-routes both
kinds). Set `CSV_INPUT`, `ELEMENTS`, `PLOT_TYPE` (`scatter`/`violin`/
`boxplot`/`contour`/`heatmap`/`corrmatrix`/`all`), and binning
(`N_BINS`/`BIN_EDGES`) or percentile trim (`PCT_LO`/`PCT_HI`) as needed.
Also produces element-ratio correlation-matrix figures. Figures are saved
to `WHOLE_GRAIN_OUTPUT_DIR`/`REGION_OUTPUT_DIR` (default `figs/whole_grain/`/
`figs/regions/`) — independent of wherever `CSV_INPUT` points, so pointing
it at `figs/data/` never dumps PNGs in among the reusable data files.

### Step 7 — Multivariate statistics: PCA, Random Forest, and SHAP
Three scripts, split so retraining a Random Forest/SHAP model is never
required just to regenerate or restyle a figure:

**Step 7a — `kyanite_pca.py`**

Reads `figs/data/<grain_id>_pixel_data.csv` (or a directory of several
files, pooled) and runs PCA (dimensionality/structure of the trace-element
space): scree plot, per-PC loadings, and PC score vs. CL intensity.
`BELOW_DETECTION`/`MAX_BELOW_DETECTION_FRAC` exclude elements that are
mostly below detection before fitting. Region CSVs (from Step 4) get a
different treatment: one PCA fit pooled across all of a grain's regions, so
every region is projected into the same PC space and can be tested for
separation (scree/loadings/PC1-vs-PC2 scatter/biplot, plus ANOVA and
centroid-distance stats). Figures go to `WHOLE_GRAIN_OUTPUT_DIR`/
`REGION_OUTPUT_DIR` (default `figs/pca/`/`figs/regions/`); the reusable CSV
tables (variance, loadings, and — region CSVs only — scores/centroid
distances) go to `DATA_OUTPUT_DIR` (default `figs/data/`, alongside the
pixel-data CSVs); the run log goes to `DIAGNOSTICS_DIR` (default
`figs/diagnostics/`, matching every other analysis log in this project).
All independent of `CSV_INPUT`.

**Step 7b — `kyanite_rf_shap.py`**

Reads the same whole-grain pixel CSV(s) as Step 7a (region CSVs are
skipped with a warning — RF/SHAP is whole-grain only) and fits `rf`
(cross-validated Random Forest regressing CL on elements, with permutation
importance — the primary quantitative result for which elements predict
CL) and/or `shap` (TreeSHAP importance and pairwise interactions from a
single RF fit on a subsample, since SHAP does not scale to a full
~300k-pixel grain), selected via `ANALYSES`. This script only fits models
and writes CSVs (`OUTPUT_DIR`, default `figs/data/`, since these CSVs are
themselves reusable data) plus a run log (`DIAGNOSTICS_DIR`, default
`figs/diagnostics/`) — no figures. The per-pixel data behind every figure
Step 7c can draw is exported: out-of-fold predictions (`_rf_predictions.csv`),
permutation importance (`_rf_importance.csv`), SHAP importance
(`_shap_importance.csv`), raw per-pixel SHAP values (`_shap_values.csv`), and
the SHAP interaction matrix (`_shap_interactions.csv`).

**Step 7c — `kyanite_rf_shap_plots.py`**

Reads Step 7b's CSVs back from `CSV_INPUT` (default `figs/data/`, matching
Step 7b's `OUTPUT_DIR`) and renders the observed-vs-predicted scatter and
permutation-importance bar chart to `RF_OUTPUT_DIR` (default `figs/rf/`),
and the SHAP importance bar chart, interaction heatmap, and dependence
panels to `SHAP_OUTPUT_DIR` (default `figs/shap/`) — no model fitting
happens here, so changing `PLOTS`/`FIG_DPI`/`SHOW_TITLE` and rerunning is
instant. A grain missing a plot's underlying CSV (e.g. it only ran
`ANALYSES=['rf']` in Step 7b) is skipped for that plot with a warning, not
an error.

**`kyanite_sample_size_convergence.py`** is a diagnostic, not a pipeline
step. Run it once per grain to check whether Step 7b's `MAX_SAMPLES`/
`SHAP_SAMPLES` subsampling has converged (importance estimates stop
changing with more data), rather than assuming convergence for
computational convenience. Since the whole script is itself a diagnostic,
its output doesn't get a dedicated analysis-family folder like Steps
7a–7c — raw sweep CSV, convergence figures, and log all go to
`DIAGNOSTICS_DIR` (default `figs/diagnostics/`), independent of `CSV_INPUT`.

### Step 8 (optional, parallel track) — XANES spot classification
This sub-pipeline relates CL and chemistry to Fe²⁺/Fe³⁺ oxidation state at
discrete spot locations, rather than full-grain maps, and is independent of
Steps 3–7:

0. **`xanes_spot_csv_rename.py`** (optional) — if processed per-spot XANES
   mu(E) CSVs arrive named by their beamline acquisition-time area identity
   (e.g. `Fe_XANES_G1287_Ky1-Fe1.001.csv`) rather than already in this
   project's `<grain_id>_spotNN.csv` convention, this renames them into
   `inputs/xanes/` — driven directly by `xrf_h5_extract_spots.py`'s own spot
   numbering (see step 3 below, including its line-scan expansion), so the
   two can never drift out of sync. Defaults to `DRY_RUN=True`/
   `OVERWRITE=False` (preview the plan, never clobbers existing output).
1. **`xanes_plot.py`** — reads raw per-spot spectrum CSVs from
   `XANES_INPUT` (default `inputs/xanes/`) and plots each spot's Fe K-edge
   pre-edge doublet as a small-multiples grid, for classification of peak
   shape by eye: **Type 1** (Fe²⁺ peak taller), **Type 2** (roughly equal),
   **Type 3** (Fe³⁺ peak taller). Automatic classification is available but
   `CLASSIFY` defaults to `False`, since the automatic method does not
   reliably match expert judgment across grains with varying peak
   separations; manual classification from the printed grids is
   recommended.
2. Hand-classify every spot into a combined CSV (`GrainID, Spot, Class` —
   1/2/3, or 4 for bad/unusable data), then run
   **`xanes_classification_split.py`** to split it into one
   `<grain_id>_pre_edge_classification.csv` per grain in
   `inputs/xanes_classification/`.
3. **`xrf_h5_extract_spots.py`** — for each spot marked in the raw h5
   (`xrmmap/areas`), extracts pixel and physical coordinates, mean element
   concentrations, and CL brightness over a small grain-mask-restricted
   circular zone (`ZONE_RADIUS_UM`), and joins in the hand classification
   from step 2. `GRAIN_IDS` may be a single grain, a list, or `None`
   (default) to auto-discover and run every grain with an h5 file in
   `H5_DIR` (default `inputs/xrf/`) in one go — a grain that fails partway
   (e.g. an unreadable/malformed h5) is skipped with a warning rather than
   aborting the batch. Also set `FIGS_DIR`, `CLASSIFICATION_DIR` (default
   `inputs/xanes_classification/`). A missing mask/CL image/classification
   CSV, or zero mask-overlap for a given spot, produce NaN (and, for
   mask-overlap, an `on_grain = False` flag) plus a warning rather than an
   error.
   - Note: h5 area names do not need to share a naming scheme with
     `GRAIN_ID` (e.g. h5 area `LLF6-Area2-spot01` under grain `LLF6-01`);
     matching is done via `NAME_FILTER` (default regex requires the
     substring `spot`). If h5 areas use a different naming scheme (`pt01`,
     `point3`, ...), `onboard_dataset.py`'s h5 check reports which
     `NAME_FILTER` to use instead. Some grains carry more than one family of
     point in the same h5 — e.g. generic `spotNN` points alongside dedicated
     `-FeN` points from a separate Fe-only XANES session — `NAME_FILTER`
     picks which family a run extracts (e.g. `r'-Fe\d+$'` for Fe-only). A
     spot's number is always the area name's *trailing digits*, regardless
     of which family/tag word precedes them (`Fe7` and `spot07` both give
     spot number 7) — don't extract both families for the same grain in one
     run, since they'd collide on spot number.
   - A XANES *line* scan is sometimes marked in the h5 as just two
     single-pixel areas, `<prefix>_linestart`/`<prefix>_linestop` — the
     points actually measured in between have no area entry of their own.
     `EXPAND_LINE_SCANS = True` (default) recovers each intermediate point's
     pixel location from its raw per-point file's stage-position header
     (`XANES_RAW_DIR`, default `inputs/xanes_raw/`) instead, and extracts
     each as its own spot, numbered continuing on from the grain's other
     spots (e.g. `Fe1..Fe7` -> spot1..spot7, then a 15-point line ->
     spot8..spot22). A line missing its raw files degrades gracefully
     (warning, that line skipped) rather than blocking the batch.
   - Output: `figs/data/<grain_id>_spot_geochemistry.csv` — reusable data,
     read back by both `kyanite_spot_analysis.py` and `xanes_rf_classifier.py`.
4. **`kyanite_spot_analysis.py`** — batch analysis pooling every grain's
   `*_spot_geochemistry.csv`: XANES class pie-chart grid, pooled
   CL-vs-element scatter/box plots colored by class, PCA (scatter,
   scree, loadings, biplot with cluster hulls) over `PCA_ELEMENTS`, and a
   per-grain spot-location map on the registered CL image, all colored
   consistently by class, with `'Bad data'`/unclassified spots shown as
   grey points for QC context rather than dropped. Off-grain spots
   (`on_grain = False` — the spot's zone missed the grain mask entirely, so
   it sampled some other phase) are handled differently per figure: the
   pie chart excludes them (it characterizes this grain's own kyanite class
   distribution), while scatter/box/PCA exclude them implicitly since their
   CL/element values are already `NaN`. The spot map is the exception — it
   keeps off-grain spots visible, marked with an `'X'` instead of `'o'` but
   still colored by class, since their oxidation state is still meaningful
   data for whatever phase they actually sampled.
5. **`xanes_rf_classifier.py`** — the classification analog of Step 7's
   regression: cross-validated Random Forest predicting XANES class from
   spot chemistry, pooled across all grains (per-grain models are not
   meaningful, since some grains are entirely one class). Off-grain spots
   are excluded the same implicit way as Step 8.4's scatter/box/PCA — their
   `NaN` chemistry is correctly dropped, since this classifier is about
   kyanite's own chemistry specifically. `CV_STRATEGY =
   'grouped'` (default) holds out folds by grain rather than by spot, so a
   fold cannot learn one grain's chemical signature instead of a general
   chemistry–oxidation relationship. Reads the same `figs/data/` CSVs as
   `kyanite_spot_analysis.py`. Output splits the same way as Steps 7a/7b:
   its two figures (importance, confusion matrix) go to `figs/spot_analysis/`
   (`OUT_DIR`) — shared with `kyanite_spot_analysis.py`, no dedicated folder
   of its own, since both fall under the same "spot analysis" umbrella even
   though they're different analyses; its reusable CSVs (importance,
   predictions) go to `figs/data/` (`DATA_OUTPUT_DIR`); its run log goes to
   `figs/diagnostics/` (`DIAGNOSTICS_DIR`).

### Utilities
- **`xrf_display.py`** — visualizes an element-map TIFF (or an
  element-ratio map) with the grain mask overlaid; used for a quick check
  before or after registration. `GRAIN_IDS = None` (default) renders every
  grain with a maps folder and a mask in one run. Contrast range is set
  from a saturation + MAD-outlier check (matching `kyanite_figures.py`'s
  default outlier convention) rather than a fixed percentile, so a few
  extreme pixels can't wash out internal zoning. Saves rendered PNGs to
  `figs/map_renders/`.
- **`sum_epma_maps.py`** — sums two or more element-line maps into one TIFF
  (e.g. `Zr_La` + `Zr_Lb`, when a single line does not capture the full
  signal).
- **`kyanite.sh`** — example SLURM batch script for running a Python step
  (default `kyanite_rf_shap.py`, the compute-heavy step) non-interactively on
  an HPC cluster; sources a dedicated venv before invoking Python (see
  Requirements above).

---

## Walkthrough: taking one new grain end to end

1. If the raw files aren't already named per convention, write a manifest
   and run `onboard_dataset.py` (dry run first).
2. If maps come from a synchrotron h5, run `xrf_h5_to_tiff.py` to populate
   `inputs/maps/<grain_id>/`.
3. Run `CL_EPMA_registration.m`: pick control points, choose a mask method,
   and review the RMSE and shift-sensitivity output before using the r
   values. This is the only step mandatory for every grain.
4. Optionally clean up the mask (`CL_mask_edit.m`), draw regions or
   texture domains (`CL_region_extraction.m`), or build local-regression
   maps (`CL_local_regression_map.py`).
5. Generate figures from the pixel CSV (`kyanite_figures.py`), run PCA
   (`kyanite_pca.py`), and/or fit Random Forest/SHAP (`kyanite_rf_shap.py`,
   then `kyanite_rf_shap_plots.py` for the figures) to obtain a quantitative
   estimate of which elements predict CL.
6. If XANES spectra are also available for spots on this grain, run that
   track in parallel (Step 8 above) and combine with the rest at the
   `kyanite_spot_analysis.py`/`xanes_rf_classifier.py` stage.

## Color conventions

Every figure-generating script draws from one canonical color spec, so the
same element, region, or XANES class renders in the same color no matter
which grain or script produced the figure — this is what makes it possible
to flip between two grains' figures and compare them directly, rather than
each grain's figures using whatever colors happened to fall out that run.
Python scripts import the spec from `kyanite_palette.py`; the MATLAB
scripts carry local functions with the identical values (MATLAB can't
import a Python module). See `CLAUDE.md`'s "Color conventions" section for
the full breakdown — briefly: a fixed color per core element (Cr/Fe/Ti/V/Mn),
a fixed color per XANES class (Type 1/2/3), regions colored deterministically
by sorted name (there's no fixed region vocabulary — today's names are
generic `roi_1`/`roi_2`/...), and one canonical diverging/sequential
colormap for signed and continuous-intensity quantities respectively.

## Common pitfalls

- **`grain_id` must match exactly across all scripts.** It is a bare
  string used as a filename prefix, not derived from metadata. A typo
  produces a second, disconnected grain rather than an error.
- **Normalization settings must match across scripts for the same grain.**
  `NORMALIZE_BY_CLOCK`/`NORMALIZE_BY_I0` and `normalize_epma`/`NORMALIZE_EPMA`
  (in `CL_EPMA_registration.m`/`CL_mask_edit.m`/`CL_local_regression_map.py`)
  must agree with the values used in Step 2, or downstream comparisons are
  invalid.
- **`classification_mode` and default mode in `CL_region_extraction.m`
  write the same `_region_pixel_data.csv` filename.** Running one mode
  after the other for the same grain overwrites the file.
- **XANES auto-classification is disabled by default.** Manual
  classification of the pre-edge grids is required; `CLASSIFY = True` in
  `xanes_plot.py` is not a reliable substitute.
- **Don't extract two point families (e.g. generic `spotNN` and dedicated
  `-FeN`) for the same grain in one `xrf_h5_extract_spots.py` run.** Spot
  numbers come from the area name's trailing digits regardless of family, so
  both would start numbering from 1 and collide. No reserved
  numbering/namespacing scheme exists yet for that case.
- **Mask edits and region/domain extraction do not re-register the
  image.** If registration itself is inaccurate, rerun Step 2 rather than
  correcting for it downstream.
- **Figure/output location is independent of `CSV_INPUT`.** `kyanite_figures.py`,
  `kyanite_pca.py`, `kyanite_rf_shap.py`, `kyanite_rf_shap_plots.py`, and
  `kyanite_sample_size_convergence.py` each have an explicit output-directory
  parameter (`WHOLE_GRAIN_OUTPUT_DIR`/`REGION_OUTPUT_DIR`/`DATA_OUTPUT_DIR`/
  `OUTPUT_DIR`/`RF_OUTPUT_DIR`/`SHAP_OUTPUT_DIR`, depending on the script)
  rather than saving next to whatever CSV they read — pointing `CSV_INPUT` at
  `figs/data/` (where the pixel-data CSVs live) is always safe and never dumps
  figures in among the reusable data files.
- **`kyanite_rf_shap_plots.py`'s `LOG_TRANSFORM` must match whatever
  `kyanite_rf_shap.py` used for that data.** It only controls axis labeling
  (`(log10)` suffix) here — the values themselves were already transformed
  (or not) upstream, so a mismatch mislabels the dependence-plot axes rather
  than erroring.
- **`CL_EPMA_registration.m`/`CL_mask_edit.m` no longer generate a CL-vs-element
  scatter figure.** They still compute and log Pearson r/linear-fit numbers as
  an immediate registration-quality check, but all exploratory figures
  (scatter/violin/contour/heatmap/corrmatrix) come from `kyanite_figures.py`.
