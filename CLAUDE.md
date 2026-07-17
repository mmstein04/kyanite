# Kyanite CL-EPMA Registration Project

## Project context
This project registers cathodoluminescence (CL) images of kyanite grains to EPMA
and synchrotron XRF element maps to study which trace elements control kyanite CL
emission. The core workflow is: export element maps → register CL image → mask the
grain → extract per-pixel CL vs. chemistry data → scatter/violin/box plots.

## Science context
- Known CL activators in kyanite: Cr³⁺, Ti
- Known CL quencher: Fe²⁺ (quenches above ~3200 µg/g)
- Goal: quantify per-pixel correlations between CL intensity and trace element
  concentrations, using both EPMA and synchrotron XRF maps

## Codebase overview

### Main workflow scripts (run in order)

| Script | Language | Purpose |
|---|---|---|
| `onboard_dataset.py` | Python | Optional first step for a new (e.g. external collaborator's) dataset: stage a grain's arbitrarily-named raw files (CL image, element-map TIFFs, XANES classification CSV, XRF HDF5) into this project's exact file/folder conventions, driven by a per-grain YAML manifest |
| `xanes_classification_split.py` | Python | Split a combined hand-classification CSV (`GrainID`, `Spot`, `Class`) into one `<grain_id>_pre_edge_classification.csv` per grain, matching the naming `xrf_h5_extract_spots.py` expects |
| `xanes_plot.py` | Python | Plot each spot's XANES pre-edge µ(E) spectrum for visual Type 1/2/3 classification, with an optional automatic classifier (relative Fe²⁺/Fe³⁺ peak height) as a starting point |
| `xrf_h5_to_tiff.py` | Python | Extract XRF element maps from a Larch/GSECARS HDF5 file; export as 32-bit float TIFFs + metadata sidecars |
| `xrf_h5_extract_spots.py` | Python | Build a per-spot CSV: pixel + physical coordinates of XANES spot locations (`xrmmap/areas`), mean element concentrations and CL brightness over a small grain-mask-restricted circular zone around each spot, and the joined XANES pre-edge classification |
| `CL_EPMA_registration.m` | MATLAB | Full registration + analysis pipeline (see workflow below) |
| `CL_region_extraction.m` | MATLAB | Draw named sub-grain polygon regions on an already-registered CL image and extract per-pixel CL + element data per region (no re-registration) |
| `CL_mask_edit.m` | MATLAB | After-the-fact touch-up of a grain mask already produced by `CL_EPMA_registration.m` (e.g. an inclusion was masked-in, or real grain was masked-out weeks earlier) — draw add/remove polygons on the already-registered CL image, then re-derive pixel data and downstream figures under the corrected mask |
| `CL_local_regression_map.py` | Python | Slide a circular window across an already-registered grain and run a per-pixel CL-vs-element linear regression inside each window, producing continuous slope/R maps that complement `CL_region_extraction.m`'s fixed polygon regions |
| `kyanite_figures.py` | Python | Standalone figure generation from exported CSV pixel data |
| `kyanite_pca.py` | Python | PCA analysis of CL vs. trace elements from whole-grain CSV pixel data; for region CSVs, instead fits one shared PCA pooled across regions (scree, loadings, PC1/PC2-by-region scatter, biplot) to test whether hand-drawn regions separate in PC space — no per-region PCA |
| `kyanite_rf_shap.py` | Python | Cross-validated Random Forest regression and TreeSHAP importance/interactions of CL vs. trace elements from whole-grain CSV pixel data; fits models and exports CSVs only, no figures — region CSVs are skipped by default, too computationally expensive to run per region (`ANALYZE_REGIONS=True` opts in; region-level analysis is otherwise `kyanite_pca.py`'s cheap pooled region-PCA) |
| `kyanite_rf_shap_plots.py` | Python | Figure generation from `kyanite_rf_shap.py`'s CSV outputs (observed-vs-predicted, permutation/SHAP importance, SHAP interactions, SHAP dependence) — decoupled from model fitting so a figure can be regenerated or restyled without retraining |
| `kyanite_sample_size_convergence.py` | Python | Diagnostic: sweeps RF/SHAP over a range of pixel subsample sizes for one grain to check whether importance estimates have converged below `kyanite_rf_shap.py`'s `MAX_SAMPLES`/`SHAP_SAMPLES`, or would still change with more data |
| `kyanite_spot_analysis.py` | Python | Batch analysis of `<grain_id>_spot_geochemistry.csv` files: XANES class distribution pie-chart grid, pooled CL-vs-element scatter plots colored by class, element-by-class box plots, PCA (PC1/PC2 scatter, scree, loadings, biplot) colored by class, and per-grain labeled spot-location maps on the registered CL image |
| `xanes_rf_classifier.py` | Python | Cross-validated Random Forest classification of XANES pre-edge class (Type 1/2/3) from per-spot trace-element geochemistry, pooled across grains — the classification analog of `kyanite_rf_shap.py` |
| `xrf_display.py` | Python | Visualize XRF element-map TIFFs with grain mask overlay and optional element-ratio maps, using this project's shared `SEQUENTIAL_CMAP` and MAD outlier conventions for the display range |
| `sum_epma_maps.py` | Python | Sum two or more element maps into a combined TIFF (e.g. Zr_La + Zr_Lb) |
| `kyanite.sh` | Bash | Example SLURM batch script: activates the project's dedicated Python virtualenv and runs one Python step (default `kyanite_rf_shap.py`, the compute-heavy step) non-interactively on an HPC cluster — adapt the script name and `#SBATCH`/environment variables per job |

### `CL_EPMA_registration.m` workflow
1. Load CL image and auto-discover all EPMA/XRF TIFFs in `epma_dir`
2. Interactively pick control points (`cpselect`) to warp CL onto the EPMA grid
3. Evaluate registration quality (RMSE in pixels and µm)
4. Build binary grain mask — methods: `otsu`, `manual`, `interactive`, `polygon`, `activecontour`
5. Apply mask; extract per-pixel CL and element concentration vectors
6. Compute Pearson r and a linear fit, CL vs. each element, as an immediate
   registration-quality check (exploratory scatter/violin/contour/etc. figures
   are generated downstream by `kyanite_figures.py` from the exported pixel
   data CSV, not here)
7. Shift-sensitivity analysis to quantify alignment-error impact on correlations
8. Write analysis log, save registered CL TIFFs (grayscale 16-bit + original color), mask TIFF, pixel data `.mat` / `.csv`

### `CL_region_extraction.m` workflow
1. Load the registered CL image, grain mask, and EPMA/XRF maps already produced by
   `CL_EPMA_registration.m` for this grain — no warping, control points, or mask
   generation here
2. Interactively draw and name one polygon per region of interest (e.g. core, rim);
   saved regions can be reloaded on rerun
3. Intersect each region with the grain mask (optional, default on)
4. Extract per-pixel CL and element vectors per region into a combined long-format
   table with a `Region` column
5. Save per-region/per-channel summary statistics (`figs/regions/`) and QC figures
   (region boundaries on the CL image and on every element map, `figs/diagnostics/`)

`classification_mode = true` switches to a different workflow: subdividing the WHOLE
grain into non-overlapping CL textural classification domains (e.g. sector,
oscillatory) instead of arbitrary/partial-coverage named ROIs.
- Domain class labels are drawn from a fixed vocabulary (`TEXTURE_CLASSES` +
  `TEXTURE_CLASS_COLORS`, mirroring `kyanite_spot_analysis.py`'s
  `CATEGORY_ORDER`/`CATEGORY_COLORS` pattern), not freeform typed names; multiple
  disjoint polygons can share one class, each getting its own instance id
  (e.g. `sector_1`, `sector_2`)
- Each newly drawn domain auto-clips to whatever grain-mask area is not yet claimed
  by an earlier domain (`poly_mask & grain_mask & ~already_classified_mask`), so
  domains are guaranteed non-overlapping by construction — draw loosely; the last
  domain can just cover whatever remains
- Any grain-mask pixels left over when you stop drawing are auto-bucketed into a
  reserved `Unclassified` class, so output always covers the full grain mask
- `Region` column holds the texture class (for grouping/plotting downstream), a new
  `DomainID` column holds the per-polygon instance id; `<grain_id>_regions.mat` /
  `_region_summary.csv` are replaced by mode-specific `_texture_domains.mat` /
  `_texture_domain_summary.csv` so the two modes never collide on disk for the same
  grain — but `_region_pixel_data.csv`/`.mat` keep their exact default-mode filenames
  in both modes (downstream scripts glob on that literal suffix), so running one mode
  after the other for the same grain overwrites that file (a runtime warning fires
  if the mode differs from what's already on disk)
- Produces an additional full-grain texture class label map
  (`<grain_id>_texture_class_map.tif`, a uint8 index raster in `TEXTURE_CLASSES`
  order + Unclassified, and a matching colored/legend PNG)
- Output folders mirror default mode: reusable data (`_texture_domains.mat`,
  `_region_pixel_data.csv`/`.mat`, `_texture_class_map.tif`) in `figs/data/`;
  QC (`_texture_domains_overlay.png`, `_texture_domains_all_maps_QC.png`) in
  `figs/diagnostics/`; the summary CSV and the class-map PNG in `figs/regions/`

### `CL_mask_edit.m` workflow
1. Load the registered CL image, EPMA/XRF maps, and current grain mask already
   produced by `CL_EPMA_registration.m` for this grain — no control points,
   warping, or from-scratch mask generation here
2. Back up every mask-dependent output file this run is about to overwrite
   (mask TIFF, pixel data, shift-sensitivity/QC PNGs, prior edit
   history) into `mask_edit_backups/<grain_id>_<timestamp>/` before touching
   anything
3. Interactive add/remove loop: draw a polygon, tag it add or remove, see it
   applied immediately with a live preview; `u` undoes the last edit, `d`
   finishes — mirrors the add/remove-by-drawing pattern, but for freeform
   whole-mask touch-ups rather than exhaustive non-overlapping domains like
   `CL_region_extraction.m`'s `classification_mode`
4. Re-applies the same morphological cleanup knobs used at registration time
   (`close_radius_px` / `min_object_px` / `fill_holes`) and overwrites
   `data/<grain_id>_mask.tif`
5. `regenerate_downstream` (default `true`): re-extracts per-pixel CL +
   element vectors under the corrected mask and overwrites
   `data/<grain_id>_pixel_data.csv`/`.mat`, then recomputes the CL-vs-element
   Pearson correlations, shift-sensitivity analysis, and all-maps/mask-check QC
   figures — so every figure/log in `figs/` matches the corrected mask, not just
   the CSV. (Exploratory scatter/violin/contour/etc. figures are regenerated
   downstream by `kyanite_figures.py` from the CSV, not by this script.)
   Registration-quality fields (`RMSE_px`/`RMSE_um`) are carried
   forward unchanged from the prior `data/pixel_data.mat` since no re-registration
   happens here. Set `false` to touch only the mask TIFF + edit history.
6. Saves a cumulative `data/<grain_id>_mask_edits.mat` (every add/remove polygon
   ever applied across runs, for audit/undo-by-inspection), a
   `diagnostics/<grain_id>_mask_edit_diff.png` (before/after boundary + added/removed
   pixel diff), and a `diagnostics/<grain_id>_mask_edit_log.txt` run record

### `onboard_dataset.py` details
- Purpose: let a collaborator with a differently-named/organized dataset feed it into
  this workflow without hand-renaming files. Reads a per-grain YAML manifest (see
  `dataset_manifest.example.yaml`) describing where their raw files actually are and
  how their element-map files map to `<Element>_<Line>`, then copies (or, for the
  multi-GB XRF h5, symlinks) everything into this project's exact naming/folder
  conventions
- Never guesses a mapping it wasn't told: element-map files resolve only via an
  explicit `files` entry or a `filename_pattern` regex that must match the *whole*
  filename (`re.fullmatch`, not `search`) — anything else is skipped with a warning
  rather than silently mis-parsed
- Defaults to `DRY_RUN = True`: prints the full plan (and any warnings/collisions,
  e.g. two source files resolving to the same `<Element>_<Line>`) without touching
  disk; set `DRY_RUN = False` to execute. Existing destination files are skipped
  (not overwritten) unless `OVERWRITE = True`
- For the raw XRF HDF5, also inspects `xrmmap/areas` key names against the
  `spot<N>` convention `xrf_h5_extract_spots.py`'s `NAME_FILTER`/spot-number regex
  expects, and — if it doesn't match — reports which built-in fallback pattern
  (trailing digits, `pt<N>`, `point<N>`) does, so `NAME_FILTER` can be updated
  instead of the h5 file itself being touched
- All operations are copies or symlinks of originals — nothing is moved, renamed, or
  modified in place
- Writes `inputs/<grain_id>_onboarding_log.txt` (skipped in dry-run mode) recording every
  planned source → destination mapping and warning, for audit purposes

### `xrf_h5_to_tiff.py` details
- Data source: `xrmmap/roimap/sum_cor` [rows × cols × n_rois], `xrmmap/roimap/sum_name`
- Scan geometry read from `xrmmap/config/scan` (step sizes, ranges, dwell time)
- Optional normalization: `NORMALIZE_BY_CLOCK` and/or `NORMALIZE_BY_I0` (applied in that order)
- Rows are flipped vertically on export (HDF5 stores bottom-to-top)
- Each TIFF gets a `.txt` sidecar with: source file, ROI index, dimensions, value stats
  (min/max/mean/median/std), normalization, scan step size, range, dwell time, UTC timestamp

### `xrf_h5_extract_spots.py` details
- Data source: `xrmmap/areas/<name>` — boolean point/region masks saved during the XRF
  scan (e.g. XANES spot locations); `xrmmap/positions/pos` for the physical stage
  position at each pixel; `xrmmap/config/scan` for axis identity, range, and per-axis
  step size; `xrmmap/roimap/sum_cor`/`sum_name` for the full element ROI list (kept as
  a live h5py Dataset and only ever sliced in small windows — these files are multi-GB)
- Reports each spot's pixel index in two frames: native HDF5 orientation
  (`row_px_h5`/`col_px_h5`, row 0 = bottom of scan) and row-flipped
  (`row_px_tiff`/`col_px_tiff`, 0-based, and `row_matlab`/`col_matlab`, 1-based) to
  match the TIFFs from `xrf_h5_to_tiff.py` (`np.flipud`) — the same pixel grid used
  by `CL_EPMA_registration.m`/`CL_region_extraction.m` for the registered CL image,
  element maps, and mask
- For each spot, also extracts the mean of every element ROI and of the registered CL
  image (`figs/<GRAIN_ID>_CL_registered.tif`, 16-bit rescaled back to the same [0,1]
  scale used in `pixel_data.csv`) over a small circular zone (`ZONE_RADIUS_UM`,
  built from the h5's true per-axis pixel size — not assumed isotropic), restricted to
  grain-mask pixels (`figs/data/<GRAIN_ID>_mask.tif`, thresholded `> 128`) — mirrors the
  registration pipeline's "region intersected with grain mask" masking semantics, and
  reads elements straight from the h5 rather than the (possibly smaller, curated)
  `inputs/maps/<grain_id>/*.tif` subset, so "all the other elements" are available, not just
  whatever was previously exported
- Joins each spot to its hand-assigned XANES pre-edge class via
  `inputs/xanes_classification/<GRAIN_ID>_pre_edge_classification.csv` (built by
  `xanes_classification_split.py`), matched on spot number
- Missing mask/CL/classification files for `GRAIN_ID`, or zero grain-mask overlap for a
  spot's zone, degrade gracefully (NaN + a printed warning) rather than crashing —
  useful at earlier pipeline stages
- `on_grain` column: `False` when a spot's zone has zero overlap with the grain mask
  (it sampled some other phase, not kyanite) — CL and every element mean are NaN for
  that spot, but `category`/`category_label` (the hand-assigned XANES pre-edge class)
  are left untouched, since oxidation state is a property of whatever phase was
  actually sampled and remains useful data on its own, just not attributable to
  kyanite. `NaN` (indeterminate) only if no grain mask was found for `GRAIN_ID` at all.
  See `kyanite_spot_analysis.py`'s handling below
- `NAME_FILTER` (default `'spot'`) selects which `xrmmap/areas` entries to include;
  set to `None` to include drawn polygon regions too (their shape is discarded —
  extraction is centered on the region's centroid, like a point spot)
- Note: spot numbering in the HDF5 area names doesn't necessarily share a prefix
  with exported XANES spot CSVs or `GRAIN_ID` (e.g. h5 area `LLF6-Area2-spot01` vs.
  grain `LLF6-01`) — everything joins on the trailing spot number + `GRAIN_ID`, not
  the h5 area's own name
- `OUTPUT_CSV` defaults to `figs/data/<GRAIN_ID>_spot_geochemistry.csv` — reusable
  data, read back by both `kyanite_spot_analysis.py` and `xanes_rf_classifier.py`,
  whose `CSV_INPUT` defaults point at the same location

### `kyanite_spot_analysis.py` details
- Input: `<grain_id>_spot_geochemistry.csv` files (see above), one per grain;
  `CSV_INPUT` may be a single file or a directory (globs `*_spot_geochemistry.csv`)
- Three independently toggleable analyses (`ANALYSES` list): `pie` (one combined
  figure, small-multiples grid of per-grain XANES class pies — `'Bad data'`/
  unclassified spots AND off-grain spots (`on_grain = False`) excluded from the pie
  counts entirely, so the pie reflects only this grain's own kyanite class
  distribution; per-grain slice order/coloring stays identical even at zero count
  for a type; a per-grain subtitle reports how many off-grain spots were excluded),
  `scatter` (CL-vs-element, one figure per element, pooling spots from every input
  grain — `'Bad data'`/unclassified spots ARE included here, as grey points/markers,
  for QC context; off-grain spots are absent, but only because their CL/element
  means are already `NaN` from extraction and get dropped by the normal
  CL/element `dropna` — no separate `on_grain` filtering needed here), `map`
  (per-grain spot-location map on the registered CL image, labeled by spot number
  — off-grain spots ARE plotted, at their real location, still colored by XANES
  class, but marked with an `'X'` marker instead of `'o'` so they read as
  off-grain-but-still-useful rather than being silently dropped), `pca` (one PCA
  fit over `PCA_ELEMENTS`, pooling spots from every input grain, producing four
  figures — PC1-vs-PC2 scatter, scree plot, PC1/PC2 loadings bar charts, and a
  PC1-vs-PC2 biplot with loading vectors — all colored by XANES class the same way
  as `scatter` — `'Bad data'`/unclassified spots ARE included as grey points;
  spots missing any `PCA_ELEMENTS` value are dropped, which — same as `scatter` —
  already excludes off-grain spots without any separate filtering)
- `on_grain` (written by `xrf_h5_extract_spots.py`): `False` means the spot's CL/
  element means are `NaN` because it sampled a different phase, not kyanite — its
  `category_label` (XANES pre-edge class / oxidation state) stays intact and
  meaningful for that other phase, just not attributable to kyanite. `on_grain_mask()`
  treats `NaN` (indeterminate — no grain mask at extraction time) and a missing
  `on_grain` column (an older CSV, extracted before this field existed) both as
  on-grain, so nothing regresses for CSVs from before this change
- `SCATTER_ELEMENTS` (default `None`) auto-detects every element column present in
  the union of input files; an element missing from some grains' CSVs (ROI lists
  vary slightly, e.g. LLF6-01 has extra REE lines) is pooled from whichever grains
  do have it, with a warning listing which grains were excluded from that plot
- `PCA_ELEMENTS` — element list the PCA considers, independent of `SCATTER_ELEMENTS`
  (defaults to the same Cr/V/Fe/Ti/Mn set, but chosen deliberately since PCA is
  sensitive to which variables are included); `PCA_LOG_TRANSFORM` log10-transforms
  elements before z-scoring/PCA, matching `kyanite_pca.py`'s convention;
  `PCA_N_PCS_SCREE` caps how many PCs the scree plot shows (`None` = all);
  `PCA_LOADING_THRESHOLD` highlights `|loading| >=` this value on the loadings bars
- `PCA_CLUSTER_OUTLINES` draws a convex-hull outline (filled at `PCA_CLUSTER_ALPHA`)
  around each class's points on the PC1-vs-PC2 scatter, so the footprint each class
  occupies in PC space is easy to compare; `PCA_CLUSTER_CLASSES` (default `None` =
  `CATEGORY_ORDER`) controls which classes get one — `'Bad data'`/unclassified is
  excluded by default since it isn't a real class to contour
- The PCA biplot fixes its axes to the PC1/PC2 score cloud's own extent before
  drawing loading-vector arrows, then scales all vectors by one shared factor —
  drawing arrows first would let matplotlib autoscale around the (unit-norm,
  disproportionately long) loadings and shrink the score cloud to a sliver
- `CATEGORY_COLORS` reuses `xanes_plot.py`'s hex values, re-keyed to this file's
  real `category_label` strings (`'Bad data'`, not `'Ambiguous'`)
- Spot maps plot directly at `row_px_tiff`/`col_px_tiff` on
  `imshow(cl_img, cmap='gray', origin='upper')` with no additional flip — this
  already matches the project's row-0-at-top convention

## File conventions
- Everything the core pipeline reads as input lives under `inputs/`, separate from
  `figs/` (all script outputs): `inputs/xrf/<grain_id>_xrf.h5` (raw synchrotron XRF
  scans), `inputs/cl/<grain_id>_CL_color.<ext>` (raw CL micrograph, and any
  alternative `mask_image_file` for `CL_EPMA_registration.m`),
  `inputs/xanes/<grain_id>_spotNN.csv` (raw per-spot XANES spectra),
  `inputs/xanes_classification/<grain_id>_pre_edge_classification.csv` (hand-assigned
  pre-edge class labels). `inputs/maps/<grain_id>/*.tif` is the one exception —
  it's generated by `xrf_h5_to_tiff.py` from `inputs/xrf/`, not raw — but every
  downstream script (`CL_EPMA_registration.m` and friends) only ever reads it as
  `epma_dir`, never writes to it, so it lives alongside the rest of `inputs/`
  rather than in `figs/`
- Output filenames: `<grain_id>_<Element>_<Line>.tif`  (e.g. `NA-CM-G12B7-02_Fe_Ka.tif`)
- Metadata sidecars: same base name, `.txt` extension
- Registered CL (grayscale, 16-bit): `<grain_id>_CL_registered.tif`
- Registered CL (original color, native bit depth): `<grain_id>_CL_registered_color.tif`
- Figures saved to `figs/`, element maps to `inputs/maps/<grain_id>/`
- Not-for-publishing sanity/alignment-check and run-metadata outputs from
  `CL_EPMA_registration.m` (and re-derived by `CL_mask_edit.m`, where
  applicable) go in `figs/diagnostics/` rather than directly in `figs/`:
  `<grain_id>_analysis_log.txt`, `<grain_id>_all_maps_QC.png`,
  `<grain_id>_mask_image_registration.png`, `<grain_id>_registration_overlay.png`,
  `<grain_id>_shift_sensitivity.png`, `<grain_id>_mask_check.png`,
  `<grain_id>_mask_edit_diff.png`, `<grain_id>_mask_edit_log.txt`. Same idea
  for `CL_region_extraction.m`'s region-mode QC: `<grain_id>_region_analysis_log.txt`,
  `<grain_id>_regions_overlay.png`, `<grain_id>_regions_all_maps_QC.png` (or the
  `_texture_domains_*` equivalents in `classification_mode`). Same idea again for
  `CL_local_regression_map.py`: `<grain_id>_local_regression_analysis_log.txt`,
  `<grain_id>_local_regression_slope_QC.png`, `<grain_id>_local_regression_R_QC.png`,
  `<grain_id>_local_regression_n_map.png`. Same idea again for `kyanite_pca.py`
  (`<label>_pca_log.txt` whole-grain, `<grain_id>_regions_pca_log.txt` region-mode),
  `kyanite_rf_shap.py` (`<grain_id>_rf_shap_log.txt`), and `xanes_rf_classifier.py`
  (`<OUTPUT_LABEL>_rf_classifier_log.txt`) — their run logs live in
  `figs/diagnostics/` (`DIAGNOSTICS_DIR`) even though their figures/CSVs go
  elsewhere (`figs/pca/`+`figs/regions/`, `figs/data/`, and
  `figs/spot_analysis/`+`figs/data/`, respectively).
  `kyanite_figures.py`'s own outlier-exclusion
  QC (see its Key Parameters entry) follows the same idea: `<grain_id>_<element>_
  outlier_exclusion_QC.png` in `figs/diagnostics/`, not alongside its analysis figures
- Whole-grain PCA figures from `kyanite_pca.py` go to `WHOLE_GRAIN_OUTPUT_DIR`
  (default `figs/pca/`); its region-PCA figures (RF/SHAP has no region-mode
  output) plus `kyanite_figures.py`'s region-mode figures and
  `CL_region_extraction.m`'s own region-summary/texture-class-map outputs
  share `REGION_OUTPUT_DIR`/`output_dir` (default `figs/regions/`).
  `kyanite_rf_shap_plots.py`'s figures split by analysis: RF figures to
  `RF_OUTPUT_DIR` (default `figs/rf/`), SHAP figures to `SHAP_OUTPUT_DIR`
  (default `figs/shap/`) — kept apart even though both are read from the same
  `figs/data/` CSVs, since they're different analyses. All of these
  are kept separate from `figs/data/` and `figs/diagnostics/` — output
  location for each Python script is an explicit parameter, independent of
  wherever `CSV_INPUT` happens to point. Same principle for the other
  analysis-family folders: `kyanite_figures.py`'s own whole-grain figures →
  `figs/whole_grain/` (unchanged; only `kyanite_pca.py`'s moved).
  `kyanite_sample_size_convergence.py` is the one exception: it's itself a
  diagnostic (not a pipeline step), so it has no dedicated analysis-family
  folder at all — its raw sweep CSV, convergence figures, and log all go to
  `figs/diagnostics/` (`DIAGNOSTICS_DIR`).
  `CL_local_regression_map.py`'s one true result figure
  (`<grain_id>_local_regression_R_Cr_vs_CL.png`) → `figs/local_regression/`,
  `kyanite_spot_analysis.py` and `xanes_rf_classifier.py` → `figs/spot_analysis/`
  (shared — both pool the same per-spot CSVs and fall under the same
  "spot analysis" umbrella, even though they're two different analyses),
  `xanes_plot.py` → `figs/xanes/`
- Reusable data files read back in by other scripts go in `figs/data/`
  rather than directly in `figs/`: grain mask (`<grain_id>_mask.tif`), pixel
  data exports (`<grain_id>_pixel_data.csv` and `.mat`), control-point MATs
  (`<grain_id>_controlpoints.mat`, `<grain_id>_mask_image_controlpoints.mat`),
  `CL_mask_edit.m`'s cumulative edit history (`<grain_id>_mask_edits.mat`),
  region polygons (`<grain_id>_regions.mat`), region pixel data exports
  (`<grain_id>_region_pixel_data.csv` and `.mat`, long-format, `Region`
  column), `CL_local_regression_map.py`'s slope/R/n maps (`<grain_id>_local_regression.npz`)
  and long-format table (`<grain_id>_local_regression_pixel_data.csv`), and
  per-spot geochemistry exports (`<grain_id>_spot_geochemistry.csv`, read by both
  `kyanite_spot_analysis.py` and `xanes_rf_classifier.py`) — whole-grain and region
  pixel-data CSVs are colocated here so `kyanite_figures.py`/`kyanite_pca.py`
  can point `CSV_INPUT` at one directory and pick up both (routed by the presence
  of a `Region` column, not by folder); `kyanite_rf_shap.py` points `CSV_INPUT` at
  the same directory but only ever processes the whole-grain ones, skipping region
  CSVs. `kyanite_pca.py`'s own reusable tables (`<label>_pca_variance.csv`,
  `<label>_pca_loadings.csv`, and — region CSVs only —
  `<label>_scores.csv`/`<label>_centroid_distances.csv`), `kyanite_rf_shap.py`'s
  CSVs (`<grain_id>_rf_importance.csv`/`_rf_predictions.csv`/`_shap_importance.csv`/
  `_shap_values.csv`/`_shap_interactions.csv`), and `xanes_rf_classifier.py`'s
  CSVs (`<OUTPUT_LABEL>_rf_classifier_importance.csv`/`_predictions.csv`) are
  likewise written here (`DATA_OUTPUT_DIR`/`OUTPUT_DIR`/`DATA_OUTPUT_DIR`
  respectively), since `kyanite_rf_shap_plots.py` (for the first two) reads
  them back — same "reusable data lives in figs/data/" rule as everything
  else in this list, just produced by an
  analysis script rather than a registration/extraction one
- Region summary stats: `<grain_id>_region_summary.csv` (in `figs/regions/`,
  alongside that grain's region figures — not reloaded by any script, so it
  doesn't live in `figs/data/`)
- Region analysis log: `<grain_id>_region_analysis_log.txt` (in `figs/diagnostics/`)
- Texture classification mode (`classification_mode = true` in `CL_region_extraction.m`)
  outputs, per grain: `<grain_id>_texture_domains.mat` (domain polygons + classes,
  reusable), `<grain_id>_region_pixel_data.csv`/`.mat` (same filename as default mode —
  `Region` = texture class, `DomainID` = per-polygon instance id),
  `<grain_id>_texture_domain_summary.csv`, `<grain_id>_texture_domains_overlay.png`,
  `<grain_id>_texture_domains_all_maps_QC.png`, `<grain_id>_texture_class_map.tif`
  (uint8 class-index raster) and `.png` (colored + legend)
- Mask edit outputs: `data/<grain_id>_mask_edits.mat` (cumulative add/remove edit
  history), `diagnostics/<grain_id>_mask_edit_diff.png` (before/after boundary + diff),
  `diagnostics/<grain_id>_mask_edit_log.txt` (per-run record); pre-edit copies of every
  file `CL_mask_edit.m` is about to overwrite are saved to
  `figs/mask_edit_backups/<grain_id>_<timestamp>/` before each run
- Local regression outputs (`CL_local_regression_map.py`): `data/<grain_id>_local_regression.npz`,
  `data/<grain_id>_local_regression_pixel_data.csv`, `diagnostics/<grain_id>_local_regression_analysis_log.txt`,
  `diagnostics/<grain_id>_local_regression_{slope,R}_QC.png`, `diagnostics/<grain_id>_local_regression_n_map.png`,
  and the one true result figure, `<grain_id>_local_regression_R_Cr_vs_CL.png`, directly in `figs/local_regression/`
- Spot coordinate exports: `<grain_id>_spot_coordinates.csv`
- Spot geochemistry/CL/XANES-class exports: `figs/data/<grain_id>_spot_geochemistry.csv`
  (reusable — read back by `kyanite_spot_analysis.py` and `xanes_rf_classifier.py`)
- Spot analysis figures saved to `figs/spot_analysis/`: `xanes_class_pie_grid.png`,
  `CL_vs_<element>_scatter.png`, `<element>_by_class_boxplot.png`,
  `pca_pc1_pc2_scatter.png`, `pca_scree.png`, `pca_loadings_pc1_pc2.png`,
  `pca_biplot.png`, `<grain_id>_spot_map.png`
- XANES pre-edge classification figures saved to `figs/xanes/` (`xanes_plot.py`):
  `<grain_id>_pre_edge_grid.png`, `<grain_id>_pre_edge_overlay.png`, `<grain_id>_xanes_overlay.png`
- XANES RF classifier outputs (`xanes_rf_classifier.py`, all prefixed with
  `OUTPUT_LABEL`, default `all_grains_`), split the same way as
  `kyanite_pca.py`/`kyanite_rf_shap.py`: figures
  (`_rf_classifier_importance.png`, `_rf_classifier_confusion_matrix.png`) in
  `figs/spot_analysis/` (`OUT_DIR` — shared with `kyanite_spot_analysis.py`,
  no dedicated folder of its own); reusable CSVs
  (`_rf_classifier_importance.csv`, `_rf_classifier_predictions.csv`) in
  `figs/data/` (`DATA_OUTPUT_DIR`); the run log
  (`_rf_classifier_log.txt`) in `figs/diagnostics/` (`DIAGNOSTICS_DIR`)
- Sample-size convergence diagnostic outputs saved to `figs/diagnostics/`
  (`kyanite_sample_size_convergence.py`'s `DIAGNOSTICS_DIR` — no dedicated
  analysis-family folder, since the whole script is itself a diagnostic):
  `<label>_convergence_raw.csv`, `<label>_convergence_{rmse,r2,
  perm_importance,shap_importance}.png`, `<label>_convergence_log.txt`
- Onboarding manifest (per new dataset, hand-written from `dataset_manifest.example.yaml`):
  `dataset_manifest.yaml` (or any name — set in `onboard_dataset.py`'s `MANIFEST_FILE`)
- Onboarding audit log: `inputs/<grain_id>_onboarding_log.txt` (written by `onboard_dataset.py`,
  only on a non-dry-run execution)

## Color conventions
One canonical spec per color role, so the same category/element/quantity
renders in the same color regardless of which script or which grain
produced the figure — this is what actually makes figures comparable
across grains, not just individually well-designed. Python scripts import
these from `kyanite_palette.py`; MATLAB scripts (`CL_EPMA_registration.m`,
`CL_mask_edit.m`, `CL_region_extraction.m`) can't import a Python module,
so they carry local functions with the identical values hand-copied in —
**if you change a value here, update both sides.**

- **House palette** — general-purpose roles, not tied to any specific
  element/region/category: `BLUE = '#3B9BDD'` (main data cloud/bar), `ORANG
  = '#D85B30'` (fit line, highlight, or "above threshold" marker).
- **Element → color** (`kyanite_palette.ELEMENT_COLORS` /
  `element_colors()`; MATLAB: `element_line_colors()` in
  `CL_EPMA_registration.m`/`CL_mask_edit.m`) — Okabe-Ito colorblind-safe
  set, fixed for the 5 elements that recur across every grain's
  whole-grain/region maps: `Cr_Ka #E69F00`, `Fe_Ka #56B4E9`, `Ti_Ka
  #009E73`, `V_Ka #F0E442`, `Mn_Ka #CC79A7`. Any other element name falls
  back to two extra Okabe-Ito colors (`#0072B2`, `#D55E00`), assigned by
  sorted order — stable within one call/figure, not individually curated.
  Applies wherever multiple elements are overlaid by color in one set of
  axes (currently: the shift-sensitivity plots, and
  `kyanite_sample_size_convergence.py`'s per-element convergence curves) —
  most figures instead facet by subplot/filename per element and use the
  house palette for that single element's data/fit, which doesn't need this.
  PCA loadings bar charts (`kyanite_pca.py`'s `plot_loadings`,
  `kyanite_spot_analysis.py`'s `plot_pca_loadings`) are a related but
  distinct case: bar *fill* is the element's fixed color (so the same
  element reads the same color across every PC/grain/script) and bar
  *border* separately encodes significance — a black outline (`lw=1.5`) if
  `|loading|` clears the threshold, no border (`edgecolor='none'`)
  otherwise — so element identity and significance are both visible at
  once, on two different visual channels rather than one overloaded color.
- **Region name → color** (`kyanite_palette.REGION_PALETTE` /
  `region_colors()`; MATLAB: `region_name_colors()` in
  `CL_region_extraction.m`) — region names are freeform and per-grain
  (today: generic `roi_1`/`roi_2`/..., not semantic labels), so there's no
  fixed vocabulary to hardcode. Colors are instead assigned deterministically
  by *sorted name* into a fixed 10-color qualitative palette (matplotlib's
  `tab10`: `#1f77b4, #ff7f0e, #2ca02c, #d62728, #9467bd, #8c564b, #e377c2,
  #7f7f7f, #bcbd22, #17becf`, repeating past 10 distinct regions) — same
  name always gets the same color in every script and every grain; if
  regions are drawn in a consistent order across grains, same position
  ends up the same color too. Used by `CL_region_extraction.m`'s freeform-mode
  boundary overlay/QC figure, `kyanite_figures.py`'s region-highlight
  figure, and `kyanite_pca.py`'s region-PCA scatter/biplot/hulls.
  (`classification_mode`'s texture domains are a separate, already-fixed
  vocabulary — `TEXTURE_CLASS_COLORS` — and don't use this.)
- **XANES pre-edge class → color** (`kyanite_palette.CATEGORY_COLORS`/
  `CATEGORY_ORDER`) — `Type 1 #D85B30`, `Type 2 #4C9F70`, `Type 3 #7A5195`,
  plus a grey (`GREY = '#999999'`) fallback for the unclassified/QC-failed
  case. `kyanite_spot_analysis.py` and `xanes_rf_classifier.py` key the grey
  fallback `'Bad data'`; `xanes_plot.py`'s own optional auto-classifier keys
  it `'Ambiguous'` instead (different vocabulary, same color, by design —
  each file builds its local `CATEGORY_COLORS` by spreading the shared dict
  and adding its own grey-fallback key).
- **Diverging colormap** (`kyanite_palette.DIVERGING_CMAP = 'RdBu_r'`) —
  every signed, zero-centered quantity: correlation matrices
  (`kyanite_figures.py`'s `corrmatrix`), local-regression slope/R maps
  (`CL_local_regression_map.py`'s slope/R map grids and its Cr-vs-CL figure) —
  now imported directly rather than hand-rolled, since (as of the Python port)
  no script in this project needs a MATLAB-compatible stand-in for it.
- **Sequential colormap** (`kyanite_palette.SEQUENTIAL_CMAP = 'inferno'`) —
  every continuous-intensity role: KDE density (`kyanite_figures.py`'s
  `heatmap`), SHAP interaction magnitude and dependence-plot coloring
  (`kyanite_rf_shap_plots.py`), `xrf_display.py`'s element/ratio map display
  range, and `CL_local_regression_map.py`'s window-coverage (n) map.

## Key parameters (set per-grain at top of each script)

**`onboard_dataset.py`**
- `MANIFEST_FILE` — path to the per-grain YAML manifest (see `dataset_manifest.example.yaml`)
- `PROJECT_ROOT` — root this project's conventional paths (`figs/`, `inputs/`)
  are resolved relative to
- `DRY_RUN` — `True` (default): print the plan and warnings only; `False`: execute it
- `OVERWRITE` — `False` (default): skip any destination file that already exists

**`CL_EPMA_registration.m`**
- `grain_id` — sample name string, used in all output filenames
- `input_dir` — folder holding the grain's raw CL image (and, if used,
  `mask_image_file`); default `inputs/cl`
- `cl_filename` — CL image (supports .tif, .png, .bmp, any `imread`-compatible format)
- `epma_dir` — folder of element map TIFFs; all `*.tif` files auto-discovered;
  default `inputs/maps/<grain_id>`
- `epma_ref_file` — reference map for control point picking (choose highest contrast)
- `epma_pixel_um_from_sidecar` (default `true`): pixel size is read from
  `xrf_h5_to_tiff.py`'s metadata sidecar (`<grain_id>_<el>_Ka.txt`'s
  `step_size_pos1_um`) for whichever auto-discovered `epma_dir` map has one,
  same mechanism as `xrf_display.py`/`CL_local_regression_map.py`.
  `epma_pixel_um` is only the fallback (with a warning) if no sidecar is
  found/parseable for this grain
- `mask_method` — grain segmentation method
- `pct_lo_cut` / `pct_hi_cut` — outlier percentile bounds for the Pearson r/fit computation

**`CL_region_extraction.m`**
- `grain_id` — must match a grain already processed by `CL_EPMA_registration.m`
- `input_dir` — folder holding that grain's registered CL TIFFs + mask TIFF
- `epma_dir` — same EPMA/XRF map folder used during registration
- `epma_pixel_um_from_sidecar` (default `true`): same sidecar auto-detection as
  `CL_EPMA_registration.m` — keeps this in sync with the value registration used
  for this grain without hand-copying it; `epma_pixel_um` is only the fallback
  (with a warning) if no sidecar is found/parseable
- `restrict_to_grain_mask` — intersect each drawn region with the grain mask (default `true`)
- `normalize_epma` — match the value used in `CL_EPMA_registration.m` for comparable values
- `classification_mode` — `false` (default): freeform/possibly-overlapping/partial-coverage
  named ROIs, unchanged from prior behavior. `true`: exhaustive, non-overlapping CL
  textural classification domains (requires `restrict_to_grain_mask = true`)
- `TEXTURE_CLASSES` / `TEXTURE_CLASS_COLORS` — fixed vocabulary + color map for
  `classification_mode` (never include `'Unclassified'` — it's reserved, auto-assigned
  to any grain-mask pixels left over when you stop drawing)

**`CL_mask_edit.m`**
- `grain_id` — must match a grain already processed by `CL_EPMA_registration.m`
- `input_dir` — folder holding that grain's registered CL TIFFs, mask TIFF, and
  pixel data (also used as `output_dir`; edits happen in place)
- `epma_dir` — same EPMA/XRF map folder used during registration
- `epma_pixel_um_from_sidecar` (default `true`): same sidecar auto-detection as
  `CL_EPMA_registration.m`/`CL_region_extraction.m` — keeps this in sync with
  the value registration used for this grain without hand-copying it;
  `epma_pixel_um` is only the fallback (with a warning) if no sidecar is
  found/parseable
- `normalize_epma`, `pct_lo_cut`/`pct_hi_cut`, `shift_range` — must
  match the values used in the grain's original `CL_EPMA_registration.m` run, or
  re-derived pixel data/plots won't be comparable to before the edit
- `close_radius_px` / `min_object_px` / `fill_holes` — post-edit mask cleanup,
  same knobs as `CL_EPMA_registration.m`'s `SECTION 5`
- `regenerate_downstream` — `true` (default): re-extract pixel data and
  recompute Pearson correlations/shift-sensitivity/QC figures after saving the
  edited mask; `false`: touch only the mask TIFF + edit history

**`CL_local_regression_map.py`**
- `GRAIN_IDS` — single string, list, or `None` (default) to auto-discover and run every grain with
  a registered CL image, a mask, and a maps folder (skipping, with a warning, any that's missing
  one); a grain that fails partway (missing input, size mismatch, etc.) is skipped with a warning
  rather than aborting the batch
- `INPUT_DIR` — folder holding each grain's registered CL TIFFs + mask TIFF (default `figs/`);
  `DATA_DIR`/`DIAGNOSTICS_DIR`/`OUTPUT_DIR` — where reusable data, QC figures/log, and the one
  true result figure are saved respectively (defaults `figs/data/`, `figs/diagnostics/`,
  `figs/local_regression/`); `MAPS_DIR` — base EPMA/XRF map folder, same one used during
  registration (per-grain maps live in `MAPS_DIR/<grain_id>/`)
- `WINDOW_RADIUS_UM` — physical radius of the circular regression window; `MIN_WINDOW_PX`
  (derived per grain, half the disk area by default) — minimum in-mask pixels required inside a
  window before a regression is computed there
- `PIXEL_UM_FROM_SIDECAR` (default `True`): each grain's µm/px is read from
  `xrf_h5_to_tiff.py`'s metadata sidecar (same mechanism as `xrf_display.py`), not a single
  hardcoded constant — grains in this project are imaged at different resolutions (e.g. 1.0 vs.
  2.0 µm/px), so a fixed `EPMA_PIXEL_UM` would silently make `WINDOW_RADIUS_UM`'s actual pixel
  radius wrong for whichever grains don't match it. `EPMA_PIXEL_UM` is only the fallback (with a
  warning) if no sidecar is found/parseable for a given grain
- `NORMALIZE_EPMA` — match the value used in `CL_EPMA_registration.m` for comparable values
- `USE_COLOR_DISPLAY` — use the color registered CL as the Cr comparison figure's background
  (falls back to grayscale, with a warning, if not found)
- Uses direct (not FFT-based) 2-D convolution (`scipy.signal.convolve2d`) to compute the moving-
  window regression sums, matching MATLAB's `conv2` exactly — verified against the original
  script's output for `RH-XA-57081P-07` (identical `n_map`, slope/R maps matching to ~1e-10).
  This also surfaced a latent bug in `CL_local_regression_map.m` worth knowing about: its R-map
  clipping line (`r_e = max(min(r_e, 1), -1)`) relies on MATLAB's `max`/`min` returning the
  non-NaN operand when one input is NaN, which silently resurrects every masked-out/invalid pixel
  in `r_maps` as exactly `1.0` instead of leaving it `NaN` — visible as a spurious red halo in the
  old `_local_regression_R_QC.png`/`_local_regression_R_Cr_vs_CL.png` figures near the grain-mask
  boundary (within one window radius of the edge). The Python port uses `np.clip`, which
  correctly preserves `NaN`, so this halo is gone in the new figures. The pixel-data CSV was never
  affected (its rows are already filtered on slope validity, which this bug didn't touch) — only
  the raw `r_maps` array and the QC/Cr-comparison figures from prior runs of the MATLAB script
  carry the artifact.
- Output: `figs/data/<grain_id>_local_regression.npz` (slope/R/n maps + metadata — `.npz`, not
  `.mat`, since nothing reads this file back in and `.npz` is numpy's native equivalent);
  `RowIdx`/`ColIdx` in `_local_regression_pixel_data.csv` are 0-based (numpy convention), unlike
  the original script's 1-based indices — nothing downstream joins on them

**`xrf_h5_to_tiff.py`**
- `H5_FILE` — raw XRF HDF5 file, default location `inputs/xrf/`
- `OUTPUT_DIR`, `GRAIN_ID`
- `ELEMENTS` — list of ROI names to export (or `None` for all)
- `NORMALIZE_BY_CLOCK`, `NORMALIZE_BY_I0`

**`xrf_h5_extract_spots.py`**
- `H5_DIR` — folder of raw XRF HDF5 files, one per grain (`H5_DIR/<grain_id>_xrf.h5`);
  default `inputs/xrf/`
- `GRAIN_IDS` — single string, list, or `None` (default discovers every grain with an h5
  file in `H5_DIR` — the only hard requirement for extraction; a missing mask/CL image/
  classification CSV for a given grain still degrades gracefully, NaN + a warning, same
  as ever). A grain that fails partway (e.g. an unreadable/malformed h5) is skipped with
  a warning rather than aborting the batch. Drives the grain's `figs/`/
  `inputs/xanes_classification/` filenames
- `FIGS_DIR`, `CLASSIFICATION_DIR` — where to look up the mask/CL TIFFs and classification CSV
  (default `inputs/xanes_classification/`)
- `OUTPUT_DIR` — where `<grain_id>_spot_geochemistry.csv` is written; default `figs/data/`.
  `SAVE_CSV` — `False` for console-only (no files written)
- `NAME_FILTER` — regex to select which `xrmmap/areas` entries count as spots (default `'spot'`)
- `ZONE_RADIUS_UM` — physical radius (µm) of the circular sampling zone around each spot
- `ELEMENTS` / `EXCLUDE_ROIS` — element ROIs to extract (`None` = all except known scaler channels)
- `NORMALIZE_BY_CLOCK`, `NORMALIZE_BY_I0` — should match whatever was used for this grain's
  exported maps, for comparable units

**`xrf_display.py`**
- `MAPS_DIR`, `MASK_DIR` — where to find `<grain_id>_<el>_Ka.tif` (default `inputs/maps/`) and
  `<grain_id>_mask.tif` (default `figs/data/`); `OUTPUT_DIR` — where rendered PNGs are saved
  (default `figs/map_renders/`), independent of the above
- `GRAIN_IDS` — single string, list, or `None` (default) to auto-discover and render every grain
  with both a maps folder (`MAPS_DIR/<grain_id>/`) and a mask (skipping, with a warning, any maps
  folder that has no matching mask); `ELEMENTS` — bare element symbols (e.g. `'Cr'`, mapped to
  `<grain_id>_Cr_Ka.tif`). A grain missing one of `ELEMENTS`/`RATIOS`' component maps has just
  that element/ratio skipped (with a warning), not the whole batch
- `RATIOS` — list of `(numerator, denominator)` element symbol pairs to render as ratio maps
  (e.g. `('Cr', 'V')`); `[]` disables ratio maps
- `CMAP` — defaults to `kyanite_palette.SEQUENTIAL_CMAP` (`'inferno'`)
- `SATURATION_FILTER`/`SATURATION_BAND_FRAC`/`SATURATION_MIN_FRAC`/`SATURATION_MIN_COUNT` and
  `MAD_K_LO`/`MAD_K_HI` (default `None`/`4`) — same saturation + MAD outlier logic as
  `kyanite_figures.py`'s `OUTLIER_METHOD='mad'` default, but used only to set the imshow
  `vmin`/`vmax` display range rather than to exclude pixels — every in-mask pixel is still drawn,
  clamped to that range, so a few extreme pixels can't pin the whole color scale and wash out
  internal zoning. `RATIO_MAD_K_LO`/`RATIO_MAD_K_HI` — same, independently tunable for ratio maps,
  which are often more skewed than raw element concentrations
- `SHOW_SCALEBAR`, `SCALEBAR_UM`, `SCALEBAR_POS`, `SCALEBAR_MARGIN` — physical scale bar drawn on
  each render. `PIXEL_UM_FROM_SIDECAR` (default `True`): pixel size is read straight from
  `xrf_h5_to_tiff.py`'s metadata sidecar (`<grain_id>_<el>_Ka.txt`'s `step_size_pos1_um`, the fast
  /X axis the scale bar spans) for whichever loaded element has one, so it can't drift out of sync
  with the actual scan geometry; `PIXEL_UM` (scalar or per-grain list) is only the fallback used
  (with a warning) if no sidecar is found/parseable for any loaded element
- Output: `figs/map_renders/<grain_id>_<el>_Ka_display.png` per element,
  `figs/map_renders/<grain_id>_<num>_<den>_ratio_display.png` per ratio

**`sum_epma_maps.py`**
- `INPUT_DIR`, `OUTPUT_DIR` — default both `inputs/maps/`
- `INPUT_FILES` — list of map filenames (relative to `INPUT_DIR`) to sum
- `OUTPUT_FILE` — output filename (saved to `OUTPUT_DIR`)
- Auto-crops all inputs to their smallest common dimensions (trimming from the right/bottom) if
  they differ in size, e.g. from colorbar width variation; RGB inputs are converted to grayscale
  (with a warning) before summing
- Output is always a 32-bit float TIFF (raw pixel sum, accumulated in float64 to avoid overflow
  regardless of input bit depth) — compatible with `CL_EPMA_registration.m`'s `epma_dir` input

**`kyanite_figures.py`**
- `CSV_INPUT`, `ELEMENTS`, `PLOT_TYPE` (`scatter`, `violin`, `boxplot`, `contour`,
  `heatmap`, `corrmatrix`, `summary`, `distributions`, `all`, or a list of these)
- `WHOLE_GRAIN_OUTPUT_DIR` / `REGION_OUTPUT_DIR` — where figures are saved
  (default `figs/whole_grain/` / `figs/regions/`), independent of `CSV_INPUT`
- `N_BINS` / `BIN_EDGES`
- `summary` (only fires when `CSV_INPUT` is a directory): pools every whole-grain
  (non-region) CSV found into one grains-x-`ELEMENTS` heatmap of CL-vs-element
  Pearson r (annotated with n per cell), so correlation strength/consistency can be
  compared across grains — skipped (with a warning) if `CSV_INPUT` isn't a directory
  or fewer than 2 whole-grain CSVs are found. `SUMMARY_OUTPUT_DIR` (default `None` →
  `WHOLE_GRAIN_OUTPUT_DIR`) and `ALL_GRAINS_LABEL` (default `'all_grains'`, the
  filename prefix) control where/how it's saved
- Outlier removal on the element axis (applied per region in region mode), two
  independent stages: (1) `SATURATION_FILTER` (default on) flags and excludes
  pixels piled up near an element's own max value — the signature of a
  saturated/clipped detector channel — controlled by `SATURATION_BAND_FRAC`/
  `SATURATION_MIN_FRAC`/`SATURATION_MIN_COUNT`; only ever fires (with a printed
  warning) on a genuine pileup, and only on the max side (a pileup near the min is
  ordinary near-zero/below-detection-limit data, not saturation). (2) `OUTLIER_METHOD`
  trims whatever's left: `'mad'` (default) — a robust modified z-score computed in
  log-space (element concentrations are right-skewed, same assumption this project
  already makes before PCA elsewhere), excluded where it exceeds `MAD_K_LO`/
  `MAD_K_HI` (`None` disables that side; default has no low-side trim, and
  `MAD_K_HI=4` — chosen as the project default after visually comparing candidate
  configurations' excluded pixels with `kyanite_outlier_method_comparison.py`),
  adapting to how spread out each element's own distribution actually is instead of
  always chopping a fixed fraction; or `'percentile'` — legacy fixed-percentile
  behavior via `PCT_LO`/`PCT_HI` (0/100 disables it)
- `OUTLIER_SPATIAL_QC` (default on, whole-grain CSVs only): every time the outlier
  logic above is applied to an element, also renders where it actually excluded
  pixels directly on the masked 2-D element map — one `<grain_id>_<element>_
  outlier_exclusion_QC.png` per element, saved to `OUTLIER_QC_DIR` (default
  `figs/diagnostics/`). `pixel_data.csv` carries no row/col, so this reloads the raw
  element TIFF (`MAPS_DIR`, default `inputs/maps/`) and grain mask TIFF (`MASK_DIR`,
  default `figs/data/`) instead of using the CSV — skipped per element (with a
  warning) if either file isn't found. Exclusion decisions are scale-invariant
  (percentile and log-space MAD are both unaffected by a positive scalar like
  whatever `normalize_epma` applied), so the pixels flagged here exactly match what
  the CSV-based filtering just computed even though the displayed concentration
  values may be in different (e.g. unnormalized raw) units. Colors: gray = kept,
  dark red (`SATURATION_QC_COLOR`) = saturation-excluded, orange = statistical-trim-
  excluded — always reflects whichever `OUTLIER_METHOD` is currently configured, not
  a comparison between methods
- `distributions` — `summary`-shaped (only fires when `CSV_INPUT` is a directory,
  pools every whole-grain CSV rather than looping per-grain): for each element,
  renders a grain-x-grain small-multiples grid (`DIST_GRID_NCOLS` columns) of its raw
  value histogram and a second grid of its log10 histogram, each with a fitted normal
  curve + skew annotation, on each grain's full *unfiltered* masked population — a
  direct sanity check, per grain/element, of the log-normal assumption
  `OUTLIER_METHOD='mad'` relies on (testing on already-trimmed data would be
  circular). Also writes `<ALL_GRAINS_LABEL>_element_distribution_stats.csv`
  (skew/kurtosis, raw and log, per grain x element) and prints each element's median
  log-skew/kurtosis across grains. Saved to `DISTRIBUTION_QC_DIR` (default `None` →
  `OUTLIER_QC_DIR`, i.e. `figs/diagnostics/`) — skipped (with a warning) if
  `CSV_INPUT` isn't a directory or no whole-grain CSVs are found

**`kyanite_pca.py`**
- `CSV_INPUT`, `ELEMENTS` — file/directory and columns to include (defaults to all element columns)
- `WHOLE_GRAIN_OUTPUT_DIR` / `REGION_OUTPUT_DIR` — where figures are saved
  (default `figs/pca/` / `figs/regions/`), independent of `CSV_INPUT`
- `DATA_OUTPUT_DIR` — where the reusable CSV tables below are saved (default
  `figs/data/`, alongside the pixel-data CSVs this script reads), independent
  of `WHOLE_GRAIN_OUTPUT_DIR`/`REGION_OUTPUT_DIR` and of `CSV_INPUT`
- `DIAGNOSTICS_DIR` — where the run log is saved (default `figs/diagnostics/`,
  matching every other analysis/registration log in this project), independent
  of the other output dirs and of `CSV_INPUT`
- `BELOW_DETECTION` / `MAX_BELOW_DETECTION_FRAC` — drop poorly-detected elements
- `LOG_TRANSFORM`, `PC_TO_PLOT`, `LOADING_THRESHOLD` — PCA options
- Region CSVs (`*_region_pixel_data.csv`, has a `Region` column) get one pooled-PCA
  analysis per grain instead of the whole-grain PCA above (skipped if the grain has
  fewer than 2 regions): a single PCA fit across all of a grain's regions together,
  with every region projected into that shared PC space — fitting PCA independently
  per region would give each region its own PC space, making scores incomparable
  across regions; this pooled fit is what actually lets you test whether hand-drawn
  regions (e.g. core vs. rim) separate out in PC space
- `REGION_PCA_PCS` — which two PCs regions are compared on (default PC1 vs PC2, used
  for the scatter, biplot, and separation stats — scree and loadings still cover all
  computed PCs/`N_PCS_SCREE` same as the whole-grain PCA); `REGION_PCA_HULLS` — draw a
  convex-hull outline around each region's point cloud on both the scatter and biplot
- Region separation is also tested quantitatively: one-way ANOVA of each
  `REGION_PCA_PCS` component across regions, plus pairwise region-centroid distances
  in that PC subspace — both written to `<grain_id>_regions_pca_log.txt` and the
  centroid distances to `<grain_id>_regions_pca_centroid_distances.csv`
- Region PCA outputs: `<grain_id>_regions_pca_pca_variance.csv`,
  `<grain_id>_regions_pca_pca_loadings.csv`, `<grain_id>_regions_pca_scores.csv`
  (per-pixel PC scores + `Region` column), and `<grain_id>_regions_pca_centroid_distances.csv`
  (all four in `DATA_OUTPUT_DIR`, i.e. `figs/data/`); `<grain_id>_regions_pca_pca_scree.png`,
  `<grain_id>_regions_pca_pca_loadings_PC<n>.png` (one per `REGION_PCA_PCS` component),
  `<grain_id>_regions_pca_pc<i>_pc<j>_scatter.png` (PC scores colored by region),
  and `<grain_id>_regions_pca_pca_biplot.png` (same scatter with element
  loading-vector arrows overlaid) in `REGION_OUTPUT_DIR` (`figs/regions/`); and
  `<grain_id>_regions_pca_log.txt` in `DIAGNOSTICS_DIR` (`figs/diagnostics/`)
- Whole-grain PCA outputs, similarly split: `<grain_id>_pca_variance.csv`/
  `<grain_id>_pca_loadings.csv` in `DATA_OUTPUT_DIR` (`figs/data/`);
  `<grain_id>_pca_scree.png`, `<grain_id>_pca_scores_vs_CL.png`, and
  `<grain_id>_pca_loadings_PC<n>.png` (one per `PC_TO_PLOT`) in
  `WHOLE_GRAIN_OUTPUT_DIR` (`figs/pca/`); `<grain_id>_pca_log.txt` in
  `DIAGNOSTICS_DIR` (`figs/diagnostics/`)

**`kyanite_rf_shap.py`**
- Fits models and exports CSVs + a log only — no figures; region CSVs (`Region`
  column present) are skipped with a warning by default, since a full CV RF +
  permutation importance + TreeSHAP fit is too expensive to also run once per
  region on top of every whole-grain grain
- `ANALYZE_REGIONS` — `False` (default): skip region CSVs as above. `True`: fit
  RF/SHAP separately on each region within a region CSV instead (each region
  treated as its own independent dataset/model, not pooled — unlike
  `kyanite_pca.py`'s region PCA); outputs are labeled `<grain_id>_<region>`
  instead of `<grain_id>`
- `CSV_INPUT`, `ELEMENTS` — file/directory and columns to include (defaults to all element columns)
- `OUTPUT_DIR` — where the CSVs are saved (default `figs/data/`, since these
  CSVs are themselves reusable data — `kyanite_rf_shap_plots.py` reads them
  back), independent of `CSV_INPUT`
- `DIAGNOSTICS_DIR` — where the run log is saved (default `figs/diagnostics/`,
  matching every other analysis/registration log in this project), independent
  of `OUTPUT_DIR` and of `CSV_INPUT`
- `ANALYSES` — `rf`, `shap`, `all`, or a list of these
- `BELOW_DETECTION` / `MAX_BELOW_DETECTION_FRAC` — drop poorly-detected elements
- `LOG_TRANSFORM` — log10-transform elements before RF/SHAP (must match what
  `kyanite_rf_shap_plots.py`'s own `LOG_TRANSFORM` assumes for axis labeling)
- `CV_FOLDS`, `N_ESTIMATORS`, `MAX_SAMPLES`, `IMPORTANCE_SIG_RATIO` — Random Forest options
- `SHAP_SAMPLES`, `SHAP_INTERACTIONS` — TreeSHAP importance and pairwise interaction values from a single RF fit on a subsample
- Outputs, per grain (in `OUTPUT_DIR`): `<grain_id>_rf_importance.csv` (mean/std
  permutation importance per element, CV-averaged, plus a `significant` flag),
  `<grain_id>_rf_predictions.csv` (`row_index` back to the source CSV,
  `observed_CL`, out-of-fold `predicted_CL`, `fold`), `<grain_id>_shap_importance.csv`
  (mean |SHAP value| per element), `<grain_id>_shap_values.csv` (`row_index` plus
  `<element>_value`/`<element>_shap` columns — the raw per-pixel data behind the
  dependence plots), `<grain_id>_shap_interactions.csv` (pairwise interaction
  matrix, only if `SHAP_INTERACTIONS`); `<grain_id>_rf_shap_log.txt` in
  `DIAGNOSTICS_DIR`

**`kyanite_rf_shap_plots.py`**
- Reads `kyanite_rf_shap.py`'s CSVs back and plots them — no model fitting, so a
  figure can be regenerated or restyled without retraining
- `CSV_INPUT` — directory to read CSVs from (default `figs/data/`, matching
  `kyanite_rf_shap.py`'s `OUTPUT_DIR`); grains are discovered from whichever
  `*_rf_predictions.csv`/`*_shap_values.csv` files are present, so a grain that only
  ran `ANALYSES=['rf']` (or `['shap']`) upstream still gets whichever plots apply,
  with the rest skipped (with a warning) rather than erroring
- `RF_OUTPUT_DIR` / `SHAP_OUTPUT_DIR` — where figures are saved (default
  `figs/rf/` / `figs/shap/`), independent of `CSV_INPUT`; RF figures
  (`observed_vs_predicted`, `importance`) go to the former, SHAP figures
  (`shap_importance`, `shap_interactions`, `shap_dependence`) to the latter
- `GRAIN_FILTER` — list of grain_ids to plot, or `None` for every grain found in `CSV_INPUT`
- `PLOTS` — `observed_vs_predicted`, `importance`, `shap_importance`,
  `shap_interactions`, `shap_dependence`, `all`, or a list of these
- `LOG_TRANSFORM` — must match whatever `kyanite_rf_shap.py` used for this data;
  only affects axis labeling here (`(log10)` suffix), since values were already
  transformed (or not) upstream
- `FIG_DPI`, `SHOW_TITLE` — figure styling
- Per-fold RMSE/R2 for the observed-vs-predicted title are recomputed directly from
  `<grain_id>_rf_predictions.csv` rather than stored separately, so there's one
  source of truth for those numbers
- SHAP dependence panels color by top interacting partner using
  `<grain_id>_shap_interactions.csv`, looked up by element name (not position), so
  a partial or differently-ordered interactions CSV still matches up correctly;
  falls back to uncolored if that CSV isn't present

**`kyanite_sample_size_convergence.py`**
- `CSV_INPUT` — a single grain's `*_pixel_data.csv` (not batch)
- `DIAGNOSTICS_DIR` — where output (raw sweep CSV, convergence figures, log)
  is saved (default `figs/diagnostics/`), independent of `CSV_INPUT`; unlike
  `kyanite_pca.py`/`kyanite_rf_shap.py` there's no separate figures/data split
  or dedicated analysis-family folder, since this whole script is itself a
  diagnostic (not a pipeline step)
- `SAMPLE_SIZES`, `N_REPEATS` — sizes to sweep and independent random subsamples per size; the true full-grain pixel count is always appended as the last point
- `MAX_DEPTH` — caps RF tree depth so TreeSHAP stays tractable at large sample sizes (unrestricted depth at 300k+ px made a single `shap_values()` call ~60s vs. ~2s capped) and keeps model complexity comparable across the sweep
- `SHAP_EXPLAIN_SAMPLES` — cap on held-out points explained by SHAP per repeat, so SHAP cost doesn't grow with sample size
- `CONVERGENCE_THRESHOLD` — step-to-step relative change below which a metric counts as converged
- Output: importance/RMSE/R2 vs. sample size plots with repeat-to-repeat spread as a shaded band, raw results CSV, and a log noting the convergence size per element/metric

**`kyanite_spot_analysis.py`**
- `CSV_INPUT` — file/directory of `*_spot_geochemistry.csv` (defaults to `figs/data/`,
  where `xrf_h5_extract_spots.py` writes it as reusable data)
- `FIGS_DIR`, `OUT_DIR` — where to find `<grain_id>_CL_registered.tif` / save figures
- `ANALYSES` — `pie`, `scatter`, `box`, `map`, `pca`, `all`, or a list of these
- `SCATTER_ELEMENTS` — element columns for the CL-vs-element and by-class box plots (`None` = auto-detect all)
- `PCA_ELEMENTS`, `PCA_LOG_TRANSFORM` — element list for the PCA scatter/scree/loadings/biplot, and
  whether to log10-transform elements before z-scoring/PCA (independent of `SCATTER_ELEMENTS`)
- `PCA_N_PCS_SCREE`, `PCA_LOADING_THRESHOLD` — how many PCs the scree plot shows (`None` = all), and
  the `|loading|` cutoff highlighted on the PC1/PC2 loadings bars
- `CATEGORY_ORDER` / `CATEGORY_COLORS` — fixed XANES class order/coloring, shared across all figures

**`xanes_rf_classifier.py`**
- `CSV_INPUT` — file/directory of `*_spot_geochemistry.csv` (defaults to `figs/data/`, same as `kyanite_spot_analysis.py`)
- `OUT_DIR` — figures only (`_rf_classifier_importance.png`, `_rf_classifier_confusion_matrix.png`);
  default `figs/spot_analysis/`, shared with `kyanite_spot_analysis.py` — both pool
  the same per-spot CSVs and fall under the same "spot analysis" umbrella, even
  though they're two different analyses (classifier vs. pooled scatter/pie/box/PCA figures)
- `DATA_OUTPUT_DIR` — reusable CSVs (`_rf_classifier_importance.csv`, `_rf_classifier_predictions.csv`);
  default `figs/data/`, alongside the `spot_geochemistry` CSVs this script reads
- `DIAGNOSTICS_DIR` — run log (`_rf_classifier_log.txt`); default `figs/diagnostics/`,
  matching every other analysis/registration log in this project
- `OUTPUT_LABEL` — filename prefix for all outputs (pooled analysis; no per-grain run)
- `ELEMENTS` — element list restricted to columns present in *every* input CSV (`None` = auto-detect)
- `CATEGORY_ORDER` — fixed XANES class order, `'Bad data'`/NaN excluded (matches `kyanite_spot_analysis.py`)
- `CV_STRATEGY` — `'grouped'` (default, `StratifiedGroupKFold` by `grain_id`: no grain's spots span
  train/test) or `'stratified'` (`StratifiedKFold` ignoring grain identity)
- Off-grain spots (`on_grain = False`) are excluded implicitly, not by an explicit
  filter: their element columns are already `NaN` (see `xrf_h5_extract_spots.py`'s
  `on_grain`), and `prepare_data()` drops any row with an incomplete feature
  vector — correct here since this classifier predicts XANES class from
  *kyanite's* chemistry specifically, unlike `kyanite_spot_analysis.py`'s spot map,
  which keeps them visible (flagged) since their classification is still
  meaningful for whatever other phase they sampled

**`xanes_plot.py`**
- `XANES_INPUT` — file/directory of raw per-spot XANES spectra (`inputs/xanes/<grain_id>_spotNN.csv`)
- `OUT_DIR` — default `figs/xanes/`
- `CLASSIFY` — off by default (automatic classification isn't reliable enough across all grains yet);
  when enabled, classifies each spot's pre-edge doublet by relative Fe²⁺/Fe³⁺ peak height (see script
  header for the full method) as a starting point for hand classification, writing a CSV in the same
  `<grain_id>_pre_edge_classification.csv` convention as `xanes_classification_split.py`

**`xanes_classification_split.py`**
- `INPUT_CSV` — combined hand-classification CSV (`GrainID`, `Spot`, `Class` columns)
- `OUTPUT_DIR` — default `inputs/xanes_classification/`, matching where `xrf_h5_extract_spots.py` looks
  for `<grain_id>_pre_edge_classification.csv`

## Requirements
- MATLAB with Image Processing Toolbox (for `cpselect`, `imwarp`, `imread`, etc.)
- Python: `h5py`, `numpy`, `tifffile`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `shap`, `pyyaml` (only needed for `onboard_dataset.py`)
- Images are 8-, 16-, or 32-bit grayscale TIFFs; EPMA maps are the fixed reference
- On a shared HPC/cluster account, install the Python dependencies above into a
  dedicated virtualenv rather than a shared `--user`/system site-packages —
  installing directly there can silently upgrade packages (e.g. `numpy`) out from
  under unrelated tools already relying on an older version. `kyanite.sh` is an
  example SLURM script that activates such a venv before running a Python step
