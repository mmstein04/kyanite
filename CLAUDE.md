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
| `xrf_h5_to_tiff.py` | Python | Extract XRF element maps from a Larch/GSECARS HDF5 file; export as 32-bit float TIFFs + metadata sidecars |
| `xrf_h5_extract_spots.py` | Python | Build a per-spot CSV: pixel + physical coordinates of XANES spot locations (`xrmmap/areas`), mean element concentrations and CL brightness over a small grain-mask-restricted circular zone around each spot, and the joined XANES pre-edge classification |
| `CL_EPMA_registration.m` | MATLAB | Full registration + analysis pipeline (see workflow below) |
| `CL_region_extraction.m` | MATLAB | Draw named sub-grain polygon regions on an already-registered CL image and extract per-pixel CL + element data per region (no re-registration) |
| `CL_mask_edit.m` | MATLAB | After-the-fact touch-up of a grain mask already produced by `CL_EPMA_registration.m` (e.g. an inclusion was masked-in, or real grain was masked-out weeks earlier) — draw add/remove polygons on the already-registered CL image, then re-derive pixel data and downstream figures under the corrected mask |
| `kyanite_figures.py` | Python | Standalone figure generation from exported CSV pixel data |
| `kyanite_pca_rf.py` | Python | PCA and cross-validated Random Forest analysis of CL vs. trace elements from whole-grain CSV pixel data; for region CSVs, instead fits one shared PCA pooled across regions (scree, loadings, PC1/PC2-by-region scatter, biplot) to test whether hand-drawn regions separate in PC space — no per-region PCA/RF/SHAP |
| `kyanite_sample_size_convergence.py` | Python | Diagnostic: sweeps RF/SHAP over a range of pixel subsample sizes for one grain to check whether importance estimates have converged below `kyanite_pca_rf.py`'s `MAX_SAMPLES`/`SHAP_SAMPLES`, or would still change with more data |
| `kyanite_spot_analysis.py` | Python | Batch analysis of `<grain_id>_spot_geochemistry.csv` files: XANES class distribution pie-chart grid, pooled CL-vs-element scatter plots colored by class, element-by-class box plots, PCA (PC1/PC2 scatter, scree, loadings, biplot) colored by class, and per-grain labeled spot-location maps on the registered CL image |
| `xrf_display.m` | MATLAB | Visualize XRF element-map TIFFs with grain mask overlay |
| `sum_epma_maps.m` | MATLAB | Sum two or more element maps into a combined TIFF (e.g. Zr_La + Zr_Lb) |

### `CL_EPMA_registration.m` workflow
1. Load CL image and auto-discover all EPMA/XRF TIFFs in `epma_dir`
2. Interactively pick control points (`cpselect`) to warp CL onto the EPMA grid
3. Evaluate registration quality (RMSE in pixels and µm)
4. Build binary grain mask — methods: `otsu`, `manual`, `interactive`, `polygon`, `activecontour`
5. Apply mask; extract per-pixel CL and element concentration vectors
6. Scatter plot CL vs. each element; compute Pearson r
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
5. Save per-region/per-channel summary statistics and QC figures (region boundaries
   on the CL image and on every element map)

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

### `CL_mask_edit.m` workflow
1. Load the registered CL image, EPMA/XRF maps, and current grain mask already
   produced by `CL_EPMA_registration.m` for this grain — no control points,
   warping, or from-scratch mask generation here
2. Back up every mask-dependent output file this run is about to overwrite
   (mask TIFF, pixel data, scatter/shift-sensitivity/QC PNGs, prior edit
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
   `data/<grain_id>_pixel_data.csv`/`.mat`, then regenerates the CL-vs-element
   scatter plots, shift-sensitivity analysis, and all-maps/mask-check QC
   figures — so every figure in `figs/` matches the corrected mask, not just
   the CSV. Registration-quality fields (`RMSE_px`/`RMSE_um`) are carried
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
- Writes `<grain_id>_onboarding_log.txt` (skipped in dry-run mode) recording every
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
- `NAME_FILTER` (default `'spot'`) selects which `xrmmap/areas` entries to include;
  set to `None` to include drawn polygon regions too (their shape is discarded —
  extraction is centered on the region's centroid, like a point spot)
- Note: spot numbering in the HDF5 area names doesn't necessarily share a prefix
  with exported XANES spot CSVs or `GRAIN_ID` (e.g. h5 area `LLF6-Area2-spot01` vs.
  grain `LLF6-01`) — everything joins on the trailing spot number + `GRAIN_ID`, not
  the h5 area's own name
- Note: the 8 real `*_spot_geochemistry.csv` files currently on disk live in
  `figs/xanes/`, not directly in `figs/` as `OUTPUT_CSV`'s own default implies —
  `kyanite_spot_analysis.py`'s `CSV_INPUT` default points at the real location

### `kyanite_spot_analysis.py` details
- Input: `<grain_id>_spot_geochemistry.csv` files (see above), one per grain;
  `CSV_INPUT` may be a single file or a directory (globs `*_spot_geochemistry.csv`)
- Three independently toggleable analyses (`ANALYSES` list): `pie` (one combined
  figure, small-multiples grid of per-grain XANES class pies — `'Bad data'`/
  unclassified spots excluded from the pie counts entirely, per-grain slice
  order/coloring stays identical even at zero count for a type), `scatter`
  (CL-vs-element, one figure per element, pooling spots from every input grain —
  `'Bad data'`/unclassified spots ARE included here, as grey points/markers, for
  QC context), `map` (per-grain spot-location map on the registered CL image,
  labeled by spot number), `pca` (one PCA fit over `PCA_ELEMENTS`, pooling spots
  from every input grain, producing four figures — PC1-vs-PC2 scatter, scree plot,
  PC1/PC2 loadings bar charts, and a PC1-vs-PC2 biplot with loading vectors — all
  colored by XANES class the same way as `scatter` — `'Bad data'`/unclassified
  spots ARE included as grey points; spots missing any `PCA_ELEMENTS` value are
  dropped)
- `SCATTER_ELEMENTS` (default `None`) auto-detects every element column present in
  the union of input files; an element missing from some grains' CSVs (ROI lists
  vary slightly, e.g. LLF6-01 has extra REE lines) is pooled from whichever grains
  do have it, with a warning listing which grains were excluded from that plot
- `PCA_ELEMENTS` — element list the PCA considers, independent of `SCATTER_ELEMENTS`
  (defaults to the same Cr/V/Fe/Ti/Mn set, but chosen deliberately since PCA is
  sensitive to which variables are included); `PCA_LOG_TRANSFORM` log10-transforms
  elements before z-scoring/PCA, matching `kyanite_pca_rf.py`'s convention;
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
  `<grain_id>_mask_edit_diff.png`, `<grain_id>_mask_edit_log.txt`
- Reusable data files read back in by other scripts go in `figs/data/`
  rather than directly in `figs/`: grain mask (`<grain_id>_mask.tif`), pixel
  data exports (`<grain_id>_pixel_data.csv` and `.mat`), control-point MATs
  (`<grain_id>_controlpoints.mat`, `<grain_id>_mask_image_controlpoints.mat`),
  and `CL_mask_edit.m`'s cumulative edit history (`<grain_id>_mask_edits.mat`)
- Region polygons (reusable): `<grain_id>_regions.mat`
- Region pixel data exports: `<grain_id>_region_pixel_data.csv` and `.mat` (long-format, `Region` column)
- Region summary stats: `<grain_id>_region_summary.csv`
- Region analysis log: `<grain_id>_region_analysis_log.txt`
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
- Spot coordinate exports: `<grain_id>_spot_coordinates.csv`
- Spot geochemistry/CL/XANES-class exports: `figs/<grain_id>_spot_geochemistry.csv`
  (currently on disk in `figs/xanes/` for all 8 processed grains)
- Spot analysis figures saved to `figs/spot_analysis/`: `xanes_class_pie_grid.png`,
  `CL_vs_<element>_scatter.png`, `<element>_by_class_boxplot.png`,
  `pca_pc1_pc2_scatter.png`, `pca_scree.png`, `pca_loadings_pc1_pc2.png`,
  `pca_biplot.png`, `<grain_id>_spot_map.png`
- Onboarding manifest (per new dataset, hand-written from `dataset_manifest.example.yaml`):
  `dataset_manifest.yaml` (or any name — set in `onboard_dataset.py`'s `MANIFEST_FILE`)
- Onboarding audit log: `<grain_id>_onboarding_log.txt` (written by `onboard_dataset.py`,
  only on a non-dry-run execution)

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
- `epma_pixel_um` — pixel size in µm (spatial calibration)
- `mask_method` — grain segmentation method
- `pct_lo_cut` / `pct_hi_cut` — outlier percentile bounds for scatter plots

**`CL_region_extraction.m`**
- `grain_id` — must match a grain already processed by `CL_EPMA_registration.m`
- `input_dir` — folder holding that grain's registered CL TIFFs + mask TIFF
- `epma_dir` — same EPMA/XRF map folder used during registration
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
- `epma_pixel_um`, `normalize_epma`, `pct_lo_cut`/`pct_hi_cut`, `shift_range` — must
  match the values used in the grain's original `CL_EPMA_registration.m` run, or
  re-derived pixel data/plots won't be comparable to before the edit
- `close_radius_px` / `min_object_px` / `fill_holes` — post-edit mask cleanup,
  same knobs as `CL_EPMA_registration.m`'s `SECTION 5`
- `regenerate_downstream` — `true` (default): re-extract pixel data and
  regenerate scatter/shift-sensitivity/QC figures after saving the edited mask;
  `false`: touch only the mask TIFF + edit history

**`xrf_h5_to_tiff.py`**
- `H5_FILE` — raw XRF HDF5 file, default location `inputs/xrf/`
- `OUTPUT_DIR`, `GRAIN_ID`
- `ELEMENTS` — list of ROI names to export (or `None` for all)
- `NORMALIZE_BY_CLOCK`, `NORMALIZE_BY_I0`

**`xrf_h5_extract_spots.py`**
- `H5_FILE` — raw XRF HDF5 file, default location `inputs/xrf/`
- `GRAIN_ID` — must match the grain's `figs/`/`inputs/xanes_classification/` filenames;
  a warning is printed if `GRAIN_ID` isn't a substring of `H5_FILE`'s filename
- `FIGS_DIR`, `CLASSIFICATION_DIR` — where to look up the mask/CL TIFFs and classification CSV
  (default `inputs/xanes_classification/`)
- `OUTPUT_CSV` — defaults to `figs/<GRAIN_ID>_spot_geochemistry.csv`; set to `None` for console-only
- `NAME_FILTER` — regex to select which `xrmmap/areas` entries count as spots (default `'spot'`)
- `ZONE_RADIUS_UM` — physical radius (µm) of the circular sampling zone around each spot
- `ELEMENTS` / `EXCLUDE_ROIS` — element ROIs to extract (`None` = all except known scaler channels)
- `NORMALIZE_BY_CLOCK`, `NORMALIZE_BY_I0` — should match whatever was used for this grain's
  exported maps, for comparable units

**`kyanite_figures.py`**
- `CSV_FILE`, `ELEMENT`, `PLOT_TYPE` (`scatter`, `violin`, `boxplot`, or `all`)
- `N_BINS` / `BIN_EDGES`, `PCT_LO` / `PCT_HI`

**`kyanite_pca_rf.py`**
- `CSV_INPUT`, `ELEMENTS` — file/directory and columns to include (defaults to all element columns)
- `ANALYSES` — `pca`, `rf`, `shap`, `all`, or a list of these — for whole-grain CSVs.
  Region CSVs only ever run the pooled region-PCA analysis below (skipped entirely,
  with a warning, if `'pca'` isn't in `ANALYSES`); `rf`/`shap` never run per-region
- `BELOW_DETECTION` / `MAX_BELOW_DETECTION_FRAC` — drop poorly-detected elements
- `LOG_TRANSFORM`, `PC_TO_PLOT`, `LOADING_THRESHOLD` — PCA options
- `CV_FOLDS`, `N_ESTIMATORS`, `MAX_SAMPLES`, `IMPORTANCE_SIG_RATIO` — Random Forest options
- `SHAP_SAMPLES`, `SHAP_INTERACTIONS`, `SHAP_DEPENDENCE_PLOTS` — TreeSHAP importance, pairwise interaction values, and element-vs-own-SHAP-value dependence plots from a single RF fit on a subsample
- Region CSVs (`*_region_pixel_data.csv`, has a `Region` column) get one pooled-PCA
  analysis per grain instead of the whole-grain PCA/RF/SHAP breakdown above (skipped
  if the grain has fewer than 2 regions): a single PCA fit across all of a grain's
  regions together, with every region projected into that shared PC space — fitting
  PCA independently per region would give each region its own PC space, making
  scores incomparable across regions; this pooled fit is what actually lets you test
  whether hand-drawn regions (e.g. core vs. rim) separate out in PC space
- `REGION_PCA_PCS` — which two PCs regions are compared on (default PC1 vs PC2, used
  for the scatter, biplot, and separation stats — scree and loadings still cover all
  computed PCs/`N_PCS_SCREE` same as the whole-grain PCA); `REGION_PCA_HULLS` — draw a
  convex-hull outline around each region's point cloud on both the scatter and biplot
- Region separation is also tested quantitatively: one-way ANOVA of each
  `REGION_PCA_PCS` component across regions, plus pairwise region-centroid distances
  in that PC subspace — both written to `<grain_id>_regions_pca_log.txt` and the
  centroid distances to `<grain_id>_regions_pca_centroid_distances.csv`
- Region PCA outputs (alongside the region CSV): `<grain_id>_regions_pca_pca_variance.csv`,
  `<grain_id>_regions_pca_pca_loadings.csv`, `<grain_id>_regions_pca_scores.csv`
  (per-pixel PC scores + `Region` column), `<grain_id>_regions_pca_centroid_distances.csv`,
  `<grain_id>_regions_pca_pca_scree.png`, `<grain_id>_regions_pca_pca_loadings_PC<n>.png`
  (one per `REGION_PCA_PCS` component), `<grain_id>_regions_pca_pc<i>_pc<j>_scatter.png`
  (PC scores colored by region), `<grain_id>_regions_pca_pca_biplot.png` (same scatter
  with element loading-vector arrows overlaid), `<grain_id>_regions_pca_log.txt`

**`kyanite_sample_size_convergence.py`**
- `CSV_INPUT` — a single grain's `*_pixel_data.csv` (not batch)
- `SAMPLE_SIZES`, `N_REPEATS` — sizes to sweep and independent random subsamples per size; the true full-grain pixel count is always appended as the last point
- `MAX_DEPTH` — caps RF tree depth so TreeSHAP stays tractable at large sample sizes (unrestricted depth at 300k+ px made a single `shap_values()` call ~60s vs. ~2s capped) and keeps model complexity comparable across the sweep
- `SHAP_EXPLAIN_SAMPLES` — cap on held-out points explained by SHAP per repeat, so SHAP cost doesn't grow with sample size
- `CONVERGENCE_THRESHOLD` — step-to-step relative change below which a metric counts as converged
- Output: importance/RMSE/R2 vs. sample size plots with repeat-to-repeat spread as a shaded band, raw results CSV, and a log noting the convergence size per element/metric

**`kyanite_spot_analysis.py`**
- `CSV_INPUT` — file/directory of `*_spot_geochemistry.csv` (defaults to `figs/xanes/`)
- `FIGS_DIR`, `OUT_DIR` — where to find `<grain_id>_CL_registered.tif` / save figures
- `ANALYSES` — `pie`, `scatter`, `box`, `map`, `pca`, `all`, or a list of these
- `SCATTER_ELEMENTS` — element columns for the CL-vs-element and by-class box plots (`None` = auto-detect all)
- `PCA_ELEMENTS`, `PCA_LOG_TRANSFORM` — element list for the PCA scatter/scree/loadings/biplot, and
  whether to log10-transform elements before z-scoring/PCA (independent of `SCATTER_ELEMENTS`)
- `PCA_N_PCS_SCREE`, `PCA_LOADING_THRESHOLD` — how many PCs the scree plot shows (`None` = all), and
  the `|loading|` cutoff highlighted on the PC1/PC2 loadings bars
- `CATEGORY_ORDER` / `CATEGORY_COLORS` — fixed XANES class order/coloring, shared across all figures

## Requirements
- MATLAB with Image Processing Toolbox (for `cpselect`, `imwarp`, `imread`, etc.)
- Python: `h5py`, `numpy`, `tifffile`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `shap`, `pyyaml` (only needed for `onboard_dataset.py`)
- Images are 8-, 16-, or 32-bit grayscale TIFFs; EPMA maps are the fixed reference
