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
| `xrf_h5_to_tiff.py` | Python | Extract XRF element maps from a Larch/GSECARS HDF5 file; export as 32-bit float TIFFs + metadata sidecars |
| `xrf_h5_extract_spots.py` | Python | Build a per-spot CSV: pixel + physical coordinates of XANES spot locations (`xrmmap/areas`), mean element concentrations and CL brightness over a small grain-mask-restricted circular zone around each spot, and the joined XANES pre-edge classification |
| `CL_EPMA_registration.m` | MATLAB | Full registration + analysis pipeline (see workflow below) |
| `CL_region_extraction.m` | MATLAB | Draw named sub-grain polygon regions on an already-registered CL image and extract per-pixel CL + element data per region (no re-registration) |
| `kyanite_figures.py` | Python | Standalone figure generation from exported CSV pixel data |
| `kyanite_pca_rf.py` | Python | PCA and cross-validated Random Forest analysis of CL vs. trace elements from exported CSV pixel data |
| `kyanite_sample_size_convergence.py` | Python | Diagnostic: sweeps RF/SHAP over a range of pixel subsample sizes for one grain to check whether importance estimates have converged below `kyanite_pca_rf.py`'s `MAX_SAMPLES`/`SHAP_SAMPLES`, or would still change with more data |
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
  grain-mask pixels (`figs/<GRAIN_ID>_mask.tif`, thresholded `> 128`) — mirrors the
  registration pipeline's "region intersected with grain mask" masking semantics, and
  reads elements straight from the h5 rather than the (possibly smaller, curated)
  `maps/<grain_id>/*.tif` subset, so "all the other elements" are available, not just
  whatever was previously exported
- Joins each spot to its hand-assigned XANES pre-edge class via
  `xanes_classification/<GRAIN_ID>_pre_edge_classification.csv` (built by
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

## File conventions
- Output filenames: `<sample>_<Element>_<Line>.tif`  (e.g. `NA-CM-G12B7-02_Fe_Ka.tif`)
- Metadata sidecars: same base name, `.txt` extension
- Registered CL (grayscale, 16-bit): `<grain_id>_CL_registered.tif`
- Registered CL (original color, native bit depth): `<grain_id>_CL_registered_color.tif`
- Pixel data exports: `<grain_id>_pixel_data.csv` and `.mat`
- Analysis log: `<grain_id>_analysis_log.txt`
- Figures saved to `figs/`, element maps to `maps/<grain_id>/`
- Region polygons (reusable): `<grain_id>_regions.mat`
- Region pixel data exports: `<grain_id>_region_pixel_data.csv` and `.mat` (long-format, `Region` column)
- Region summary stats: `<grain_id>_region_summary.csv`
- Region analysis log: `<grain_id>_region_analysis_log.txt`
- Spot coordinate exports: `<sample>_spot_coordinates.csv`
- Spot geochemistry/CL/XANES-class exports: `figs/<grain_id>_spot_geochemistry.csv`

## Key parameters (set per-grain at top of each script)

**`CL_EPMA_registration.m`**
- `grain_id` — sample name string, used in all output filenames
- `cl_filename` — CL image (supports .tif, .png, .bmp, any `imread`-compatible format)
- `epma_dir` — folder of element map TIFFs; all `*.tif` files auto-discovered
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

**`xrf_h5_to_tiff.py`**
- `H5_FILE`, `OUTPUT_DIR`, `SAMPLE`
- `ELEMENTS` — list of ROI names to export (or `None` for all)
- `NORMALIZE_BY_CLOCK`, `NORMALIZE_BY_I0`

**`xrf_h5_extract_spots.py`**
- `H5_FILE`, `GRAIN_ID` — `GRAIN_ID` must match the grain's `figs/`/`xanes_classification/` filenames;
  a warning is printed if `GRAIN_ID` isn't a substring of `H5_FILE`'s filename
- `FIGS_DIR`, `CLASSIFICATION_DIR` — where to look up the mask/CL TIFFs and classification CSV
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
- `ANALYSES` — `pca`, `rf`, `shap`, `all`, or a list of these
- `BELOW_DETECTION` / `MAX_BELOW_DETECTION_FRAC` — drop poorly-detected elements
- `LOG_TRANSFORM`, `PC_TO_PLOT`, `LOADING_THRESHOLD` — PCA options
- `CV_FOLDS`, `N_ESTIMATORS`, `MAX_SAMPLES`, `IMPORTANCE_SIG_RATIO` — Random Forest options
- `SHAP_SAMPLES`, `SHAP_INTERACTIONS`, `SHAP_DEPENDENCE_PLOTS` — TreeSHAP importance, pairwise interaction values, and element-vs-own-SHAP-value dependence plots from a single RF fit on a subsample

**`kyanite_sample_size_convergence.py`**
- `CSV_INPUT` — a single grain's `*_pixel_data.csv` (not batch)
- `SAMPLE_SIZES`, `N_REPEATS` — sizes to sweep and independent random subsamples per size; the true full-grain pixel count is always appended as the last point
- `MAX_DEPTH` — caps RF tree depth so TreeSHAP stays tractable at large sample sizes (unrestricted depth at 300k+ px made a single `shap_values()` call ~60s vs. ~2s capped) and keeps model complexity comparable across the sweep
- `SHAP_EXPLAIN_SAMPLES` — cap on held-out points explained by SHAP per repeat, so SHAP cost doesn't grow with sample size
- `CONVERGENCE_THRESHOLD` — step-to-step relative change below which a metric counts as converged
- Output: importance/RMSE/R2 vs. sample size plots with repeat-to-repeat spread as a shaded band, raw results CSV, and a log noting the convergence size per element/metric

## Requirements
- MATLAB with Image Processing Toolbox (for `cpselect`, `imwarp`, `imread`, etc.)
- Python: `h5py`, `numpy`, `tifffile`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `shap`
- Images are 8-, 16-, or 32-bit grayscale TIFFs; EPMA maps are the fixed reference
