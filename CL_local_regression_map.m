% =========================================================================
% CL_LOCAL_REGRESSION_MAP.m
%
% PURPOSE:
%   Slide a circular window across every pixel of an already-registered
%   grain, run a per-pixel CL-vs-element linear regression on the pixels
%   inside that window (intersected with the grain mask), and store the
%   resulting slope and Pearson r back at the center pixel. Produces
%   continuous "slope map" and "R map" images showing how the CL-element
%   relationship varies spatially across the grain, complementing the
%   fixed polygon regions of CL_region_extraction.m.
%
% WORKFLOW:
%   1. Load the registered CL image, grain mask, and EPMA/XRF maps already
%      produced by CL_EPMA_registration.m for this grain — no warping,
%      control-point picking, or grain-mask generation is done here.
%   2. Build a binary circular kernel of the requested physical radius.
%   3. Compute local regression sums (n, Sx, Sy, Sxx, Syy, Sxy) via 2-D
%      convolution with that kernel — this is mathematically equivalent to
%      running an explicit per-window linear regression at every pixel,
%      but runs as a handful of convolutions instead of a double loop.
%   4. Derive per-pixel slope and Pearson r maps for every element from
%      those sums; mask out pixels outside the grain or with too few valid
%      neighbors in the window.
%   5. Save maps (.mat + long-format .csv) and QC figures (slope map grid,
%      R map grid, window-coverage map).
%
% INPUTS (set in PARAMETERS section below):
%   - Registered CL image + grain mask (outputs of CL_EPMA_registration.m)
%   - Folder of EPMA/XRF element map TIFFs for the same grain
%
% OUTPUTS:
%   - [grain_id]_local_regression.mat            — slope/R/n maps + metadata
%   - [grain_id]_local_regression_pixel_data.csv — long-format per-pixel table
%   - [grain_id]_local_regression_slope_QC.png   — slope map, all elements
%   - [grain_id]_local_regression_R_QC.png       — R map, all elements
%   - [grain_id]_local_regression_n_map.png      — window coverage map
%   - [grain_id]_local_regression_analysis_log.txt — comprehensive run record
%
% REQUIREMENTS:
%   - MATLAB Image Processing Toolbox (for imread of TIFFs, visboundaries)
%
% AUTHOR:  M. Stein
% DATE:    2026-07-01
% VERSION: 1.0
% =========================================================================

clear; clc; close all;

set(0, 'DefaultTextInterpreter',        'none');
set(0, 'DefaultAxesTickLabelInterpreter','none');
set(0, 'DefaultLegendInterpreter',      'none');

% =========================================================================
%% SECTION 1: PARAMETERS  — edit this section for each new grain / radius
% =========================================================================

grain_id = 'NA-GS-P84-06';

% Directory containing the outputs of CL_EPMA_registration.m for this grain
% (registered CL TIFFs and the grain mask TIFF).
input_dir = '/Users/mstein/bin/kyanite/figs';

cl_filename   = [grain_id, '_CL_registered.tif'];   % 16-bit grayscale
mask_filename = [grain_id, '_mask.tif'];

% Folder containing EPMA/XRF element map TIFFs (same folder used by
% CL_EPMA_registration.m for this grain). All *.tif files auto-discovered.
epma_dir = ['/Users/mstein/bin/kyanite/maps/', grain_id];

output_dir = '/Users/mstein/bin/kyanite/local_regression_figs';

% --- Spatial calibration --------------------------------------------------
epma_pixel_um = 1.0;     % µm per pixel — must match the value used during registration

% --- Moving-window regression parameters -----------------------------------
% Physical radius of the circular regression window, in µm. Kept in µm
% (rather than px) so it stays meaningful across grains imaged at
% different pixel sizes; converted to px below using epma_pixel_um.
window_radius_um = 20.0;

% Minimum number of in-mask pixels required inside a window before a
% regression is computed there; windows with fewer valid pixels are set
% to NaN in the output maps. Default: half the full disk area.
window_radius_px = round(window_radius_um / epma_pixel_um);
min_window_px    = ceil(0.5 * pi * window_radius_px^2);

% --- Element map normalization ---------------------------------------------
% true:  re-normalize each element map to [0 1] using its min/max within the
%        grain mask (matches CL_EPMA_registration.m's normalize_epma = true).
% false: keep raw pixel values (counts/intensity units from the TIFF).
normalize_epma = false;

% Percentile range used to contrast-stretch element maps in QC figures only.
display_pct = [0, 97];

% =========================================================================
%% SECTION 2: SETUP
% =========================================================================

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% --- Auto-discover EPMA maps from epma_dir --------------------------------
tif_listing = dir(fullfile(epma_dir, '*.tif'));
if isempty(tif_listing)
    error('No *.tif files found in epma_dir: %s', epma_dir);
end
tif_names = {tif_listing.name};

% Exclude the CL image and any script-generated output TIFFs
output_suffixes = {'_CL_registered.tif', '_CL_registered_color.tif', '_mask.tif'};
keep_flags = true(1, numel(tif_names));
for k = 1:numel(tif_names)
    fname = tif_names{k};
    for p = 1:numel(output_suffixes)
        if endsWith(fname, output_suffixes{p})
            keep_flags(k) = false;
        end
    end
end
epma_files = sort(tif_names(keep_flags));

if isempty(epma_files)
    error('No EPMA map TIFFs remain after filtering in: %s', epma_dir);
end

% Extract labels: strip grain_id prefix and trailing _itN iteration suffix
epma_labels = cell(1, numel(epma_files));
for k = 1:numel(epma_files)
    [~, base] = fileparts(epma_files{k});
    label = base;
    prefix = [grain_id, '_'];
    if startsWith(label, prefix)
        label = label(numel(prefix)+1:end);
    end
    label = regexprep(label, '_it\d+$', '');
    epma_labels{k} = label;
end

n_elements = numel(epma_files);
fprintf('Auto-discovered %d EPMA maps in: %s\n', n_elements, epma_dir);
fprintf('=== CL Local Regression Map: %s ===\n\n', grain_id);

% ---- Open analysis log ---------------------------------------------------
log_file = fullfile(output_dir, [grain_id '_local_regression_analysis_log.txt']);
log_fid  = fopen(log_file, 'w');
if log_fid == -1
    error('Cannot open log file for writing: %s', log_file);
end
lprintf = @(varargin) fprintf(log_fid, varargin{:});

DIV = ['================================================================================\n'];
SEC = ['--------------------------------------------------------------------------------\n'];

lprintf(DIV);
lprintf('CL LOCAL REGRESSION MAP ANALYSIS LOG\n');
lprintf('Grain ID:   %s\n', grain_id);
lprintf('Run time:   %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
lprintf(DIV);

lprintf('\n--- SYSTEM INFO ---\n');
lprintf('  MATLAB version:  %s\n', version);
try
    itb = ver('images');
    lprintf('  Image Processing Toolbox:  %s  (v%s)\n', itb.Name, itb.Version);
catch
    lprintf('  Image Processing Toolbox:  (version query failed)\n');
end
lprintf('  Platform:  %s\n', computer);
uname = getenv('USER');
if isempty(uname), uname = getenv('USERNAME'); end
lprintf('  User:      %s\n', uname);
lprintf('  Script:    CL_local_regression_map.m  v1.0\n');

lprintf('\n--- PARAMETERS ---\n');
lprintf('  Grain ID:              %s\n', grain_id);
lprintf('  Input directory:       %s\n', input_dir);
lprintf('  EPMA directory:        %s\n', epma_dir);
lprintf('  Output directory:      %s\n', output_dir);
lprintf('  Registered CL (gray):  %s\n', cl_filename);
lprintf('  Grain mask:            %s\n', mask_filename);
lprintf('  EPMA maps (%d total):\n', n_elements);
for e = 1:n_elements
    lprintf('    [%d]  %-20s  label: %-10s\n', e, epma_files{e}, epma_labels{e});
end
lprintf('\n  Window radius:          %.4f um  (%d px)\n', window_radius_um, window_radius_px);
lprintf('  Min valid px per window: %d\n', min_window_px);
lprintf('  Normalize EPMA maps:     %s\n', mat2str(normalize_epma));
lprintf('  Spatial calibration:     %.4f um/px\n', epma_pixel_um);
lprintf(SEC);

% =========================================================================
%% SECTION 3: LOAD REGISTERED CL, GRAIN MASK, AND EPMA MAPS
% =========================================================================

fprintf('Loading registered CL, grain mask, and EPMA maps...\n');
lprintf('\n--- INPUT FILE METADATA ---\n');

cl_path = fullfile(input_dir, cl_filename);
if ~exist(cl_path, 'file')
    fclose(log_fid);
    error('Registered CL image not found: %s\nRun CL_EPMA_registration.m for this grain first.', cl_path);
end
cl_reg = im2double(imread(cl_path));
[nrows, ncols] = size(cl_reg);
fprintf('  Registered CL loaded: %d x %d pixels\n', nrows, ncols);
log_file_info(log_fid, cl_path, 'Registered CL image (grayscale)');

mask_path = fullfile(input_dir, mask_filename);
if ~exist(mask_path, 'file')
    fclose(log_fid);
    error('Grain mask not found: %s\nRun CL_EPMA_registration.m for this grain first.', mask_path);
end
grain_mask = imread(mask_path) > 128;
if ~isequal(size(grain_mask), [nrows, ncols])
    fclose(log_fid);
    error('Grain mask size (%d x %d) does not match registered CL size (%d x %d).', ...
          size(grain_mask,1), size(grain_mask,2), nrows, ncols);
end
fprintf('  Grain mask loaded: %d px in grain.\n', sum(grain_mask(:)));
log_file_info(log_fid, mask_path, 'Grain mask');

% Warn if the grain touches the image border — outside-image pixels are
% zero-padded by conv2, which would otherwise silently bias windows there.
touches_border = any(grain_mask(1,:)) || any(grain_mask(end,:)) || ...
                 any(grain_mask(:,1)) || any(grain_mask(:,end));
if touches_border
    warning('Grain mask touches the image border — windows near the border may be biased by zero-padding.');
    lprintf('\n  ** WARNING: grain mask touches the image border. **\n');
end

% EPMA maps
epma_raw     = cell(1, n_elements);   % normalized 0-1, whole-image basis (display only)
epma_raw_abs = cell(1, n_elements);   % raw pixel values
for e = 1:n_elements
    raw_img = imread(fullfile(epma_dir, epma_files{e}));
    epma_raw_abs{e} = double(raw_img);
    epma_raw{e}     = normalize_image(raw_img);
    fprintf('  %s map loaded:  %d x %d pixels\n', ...
            epma_labels{e}, size(epma_raw{e},1), size(epma_raw{e},2));
    log_file_info(log_fid, fullfile(epma_dir, epma_files{e}), sprintf('%s map', epma_labels{e}));
end

% Sanity check: all EPMA maps must be the same size; auto-crop to smallest
% common dimensions if they differ (e.g. colorbar width variation).
epma_nrows = cellfun(@(x) size(x,1), epma_raw);
epma_ncols = cellfun(@(x) size(x,2), epma_raw);
min_rows = min(epma_nrows);
min_cols = min(epma_ncols);
if numel(unique(epma_nrows)) > 1 || numel(unique(epma_ncols)) > 1
    fprintf('  EPMA maps are not all the same size — auto-cropping to %d x %d px.\n', min_rows, min_cols);
    lprintf('\n  NOTE: EPMA maps had inconsistent sizes — auto-cropped to %d x %d.\n', min_rows, min_cols);
    for e = 1:n_elements
        epma_raw{e}     = epma_raw{e}(1:min_rows, 1:min_cols);
        epma_raw_abs{e} = epma_raw_abs{e}(1:min_rows, 1:min_cols);
    end
end

if min_rows ~= nrows || min_cols ~= ncols
    fclose(log_fid);
    error(['EPMA map grid (%d x %d) does not match registered CL size (%d x %d).\n' ...
           'epma_dir may not be the folder used during registration for this grain.'], ...
          min_rows, min_cols, nrows, ncols);
end

fprintf('\nWorking grid: %d rows x %d cols\n\n', nrows, ncols);
lprintf('\n  Working grid: %d rows x %d cols  (%.1f x %.1f um at %.4f um/px)\n', ...
        nrows, ncols, nrows*epma_pixel_um, ncols*epma_pixel_um, epma_pixel_um);
lprintf(SEC);

% --- Normalization basis for elements (mirrors CL_EPMA_registration.m) ----
epma_norm = cell(1, n_elements);
if normalize_epma
    for e = 1:n_elements
        v = epma_raw_abs{e}(grain_mask);
        vmin = min(v); vmax = max(v);
        if vmax > vmin
            epma_norm{e} = (epma_raw_abs{e} - vmin) / (vmax - vmin);
        else
            epma_norm{e} = zeros(nrows, ncols);
        end
    end
end

% =========================================================================
%% SECTION 4: CIRCULAR KERNEL AND SHARED WINDOW-COVERAGE MAP
% =========================================================================

fprintf('--- BUILDING MOVING-WINDOW KERNEL ---\n');

[Kx, Ky] = meshgrid(-window_radius_px:window_radius_px, -window_radius_px:window_radius_px);
K = double(Kx.^2 + Ky.^2 <= window_radius_px^2);
disk_area_px = sum(K(:));

m = double(grain_mask);
n_map = conv2(m, K, 'same');   % # in-mask pixels within the window, per center pixel — shared across elements

fprintf('  Window radius: %.2f um (%d px)  |  disk area: %d px  |  min valid px: %d\n', ...
        window_radius_um, window_radius_px, disk_area_px, min_window_px);
fprintf('  n_map range within grain mask: [%d, %d]\n', ...
        min(n_map(grain_mask)), max(n_map(grain_mask)));

lprintf('\n--- MOVING-WINDOW KERNEL ---\n');
lprintf('  Radius:      %.4f um  (%d px)\n', window_radius_um, window_radius_px);
lprintf('  Disk area:   %d px\n', disk_area_px);
lprintf('  Min valid px per window: %d\n', min_window_px);
lprintf('  n_map range within grain mask: [%d, %d]\n', ...
        min(n_map(grain_mask)), max(n_map(grain_mask)));
lprintf(SEC);

% =========================================================================
%% SECTION 5: PER-ELEMENT MOVING-WINDOW REGRESSION
% =========================================================================

fprintf('\n--- COMPUTING PER-PIXEL LOCAL REGRESSIONS ---\n');

y = cl_reg;
y(~grain_mask) = 0;
Sy_map  = conv2(y,    K, 'same');
Syy_map = conv2(y.^2, K, 'same');

valid_window = grain_mask & (n_map >= min_window_px);

slope_maps = NaN(nrows, ncols, n_elements);
r_maps     = NaN(nrows, ncols, n_elements);

sum_elem_label = cell(n_elements, 1);
sum_slope_mean = zeros(n_elements, 1);
sum_slope_std  = zeros(n_elements, 1);
sum_slope_min  = zeros(n_elements, 1);
sum_slope_max  = zeros(n_elements, 1);
sum_r_mean     = zeros(n_elements, 1);
sum_r_std      = zeros(n_elements, 1);
sum_r_min      = zeros(n_elements, 1);
sum_r_max      = zeros(n_elements, 1);
sum_n_valid    = zeros(n_elements, 1);
sum_pct_nan    = zeros(n_elements, 1);

for e = 1:n_elements
    if normalize_epma
        x = epma_norm{e};
    else
        x = epma_raw_abs{e};
    end
    x(~grain_mask) = 0;

    Sx_map  = conv2(x,    K, 'same');
    Sxx_map = conv2(x.^2, K, 'same');
    Sxy_map = conv2(x.*y, K, 'same');

    denom_slope = n_map.*Sxx_map - Sx_map.^2;
    denom_r_sq  = (n_map.*Sxx_map - Sx_map.^2) .* (n_map.*Syy_map - Sy_map.^2);

    slope_e = (n_map.*Sxy_map - Sx_map.*Sy_map) ./ denom_slope;
    r_e     = (n_map.*Sxy_map - Sx_map.*Sy_map) ./ sqrt(denom_r_sq);

    % Degenerate windows (near-zero denominator, e.g. constant x within the
    % window) produce Inf/NaN from 0/0 or division by ~0 — treat as invalid.
    degenerate = denom_slope <= eps(class(denom_slope)) * disk_area_px^2 | denom_r_sq <= 0;

    invalid = ~valid_window | degenerate;
    slope_e(invalid) = NaN;
    r_e(invalid)     = NaN;
    r_e = max(min(r_e, 1), -1);   % clip tiny numerical overshoot past +-1

    slope_maps(:,:,e) = slope_e;
    r_maps(:,:,e)     = r_e;

    valid_here = ~isnan(slope_e) & grain_mask;
    sum_elem_label{e} = epma_labels{e};
    sum_n_valid(e)    = sum(valid_here(:));
    sum_pct_nan(e)    = 100 * (sum(grain_mask(:)) - sum_n_valid(e)) / sum(grain_mask(:));
    sum_slope_mean(e) = mean(slope_e(valid_here));
    sum_slope_std(e)  = std(slope_e(valid_here));
    sum_slope_min(e)  = min(slope_e(valid_here));
    sum_slope_max(e)  = max(slope_e(valid_here));
    sum_r_mean(e)     = mean(r_e(valid_here));
    sum_r_std(e)       = std(r_e(valid_here));
    sum_r_min(e)      = min(r_e(valid_here));
    sum_r_max(e)      = max(r_e(valid_here));

    fprintf('  %-10s  valid px: %6d / %6d (%.1f%% NaN)  |  slope [%.4g, %.4g]  |  r [%.3f, %.3f]\n', ...
            epma_labels{e}, sum_n_valid(e), sum(grain_mask(:)), sum_pct_nan(e), ...
            sum_slope_min(e), sum_slope_max(e), sum_r_min(e), sum_r_max(e));
end

lprintf('\n--- PER-ELEMENT LOCAL REGRESSION SUMMARY (valid pixels only) ---\n');
lprintf('  %-10s  %-10s  %-8s  %-12s  %-12s  %-12s  %-12s  %-10s  %-10s  %-10s  %-10s\n', ...
        'Element', 'N_valid', 'PctNaN', 'SlopeMean', 'SlopeStd', 'SlopeMin', 'SlopeMax', 'RMean', 'RStd', 'RMin', 'RMax');
for e = 1:n_elements
    lprintf('  %-10s  %-10d  %-8.1f  %-12.4g  %-12.4g  %-12.4g  %-12.4g  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n', ...
            sum_elem_label{e}, sum_n_valid(e), sum_pct_nan(e), sum_slope_mean(e), sum_slope_std(e), ...
            sum_slope_min(e), sum_slope_max(e), sum_r_mean(e), sum_r_std(e), sum_r_min(e), sum_r_max(e));
end
lprintf(SEC);

% =========================================================================
%% SECTION 6: SAVE OUTPUTS
% =========================================================================

fprintf('\n--- SAVING OUTPUTS ---\n');

mat_file = fullfile(output_dir, [grain_id '_local_regression.mat']);
save(mat_file, 'slope_maps', 'r_maps', 'n_map', 'epma_labels', 'grain_mask', ...
     'window_radius_px', 'window_radius_um', 'min_window_px', 'normalize_epma', ...
     'epma_pixel_um', 'grain_id');
fprintf('  Maps saved to: %s\n', mat_file);

% ---- Long-format pixel data CSV: one row per valid (pixel, element) -------
[rows_idx, cols_idx] = find(grain_mask);
n_grain_px = numel(rows_idx);

csv_row    = [];
csv_col    = [];
csv_elem   = {};
csv_n      = [];
csv_slope  = [];
csv_r      = [];

for e = 1:n_elements
    slope_e = slope_maps(:,:,e);
    r_e     = r_maps(:,:,e);
    lin_idx = sub2ind([nrows, ncols], rows_idx, cols_idx);
    valid_e = ~isnan(slope_e(lin_idx));

    csv_row   = [csv_row;   rows_idx(valid_e)];                              %#ok<AGROW>
    csv_col   = [csv_col;   cols_idx(valid_e)];                              %#ok<AGROW>
    csv_elem  = [csv_elem;  repmat({epma_labels{e}}, sum(valid_e), 1)];       %#ok<AGROW>
    csv_n     = [csv_n;     n_map(lin_idx(valid_e))];                        %#ok<AGROW>
    csv_slope = [csv_slope; slope_e(lin_idx(valid_e))];                      %#ok<AGROW>
    csv_r     = [csv_r;     r_e(lin_idx(valid_e))];                          %#ok<AGROW>
end

pixel_T = table(csv_row, csv_col, csv_elem, csv_n, csv_slope, csv_r, ...
    'VariableNames', {'RowIdx', 'ColIdx', 'Element', 'N', 'Slope', 'R'});
csv_file = fullfile(output_dir, [grain_id '_local_regression_pixel_data.csv']);
writetable(pixel_T, csv_file);
fprintf('  Pixel data CSV saved to: %s  (%d rows)\n', csv_file, size(pixel_T,1));

lprintf('\n--- OUTPUT DATA ---\n');
lprintf('  Grain pixels (mask):        %d\n', n_grain_px);
lprintf('  Pixel data CSV rows:        %d  (valid pixel x element pairs)\n', size(pixel_T,1));
if normalize_epma
    lprintf('  EPMA normalisation:         in-grain-mask min/max\n');
else
    lprintf('  EPMA normalisation:         none — raw pixel counts preserved\n');
end
lprintf(SEC);

% =========================================================================
%% SECTION 7: QC FIGURES
% =========================================================================

fprintf('\n--- SAVING QC FIGURES ---\n');

n_cols2 = 3;
n_rows2 = ceil(n_elements / n_cols2);
cmap_div = diverging_cmap(256);

% ---- Slope map grid --------------------------------------------------------
fig_slope = figure('Name', 'Local slope maps', 'Position', [50 50 380*n_cols2, 340*n_rows2]);
for e = 1:n_elements
    subplot(n_rows2, n_cols2, e);
    slope_e = slope_maps(:,:,e);
    valid_here = ~isnan(slope_e);
    if any(valid_here(:))
        clim_abs = prctile(abs(slope_e(valid_here)), 98);
        if clim_abs <= 0, clim_abs = 1; end
    else
        clim_abs = 1;
    end
    disp_slope = slope_e;
    disp_slope(~grain_mask) = NaN;
    imagesc(disp_slope, [-clim_abs, clim_abs]);
    axis image off;
    colormap(gca, cmap_div);
    set(gca, 'Color', [0.85 0.85 0.85]);   % NaN background
    colorbar;
    title(sprintf('%s (slope)', epma_labels{e}), 'FontSize', 9);
end
sgtitle(sprintf('%s — local CL-vs-element slope (r = %.1f um)', grain_id, window_radius_um), 'Interpreter', 'none');
slope_qc_file = fullfile(output_dir, [grain_id '_local_regression_slope_QC.png']);
saveas(fig_slope, slope_qc_file);
fprintf('  Slope QC figure saved to: %s\n', slope_qc_file);

% ---- R map grid -------------------------------------------------------------
fig_r = figure('Name', 'Local R maps', 'Position', [50 50 380*n_cols2, 340*n_rows2]);
for e = 1:n_elements
    subplot(n_rows2, n_cols2, e);
    r_e = r_maps(:,:,e);
    disp_r = r_e;
    disp_r(~grain_mask) = NaN;
    imagesc(disp_r, [-1, 1]);
    axis image off;
    colormap(gca, cmap_div);
    set(gca, 'Color', [0.85 0.85 0.85]);
    colorbar;
    title(sprintf('%s (R)', epma_labels{e}), 'FontSize', 9);
end
sgtitle(sprintf('%s — local CL-vs-element Pearson R (r = %.1f um)', grain_id, window_radius_um), 'Interpreter', 'none');
r_qc_file = fullfile(output_dir, [grain_id '_local_regression_R_QC.png']);
saveas(fig_r, r_qc_file);
fprintf('  R QC figure saved to: %s\n', r_qc_file);

% ---- Window-coverage (n) map -----------------------------------------------
fig_n = figure('Name', 'Window coverage', 'Position', [100 100 550 500]);
disp_n = n_map;
disp_n(~grain_mask) = NaN;
imagesc(disp_n);
axis image off;
colormap(gca, parula(256));
set(gca, 'Color', [0.85 0.85 0.85]);
colorbar;
title(sprintf('%s — window coverage (n, radius = %.1f um) | min valid = %d', ...
      grain_id, window_radius_um, min_window_px), 'FontSize', 10, 'Interpreter', 'none');
n_map_file = fullfile(output_dir, [grain_id '_local_regression_n_map.png']);
saveas(fig_n, n_map_file);
fprintf('  Window coverage figure saved to: %s\n', n_map_file);

% =========================================================================
%% DONE — write log footer and close
% =========================================================================

all_outputs = { ...
    mat_file,       'Local regression maps (.mat)'; ...
    csv_file,       'Local regression pixel data (.csv)'; ...
    slope_qc_file,  'Slope map QC figure (PNG)'; ...
    r_qc_file,      'R map QC figure (PNG)'; ...
    n_map_file,     'Window coverage map (PNG)'; ...
    log_file,       'Analysis log (this file)'; ...
};

lprintf('\n--- OUTPUT FILE INVENTORY ---\n');
lprintf('  %-40s  %-12s  %s\n', 'Description', 'Size (bytes)', 'Path');
lprintf('  %-40s  %-12s  %s\n', repmat('-',1,40), repmat('-',1,12), repmat('-',1,20));
for f = 1:size(all_outputs, 1)
    fpath = all_outputs{f,1};
    label = all_outputs{f,2};
    d = dir(fpath);
    if ~isempty(d)
        lprintf('  %-40s  %-12d  %s\n', label, d.bytes, fpath);
    else
        lprintf('  %-40s  %-12s  %s\n', label, '[not found]', fpath);
    end
end

lprintf('\n');
lprintf(DIV);
lprintf('END OF LOG\n');
lprintf(DIV);
fclose(log_fid);

fprintf('\n=== COMPLETE ===\n');
fprintf('All outputs written to: %s\n', output_dir);
fprintf('Key files:\n');
fprintf('  %s_local_regression_analysis_log.txt   — comprehensive run record\n', grain_id);
fprintf('  %s_local_regression.mat                — slope/R/n maps + metadata\n', grain_id);
fprintf('  %s_local_regression_pixel_data.csv     — long-format per-pixel table\n', grain_id);
fprintf('  %s_local_regression_slope_QC.png\n', grain_id);
fprintf('  %s_local_regression_R_QC.png\n', grain_id);
fprintf('  %s_local_regression_n_map.png\n\n', grain_id);

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================

function img_norm = normalize_image(img_raw)
% Accepts any bit-depth image and returns double in [0, 1].
% RGB inputs are converted to grayscale with standard luminance weights.
    img_d = double(img_raw);
    if ndims(img_d) == 3
        warning(['Input image has %d channels — converting to grayscale. ' ...
                 'If this is a false-color map, check that grayscale ' ...
                 'conversion preserves the intended intensity gradient.'], ...
                size(img_d,3));
        img_d = 0.2989*img_d(:,:,1) + 0.5870*img_d(:,:,2) + 0.1140*img_d(:,:,3);
    end
    img_min = min(img_d(:));
    img_max = max(img_d(:));
    if img_max == img_min
        warning('Image has uniform intensity — normalization will produce zeros.');
        img_norm = zeros(size(img_d));
    else
        img_norm = (img_d - img_min) / (img_max - img_min);
    end
end

function log_file_info(fid, fpath, label)
% Write image file metadata (dimensions, bit depth, size, date) to log.
    d = dir(fpath);
    try
        info = imfinfo(fpath);
        nchan = 1;
        if isfield(info(1), 'SamplesPerPixel'), nchan = info(1).SamplesPerPixel; end
        fprintf(fid, '  %s:\n', label);
        fprintf(fid, '    Path:       %s\n', fpath);
        fprintf(fid, '    Dimensions: %d x %d px  (%d channel(s))\n', ...
                info(1).Height, info(1).Width, nchan);
        fprintf(fid, '    Bit depth:  %d-bit\n', info(1).BitDepth);
        if ~isempty(d)
            fprintf(fid, '    File size:  %d bytes\n', d.bytes);
            fprintf(fid, '    Modified:   %s\n', d.date);
        end
    catch
        fprintf(fid, '  %s:  %s\n', label, fpath);
        if ~isempty(d)
            fprintf(fid, '    File size:  %d bytes  |  Modified: %s\n', d.bytes, d.date);
        else
            fprintf(fid, '    (imfinfo failed; file may not exist yet)\n');
        end
    end
end

function cmap = diverging_cmap(n)
% Blue-white-red diverging colormap, built by linear RGB interpolation.
% Avoids a Bioinformatics Toolbox dependency (redbluecmap).
    control_pos = [0, 0.5, 1];
    control_rgb = [0.15 0.25 0.65;    % blue
                   1.00 1.00 1.00;    % white
                   0.70 0.10 0.15];   % red
    t = linspace(0, 1, n)';
    cmap = [interp1(control_pos, control_rgb(:,1), t, 'linear'), ...
            interp1(control_pos, control_rgb(:,2), t, 'linear'), ...
            interp1(control_pos, control_rgb(:,3), t, 'linear')];
end
