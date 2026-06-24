% =========================================================================
% CL_EPMA_REGISTRATION.m
%
% PURPOSE:
%   register a cathodoluminescence (CL) image to one or more EPMA element
%   maps of the same grain, create a grain mask from the registered CL
%   image, extract per-pixel chemistry and CL intensity data, and produce
%   exploratory scatter plots of CL vs. element concentrations.
%
% WORKFLOW:
%   1. Load CL and EPMA images (handyles 8-, 16-, and 32-bit TIFFs)
%   2. Interactively pick control points to register CL onto EPMA grid
%   3. Evaluate registration quality (RMSE in pixels and microns)
%   4. Build a binary grain mask from the registered CL image
%   5. Apply mask to extract pixel vectors for all maps
%   6. Scatter plot CL vs. each element; compute Pearson r
%   7. Run shift-sensitivity analysis to quantify alignment error impact
%   8. Write comprehensive analysis log throughout
%
% INPUTS (set in PARAMETERS section below):
%   - CL image (.tif, .png, .bmp, or any format supported by imread)
%   - Folder of EPMA element map TIFFs (auto-discovered; labels from filenames)
%   - Pixel size of EPMA map in microns (for spatial error reporting)
%
% OUTPUTS:
%   - Registered CL image saved as TIFF
%   - Grain mask saved as TIFF
%   - [grain_id]_analysis_log.txt — comprehensive run record
%   - Scatter plots saved as .png
%   - Pixel data matrix saved as .mat and .csv
%
% REQUIREMENTS:
%   - MATLAB Image Processing Toolbox (for cpselect, imwarp, etc.)
%
% AUTHOR:  M. Stein
% DATE:    [date]
% VERSION: 1.1
% =========================================================================

clear; clc; close all;

% Disable TeX interpreter globally so underscores in labels (e.g. Fe_Ka)
% are displayed literally rather than as subscript markup.
set(0, 'DefaultTextInterpreter',        'none');
set(0, 'DefaultAxesTickLabelInterpreter','none');
set(0, 'DefaultLegendInterpreter',      'none');

% =========================================================================
%% SECTION 1: PARAMETERS  — edit this section for each new grain
% =========================================================================

% --- File paths -----------------------------------------------------------
input_dir  = '/Users/mstein/bin/kyanite';

grain_id    = 'NA-SS-P1156-02';

cl_filename = [grain_id, '_CL_color.bmp'];

% Folder containing EPMA element map TIFFs.
% All *.tif files in this folder are auto-discovered as element maps.
% The CL file and any script-generated outputs are excluded automatically.
% Labels are extracted from filenames by stripping the grain_id prefix
% and any trailing _itN iteration suffix (e.g. _Fe_Ka_it5 -> Fe_Ka).
epma_dir = ['/Users/mstein/bin/kyanite/maps/', grain_id];

% Reference map filename (filename only, no path).
% This map is the FIXED reference for control point selection —
% choose your highest-quality, highest-contrast map.
% Leave empty to use the first file found alphabetically.
epma_ref_file = [grain_id, '_Cr_Ka.tif'];   % e.g., 'NA-CM-G12B4-02_Fe_Ka_it5.tif'

output_dir  = '/Users/mstein/bin/kyanite/figs';

% --- Spatial calibration --------------------------------------------------
epma_pixel_um = 2.0;            % µm per pixel

% --- Mask parameters ------------------------------------------------------
mask_method   = 'otsu';         % otsu | manual | interactive | polygon | activecontour
thresh_manual = 0.12;             % used for manual; fallback if interactive receives no clicks
min_object_px = 500;
fill_holes    = false;

% Active contour parameters (used only when mask_method = 'activecontour')
ac_niter        = 200;      % number of Chan-Vese iterations
ac_smoothfactor = 1.0;      % contour smoothness (higher = smoother boundary)
ac_init_method  = 'polygon';   % initial mask source: 'otsu' | 'manual' | 'polygon'

% --- Registration parameters ----------------------------------------------
transform_type = 'affine';

% Percentile range used to contrast-stretch element maps in all figures.
% Does not affect registration, scatter plot data, or any outputs — display only.
% Tighten (e.g. [5, 95]) for maps with many low-signal pixels; widen for
% already-uniform images.  Set to [0, 100] to disable stretching.
display_pct = [0, 97];

% --- Shift-sensitivity analysis parameters --------------------------------
shift_range   = -30:1:30;

% --- Outlier removal ------------------------------------------------------
% Percentile cutoffs applied per-element on the x-axis in scatter plots and
% for Pearson r. Set pct_lo_cut = 0 and pct_hi_cut = 100 to disable.
pct_lo_cut =  0;   % lower cutoff: exclude bottom pct_lo_cut % of element values
pct_hi_cut = 99;   % upper cutoff: exclude top (100 - pct_hi_cut) % of element values

% --- Element map normalization --------------------------------------------
% true  (default): normalize each EPMA/XRF map to [0 1] using its in-mask
%         min/max — good for EPMA maps where absolute counts are arbitrary.
% false: keep raw pixel values (counts/intensity units from the TIFF) —
%         use for XAFS/XRF maps whose intensities are comparable across grains.
normalize_epma = false;

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
output_suffixes = {'_CL_registered.tif', '_mask.tif'};
keep_flags = true(1, numel(tif_names));
for k = 1:numel(tif_names)
    fname = tif_names{k};
    if strcmp(fname, cl_filename)
        keep_flags(k) = false;
        continue;
    end
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

% Move the reference file to front if specified
if ~isempty(epma_ref_file)
    ref_idx = find(strcmp(epma_files, epma_ref_file), 1);
    if isempty(ref_idx)
        warning('epma_ref_file ''%s'' not found; using first file alphabetically.', epma_ref_file);
    else
        epma_files = [epma_files(ref_idx), epma_files([1:ref_idx-1, ref_idx+1:end])];
    end
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

fprintf('=== CL-EPMA Registration: %s ===\n\n', grain_id);

% ---- Open analysis log ---------------------------------------------------
% The log is written incrementally throughout script execution, so a
% partial log survives even if the script fails partway through.
log_file = fullfile(output_dir, [grain_id '_analysis_log.txt']);
log_fid  = fopen(log_file, 'w');
if log_fid == -1
    error('Cannot open log file for writing: %s', log_file);
end

% lprintf writes to the log file with the same syntax as fprintf.
% log_fid is captured by value (it is an integer handle) so this is stable.
lprintf = @(varargin) fprintf(log_fid, varargin{:});

DIV = ['================================================================================\n'];
SEC = ['--------------------------------------------------------------------------------\n'];

% ---- Log header ----------------------------------------------------------
lprintf(DIV);
lprintf('CL-EPMA REGISTRATION ANALYSIS LOG\n');
lprintf('Grain ID:   %s\n', grain_id);
lprintf('Run time:   %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
lprintf(DIV);

% ---- System info ---------------------------------------------------------
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
lprintf('  Script:    CL_EPMA_registration.m  v1.1\n');

% ---- Parameters ----------------------------------------------------------
lprintf('\n--- PARAMETERS ---\n');
lprintf('  Grain ID:            %s\n', grain_id);
lprintf('  CL directory:        %s\n', input_dir);
lprintf('  EPMA directory:      %s\n', epma_dir);
lprintf('  Output directory:    %s\n', output_dir);
lprintf('\n  CL image (moving):   %s\n', cl_filename);
lprintf('  EPMA maps (%d total):\n', n_elements);
for e = 1:n_elements
    if e == 1
        ref_tag = '  <-- fixed reference for registration';
    else
        ref_tag = '';
    end
    lprintf('    [%d]  %-20s  label: %-10s%s\n', ...
            e, epma_files{e}, epma_labels{e}, ref_tag);
end
lprintf('\n  Registration:\n');
lprintf('    Transform type:    %s\n', transform_type);
lprintf('    Display stretch:   %g–%g pct (all element map figures)\n', display_pct(1), display_pct(2));
switch transform_type
    case 'affine';     lprintf('    Min control pts:   6\n');
    case 'similarity'; lprintf('    Min control pts:   4\n');
    case 'rigid';      lprintf('    Min control pts:   3\n');
    otherwise;         lprintf('    Min control pts:   unknown transform type\n');
end
lprintf('\n  Mask:\n');
lprintf('    Method:            %s\n', mask_method);
switch mask_method
    case 'manual'
        lprintf('    Manual threshold:  %.4f  [active]\n', thresh_manual);
    case 'interactive'
        lprintf('    Manual threshold:  %.4f  [fallback if no clicks received]\n', thresh_manual);
    case 'otsu'
        lprintf('    Manual threshold:  %.4f  [inactive — otsu used]\n', thresh_manual);
    case 'activecontour'
        lprintf('    AC iterations:     %d\n', ac_niter);
        lprintf('    AC smooth factor:  %.4f\n', ac_smoothfactor);
        lprintf('    AC init method:    %s\n', ac_init_method);
        if strcmp(ac_init_method, 'manual')
            lprintf('    AC init threshold: %.4f\n', thresh_manual);
        end
    % polygon: no threshold parameters
end
lprintf('    Min object size:   %d px\n', min_object_px);
lprintf('    Fill holes:        %s\n', mat2str(fill_holes));
lprintf('\n  Spatial calibration: %.4f µm/px\n', epma_pixel_um);
if length(shift_range) > 1
    lprintf('  Shift test range:    %d to %d px  (step %g)\n', ...
            min(shift_range), max(shift_range), shift_range(2)-shift_range(1));
else
    lprintf('  Shift test range:    %d px\n', shift_range(1));
end
if pct_lo_cut > 0 || pct_hi_cut < 100
    lprintf('  Outlier removal:     %.4g–%.4gth pct per element used in plots/r\n', pct_lo_cut, pct_hi_cut);
else
    lprintf('  Outlier removal:     disabled (pct_lo_cut = 0, pct_hi_cut = 100)\n');
end
lprintf(SEC);

% =========================================================================
%% SECTION 3: LOAD IMAGES
% =========================================================================

fprintf('Loading images...\n');
lprintf('\n--- INPUT FILE METADATA ---\n');

load_normalize = @(fpath) normalize_image(imread(fullfile(input_dir, fpath)));

% CL image
cl_raw = load_normalize(cl_filename);
fprintf('  CL image loaded:   %d x %d pixels\n', size(cl_raw,1), size(cl_raw,2));
log_file_info(log_fid, fullfile(input_dir, cl_filename), 'CL image (moving)');
lprintf('    Normalized stats:  min=%.4f  max=%.4f  mean=%.4f  std=%.4f\n', ...
        min(cl_raw(:)), max(cl_raw(:)), mean(cl_raw(:)), std(cl_raw(:)));

% EPMA maps
epma_raw = cell(1, n_elements);
if ~normalize_epma; epma_raw_abs = cell(1, n_elements); end
for e = 1:n_elements
    raw_img = imread(fullfile(epma_dir, epma_files{e}));
    if ~normalize_epma
        epma_raw_abs{e} = double(raw_img);
    end
    epma_raw{e} = normalize_image(raw_img);
    fprintf('  %s map loaded:  %d x %d pixels\n', ...
            epma_labels{e}, size(epma_raw{e},1), size(epma_raw{e},2));
    log_file_info(log_fid, fullfile(epma_dir, epma_files{e}), ...
                  sprintf('%s map', epma_labels{e}));
    lprintf('    Normalized stats:  min=%.4f  max=%.4f  mean=%.4f  std=%.4f\n', ...
            min(epma_raw{e}(:)), max(epma_raw{e}(:)), ...
            mean(epma_raw{e}(:)), std(epma_raw{e}(:)));
end

% Sanity check: all EPMA maps must be the same size
% If maps differ in size (e.g. colorbar widths vary), crop all to the
% smallest dimensions by trimming the right and/or bottom edges.
epma_nrows = cellfun(@(x) size(x,1), epma_raw);
epma_ncols = cellfun(@(x) size(x,2), epma_raw);
min_rows = min(epma_nrows);
min_cols = min(epma_ncols);

if numel(unique(epma_nrows)) > 1 || numel(unique(epma_ncols)) > 1
    fprintf('  EPMA maps are not all the same size — auto-cropping to %d x %d px.\n', ...
            min_rows, min_cols);
    lprintf('\n  NOTE: EPMA maps had inconsistent sizes (likely colorbar width variation).\n');
    lprintf('  Auto-cropped all maps to smallest dimensions: %d rows x %d cols.\n', ...
            min_rows, min_cols);
    lprintf('  Per-map original sizes:\n');
    for e = 1:n_elements
        lprintf('    [%d] %-20s  %d x %d px', e, epma_files{e}, epma_nrows(e), epma_ncols(e));
        if epma_nrows(e) ~= min_rows || epma_ncols(e) ~= min_cols
            lprintf('  --> cropped %d col(s), %d row(s) from right/bottom', ...
                    epma_ncols(e)-min_cols, epma_nrows(e)-min_rows);
        end
        lprintf('\n');
    end
    for e = 1:n_elements
        epma_raw{e} = epma_raw{e}(1:min_rows, 1:min_cols);
    end
end

epma_ref = epma_raw{1};
[nrows_epma, ncols_epma] = size(epma_ref);
fprintf('\nEPMA grid (reference): %d rows x %d cols\n\n', nrows_epma, ncols_epma);
lprintf('  EPMA grid:  %d rows x %d cols  (%.1f x %.1f µm at %.4f µm/px)\n', ...
        nrows_epma, ncols_epma, ...
        nrows_epma*epma_pixel_um, ncols_epma*epma_pixel_um, epma_pixel_um);
lprintf(SEC);

% =========================================================================
%% SECTION 4: CONTROL POINT REGISTRATION
% =========================================================================

fprintf('--- CONTROL POINT SELECTION ---\n');
fprintf('Instructions:\n');
fprintf('  Left  panel = CL image (moving)\n');
fprintf('  Right panel = %s EPMA map (fixed)\n', epma_labels{1});
fprintf('  Pick at least 6 well-distributed pairs of matching points.\n');
fprintf('  Close the window when done.\n\n');

cp_savefile = fullfile(output_dir, [grain_id '_controlpoints.mat']);
cp_source   = 'newly selected via cpselect';

if exist(cp_savefile, 'file')
    resp = input('Saved control points found. Use them? (y/n): ', 's');
    if strcmpi(resp, 'y')
        load(cp_savefile, 'moving_pts', 'fixed_pts');
        fprintf('Loaded saved control points (%d pairs).\n', size(moving_pts,1));
        cp_source = ['loaded from file: ', cp_savefile];
        skip_cpselect = true;
    else
        skip_cpselect = false;
    end
else
    skip_cpselect = false;
end

if ~skip_cpselect
    epma_ref_disp = pct_stretch(epma_ref, display_pct(1), display_pct(2));
    [moving_pts, fixed_pts] = cpselect(cl_raw, epma_ref_disp, 'Wait', true);
    if size(moving_pts, 1) < 3
        fclose(log_fid);
        error('You need at least 3 control point pairs. Please rerun.');
    end
    fprintf('%d control point pairs selected.\n', size(moving_pts,1));
    save(cp_savefile, 'moving_pts', 'fixed_pts');
    fprintf('Control points saved to: %s\n', cp_savefile);
end

% ---- Log control points --------------------------------------------------
n_cp = size(moving_pts, 1);
lprintf('\n--- CONTROL POINTS ---\n');
lprintf('  Source:  %s\n', cp_source);
lprintf('  Count:   %d pairs\n', n_cp);
lprintf('  Coordinate system: (X,Y) = (column, row), 1-indexed, pixel centres.\n');
lprintf('  Moving = CL image.  Fixed = %s EPMA map (reference).\n\n', epma_labels{1});
lprintf('  %-5s  %-12s  %-12s  %-12s  %-12s\n', ...
        '#', 'Moving_X', 'Moving_Y', 'Fixed_X', 'Fixed_Y');
lprintf('  %-5s  %-12s  %-12s  %-12s  %-12s\n', ...
        '-----', '------------', '------------', '------------', '------------');
for k = 1:n_cp
    lprintf('  %-5d  %-12.4f  %-12.4f  %-12.4f  %-12.4f\n', ...
            k, moving_pts(k,1), moving_pts(k,2), ...
               fixed_pts(k,1),  fixed_pts(k,2));
end

% ---- Fit the geometric transform -----------------------------------------
fprintf('\nFitting %s transform...\n', transform_type);
tform = fitgeotrans(moving_pts, fixed_pts, transform_type);

% ---- Log transform matrix and reconstruction recipe ----------------------
Tmat = tform.T;   % 3x3 matrix; use Tmat to avoid clash with table var later
lprintf('\n--- GEOMETRIC TRANSFORM ---\n');
lprintf('  Type:  %s\n\n', transform_type);
lprintf('  Transformation matrix T (3x3):\n');
lprintf('  Convention:  [u, v, 1] = [x, y, 1] * T\n');
lprintf('  where (x,y) are CL pixel coords and (u,v) are EPMA pixel coords.\n\n');
lprintf('         col1 (u)        col2 (v)        col3\n');
for r = 1:3
    lprintf('  row%d   %-15.10f  %-15.10f  %-15.10f\n', r, Tmat(r,1), Tmat(r,2), Tmat(r,3));
end
lprintf('\n  Scalar forward mapping:\n');
lprintf('    u  =  %.10f * x  +  %.10f * y  +  %.10f\n', Tmat(1,1), Tmat(2,1), Tmat(3,1));
lprintf('    v  =  %.10f * x  +  %.10f * y  +  %.10f\n', Tmat(1,2), Tmat(2,2), Tmat(3,2));
lprintf('\n  MATLAB reconstruction recipe:\n');
lprintf('    T = [%.10f  %.10f  %.10f; ...\n', Tmat(1,:));
lprintf('         %.10f  %.10f  %.10f; ...\n', Tmat(2,:));
lprintf('         %.10f  %.10f  %.10f];\n',    Tmat(3,:));
lprintf('    tform     = affine2d(T);\n');
lprintf('    ref_out   = imref2d([%d, %d]);  %% [nrows ncols] of EPMA grid\n', ...
        nrows_epma, ncols_epma);
lprintf('    cl_reg    = imwarp(cl_raw, tform, ''OutputView'', ref_out, ...\n');
lprintf('                       ''Interp'', ''bilinear'');\n');

% ---- Evaluate registration quality ---------------------------------------
moving_pts_warped = transformPointsForward(tform, moving_pts);
residuals_px = fixed_pts - moving_pts_warped;
residual_dist = sqrt(sum(residuals_px.^2, 2));
RMSE_px  = sqrt(mean(residual_dist.^2));
RMSE_um  = RMSE_px * epma_pixel_um;
max_res  = max(residual_dist);

fprintf('\n  Registration quality:\n');
fprintf('    RMSE:            %.3f pixels  (%.2f µm)\n', RMSE_px, RMSE_um);
fprintf('    Max residual:    %.3f pixels  (%.2f µm)\n', max_res, max_res*epma_pixel_um);
fprintf('    Control points:  %d pairs\n\n', n_cp);

if RMSE_px > 3
    warning('RMSE > 3 pixels. Consider picking more or better-distributed control points.');
end

lprintf('\n--- REGISTRATION QUALITY ---\n');
lprintf('  RMSE:           %.6f px   (%.4f µm)\n', RMSE_px, RMSE_um);
lprintf('  Max residual:   %.6f px   (%.4f µm)\n', max_res, max_res*epma_pixel_um);
lprintf('  Control points: %d pairs\n', n_cp);
if RMSE_px > 3
    lprintf('  ** WARNING: RMSE > 3 pixels — consider re-picking control points **\n');
end
lprintf('\n  Per-point residuals (in EPMA-grid pixels; 1 px = %.4f µm):\n', epma_pixel_um);
lprintf('  %-5s  %-10s  %-10s  %-11s  %-11s  %-10s\n', ...
        '#', 'Fixed_X', 'Fixed_Y', 'Resid_X', 'Resid_Y', 'Dist_px');
lprintf('  %-5s  %-10s  %-10s  %-11s  %-11s  %-10s\n', ...
        '-----', '----------', '----------', '-----------', '-----------', '----------');
for k = 1:n_cp
    lprintf('  %-5d  %-10.3f  %-10.3f  %-11.5f  %-11.5f  %-10.5f\n', ...
            k, fixed_pts(k,1), fixed_pts(k,2), ...
            residuals_px(k,1), residuals_px(k,2), residual_dist(k));
end
lprintf(SEC);

% ---- Apply transform to warp CL onto EPMA grid ---------------------------
ref_epma = imref2d([nrows_epma, ncols_epma]);
cl_reg   = imwarp(cl_raw, tform, 'OutputView', ref_epma, 'Interp', 'bilinear');
fprintf('CL image registered to EPMA grid.\n');

cl_reg_file = fullfile(output_dir, [grain_id '_CL_registered.tif']);
imwrite(uint16(cl_reg * 65535), cl_reg_file);
fprintf('Registered CL saved to: %s\n\n', cl_reg_file);

% ---- Visualize registration quality --------------------------------------
figure('Name', 'Registration check', 'Position', [100 100 1200 400]);
subplot(1,3,1); imshow(cl_raw);                                                   title('CL: original');
subplot(1,3,2); imshow(pct_stretch(epma_ref, display_pct(1), display_pct(2)));    title([epma_labels{1}, ' EPMA (reference)']);
subplot(1,3,3); imshow(cl_reg);                                                   title('CL: registered to EPMA grid');
sgtitle(sprintf('%s — Registration RMSE: %.2f px (%.2f µm)', ...
        grain_id, RMSE_px, RMSE_um));

figure('Name', 'Overlay check');
overlay = cat(3, pct_stretch(epma_ref, display_pct(1), display_pct(2)), cl_reg, zeros(size(epma_ref)));
imshow(overlay);
title({'Overlay check: red = EPMA ref, green = registered CL', ...
       'Edges should align at grain boundary'});
saveas(gcf, fullfile(output_dir, [grain_id '_registration_overlay.png']));

% =========================================================================
%% SECTION 5: BUILD GRAIN MASK
% =========================================================================

fprintf('\n--- BUILDING GRAIN MASK ---\n');

cl_disp  = pct_stretch(cl_reg, display_pct(1), display_pct(2));
thresh   = NaN;    % set for threshold-based methods; NaN otherwise
poly_pos = [];     % set for polygon-based methods

switch mask_method

    case 'otsu'
        thresh = graythresh(cl_reg);
        fprintf('  Otsu threshold: %.4f\n', thresh);
        mask = cl_reg > thresh;

    case 'manual'
        thresh = thresh_manual;
        fprintf('  Manual threshold: %.4f\n', thresh);
        mask = cl_reg > thresh;

    case 'interactive'
        all_bg_x    = [];
        all_bg_y    = [];
        all_bg_vals = [];
        thresh      = thresh_manual;   % fallback if no points are selected

        fprintf('  Interactive mask mode.\n');
        fprintf('  Click on BACKGROUND pixels (non-grain areas) in the figure.\n');
        fprintf('  Press Enter when done clicking each round.\n\n');

        done = false;
        while ~done
            fig_pk = figure('Name', 'Interactive mask — click background pixels', ...
                            'Position', [50 50 900 700]);
            imshow(cl_disp);
            title({'Click on BACKGROUND pixels (outside the grain).', ...
                   'Press Enter when done clicking.'}, 'FontSize', 11);
            hold on;
            if ~isempty(all_bg_x)
                scatter(all_bg_x, all_bg_y, 60, 'r', 'filled', ...
                        'MarkerEdgeColor', 'w', 'LineWidth', 0.8);
            end

            [new_x, new_y] = ginput();
            close(fig_pk);

            if ~isempty(new_x)
                new_x    = max(1, min(ncols_epma, round(new_x)));
                new_y    = max(1, min(nrows_epma, round(new_y)));
                new_vals = cl_reg(sub2ind(size(cl_reg), new_y, new_x));
                all_bg_x    = [all_bg_x;    new_x(:)];
                all_bg_y    = [all_bg_y;    new_y(:)];
                all_bg_vals = [all_bg_vals; new_vals(:)];
            end

            if isempty(all_bg_vals)
                fprintf('  No points selected yet — using manual threshold %.4f.\n', thresh_manual);
                thresh = thresh_manual;
            else
                thresh = max(all_bg_vals);
                fprintf('  %d background sample(s): values in [%.4f, %.4f]\n', ...
                        numel(all_bg_vals), min(all_bg_vals), max(all_bg_vals));
                fprintf('  Threshold = max background value: %.4f\n', thresh);
            end

            % Build preview mask with current threshold
            mask_preview = cl_reg > thresh;
            if min_object_px > 0
                mask_preview = bwareaopen(mask_preview, min_object_px);
            end
            if fill_holes
                mask_preview = imfill(mask_preview, 'holes');
            end

            fig_prev = figure('Name', 'Mask preview', 'Position', [100 100 1000 420]);
            subplot(1,2,1);
            imshow(cl_disp); hold on;
            if ~isempty(all_bg_x)
                scatter(all_bg_x, all_bg_y, 60, 'r', 'filled', ...
                        'MarkerEdgeColor', 'w', 'LineWidth', 0.8);
            end
            title(sprintf('Background samples (%d pts)', numel(all_bg_vals)));
            subplot(1,2,2);
            imshow(cl_disp); hold on;
            visboundaries(mask_preview, 'Color', 'r', 'LineWidth', 1.5);
            title(sprintf('Mask preview  thresh: %.4f  |  %d px in grain', ...
                          thresh, sum(mask_preview(:))));
            sgtitle(sprintf('%s — interactive mask', grain_id));
            drawnow;

            resp = input(['  Accept? (y = yes,  n = add more points,\n' ...
                          '          or enter a number to override threshold): '], 's');
            resp = strtrim(resp);

            if strcmpi(resp, 'y')
                done = true;
            elseif ~isempty(resp) && ~isnan(str2double(resp))
                thresh = str2double(resp);
                fprintf('  Threshold manually overridden to: %.4f\n', thresh);
                done = true;
            end
            % 'n' or unrecognized: close preview and loop back for more clicks
            close(fig_prev);
        end
        fprintf('  Interactive threshold accepted: %.4f\n', thresh);
        mask = cl_reg > thresh;

    case 'polygon'
        % User traces the grain outline directly — no brightness threshold.
        % Robust to any CL intensity variation inside or outside the grain.
        fprintf('  Polygon mask mode.\n');
        fprintf('  Click to place vertices around the grain boundary.\n');
        fprintf('  Double-click the last vertex (or first) to close the polygon.\n\n');

        done = false;
        while ~done
            fig_poly = figure('Name', 'Draw grain boundary — double-click to close', ...
                              'Position', [50 50 900 700]);
            imshow(cl_disp);
            title({'Click to place vertices around the grain boundary.', ...
                   'Double-click the last vertex to close and accept.'}, 'FontSize', 11);

            h_poly = drawpolygon();
            wait(h_poly);

            if isempty(h_poly.Position)
                close(fig_poly);
                fclose(log_fid);
                error('No polygon drawn. Please rerun and place at least 3 vertices.');
            end
            poly_pos = h_poly.Position;   % Nx2: [x (col), y (row)]
            close(fig_poly);

            mask_preview = poly2mask(poly_pos(:,1), poly_pos(:,2), nrows_epma, ncols_epma);
            if min_object_px > 0
                mask_preview = bwareaopen(mask_preview, min_object_px);
            end
            if fill_holes
                mask_preview = imfill(mask_preview, 'holes');
            end

            fig_prev = figure('Name', 'Polygon mask preview', 'Position', [100 100 1000 420]);
            subplot(1,2,1);
            imshow(cl_disp); hold on;
            plot([poly_pos(:,1); poly_pos(1,1)], [poly_pos(:,2); poly_pos(1,2)], ...
                 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', 'b');
            title(sprintf('Polygon (%d vertices)', size(poly_pos,1)));
            subplot(1,2,2);
            imshow(cl_disp); hold on;
            visboundaries(mask_preview, 'Color', 'r', 'LineWidth', 1.5);
            title(sprintf('Mask preview  |  %d px in grain', sum(mask_preview(:))));
            sgtitle(sprintf('%s — polygon mask', grain_id));
            drawnow;

            resp = input('  Accept polygon? (y = yes, n = redraw): ', 's');
            close(fig_prev);

            if strcmpi(strtrim(resp), 'y')
                done = true;
            end
        end

        mask = poly2mask(poly_pos(:,1), poly_pos(:,2), nrows_epma, ncols_epma);
        fprintf('  Polygon accepted: %d vertices, %d px enclosed.\n', ...
                size(poly_pos,1), sum(mask(:)));

    case 'activecontour'
        % Chan-Vese active contours: evolves a contour to fit the grain boundary
        % using image statistics rather than a fixed threshold. Robust to
        % heterogeneous CL brightness and bright inclusions/holes inside the grain.
        fprintf('  Active contour mask mode  (Chan-Vese, %d iter, smoothfactor=%.2f).\n', ...
                ac_niter, ac_smoothfactor);
        fprintf('  Initializing contour from: %s\n\n', ac_init_method);

        switch ac_init_method
            case 'otsu'
                init_thresh = graythresh(cl_reg);
                init_mask   = cl_reg > init_thresh;
                init_mask   = bwareaopen(init_mask, min_object_px);
                fprintf('  Otsu seed threshold: %.4f  (%d px)\n', init_thresh, sum(init_mask(:)));
            case 'manual'
                init_thresh = thresh_manual;
                init_mask   = cl_reg > init_thresh;
                init_mask   = bwareaopen(init_mask, min_object_px);
                fprintf('  Manual seed threshold: %.4f  (%d px)\n', init_thresh, sum(init_mask(:)));
            case 'polygon'
                fprintf('  Draw a rough polygon to seed the active contour.\n');
                fprintf('  Precision is not required — just enclose the grain.\n\n');
                fig_ac_init = figure('Name', 'Draw seed region for active contours', ...
                                     'Position', [50 50 900 700]);
                imshow(cl_disp);
                title({'Draw a rough polygon enclosing the grain (seed for active contours).', ...
                       'It does not need to be precise — double-click to close.'}, 'FontSize', 11);
                h_init = drawpolygon();
                wait(h_init);
                if isempty(h_init.Position)
                    close(fig_ac_init);
                    fclose(log_fid);
                    error('No seed polygon drawn. Please rerun.');
                end
                pos_init  = h_init.Position;
                close(fig_ac_init);
                init_mask = poly2mask(pos_init(:,1), pos_init(:,2), nrows_epma, ncols_epma);
                fprintf('  Outer polygon: %d vertices, %d px.\n', size(pos_init,1), sum(init_mask(:)));

                % Optionally subtract hole polygons from the seed mask
                n_holes = 0;
                while true
                    resp_hole = input('  Mark a hole in the seed mask? (y = draw hole, n = done): ', 's');
                    if ~strcmpi(strtrim(resp_hole), 'y'), break; end

                    fig_hole = figure('Name', 'Draw hole region — double-click to close', ...
                                      'Position', [50 50 900 700]);
                    imshow(cl_disp); hold on;
                    visboundaries(init_mask, 'Color', 'g', 'LineWidth', 1.5);
                    title({'Draw a polygon over a HOLE to exclude from the seed mask.', ...
                           'Green boundary = current seed.  Double-click to close.'}, 'FontSize', 11);
                    h_hole = drawpolygon();
                    wait(h_hole);
                    if ~isempty(h_hole.Position)
                        pos_hole  = h_hole.Position;
                        hole_mask = poly2mask(pos_hole(:,1), pos_hole(:,2), nrows_epma, ncols_epma);
                        init_mask = init_mask & ~hole_mask;
                        n_holes   = n_holes + 1;
                        fprintf('  Hole %d: %d vertices, %d px removed. Seed now: %d px.\n', ...
                                n_holes, size(pos_hole,1), sum(hole_mask(:)), sum(init_mask(:)));
                    end
                    close(fig_hole);
                end
                if n_holes > 0
                    fprintf('  Total holes drawn: %d\n', n_holes);
                end
            otherwise
                fclose(log_fid);
                error('ac_init_method must be ''otsu'', ''manual'', or ''polygon''.');
        end

        fprintf('  Running active contours...\n');
        mask = activecontour(cl_reg, init_mask, ac_niter, 'Chan-Vese', ...
                             'SmoothFactor', ac_smoothfactor);
        fprintf('  Converged: %d px in grain.\n', sum(mask(:)));

        fig_ac_result = figure('Name', 'Active contour result', 'Position', [100 100 1000 420]);
        subplot(1,2,1);
        imshow(cl_disp); hold on;
        visboundaries(init_mask, 'Color', 'b', 'LineWidth', 1.5);
        title('Seed mask (blue)');
        subplot(1,2,2);
        imshow(cl_disp); hold on;
        visboundaries(mask, 'Color', 'r', 'LineWidth', 1.5);
        title(sprintf('Active contour result  |  %d px', sum(mask(:))));
        sgtitle(sprintf('%s — active contour mask (%d iter)', grain_id, ac_niter));
        drawnow;

    otherwise
        fclose(log_fid);
        error('mask_method must be ''otsu'', ''manual'', ''interactive'', ''polygon'', or ''activecontour''.');
end

n_px_raw = sum(mask(:));

if min_object_px > 0
    mask = bwareaopen(mask, min_object_px);
end
if fill_holes
    mask = imfill(mask, 'holes');
end

n_grain_px = sum(mask(:));
fprintf('  Grain pixels in mask: %d  (%.1f%% of image)\n', ...
        n_grain_px, 100*n_grain_px/numel(mask));

mask_file = fullfile(output_dir, [grain_id '_mask.tif']);
imwrite(uint8(mask)*255, mask_file);
fprintf('  Mask saved to: %s\n', mask_file);

% ---- Log mask info -------------------------------------------------------
lprintf('\n--- GRAIN MASK ---\n');
lprintf('  Method:               %s\n', mask_method);
if ~isnan(thresh)
    lprintf('  Threshold applied:    %.6f\n', thresh);
else
    lprintf('  Threshold applied:    N/A (method: %s)\n', mask_method);
end
lprintf('  Pixels before cleanup:%d\n', n_px_raw);
lprintf('  Min object size:      %d px  (bwareaopen applied)\n', min_object_px);
lprintf('  Fill holes:           %s  (imfill applied)\n', mat2str(fill_holes));
lprintf('  Final grain pixels:   %d  (%.2f%% of %d x %d image)\n', ...
        n_grain_px, 100*n_grain_px/numel(mask), nrows_epma, ncols_epma);
lprintf('  Grain area:           %.2f µm²  (at %.4f µm/px)\n', ...
        n_grain_px * epma_pixel_um^2, epma_pixel_um);
if strcmp(mask_method, 'interactive')
    if isempty(all_bg_vals)
        lprintf('  Background samples:   none — fell back to manual threshold\n');
    else
        lprintf('  Background samples:   %d points\n', numel(all_bg_vals));
        lprintf('  Background val range: %.6f – %.6f\n', min(all_bg_vals), max(all_bg_vals));
        lprintf('  Sample pts (col, row, value):\n');
        for k = 1:numel(all_bg_vals)
            lprintf('    pt%-3d  col=%4d  row=%4d  val=%.6f\n', ...
                    k, all_bg_x(k), all_bg_y(k), all_bg_vals(k));
        end
    end
elseif strcmp(mask_method, 'polygon')
    lprintf('  Polygon vertices:     %d\n', size(poly_pos,1));
    lprintf('  Vertex coords (col, row):\n');
    for k = 1:size(poly_pos,1)
        lprintf('    pt%-3d  col=%8.2f  row=%8.2f\n', k, poly_pos(k,1), poly_pos(k,2));
    end
elseif strcmp(mask_method, 'activecontour')
    lprintf('  AC method:            Chan-Vese\n');
    lprintf('  AC iterations:        %d\n', ac_niter);
    lprintf('  AC smooth factor:     %.4f\n', ac_smoothfactor);
    lprintf('  AC init method:       %s\n', ac_init_method);
end
lprintf(SEC);

% ---- Visualize mask ------------------------------------------------------
figure('Name', 'Grain mask');
subplot(1,3,1); imshow(cl_reg);  title('Registered CL');
subplot(1,3,2); imshow(mask);    title('Grain mask');
subplot(1,3,3);
imshow(cl_reg); hold on;
visboundaries(mask, 'Color', 'r', 'LineWidth', 1);
title('Mask boundary on CL');
sgtitle(sprintf('%s — Mask (method: %s, thresh: %.3f)', ...
        grain_id, mask_method, thresh));
saveas(gcf, fullfile(output_dir, [grain_id '_mask_check.png']));

% =========================================================================
%% SECTION 6: EXTRACT PIXEL DATA VECTORS
% =========================================================================

fprintf('\n--- EXTRACTING PIXEL DATA ---\n');

cl_px = cl_reg(mask);

epma_px = zeros(n_grain_px, n_elements);
for e = 1:n_elements
    if normalize_epma
        v = double(epma_raw{e}(mask));
        vmin = min(v);
        vmax = max(v);
        if vmax > vmin
            epma_px(:, e) = (v - vmin) / (vmax - vmin);
        else
            epma_px(:, e) = zeros(n_grain_px, 1);
        end
    else
        epma_px(:, e) = epma_raw_abs{e}(mask);
    end
end

fprintf('  Pixels extracted per map: %d\n', n_grain_px);
if normalize_epma
    fprintf('  EPMA pixel vectors re-normalised to [0 1] using in-mask range.\n');
else
    fprintf('  EPMA pixel vectors kept as raw counts (normalize_epma = false).\n');
end

col_names   = [{'CL'}, epma_labels];
data_matrix = [cl_px, epma_px];

mat_file = fullfile(output_dir, [grain_id '_pixel_data.mat']);
save(mat_file, 'data_matrix', 'col_names', 'mask', ...
     'grain_id', 'epma_pixel_um', 'RMSE_px', 'RMSE_um');
fprintf('  Pixel data saved to: %s\n', mat_file);

csv_file = fullfile(output_dir, [grain_id '_pixel_data.csv']);
Tbl = array2table(data_matrix, 'VariableNames', col_names);
writetable(Tbl, csv_file);
fprintf('  Pixel data CSV saved to: %s\n', csv_file);

% ---- Log extraction info -------------------------------------------------
lprintf('\n--- PIXEL DATA EXTRACTION ---\n');
lprintf('  Pixels per map:   %d\n', n_grain_px);
lprintf('  Columns:          %s\n', strjoin(col_names, ', '));
lprintf('  Matrix size:      %d rows x %d columns\n', size(data_matrix,1), size(data_matrix,2));
lprintf('  CL normalisation:   full image min/max (includes background)\n');
if normalize_epma
    lprintf('  EPMA normalisation: in-mask min/max (grain interior only)\n');
else
    lprintf('  EPMA normalisation: none — raw pixel counts preserved (normalize_epma = false)\n');
end
lprintf('\n  Per-channel statistics (within mask, normalised 0-1):\n');
lprintf('  %-10s  %-8s  %-8s  %-8s  %-8s\n', 'Channel', 'Min', 'Max', 'Mean', 'Std');
lprintf('  %-10s  %-8s  %-8s  %-8s  %-8s\n', '----------', '--------', '--------', '--------', '--------');
for c = 1:size(data_matrix,2)
    col = data_matrix(:,c);
    lprintf('  %-10s  %-8.4f  %-8.4f  %-8.4f  %-8.4f\n', ...
            col_names{c}, min(col), max(col), mean(col), std(col));
end
lprintf(SEC);

% =========================================================================
%% SECTION 7: SCATTER PLOTS — CL vs. each element
% =========================================================================

fprintf('\n--- GENERATING SCATTER PLOTS ---\n');

n_cols = 3;
n_rows = ceil(n_elements / n_cols);
figure('Name', 'CL vs. element maps', ...
       'Position', [100, 100, 400*n_cols, 350*n_rows]);

r_vals     = zeros(1, n_elements);
pfit       = zeros(n_elements, 2);   % linear fit [slope, intercept] for each element
n_outliers = zeros(1, n_elements);
n_levels   = zeros(1, n_elements);   % unique quantization levels per element (native precision)

for e = 1:n_elements
    % Per-element outlier removal: keep pct_lo_cut–pct_hi_cut percentile range on element axis
    if pct_lo_cut > 0 || pct_hi_cut < 100
        lo = prctile(epma_px(:,e), pct_lo_cut);
        hi = prctile(epma_px(:,e), pct_hi_cut);
        keep = epma_px(:,e) >= lo & epma_px(:,e) <= hi;
    else
        keep = true(size(cl_px));
    end
    n_outliers(e) = sum(~keep);
    x_e = epma_px(keep, e);
    y_e = cl_px(keep);
    n_levels(e) = numel(unique(epma_px(:,e)));   % count on full (unclipped) column

    subplot(n_rows, n_cols, e);
    scatter(x_e, y_e, 8, 'filled', ...
            'MarkerFaceAlpha', 0.08, 'MarkerFaceColor', [0.2 0.2 0.2]);
    if normalize_epma
        xlabel([epma_labels{e}, ' (norm.)'], 'FontSize', 10);
    else
        xlabel(epma_labels{e}, 'FontSize', 10);
    end
    ylabel('CL intensity (norm.)', 'FontSize', 10);
    pfit(e,:) = polyfit(x_e, y_e, 1);
    xfit = linspace(min(x_e), max(x_e), 200);
    hold on;
    plot(xfit, polyval(pfit(e,:), xfit), 'k-', 'LineWidth', 1.5);
    r_vals(e) = corr(x_e, y_e);

    % Place annotations in the emptiest corner of the scatter plot.
    % Check point density in each of the four corners (30% of x-range,
    % 30% of y-range), then anchor all text lines there.
    xl     = xlim;
    xspan  = xl(2) - xl(1);
    cdense = [
        sum(x_e < xl(1)+0.3*xspan & y_e > 0.7),   % top-left
        sum(x_e > xl(2)-0.3*xspan & y_e > 0.7),   % top-right
        sum(x_e < xl(1)+0.3*xspan & y_e < 0.3),   % bottom-left
        sum(x_e > xl(2)-0.3*xspan & y_e < 0.3)    % bottom-right
    ];
    [~, bc] = min(cdense);
    tx = [0.05 0.95 0.05 0.95];
    ty = [0.95 0.95 0.12 0.12];
    ha = {'left','right','left','right'};
    va = {'top','top','bottom','bottom'};
    dy = [-0.10 -0.10 0.10 0.10];

    text(tx(bc), ty(bc), sprintf('r = %.3f', r_vals(e)), ...
         'Units', 'normalized', 'FontSize', 9, 'Color', 'k', ...
         'HorizontalAlignment', ha{bc}, 'VerticalAlignment', va{bc});
    text(tx(bc), ty(bc)+dy(bc), sprintf('n = %d', numel(x_e)), ...
         'Units', 'normalized', 'FontSize', 8, 'Color', [0.4 0.4 0.4], ...
         'HorizontalAlignment', ha{bc}, 'VerticalAlignment', va{bc});
    if n_outliers(e) > 0
        text(tx(bc), ty(bc)+2*dy(bc), sprintf('%d px outside %g–%gth pct', n_outliers(e), pct_lo_cut, pct_hi_cut), ...
             'Units', 'normalized', 'FontSize', 7, 'Color', [0.7 0.2 0.2], ...
             'HorizontalAlignment', ha{bc}, 'VerticalAlignment', va{bc});
    end
    title(sprintf('CL vs. %s', epma_labels{e}));
    ylim([0 1]); grid on; box on;
end

sgtitle(sprintf('%s — CL vs. element maps (RMSE: %.2f px = %.2f µm)', ...
        grain_id, RMSE_px, RMSE_um), 'FontSize', 12);
saveas(gcf, fullfile(output_dir, [grain_id '_CL_vs_elements.png']));

fprintf('\n  Pearson r summary:\n');
fprintf('  %-8s  %8s\n', 'Element', 'r');
fprintf('  %-8s  %8s\n', '-------', '--------');
for e = 1:n_elements
    fprintf('  %-8s  %8.4f\n', epma_labels{e}, r_vals(e));
end

% ---- Log correlations and linear fits ------------------------------------
lprintf('\n--- PEARSON CORRELATIONS  (CL vs. element, per pixel) ---\n');
lprintf('  n_grain = %d pixels  |  RMSE = %.4f px = %.4f µm\n', n_grain_px, RMSE_px, RMSE_um);
if pct_lo_cut > 0 || pct_hi_cut < 100
    lprintf('  Outlier removal: %.4g–%.4gth pct per element (applied before r and fit)\n', pct_lo_cut, pct_hi_cut);
else
    lprintf('  Outlier removal: disabled (pct_lo_cut = 0, pct_hi_cut = 100)\n');
end
lprintf('\n');
lprintf('  %-10s  %-12s  %-14s  %-14s  %-12s  %-10s  %-10s\n', ...
        'Element', 'r', 'Slope', 'Intercept', 'n_used', 'n_removed', 'x_levels');
lprintf('  %-10s  %-12s  %-14s  %-14s  %-12s  %-10s  %-10s\n', ...
        '----------', '------------', '--------------', '--------------', '------------', '----------', '----------');
for e = 1:n_elements
    n_used = n_grain_px - n_outliers(e);
    lprintf('  %-10s  %-12.6f  %-14.6f  %-14.6f  %-12d  %-10d  %-10d\n', ...
            epma_labels{e}, r_vals(e), pfit(e,1), pfit(e,2), n_used, n_outliers(e), n_levels(e));
end
lprintf('  (Linear fit: CL = slope * element + intercept, element axis normalised 0-1)\n');
lprintf(SEC);

% =========================================================================
%% SECTION 8: SHIFT-SENSITIVITY ANALYSIS
% =========================================================================

fprintf('\n--- SHIFT SENSITIVITY ANALYSIS ---\n');

n_shifts  = length(shift_range);
r_shift_x = zeros(n_shifts, n_elements);
r_shift_y = zeros(n_shifts, n_elements);

% Pre-compute per-element outlier keep-masks using the same pct thresholds
% as the scatter plots so the r value at shift=0 matches r_vals exactly.
keep_shift = true(n_grain_px, n_elements);
if pct_lo_cut > 0 || pct_hi_cut < 100
    for e = 1:n_elements
        lo_e = prctile(epma_px(:,e), pct_lo_cut);
        hi_e = prctile(epma_px(:,e), pct_hi_cut);
        keep_shift(:,e) = epma_px(:,e) >= lo_e & epma_px(:,e) <= hi_e;
    end
end

for s = 1:n_shifts
    dx = shift_range(s);
    cl_shift_x = circshift(cl_reg, [0,  dx]);
    cl_shift_y = circshift(cl_reg, [dx, 0 ]);
    cl_vec_x = cl_shift_x(mask);
    cl_vec_y = cl_shift_y(mask);
    for e = 1:n_elements
        ke = keep_shift(:,e);
        r_shift_x(s,e) = corr(epma_px(ke,e), cl_vec_x(ke));
        r_shift_y(s,e) = corr(epma_px(ke,e), cl_vec_y(ke));
    end
end

figure('Name', 'Shift sensitivity', 'Position', [100 100 900 400]);
subplot(1,2,1);
plot(shift_range, r_shift_x, '-', 'MarkerSize', 4, 'LineWidth', 1.5);
xline(0, 'k--', 'LineWidth', 1);
xlabel('X shift (pixels)'); ylabel('Pearson r');
legend(epma_labels, 'Location', 'best', 'FontSize', 8);
title('Sensitivity to X-shift'); grid on;

subplot(1,2,2);
plot(shift_range, r_shift_y, '-', 'MarkerSize', 4, 'LineWidth', 1.5);
xline(0, 'k--', 'LineWidth', 1);
xlabel('Y shift (pixels)'); ylabel('Pearson r');
legend(epma_labels, 'Location', 'best', 'FontSize', 8);
title('Sensitivity to Y-shift'); grid on;

sgtitle(sprintf('%s — Shift sensitivity (RMSE = %.2f px)', grain_id, RMSE_px), ...
        'FontSize', 11);
saveas(gcf, fullfile(output_dir, [grain_id '_shift_sensitivity.png']));

delta_r_x = max(r_shift_x) - min(r_shift_x);
delta_r_y = max(r_shift_y) - min(r_shift_y);

fprintf('\n  Max delta-r over +/-%d pixel shift:\n', max(abs(shift_range)));
fprintf('  %-8s  %12s  %12s\n', 'Element', 'Dr (X-shift)', 'Dr (Y-shift)');
fprintf('  %-8s  %12s  %12s\n', '-------', '------------', '------------');
for e = 1:n_elements
    fprintf('  %-8s  %12.4f  %12.4f\n', epma_labels{e}, delta_r_x(e), delta_r_y(e));
end

% ---- Log shift sensitivity -----------------------------------------------
lprintf('\n--- SHIFT SENSITIVITY ANALYSIS ---\n');
lprintf('  Method:  circshift(cl_reg, [0 dx]) for X; circshift(cl_reg, [dx 0]) for Y\n');
lprintf('  Note:  circshift wraps edges — valid for internal shifts, not boundary regions.\n');
lprintf('  Outlier removal: same per-element pct thresholds as scatter plots applied (r at shift=0 matches r_vals).\n');
if length(shift_range) > 1
    lprintf('  Shift range:  %d to %d px  (step %g)\n', ...
            min(shift_range), max(shift_range), shift_range(2)-shift_range(1));
else
    lprintf('  Shift range:  %d px\n', shift_range(1));
end

lprintf('\n  Pearson r vs. X-shift (columns = elements):\n');
lprintf('  %-10s', 'Shift(px)');
for e = 1:n_elements, lprintf('  %-12s', epma_labels{e}); end
lprintf('\n');
for s = 1:n_shifts
    lprintf('  %-10d', shift_range(s));
    for e = 1:n_elements, lprintf('  %-12.6f', r_shift_x(s,e)); end
    lprintf('\n');
end

lprintf('\n  Pearson r vs. Y-shift (columns = elements):\n');
lprintf('  %-10s', 'Shift(px)');
for e = 1:n_elements, lprintf('  %-12s', epma_labels{e}); end
lprintf('\n');
for s = 1:n_shifts
    lprintf('  %-10d', shift_range(s));
    for e = 1:n_elements, lprintf('  %-12.6f', r_shift_y(s,e)); end
    lprintf('\n');
end

lprintf('\n  Max delta-r over full shift range (robustness metric):\n');
lprintf('  %-10s  %-14s  %-14s\n', 'Element', 'Delta-r (X)', 'Delta-r (Y)');
lprintf('  %-10s  %-14s  %-14s\n', '----------', '--------------', '--------------');
for e = 1:n_elements
    lprintf('  %-10s  %-14.6f  %-14.6f\n', epma_labels{e}, delta_r_x(e), delta_r_y(e));
end
lprintf(SEC);

% =========================================================================
%% SECTION 9: EXPORT ELEMENT MAPS WITH MASK OVERLAY (QC figures)
% =========================================================================

fprintf('\n--- SAVING QC FIGURE ---\n');

n_maps  = n_elements + 1;
n_cols2 = 3;
n_rows2 = ceil(n_maps / n_cols2);
figure('Name', 'All maps with mask', ...
       'Position', [50 50 380*n_cols2, 340*n_rows2]);

all_maps   = [{cl_reg}, epma_raw];
all_labels = [{'CL (registered)'}, epma_labels];

for m = 1:n_maps
    subplot(n_rows2, n_cols2, m);
    if m == 1
        imshow(all_maps{m});   % CL — no stretch
    else
        imshow(pct_stretch(all_maps{m}, display_pct(1), display_pct(2)));
    end
    hold on;
    visboundaries(mask, 'Color', 'r', 'LineWidth', 0.8);
    title(all_labels{m}, 'FontSize', 9);
end

sgtitle(sprintf('%s — All maps with grain mask boundary', grain_id));
saveas(gcf, fullfile(output_dir, [grain_id '_all_maps_QC.png']));

% =========================================================================
%% DONE — write log footer and close
% =========================================================================

% ---- Output file inventory -----------------------------------------------
all_outputs = { ...
    cl_reg_file,                                                   'Registered CL image (16-bit TIFF)'; ...
    mask_file,                                                     'Grain mask (8-bit TIFF)'; ...
    cp_savefile,                                                   'Control points (.mat)'; ...
    mat_file,                                                      'Pixel data (.mat)'; ...
    csv_file,                                                      'Pixel data (.csv)'; ...
    fullfile(output_dir, [grain_id '_CL_vs_elements.png']),        'Scatter plots (PNG)'; ...
    fullfile(output_dir, [grain_id '_shift_sensitivity.png']),     'Shift sensitivity (PNG)'; ...
    fullfile(output_dir, [grain_id '_all_maps_QC.png']),           'All-maps QC figure (PNG)'; ...
    fullfile(output_dir, [grain_id '_registration_overlay.png']),  'Registration overlay (PNG)'; ...
    fullfile(output_dir, [grain_id '_mask_check.png']),            'Mask check figure (PNG)'; ...
    log_file,                                                      'Analysis log (this file)'; ...
};

lprintf('\n--- OUTPUT FILE INVENTORY ---\n');
lprintf('  %-40s  %-12s  %s\n', 'Description', 'Size (bytes)', 'Path');
lprintf('  %-40s  %-12s  %s\n', repmat('-',1,40), repmat('-',1,12), repmat('-',1,20));
for f = 1:size(all_outputs,1)
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
fprintf('  %s_analysis_log.txt         — comprehensive run record\n', grain_id);
fprintf('  %s_controlpoints.mat        — saved control points (reusable)\n', grain_id);
fprintf('  %s_CL_registered.tif\n', grain_id);
fprintf('  %s_mask.tif\n', grain_id);
fprintf('  %s_pixel_data.mat/.csv      — main analysis data\n', grain_id);
fprintf('  %s_CL_vs_elements.png\n', grain_id);
fprintf('  %s_shift_sensitivity.png\n', grain_id);
fprintf('  %s_all_maps_QC.png\n\n', grain_id);

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

function img_out = pct_stretch(img, lo_pct, hi_pct)
% Clip and rescale a [0,1] image to the given percentile range for display.
    lo = prctile(img(:), lo_pct);
    hi = prctile(img(:), hi_pct);
    if hi > lo
        img_out = (img - lo) / (hi - lo);
        img_out = max(0, min(1, img_out));
    else
        img_out = img;
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
