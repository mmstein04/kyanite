% =========================================================================
% CL_EPMA_REGISTRATION.m
%
% PURPOSE:
%   Register a cathodoluminescence (CL) image to one or more EPMA element
%   maps of the same grain, create a grain mask from the registered CL
%   image, extract per-pixel chemistry and CL intensity data, and produce
%   exploratory scatter plots of CL vs. element concentrations.
%
% WORKFLOW:
%   1. Load CL and EPMA images (handles 8-, 16-, and 32-bit TIFFs)
%   2. Interactively pick control points to register CL onto EPMA grid
%   3. Evaluate registration quality (RMSE in pixels and microns)
%   4. Build a binary grain mask from the registered CL image
%   5. Apply mask to extract pixel vectors for all maps
%   6. Scatter plot CL vs. each element; compute Pearson r
%   7. Run shift-sensitivity analysis to quantify alignment error impact
%   8. Write comprehensive analysis log throughout
%
% INPUTS (set in PARAMETERS section below):
%   - CL image (.tif or .png, any bit depth)
%   - One or more EPMA element map images (.tif, any bit depth)
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

% =========================================================================
%% SECTION 1: PARAMETERS  — edit this section for each new grain
% =========================================================================

% --- File paths -----------------------------------------------------------
input_dir  = '/Users/mstein/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/claude_test';

cl_filename = 'CL.png';

% EPMA element map filenames and their labels.
% The FIRST map in this list is used as the FIXED reference for control
% point selection. Choose your highest-quality, highest-contrast map.
epma_files  = {'Mn.tif', 'sumP.tif'};
epma_labels = {'Mn', 'sumP'};

output_dir  = '/Users/mstein/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/claude_test';
grain_id    = 'NA-CM-G12B8-06';

% --- Spatial calibration --------------------------------------------------
epma_pixel_um = 2.0;            % µm per pixel

% --- Mask parameters ------------------------------------------------------
mask_method   = 'otsu';
thresh_manual = 0.15;           % used only if mask_method = 'manual'
min_object_px = 500;
fill_holes    = true;

% --- Registration parameters ----------------------------------------------
transform_type = 'affine';

% --- Shift-sensitivity analysis parameters --------------------------------
shift_range   = -5:1:5;

% =========================================================================
%% SECTION 2: SETUP
% =========================================================================

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% Compute n_elements here so it is available for the log header below
n_elements = length(epma_files);

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
lprintf('  Input directory:     %s\n', input_dir);
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
switch transform_type
    case 'affine';     lprintf('    Min control pts:   6\n');
    case 'similarity'; lprintf('    Min control pts:   4\n');
    case 'rigid';      lprintf('    Min control pts:   3\n');
    otherwise;         lprintf('    Min control pts:   unknown transform type\n');
end
lprintf('\n  Mask:\n');
lprintf('    Method:            %s\n', mask_method);
if strcmp(mask_method, 'manual')
    lprintf('    Manual threshold:  %.4f  [active]\n', thresh_manual);
else
    lprintf('    Manual threshold:  %.4f  [inactive — otsu used]\n', thresh_manual);
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
for e = 1:n_elements
    epma_raw{e} = load_normalize(epma_files{e});
    fprintf('  %s map loaded:  %d x %d pixels\n', ...
            epma_labels{e}, size(epma_raw{e},1), size(epma_raw{e},2));
    log_file_info(log_fid, fullfile(input_dir, epma_files{e}), ...
                  sprintf('%s map', epma_labels{e}));
    lprintf('    Normalized stats:  min=%.4f  max=%.4f  mean=%.4f  std=%.4f\n', ...
            min(epma_raw{e}(:)), max(epma_raw{e}(:)), ...
            mean(epma_raw{e}(:)), std(epma_raw{e}(:)));
end

% Sanity check: all EPMA maps must be the same size
epma_sizes = cellfun(@(x) size(x,1)*1000 + size(x,2), epma_raw);
if numel(unique(epma_sizes)) > 1
    fclose(log_fid);
    error(['EPMA maps are not all the same size. ' ...
           'Check your input files before proceeding.']);
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
    [moving_pts, fixed_pts] = cpselect(cl_raw, epma_ref, 'Wait', true);
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
subplot(1,3,1); imshow(cl_raw);    title('CL: original');
subplot(1,3,2); imshow(epma_ref);  title([epma_labels{1}, ' EPMA (reference)']);
subplot(1,3,3); imshow(cl_reg);    title('CL: registered to EPMA grid');
sgtitle(sprintf('%s — Registration RMSE: %.2f px (%.2f µm)', ...
        grain_id, RMSE_px, RMSE_um));

figure('Name', 'Overlay check');
overlay = cat(3, epma_ref, cl_reg, zeros(size(epma_ref)));
imshow(overlay);
title({'Overlay check: red = EPMA ref, green = registered CL', ...
       'Edges should align at grain boundary'});
saveas(gcf, fullfile(output_dir, [grain_id '_registration_overlay.png']));

% =========================================================================
%% SECTION 5: BUILD GRAIN MASK
% =========================================================================

fprintf('\n--- BUILDING GRAIN MASK ---\n');

switch mask_method
    case 'otsu'
        thresh = graythresh(cl_reg);
        fprintf('  Otsu threshold: %.4f\n', thresh);
    case 'manual'
        thresh = thresh_manual;
        fprintf('  Manual threshold: %.4f\n', thresh);
    otherwise
        fclose(log_fid);
        error('mask_method must be ''otsu'' or ''manual''.');
end

mask = cl_reg > thresh;
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
lprintf('  Threshold applied:    %.6f\n', thresh);
lprintf('  Pixels above thresh:  %d  (before morphological cleanup)\n', n_px_raw);
lprintf('  Min object size:      %d px  (bwareaopen applied)\n', min_object_px);
lprintf('  Fill holes:           %s  (imfill applied)\n', mat2str(fill_holes));
lprintf('  Final grain pixels:   %d  (%.2f%% of %d x %d image)\n', ...
        n_grain_px, 100*n_grain_px/numel(mask), nrows_epma, ncols_epma);
lprintf('  Grain area:           %.2f µm²  (at %.4f µm/px)\n', ...
        n_grain_px * epma_pixel_um^2, epma_pixel_um);
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
    epma_px(:, e) = epma_raw{e}(mask);
end

fprintf('  Pixels extracted per map: %d\n', n_grain_px);

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

r_vals = zeros(1, n_elements);
pfit   = zeros(n_elements, 2);   % linear fit [slope, intercept] for each element

for e = 1:n_elements
    subplot(n_rows, n_cols, e);
    scatter(epma_px(:,e), cl_px, 2, 'filled', ...
            'MarkerFaceAlpha', 0.08, 'MarkerFaceColor', [0.2 0.2 0.2]);
    xlabel([epma_labels{e}, ' (norm.)'], 'FontSize', 10);
    ylabel('CL intensity (norm.)', 'FontSize', 10);
    pfit(e,:) = polyfit(epma_px(:,e), cl_px, 1);
    xfit = linspace(min(epma_px(:,e)), max(epma_px(:,e)), 200);
    hold on;
    plot(xfit, polyval(pfit(e,:), xfit), 'r-', 'LineWidth', 1.5);
    r_vals(e) = corr(epma_px(:,e), cl_px);
    text(0.05, 0.92, sprintf('r = %.3f', r_vals(e)), ...
         'Units', 'normalized', 'FontSize', 9, 'Color', 'r');
    text(0.05, 0.82, sprintf('n = %d', n_grain_px), ...
         'Units', 'normalized', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);
    title(sprintf('CL vs. %s', epma_labels{e}));
    xlim([0 1]); ylim([0 1]);
    grid on; box on;
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
lprintf('  n = %d pixels  |  RMSE = %.4f px = %.4f µm\n\n', n_grain_px, RMSE_px, RMSE_um);
lprintf('  %-10s  %-12s  %-14s  %-14s\n', 'Element', 'r', 'Slope', 'Intercept');
lprintf('  %-10s  %-12s  %-14s  %-14s\n', '----------', '------------', '--------------', '--------------');
for e = 1:n_elements
    lprintf('  %-10s  %-12.6f  %-14.6f  %-14.6f\n', ...
            epma_labels{e}, r_vals(e), pfit(e,1), pfit(e,2));
end
lprintf('  (Linear fit: CL = slope * element + intercept, both axes normalised 0-1)\n');
lprintf(SEC);

% =========================================================================
%% SECTION 8: SHIFT-SENSITIVITY ANALYSIS
% =========================================================================

fprintf('\n--- SHIFT SENSITIVITY ANALYSIS ---\n');

n_shifts  = length(shift_range);
r_shift_x = zeros(n_shifts, n_elements);
r_shift_y = zeros(n_shifts, n_elements);

for s = 1:n_shifts
    dx = shift_range(s);
    cl_shift_x = circshift(cl_reg, [0,  dx]);
    cl_shift_y = circshift(cl_reg, [dx, 0 ]);
    for e = 1:n_elements
        r_shift_x(s,e) = corr(epma_raw{e}(mask), cl_shift_x(mask));
        r_shift_y(s,e) = corr(epma_raw{e}(mask), cl_shift_y(mask));
    end
end

figure('Name', 'Shift sensitivity', 'Position', [100 100 900 400]);
subplot(1,2,1);
plot(shift_range, r_shift_x, '-o', 'MarkerSize', 4, 'LineWidth', 1.2);
xline(0, 'k--', 'LineWidth', 1);
xlabel('X shift (pixels)'); ylabel('Pearson r');
legend(epma_labels, 'Location', 'best', 'FontSize', 8);
title('Sensitivity to X-shift'); grid on;

subplot(1,2,2);
plot(shift_range, r_shift_y, '-o', 'MarkerSize', 4, 'LineWidth', 1.2);
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
    imshow(all_maps{m}); hold on;
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
