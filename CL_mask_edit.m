% =========================================================================
% CL_MASK_EDIT.m
%
% PURPOSE:
%   fix a grain mask after the fact — e.g. you notice weeks later that the
%   mask produced by CL_EPMA_registration.m swallowed an inclusion or missed
%   a sliver of real kyanite. Loads the grain's already-registered CL image,
%   element maps, and mask (no control points, no warping), lets you draw
%   add/remove polygons to touch up the mask, then re-derives everything
%   that depends on the mask so figs/ stays internally consistent.
%
% WORKFLOW:
%   1. Load the registered CL image, EPMA/XRF maps, and current grain mask
%      already produced by CL_EPMA_registration.m for this grain.
%   2. Back up every existing mask-dependent output file before touching
%      anything.
%   3. Interactively draw polygons tagged 'add' or 'remove'; each is applied
%      immediately with a live preview, and can be undone.
%   4. Re-apply the same morphological cleanup used at registration time
%      (close / min-object / fill-holes).
%   5. Re-extract per-pixel CL + element vectors under the new mask and
%      overwrite <grain_id>_pixel_data.csv/.mat in place.
%   6. Regenerate the CL-vs-element Pearson correlations, shift-sensitivity
%      analysis, and QC figures so every downstream figure/log matches the
%      corrected mask. (Exploratory scatter/violin/contour/etc. figures are
%      generated downstream by kyanite_figures.py from the pixel data CSV,
%      not here.)
%   7. Save a cumulative edit history (<grain_id>_mask_edits.mat) and an
%      edit-run log.
%
% INPUTS (set in PARAMETERS section below):
%   - Registered CL image + grain mask + pixel data (outputs of
%     CL_EPMA_registration.m for this grain)
%   - Folder of EPMA/XRF element map TIFFs for the same grain
%
% OUTPUTS:
%   - data/<grain_id>_mask.tif              — overwritten with the edited mask
%   - data/<grain_id>_pixel_data.csv / .mat — overwritten, re-extracted under new mask
%   - diagnostics/<grain_id>_shift_sensitivity.png  — overwritten (not for publishing)
%   - diagnostics/<grain_id>_all_maps_QC.png        — overwritten (not for publishing)
%   - diagnostics/<grain_id>_mask_check.png         — overwritten (not for publishing)
%   - data/<grain_id>_mask_edits.mat        — cumulative add/remove edit history (reusable/auditable)
%   - diagnostics/<grain_id>_mask_edit_diff.png     — old vs. new mask boundary comparison
%   - diagnostics/<grain_id>_mask_edit_log.txt      — this run's record
%   - mask_edit_backups/<grain_id>_<timestamp>/  — pre-edit copies of every file this
%     script is about to overwrite
%
% REQUIREMENTS:
%   - MATLAB Image Processing Toolbox (for drawpolygon, poly2mask, etc.)
%
% AUTHOR:  M. Stein
% DATE:    2026-07-09
% VERSION: 1.0
% =========================================================================

clear; clc; close all;

set(0, 'DefaultTextInterpreter',        'none');
set(0, 'DefaultAxesTickLabelInterpreter','none');
set(0, 'DefaultLegendInterpreter',      'none');

% =========================================================================
%% SECTION 1: PARAMETERS  — edit this section for each grain / edit session
% =========================================================================

grain_id = 'RH-XA-57081P-05';

% Directory holding the outputs of CL_EPMA_registration.m for this grain.
% This script reads from and writes back into the same folder (in place).
input_dir  = '/Users/mstein/bin/kyanite/figs';
output_dir = input_dir;

cl_filename       = [grain_id, '_CL_registered.tif'];         % 16-bit grayscale
cl_color_filename = [grain_id, '_CL_registered_color.tif'];   % native color
mask_filename     = [grain_id, '_mask.tif'];

% Use the color registered CL as the background for drawing/QC figures.
% Falls back to the grayscale registered CL (contrast-stretched) if false
% or if cl_color_filename is not found.
use_color_display = true;

% Folder containing EPMA/XRF element map TIFFs (same folder used by
% CL_EPMA_registration.m for this grain). All *.tif files auto-discovered.
epma_dir = ['/Users/mstein/bin/kyanite/inputs/maps/', grain_id];

% --- Spatial calibration --------------------------------------------------
% Must match the value used in CL_EPMA_registration.m for this grain.
epma_pixel_um = 2.0;     % µm per pixel

% --- Post-edit mask cleanup ------------------------------------------------
% Re-applied after add/remove edits, same knobs as CL_EPMA_registration.m's
% SECTION 5 — keep these matched to the original run unless you have a
% specific reason to change the cleanup behavior for this fix.
min_object_px   = 500;
fill_holes      = false;
close_radius_px = 1;    % morphological closing radius (px); 0 = disabled

% --- Downstream re-derivation ----------------------------------------------
% true (default, matches this project's convention): after saving the edited
% mask, fully re-run pixel extraction, Pearson correlations, and
% shift-sensitivity analysis so every figure/log in figs/ matches the
% corrected mask, not just the CSV. Set false to touch only the mask TIFF +
% edit history and leave pixel_data/plots for a separate manual re-run.
regenerate_downstream = true;

% Element map normalization — must match normalize_epma used in
% CL_EPMA_registration.m for this grain, or pixel_data.csv values won't be
% comparable to the original run.
normalize_epma = false;

% Percentile cutoffs for Pearson r (only used if
% regenerate_downstream = true) — match the original registration run.
pct_lo_cut =  0;
pct_hi_cut = 99;

% Shift-sensitivity range (only used if regenerate_downstream = true).
shift_range = -30:1:30;

% Percentile range for contrast-stretching element maps in figures (display only).
display_pct = [0, 97];

% =========================================================================
%% SECTION 2: SETUP
% =========================================================================

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% Sanity-check/QC/metadata outputs (shift-sensitivity, all-maps QC) live in
% their own subfolder, matching CL_EPMA_registration.m's convention — not
% meant for publishing, just re-derived alongside the rest.
diagnostics_dir = fullfile(output_dir, 'diagnostics');
if ~exist(diagnostics_dir, 'dir'), mkdir(diagnostics_dir); end

% Reusable data files (grain mask, pixel-data CSV/MAT, edit-history MAT)
% live in their own subfolder, matching CL_EPMA_registration.m's convention.
data_dir = fullfile(output_dir, 'data');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end

% --- Auto-discover EPMA maps from epma_dir --------------------------------
tif_listing = dir(fullfile(epma_dir, '*.tif'));
if isempty(tif_listing)
    error('No *.tif files found in epma_dir: %s', epma_dir);
end
tif_names = {tif_listing.name};

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
fprintf('=== CL Mask Edit: %s ===\n\n', grain_id);

run_timestamp = datestr(now, 'yyyymmdd_HHMMSS');

log_file = fullfile(diagnostics_dir, [grain_id '_mask_edit_log.txt']);
log_fid  = fopen(log_file, 'w');
if log_fid == -1
    error('Cannot open log file for writing: %s', log_file);
end
lprintf = @(varargin) fprintf(log_fid, varargin{:});

DIV = ['================================================================================\n'];
SEC = ['--------------------------------------------------------------------------------\n'];

lprintf(DIV);
lprintf('CL MASK EDIT LOG\n');
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
lprintf('  Script:    CL_mask_edit.m  v1.0\n');

lprintf('\n--- PARAMETERS ---\n');
lprintf('  Grain ID:               %s\n', grain_id);
lprintf('  Input/output directory: %s\n', input_dir);
lprintf('  EPMA directory:         %s\n', epma_dir);
lprintf('  Registered CL (gray):   %s\n', cl_filename);
lprintf('  Registered CL (color):  %s  (display: %s)\n', cl_color_filename, mat2str(use_color_display));
lprintf('  Grain mask:             %s\n', mask_filename);
lprintf('  Spatial calibration:    %.4f µm/px\n', epma_pixel_um);
lprintf('  Post-edit cleanup:      close=%d px, min_object=%d px, fill_holes=%s\n', ...
        close_radius_px, min_object_px, mat2str(fill_holes));
lprintf('  Regenerate downstream:  %s\n', mat2str(regenerate_downstream));
if regenerate_downstream
    lprintf('  Normalize EPMA:         %s\n', mat2str(normalize_epma));
    lprintf('  Outlier cutoffs:        %.4g - %.4g pct\n', pct_lo_cut, pct_hi_cut);
    lprintf('  Shift range:            %d to %d px\n', min(shift_range), max(shift_range));
end
lprintf(SEC);

% =========================================================================
%% SECTION 3: LOAD REGISTERED CL, EPMA MAPS, AND CURRENT MASK
% =========================================================================

fprintf('Loading registered CL, EPMA maps, and current grain mask...\n');
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

cl_color_path = fullfile(input_dir, cl_color_filename);
have_color = exist(cl_color_path, 'file') == 2;
if use_color_display && have_color
    cl_disp = imread(cl_color_path);
    fprintf('  Using registered color CL for display.\n');
    log_file_info(log_fid, cl_color_path, 'Registered CL image (color, display only)');
else
    if use_color_display && ~have_color
        warning('use_color_display = true but %s not found; falling back to grayscale.', cl_color_filename);
    end
    cl_disp = pct_stretch(cl_reg, display_pct(1), display_pct(2));
end

mask_path = fullfile(data_dir, mask_filename);
if ~exist(mask_path, 'file')
    fclose(log_fid);
    error('Grain mask not found: %s\nRun CL_EPMA_registration.m for this grain first.', mask_path);
end
mask_orig = imread(mask_path) > 128;
if ~isequal(size(mask_orig), [nrows, ncols])
    fclose(log_fid);
    error('Grain mask size (%d x %d) does not match registered CL size (%d x %d).', ...
          size(mask_orig,1), size(mask_orig,2), nrows, ncols);
end
n_grain_px_orig = sum(mask_orig(:));
fprintf('  Current mask loaded: %d px in grain (%.1f%% of image)\n', ...
        n_grain_px_orig, 100*n_grain_px_orig/numel(mask_orig));
log_file_info(log_fid, mask_path, 'Grain mask (pre-edit)');

epma_raw     = cell(1, n_elements);   % normalized 0-1 (display)
epma_raw_abs = cell(1, n_elements);   % raw pixel values
for e = 1:n_elements
    raw_img = imread(fullfile(epma_dir, epma_files{e}));
    epma_raw_abs{e} = double(raw_img);
    epma_raw{e}     = normalize_image(raw_img);
    fprintf('  %s map loaded:  %d x %d pixels\n', ...
            epma_labels{e}, size(epma_raw{e},1), size(epma_raw{e},2));
    log_file_info(log_fid, fullfile(epma_dir, epma_files{e}), sprintf('%s map', epma_labels{e}));
end

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

% Attempt to recover RMSE_px/RMSE_um from the existing pixel_data.mat so the
% re-extracted .mat retains correct registration-quality provenance (this
% script never re-registers, so these values are carried forward unchanged).
old_mat_file = fullfile(data_dir, [grain_id '_pixel_data.mat']);
RMSE_px = NaN; RMSE_um = NaN;
if exist(old_mat_file, 'file')
    old_vars = load(old_mat_file, 'RMSE_px', 'RMSE_um');
    if isfield(old_vars, 'RMSE_px'), RMSE_px = old_vars.RMSE_px; end
    if isfield(old_vars, 'RMSE_um'), RMSE_um = old_vars.RMSE_um; end
end
if isnan(RMSE_px)
    warning('Could not recover RMSE_px/RMSE_um from %s; saving NaN in re-extracted pixel_data.mat.', old_mat_file);
end

fprintf('\nWorking grid: %d rows x %d cols\n\n', nrows, ncols);
lprintf('\n  Working grid: %d rows x %d cols  (%.1f x %.1f µm at %.4f µm/px)\n', ...
        nrows, ncols, nrows*epma_pixel_um, ncols*epma_pixel_um, epma_pixel_um);
lprintf(SEC);

% =========================================================================
%% SECTION 4: BACK UP EVERY FILE ABOUT TO BE OVERWRITTEN
% =========================================================================

backup_dir = fullfile(output_dir, 'mask_edit_backups', [grain_id '_' run_timestamp]);
mkdir(backup_dir);
fprintf('\n--- BACKING UP EXISTING OUTPUTS ---\n');
fprintf('  Backup directory: %s\n', backup_dir);

% {filename, source directory} — mask/pixel_data/mask_edits read from
% data_dir, shift_sensitivity/all_maps_QC read from diagnostics_dir,
% everything else from input_dir; backups themselves stay flat (by
% basename) regardless of source subfolder.
files_to_backup = { ...
    mask_filename,                        data_dir; ...
    [grain_id '_pixel_data.csv'],         data_dir; ...
    [grain_id '_pixel_data.mat'],         data_dir; ...
    [grain_id '_shift_sensitivity.png'],  diagnostics_dir; ...
    [grain_id '_all_maps_QC.png'],        diagnostics_dir; ...
    [grain_id '_mask_check.png'],         diagnostics_dir; ...
    [grain_id '_mask_edits.mat'],         data_dir; ...
};

lprintf('\n--- PRE-EDIT BACKUP ---\n');
lprintf('  Backup directory: %s\n', backup_dir);
n_backed_up = 0;
for k = 1:size(files_to_backup, 1)
    src = fullfile(files_to_backup{k, 2}, files_to_backup{k, 1});
    if exist(src, 'file')
        copyfile(src, fullfile(backup_dir, files_to_backup{k, 1}));
        fprintf('  Backed up: %s\n', files_to_backup{k, 1});
        lprintf('  Backed up: %s\n', files_to_backup{k, 1});
        n_backed_up = n_backed_up + 1;
    end
end
fprintf('  %d file(s) backed up.\n', n_backed_up);
lprintf('  %d file(s) backed up.\n', n_backed_up);
lprintf(SEC);

% =========================================================================
%% SECTION 5: INTERACTIVE ADD/REMOVE MASK EDITING
% =========================================================================

fprintf('\n--- INTERACTIVE MASK EDITING ---\n');
fprintf('  For each round: draw a polygon, then it is tagged add or remove.\n');
fprintf('  Menu each round: a = add region, r = remove region, u = undo last edit, d = done.\n\n');

mask = mask_orig;
mask_stack   = {};   % snapshots of mask BEFORE each edit, for undo
edit_history = struct('round', {}, 'action', {}, 'polygon', {}, ...
                       'n_px_before', {}, 'n_px_after', {}, 'timestamp', {});
round_num = 0;

done = false;
while ~done
    fig_edit = figure('Name', 'Mask edit — current state', 'Position', [50 50 900 700]);
    imshow(cl_disp); hold on;
    visboundaries(mask, 'Color', 'r', 'LineWidth', 1.2);
    title(sprintf('%s — current mask (%d px). a=add, r=remove, u=undo, d=done', ...
          grain_id, sum(mask(:))), 'FontSize', 10);
    drawnow;

    resp = input('  Action (a/r/u/d): ', 's');
    resp = lower(strtrim(resp));

    switch resp
        case 'd'
            close(fig_edit);
            done = true;

        case 'u'
            close(fig_edit);
            if isempty(mask_stack)
                fprintf('  Nothing to undo.\n');
            else
                mask = mask_stack{end};
                mask_stack(end) = [];
                edit_history(end) = [];
                round_num = round_num - 1;
                fprintf('  Last edit undone. Mask now: %d px.\n', sum(mask(:)));
            end

        case {'a', 'r'}
            if strcmp(resp, 'a'), action = 'add'; else, action = 'remove'; end
            title(sprintf('Draw polygon to %s. Double-click last vertex to close.', upper(action)));
            h_poly = drawpolygon();
            wait(h_poly);
            poly_pos = h_poly.Position;
            close(fig_edit);

            if isempty(poly_pos) || size(poly_pos,1) < 3
                fprintf('  No polygon drawn (need >= 3 vertices) — skipping this round.\n');
                continue;
            end

            poly_mask = poly2mask(poly_pos(:,1), poly_pos(:,2), nrows, ncols);
            n_before = sum(mask(:));

            mask_stack{end+1} = mask;   %#ok<SAGROW>
            if strcmp(action, 'add')
                mask = mask | poly_mask;
            else
                mask = mask & ~poly_mask;
            end
            n_after = sum(mask(:));

            round_num = round_num + 1;
            edit_history(end+1) = struct( ...
                'round', round_num, 'action', action, 'polygon', poly_pos, ...
                'n_px_before', n_before, 'n_px_after', n_after, ...
                'timestamp', datestr(now, 'yyyy-mm-dd HH:MM:SS'));  %#ok<SAGROW>

            fprintf('  %s: %d vertices, mask %d -> %d px (%+d).\n', ...
                    upper(action), size(poly_pos,1), n_before, n_after, n_after-n_before);

        otherwise
            close(fig_edit);
            fprintf('  Unrecognized input ''%s'' — use a, r, u, or d.\n', resp);
    end
end

n_edits = numel(edit_history);
fprintf('\n  %d edit(s) applied this session.\n', n_edits);

% =========================================================================
%% SECTION 6: POST-EDIT CLEANUP AND SAVE MASK
% =========================================================================

n_px_before_cleanup = sum(mask(:));
if close_radius_px > 0
    mask = imclose(mask, strel('disk', close_radius_px));
end
if min_object_px > 0
    mask = bwareaopen(mask, min_object_px);
end
if fill_holes
    mask = imfill(mask, 'holes');
end
n_grain_px = sum(mask(:));

fprintf('  Post-edit cleanup: %d -> %d px.\n', n_px_before_cleanup, n_grain_px);
fprintf('  Final grain pixels: %d  (%.2f%% of %d x %d image)\n', ...
        n_grain_px, 100*n_grain_px/numel(mask), nrows, ncols);

imwrite(uint8(mask)*255, mask_path);
fprintf('  Edited mask saved to: %s\n', mask_path);

lprintf('\n--- MASK EDITS THIS SESSION ---\n');
lprintf('  Edits applied:          %d\n', n_edits);
for k = 1:n_edits
    ev = edit_history(k);
    lprintf('  [%d] %-6s  %2d vertices  %6d -> %6d px  (%+d)   %s\n', ...
            ev.round, ev.action, size(ev.polygon,1), ev.n_px_before, ev.n_px_after, ...
            ev.n_px_after - ev.n_px_before, ev.timestamp);
end
lprintf('  Post-edit cleanup:       close=%d px, min_object=%d px, fill_holes=%s\n', ...
        close_radius_px, min_object_px, mat2str(fill_holes));
lprintf('  Pixels before this run:  %d\n', n_grain_px_orig);
lprintf('  Pixels after this run:   %d  (%+d, %.2f%% -> %.2f%% of image)\n', ...
        n_grain_px, n_grain_px - n_grain_px_orig, ...
        100*n_grain_px_orig/numel(mask), 100*n_grain_px/numel(mask));
lprintf(SEC);

% ---- Cumulative edit history (.mat) --------------------------------------
edits_mat_file = fullfile(data_dir, [grain_id '_mask_edits.mat']);
if exist(edits_mat_file, 'file')
    prior = load(edits_mat_file, 'edit_history');
    all_edit_history = [prior.edit_history, edit_history];
else
    all_edit_history = edit_history;
end
save(edits_mat_file, 'edit_history', 'all_edit_history', 'grain_id', 'run_timestamp');
fprintf('  Mask edit history saved to: %s\n', edits_mat_file);

% ---- Before/after diff figure ---------------------------------------------
fig_diff = figure('Name', 'Mask edit diff', 'Position', [100 100 1200 400]);
subplot(1,3,1); imshow(cl_disp); hold on;
visboundaries(mask_orig, 'Color', 'b', 'LineWidth', 1.2);
title(sprintf('Before  (%d px)', n_grain_px_orig));
subplot(1,3,2); imshow(cl_disp); hold on;
visboundaries(mask, 'Color', 'r', 'LineWidth', 1.2);
title(sprintf('After  (%d px)', n_grain_px));
subplot(1,3,3);
diff_img = cat(3, double(mask & ~mask_orig), double(mask_orig & ~mask), zeros(nrows, ncols));
imshow(diff_img);
title('Green = added, Red = removed');
sgtitle(sprintf('%s — mask edit diff (%d edits, net %+d px)', ...
        grain_id, n_edits, n_grain_px - n_grain_px_orig));
saveas(fig_diff, fullfile(diagnostics_dir, [grain_id '_mask_edit_diff.png']));
fprintf('  Diff figure saved to: %s\n', fullfile(diagnostics_dir, [grain_id '_mask_edit_diff.png']));

% ---- Mask check figure (matches CL_EPMA_registration.m's own) ------------
fig_check = figure('Name', 'Grain mask');
subplot(1,3,1); imshow(cl_reg);   title('Registered CL');
subplot(1,3,2); imshow(mask);     title('Grain mask (edited)');
subplot(1,3,3); imshow(cl_reg); hold on;
visboundaries(mask, 'Color', 'r', 'LineWidth', 1);
title('Mask boundary on CL');
sgtitle(sprintf('%s — Mask (edited, %d px)', grain_id, n_grain_px));
saveas(fig_check, fullfile(diagnostics_dir, [grain_id '_mask_check.png']));

if ~regenerate_downstream
    lprintf('\nregenerate_downstream = false — pixel data, correlations, and shift plots NOT re-derived.\n');
    lprintf(DIV);
    lprintf('END OF LOG\n');
    lprintf(DIV);
    fclose(log_fid);
    fprintf('\n=== COMPLETE (mask only) ===\n');
    fprintf('Re-run CL_mask_edit.m with regenerate_downstream = true, or re-run\n');
    fprintf('CL_EPMA_registration.m''s extraction manually, to refresh pixel_data/plots.\n');
    return;
end

% =========================================================================
%% SECTION 7: RE-EXTRACT PIXEL DATA UNDER THE EDITED MASK
% =========================================================================

fprintf('\n--- RE-EXTRACTING PIXEL DATA ---\n');

cl_px = cl_reg(mask);
epma_px = zeros(n_grain_px, n_elements);
for e = 1:n_elements
    if normalize_epma
        v = double(epma_raw{e}(mask));
        vmin = min(v); vmax = max(v);
        if vmax > vmin
            epma_px(:, e) = (v - vmin) / (vmax - vmin);
        else
            epma_px(:, e) = zeros(n_grain_px, 1);
        end
    else
        epma_px(:, e) = epma_raw_abs{e}(mask);
    end
end

col_names   = [{'CL'}, epma_labels];
data_matrix = [cl_px, epma_px];

mat_file = fullfile(data_dir, [grain_id '_pixel_data.mat']);
save(mat_file, 'data_matrix', 'col_names', 'mask', ...
     'grain_id', 'epma_pixel_um', 'RMSE_px', 'RMSE_um');
fprintf('  Pixel data saved to: %s\n', mat_file);

csv_file = fullfile(data_dir, [grain_id '_pixel_data.csv']);
Tbl = array2table(data_matrix, 'VariableNames', col_names);
writetable(Tbl, csv_file);
fprintf('  Pixel data CSV saved to: %s\n', csv_file);

lprintf('\n--- PIXEL DATA RE-EXTRACTION ---\n');
lprintf('  Pixels per map:   %d\n', n_grain_px);
lprintf('  Columns:          %s\n', strjoin(col_names, ', '));
lprintf('  RMSE carried forward from prior registration: %.6f px (%.4f µm)\n', RMSE_px, RMSE_um);
lprintf(SEC);

% =========================================================================
%% SECTION 8: PEARSON CORRELATIONS — CL vs. each element
% =========================================================================
% Exploratory figures (scatter/violin/contour/heatmap/corrmatrix) are
% generated downstream by kyanite_figures.py from the re-extracted pixel
% data CSV — this section only recomputes the r/fit numbers for an
% immediate, in-MATLAB post-edit sanity check.

fprintf('\n--- RECOMPUTING PEARSON CORRELATIONS ---\n');

r_vals = zeros(1, n_elements);
pfit   = zeros(n_elements, 2);
n_outliers = zeros(1, n_elements);

for e = 1:n_elements
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

    pfit(e,:) = polyfit(x_e, y_e, 1);
    r_vals(e) = corr(x_e, y_e);
end

lprintf('\n--- PEARSON CORRELATIONS  (CL vs. element, per pixel, post-edit) ---\n');
lprintf('  %-10s  %-12s  %-10s  %-10s\n', 'Element', 'r', 'n_used', 'n_removed');
for e = 1:n_elements
    lprintf('  %-10s  %-12.6f  %-10d  %-10d\n', ...
            epma_labels{e}, r_vals(e), n_grain_px - n_outliers(e), n_outliers(e));
end
lprintf(SEC);

% =========================================================================
%% SECTION 9: SHIFT-SENSITIVITY ANALYSIS
% =========================================================================

fprintf('\n--- REGENERATING SHIFT SENSITIVITY ANALYSIS ---\n');

n_shifts  = length(shift_range);
r_shift_x = zeros(n_shifts, n_elements);
r_shift_y = zeros(n_shifts, n_elements);

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
    cl_shift_y = circshift(cl_reg, [dx, 0]);
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
plot(shift_range, r_shift_x, '-', 'LineWidth', 1.5);
xline(0, 'k--', 'LineWidth', 1);
xlabel('X shift (pixels)'); ylabel('Pearson r');
legend(epma_labels, 'Location', 'best', 'FontSize', 8);
title('Sensitivity to X-shift'); grid on;

subplot(1,2,2);
plot(shift_range, r_shift_y, '-', 'LineWidth', 1.5);
xline(0, 'k--', 'LineWidth', 1);
xlabel('Y shift (pixels)'); ylabel('Pearson r');
legend(epma_labels, 'Location', 'best', 'FontSize', 8);
title('Sensitivity to Y-shift'); grid on;

sgtitle(sprintf('%s — Shift sensitivity (mask edited)', grain_id), 'FontSize', 11);
saveas(gcf, fullfile(diagnostics_dir, [grain_id '_shift_sensitivity.png']));

delta_r_x = max(r_shift_x) - min(r_shift_x);
delta_r_y = max(r_shift_y) - min(r_shift_y);

lprintf('\n--- SHIFT SENSITIVITY ANALYSIS (post-edit) ---\n');
lprintf('  %-10s  %-14s  %-14s\n', 'Element', 'Delta-r (X)', 'Delta-r (Y)');
for e = 1:n_elements
    lprintf('  %-10s  %-14.6f  %-14.6f\n', epma_labels{e}, delta_r_x(e), delta_r_y(e));
end
lprintf(SEC);

% =========================================================================
%% SECTION 10: ALL-MAPS QC FIGURE
% =========================================================================

fprintf('\n--- REGENERATING ALL-MAPS QC FIGURE ---\n');

n_maps  = n_elements + 1;
n_cols2 = 3;
n_rows2 = ceil(n_maps / n_cols2);
figure('Name', 'All maps with mask', 'Position', [50 50 380*n_cols2, 340*n_rows2]);

all_maps   = [{cl_reg}, epma_raw];
all_labels = [{'CL (registered)'}, epma_labels];

for m = 1:n_maps
    subplot(n_rows2, n_cols2, m);
    if m == 1
        imshow(all_maps{m});
    else
        imshow(pct_stretch(all_maps{m}, display_pct(1), display_pct(2)));
    end
    hold on;
    visboundaries(mask, 'Color', 'r', 'LineWidth', 0.8);
    title(all_labels{m}, 'FontSize', 9);
end
sgtitle(sprintf('%s — All maps with edited grain mask boundary', grain_id));
saveas(gcf, fullfile(diagnostics_dir, [grain_id '_all_maps_QC.png']));

% =========================================================================
%% DONE — write log footer and close
% =========================================================================

lprintf('\n--- OUTPUT FILE INVENTORY ---\n');
all_outputs = { ...
    mask_path,                                                    'Grain mask (edited, 8-bit TIFF)'; ...
    mat_file,                                                      'Pixel data (.mat, re-extracted)'; ...
    csv_file,                                                      'Pixel data (.csv, re-extracted)'; ...
    fullfile(diagnostics_dir, [grain_id '_shift_sensitivity.png']),     'Shift sensitivity (PNG, regenerated)'; ...
    fullfile(diagnostics_dir, [grain_id '_all_maps_QC.png']),           'All-maps QC figure (PNG, regenerated)'; ...
    fullfile(diagnostics_dir, [grain_id '_mask_check.png']),       'Mask check figure (PNG, regenerated)'; ...
    fullfile(diagnostics_dir, [grain_id '_mask_edit_diff.png']),   'Before/after mask diff (PNG)'; ...
    edits_mat_file,                                                'Cumulative mask edit history (.mat)'; ...
    log_file,                                                      'Mask edit log (this file)'; ...
    backup_dir,                                                    'Pre-edit backup of overwritten files'; ...
};
lprintf('  %-45s  %s\n', 'Description', 'Path');
lprintf('  %-45s  %s\n', repmat('-',1,45), repmat('-',1,20));
for f = 1:size(all_outputs,1)
    lprintf('  %-45s  %s\n', all_outputs{f,2}, all_outputs{f,1});
end

lprintf('\n');
lprintf(DIV);
lprintf('END OF LOG\n');
lprintf(DIV);
fclose(log_fid);

fprintf('\n=== COMPLETE ===\n');
fprintf('Mask edited: %d px -> %d px (%+d), %d edit(s) applied.\n', ...
        n_grain_px_orig, n_grain_px, n_grain_px - n_grain_px_orig, n_edits);
fprintf('Pre-edit files backed up to: %s\n', backup_dir);
fprintf('All outputs written to: %s\n\n', output_dir);

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================

function img_norm = normalize_image(img_raw)
    img_d = double(img_raw);
    if ndims(img_d) == 3
        img_d = 0.2989*img_d(:,:,1) + 0.5870*img_d(:,:,2) + 0.1140*img_d(:,:,3);
    end
    img_min = min(img_d(:));
    img_max = max(img_d(:));
    if img_max == img_min
        img_norm = zeros(size(img_d));
    else
        img_norm = (img_d - img_min) / (img_max - img_min);
    end
end

function img_out = pct_stretch(img, lo_pct, hi_pct)
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
