% =========================================================================
% CL_REGION_EXTRACTION.m
%
% PURPOSE:
%   draw one or more polygon regions of interest on an already-registered
%   CL image (i.e. a grain previously processed by CL_EPMA_registration.m)
%   and extract per-pixel CL intensity and element concentration vectors
%   for each region. Lets you compare sub-grain crystal regions (e.g. core
%   vs. rim, growth zones) rather than whole-grain statistics.
%
%   Set classification_mode = true for a different workflow: subdividing
%   the WHOLE grain into non-overlapping CL textural classification domains
%   (e.g. sector zoning, oscillatory zoning) drawn from a fixed vocabulary
%   (TEXTURE_CLASSES). Domains auto-clip to whatever grain-mask area isn't
%   yet claimed, so they can never overlap, and any pixels left over when
%   you stop drawing are auto-bucketed into 'Unclassified' — so every
%   grain-mask pixel always ends up classified. classification_mode = false
%   (default) keeps the original freeform/partial-coverage ROI behavior
%   above, byte-for-byte.
%
% WORKFLOW (classification_mode = false, default):
%   1. Load the registered CL image, grain mask, and EPMA/XRF maps already
%      produced by CL_EPMA_registration.m for this grain — no warping,
%      control-point picking, or grain-mask generation is done here.
%   2. Interactively draw and name one polygon per region of interest
%      (or reload a previously saved set of regions).
%   3. Intersect each region polygon with the grain mask (optional).
%   4. Extract per-pixel CL and element vectors for each region.
%   5. Save combined long-format pixel data (with a Region column) and a
%      per-region/per-channel summary statistics table.
%   6. Save QC figures: region boundaries on the CL image, and region
%      boundaries overlaid on every element map.
%
% WORKFLOW (classification_mode = true):
%   1-6 as above, except: each drawn domain is assigned a class from
%   TEXTURE_CLASSES (not a freeform name) and auto-clipped to unclaimed
%   grain-mask area; leftover pixels become an 'Unclassified' domain; the
%   Region column holds the texture class (for grouping/plotting) and a
%   DomainID column holds the per-polygon instance id; an additional
%   full-grain texture class label map (TIFF + colored PNG) is saved.
%
% INPUTS (set in PARAMETERS section below):
%   - Registered CL image + grain mask (outputs of CL_EPMA_registration.m)
%   - Folder of EPMA/XRF element map TIFFs for the same grain
%
% OUTPUTS (classification_mode = false, default):
%   - [grain_id]_regions.mat            — region polygons + names (reusable)
%   - [grain_id]_region_pixel_data.mat  — per-pixel data, all regions
%   - [grain_id]_region_pixel_data.csv  — per-pixel data, all regions
%   - [grain_id]_region_summary.csv     — per-region/per-channel stats
%   - [grain_id]_regions_overlay.png    — region boundaries on CL image
%   - [grain_id]_regions_all_maps_QC.png — region boundaries on all maps
%   - [grain_id]_region_analysis_log.txt — comprehensive run record
%
% OUTPUTS (classification_mode = true, additional/renamed):
%   - [grain_id]_texture_domains.mat        — domain polygons + classes (reusable)
%   - [grain_id]_region_pixel_data.mat/.csv — per-pixel data, Region=class, DomainID=instance
%   - [grain_id]_texture_domain_summary.csv — per-class/per-instance stats
%   - [grain_id]_texture_domains_overlay.png
%   - [grain_id]_texture_domains_all_maps_QC.png
%   - [grain_id]_texture_class_map.tif  — full-grain class index raster
%   - [grain_id]_texture_class_map.png  — full-grain class-colored figure
%   - [grain_id]_region_analysis_log.txt — comprehensive run record
%
% REQUIREMENTS:
%   - MATLAB Image Processing Toolbox (for drawpolygon, poly2mask, etc.)
%
% AUTHOR:  M. Stein
% DATE:    2026-06-30
% VERSION: 1.0
% =========================================================================

clear; clc; close all;

set(0, 'DefaultTextInterpreter',        'none');
set(0, 'DefaultAxesTickLabelInterpreter','none');
set(0, 'DefaultLegendInterpreter',      'none');

% =========================================================================
%% SECTION 1: PARAMETERS  — edit this section for each new grain / region set
% =========================================================================

grain_id = 'NA-GS-P84-06';

% Directory containing the outputs of CL_EPMA_registration.m for this grain
% (registered CL TIFFs and the grain mask TIFF).
input_dir = '/Users/mstein/bin/kyanite/figs';

% Reusable data files (grain mask, pixel data, etc.) live in their own
% subfolder of input_dir, per CL_EPMA_registration.m's convention.
data_dir = fullfile(input_dir, 'data');

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

output_dir = '/Users/mstein/bin/kyanite/figs/regions';

% --- Spatial calibration --------------------------------------------------
epma_pixel_um = 2.0;     % µm per pixel — must match the value used during registration

% --- Region parameters -----------------------------------------------------
% Intersect every drawn polygon with the grain mask, so stray clicks outside
% the grain boundary never leak into a region.
restrict_to_grain_mask = true;

% Warn (not exclude) if a region ends up smaller than this after clipping.
% Set to 0 to disable.
min_region_px = 25;

% --- Texture classification mode --------------------------------------------
% false (default): freeform-named, possibly-overlapping, partial-coverage ROI
%   workflow above — unchanged from prior versions of this script.
% true: subdivide the WHOLE grain into non-overlapping CL textural
%   classification domains drawn from TEXTURE_CLASSES below. Every newly drawn
%   domain auto-clips to whatever grain-mask area is still unclaimed by a
%   prior domain, so domains can never overlap. Any grain-mask pixels left
%   over when you stop drawing are auto-bucketed into a reserved
%   'Unclassified' class, so the output always covers the full grain mask.
classification_mode = true;

% Fixed vocabulary of CL textural classes (classification_mode only) — mirrors
% kyanite_spot_analysis.py's CATEGORY_ORDER/CATEGORY_COLORS pattern so a class
% name means the same thing across every grain. Do NOT include 'Unclassified'
% here — it is reserved and assigned automatically, never offered as a manual
% choice.
TEXTURE_CLASSES = {'sector', 'oscillatory', 'feathered', 'homogenous'};
TEXTURE_CLASS_COLORS = containers.Map( ...
    [TEXTURE_CLASSES, {'Unclassified'}], ...
    { [0.85 0.33 0.10], [0.30 0.60 0.45], [0.47 0.32 0.58], [0.40 0.40 0.40], ...
      [0.60 0.60 0.60] });

% --- Element map normalization ---------------------------------------------
% true:  re-normalize each element map to [0 1] using its min/max within the
%        grain mask (matches CL_EPMA_registration.m's normalize_epma = true,
%        so values are directly comparable to the whole-grain pixel_data.csv).
% false: keep raw pixel values (counts/intensity units from the TIFF).
normalize_epma = false;

% Percentile range used to contrast-stretch element maps in QC figures only.
% Does not affect extracted pixel data.
display_pct = [0, 97];

% =========================================================================
%% SECTION 2: SETUP
% =========================================================================

if classification_mode && ~restrict_to_grain_mask
    error(['classification_mode = true requires restrict_to_grain_mask = true ' ...
           '(texture domains are defined relative to, and must fully cover, the grain mask).']);
end

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
fprintf('=== CL Region Extraction: %s ===\n\n', grain_id);

% ---- Open analysis log ---------------------------------------------------
log_file = fullfile(output_dir, [grain_id '_region_analysis_log.txt']);
log_fid  = fopen(log_file, 'w');
if log_fid == -1
    error('Cannot open log file for writing: %s', log_file);
end
lprintf = @(varargin) fprintf(log_fid, varargin{:});

DIV = ['================================================================================\n'];
SEC = ['--------------------------------------------------------------------------------\n'];

lprintf(DIV);
lprintf('CL REGION EXTRACTION ANALYSIS LOG\n');
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
lprintf('  Script:    CL_region_extraction.m  v1.0\n');

lprintf('\n--- PARAMETERS ---\n');
lprintf('  Grain ID:              %s\n', grain_id);
lprintf('  Input directory:       %s\n', input_dir);
lprintf('  EPMA directory:        %s\n', epma_dir);
lprintf('  Output directory:      %s\n', output_dir);
lprintf('  Registered CL (gray):  %s\n', cl_filename);
lprintf('  Registered CL (color): %s  (display: %s)\n', cl_color_filename, mat2str(use_color_display));
lprintf('  Grain mask:            %s\n', mask_filename);
lprintf('  EPMA maps (%d total):\n', n_elements);
for e = 1:n_elements
    lprintf('    [%d]  %-20s  label: %-10s\n', e, epma_files{e}, epma_labels{e});
end
lprintf('\n  Restrict regions to grain mask: %s\n', mat2str(restrict_to_grain_mask));
lprintf('  Min region size warning:        %d px\n', min_region_px);
lprintf('  Normalize EPMA maps:            %s\n', mat2str(normalize_epma));
lprintf('  Spatial calibration:            %.4f µm/px\n', epma_pixel_um);
lprintf('  Classification mode:            %s\n', mat2str(classification_mode));
if classification_mode
    lprintf('  Texture class vocabulary:       %s  (+ auto ''Unclassified'')\n', strjoin(TEXTURE_CLASSES, ', '));
end
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
if restrict_to_grain_mask
    if ~exist(mask_path, 'file')
        fclose(log_fid);
        error('Grain mask not found: %s\nRun CL_EPMA_registration.m for this grain first, or set restrict_to_grain_mask = false.', mask_path);
    end
    grain_mask = imread(mask_path) > 128;
    if ~isequal(size(grain_mask), [nrows, ncols])
        fclose(log_fid);
        error('Grain mask size (%d x %d) does not match registered CL size (%d x %d).', ...
              size(grain_mask,1), size(grain_mask,2), nrows, ncols);
    end
    fprintf('  Grain mask loaded: %d px in grain.\n', sum(grain_mask(:)));
    log_file_info(log_fid, mask_path, 'Grain mask');
else
    grain_mask = true(nrows, ncols);
    fprintf('  restrict_to_grain_mask = false — regions not clipped to a grain mask.\n');
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
lprintf('\n  Working grid: %d rows x %d cols  (%.1f x %.1f µm at %.4f µm/px)\n', ...
        nrows, ncols, nrows*epma_pixel_um, ncols*epma_pixel_um, epma_pixel_um);
lprintf(SEC);

% --- Normalization basis for elements (mirrors CL_EPMA_registration.m) ----
% When normalize_epma is true, each element map is rescaled to [0 1] using
% its min/max within norm_basis_mask, so values match the whole-grain
% pixel_data.csv produced by CL_EPMA_registration.m.
if restrict_to_grain_mask
    norm_basis_mask  = grain_mask;
    norm_basis_label = 'grain mask';
else
    norm_basis_mask  = true(nrows, ncols);
    norm_basis_label = 'whole image (no grain mask loaded)';
end

epma_norm = cell(1, n_elements);
if normalize_epma
    for e = 1:n_elements
        v = epma_raw_abs{e}(norm_basis_mask);
        vmin = min(v); vmax = max(v);
        if vmax > vmin
            epma_norm{e} = (epma_raw_abs{e} - vmin) / (vmax - vmin);
        else
            epma_norm{e} = zeros(nrows, ncols);
        end
    end
end

% =========================================================================
%% SECTION 4: DEFINE REGIONS  (interactive draw-and-name loop, or reload)
% =========================================================================

if classification_mode
    regions_savefile = fullfile(output_dir, [grain_id '_texture_domains.mat']);
else
    regions_savefile = fullfile(output_dir, [grain_id '_regions.mat']);
end
regions_source   = 'newly drawn';
skip_drawing     = false;

if exist(regions_savefile, 'file')
    if classification_mode
        prompt_msg = sprintf('Saved texture domains found (%s). Use them? (y/n): ', regions_savefile);
    else
        prompt_msg = sprintf('Saved regions found (%s). Use them? (y/n): ', regions_savefile);
    end
    resp = input(prompt_msg, 's');
    if strcmpi(strtrim(resp), 'y')
        if classification_mode
            load(regions_savefile, 'domain_ids', 'domain_classes', 'domain_polys', 'domain_masks');
            valid_classes = [TEXTURE_CLASSES, {'Unclassified'}];
            bad_classes = unique(domain_classes(~ismember(domain_classes, valid_classes)));
            if ~isempty(bad_classes)
                fclose(log_fid);
                error(['Saved texture domain(s) use class(es) not in the current TEXTURE_CLASSES ' ...
                       'vocabulary (or ''Unclassified''): %s.\nUpdate TEXTURE_CLASSES to match, or ' ...
                       'delete %s and redraw.'], strjoin(bad_classes, ', '), regions_savefile);
            end
            region_names = domain_ids;
            region_polys = domain_polys;
            region_masks = domain_masks;
            fprintf('Loaded %d saved texture domain(s): %s\n', numel(domain_ids), strjoin(domain_ids, ', '));
        else
            load(regions_savefile, 'region_names', 'region_polys');
            fprintf('Loaded %d saved region(s): %s\n', numel(region_names), strjoin(region_names, ', '));
        end
        regions_source = ['loaded from file: ', regions_savefile];
        skip_drawing = true;
    end
end

if ~skip_drawing && classification_mode
    % ---- classification mode: exhaustive, non-overlapping texture domains ----
    domain_ids               = {};
    domain_classes            = {};
    domain_polys              = {};
    domain_masks              = {};
    already_classified_mask  = false(nrows, ncols);
    class_counts = containers.Map(TEXTURE_CLASSES, num2cell(zeros(1, numel(TEXTURE_CLASSES))));

    fprintf('\n--- DEFINE TEXTURE DOMAINS ---\n');
    fprintf('Draw a polygon for each CL textural domain. Each domain auto-clips to\n');
    fprintf('whatever grain-mask area is not yet claimed by an earlier domain, so\n');
    fprintf('domains can never overlap — draw loosely; the last domain can simply\n');
    fprintf('cover whatever remains.\n');
    fprintf('Double-click the last vertex (or first) to close each polygon.\n\n');

    grain_px_total = sum(grain_mask(:));
    keep_drawing   = true;
    domain_num     = 0;

    while keep_drawing
        domain_num = domain_num + 1;

        fig_r = figure('Name', 'Draw texture domain — double-click to close', 'Position', [50 50 900 700]);
        imshow(cl_disp); hold on;
        for rr = 1:numel(domain_polys)
            p = domain_polys{rr};
            if isempty(p), continue; end   % auto-filled 'Unclassified' entry — no drawn polygon
            plot([p(:,1); p(1,1)], [p(:,2); p(1,2)], '-', ...
                 'Color', TEXTURE_CLASS_COLORS(domain_classes{rr}), 'LineWidth', 1.5);
            text(mean(p(:,1)), mean(p(:,2)), domain_ids{rr}, ...
                 'Color', 'w', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        end
        remaining_now = grain_px_total - sum(already_classified_mask(:));
        title(sprintf('Domain %d: click vertices, double-click to close  (%d px remain unclassified)', ...
              domain_num, remaining_now), 'FontSize', 11);

        h_poly = drawpolygon();
        wait(h_poly);

        % Read Position before closing the figure — closing deletes the
        % underlying ROI object, which invalidates any later access to it.
        if isempty(h_poly.Position) || size(h_poly.Position, 1) < 3
            close(fig_r);
            fprintf('  No valid polygon drawn (need at least 3 vertices).\n');
            resp = input('  Try again? (y/n): ', 's');
            domain_num = domain_num - 1;
            if strcmpi(strtrim(resp), 'y'), continue; else, break; end
        end

        pos = h_poly.Position;   % Nx2: [x (col), y (row)]
        close(fig_r);

        % ---- texture class selection: numbered menu, validated input ----
        fprintf('  Texture classes:\n');
        for c = 1:numel(TEXTURE_CLASSES)
            fprintf('    [%d]  %s\n', c, TEXTURE_CLASSES{c});
        end
        chosen_class = '';
        while isempty(chosen_class)
            class_resp = strtrim(input('  Class for this domain (number or name): ', 's'));
            idx = str2double(class_resp);
            if ~isnan(idx) && idx >= 1 && idx <= numel(TEXTURE_CLASSES) && idx == round(idx)
                chosen_class = TEXTURE_CLASSES{idx};
            elseif strcmpi(class_resp, 'Unclassified')
                fprintf('  ''Unclassified'' is reserved — assigned automatically only to leftover pixels.\n');
            else
                match = find(strcmpi(TEXTURE_CLASSES, class_resp), 1);
                if ~isempty(match)
                    chosen_class = TEXTURE_CLASSES{match};
                else
                    fprintf('  Not a valid class. Enter a number 1-%d or an exact class name.\n', numel(TEXTURE_CLASSES));
                end
            end
        end

        domain_id_preview = sprintf('%s_%d', chosen_class, class_counts(chosen_class) + 1);

        poly_mask_raw   = poly2mask(pos(:,1), pos(:,2), nrows, ncols);
        new_domain_mask = poly_mask_raw & grain_mask & ~already_classified_mask;

        px_raw             = sum(poly_mask_raw(:));
        px_outside_grain   = px_raw - sum(sum(poly_mask_raw & grain_mask));
        px_overlap_claimed = sum(sum(poly_mask_raw & grain_mask)) - sum(new_domain_mask(:));
        remaining_after    = grain_px_total - sum(already_classified_mask(:) | new_domain_mask(:));

        fig_p = figure('Name', 'Domain preview', 'Position', [100 100 500 500]);
        imshow(cl_disp); hold on;
        visboundaries(new_domain_mask, 'Color', 'y', 'LineWidth', 1.5);
        title({sprintf('%s (%s) — %d px  [%d clipped: outside grain / already claimed]', ...
                        domain_id_preview, chosen_class, sum(new_domain_mask(:)), ...
                        px_outside_grain + px_overlap_claimed), ...
               sprintf('%d px remain unclassified', remaining_after)});
        drawnow;

        resp = lower(strtrim(input('  Accept this domain? (y = accept, n = redraw, s = stop adding domains): ', 's')));
        close(fig_p);

        if strcmp(resp, 'y')
            class_counts(chosen_class) = class_counts(chosen_class) + 1;
            domain_id = sprintf('%s_%d', chosen_class, class_counts(chosen_class));
            domain_ids{end+1}     = domain_id;
            domain_classes{end+1} = chosen_class;
            domain_polys{end+1}   = pos;
            domain_masks{end+1}   = new_domain_mask;
            already_classified_mask = already_classified_mask | new_domain_mask;

            % Re-save after every accepted domain so a mid-session crash
            % doesn't discard already-completed work.
            save(regions_savefile, 'domain_ids', 'domain_classes', 'domain_polys', 'domain_masks');
            fprintf('  Domain "%s" (%s) accepted: %d px. %d grain-mask px remain unclassified (%.1f%%).\n', ...
                    domain_id, chosen_class, sum(new_domain_mask(:)), remaining_after, ...
                    100 * remaining_after / max(grain_px_total, 1));
        elseif strcmp(resp, 's')
            keep_drawing = false;
        else
            domain_num = domain_num - 1;   % redraw — don't consume the domain number
            continue;
        end

        if keep_drawing && remaining_after == 0
            fprintf('  Full grain-mask coverage reached — no pixels left to classify.\n');
            keep_drawing = false;
        elseif keep_drawing
            resp2 = strtrim(input('  Draw another domain? (y/n): ', 's'));
            if ~strcmpi(resp2, 'y')
                keep_drawing = false;
            end
        end
    end

    % ---- auto-fill any leftover grain-mask pixels as 'Unclassified' ----
    n_user_domains = numel(domain_ids);
    leftover_mask  = grain_mask & ~already_classified_mask;
    if any(leftover_mask(:))
        domain_ids{end+1}     = 'Unclassified';
        domain_classes{end+1} = 'Unclassified';
        domain_polys{end+1}   = [];
        domain_masks{end+1}   = leftover_mask;
        fprintf('  %d px left unclassified — auto-assigned to "Unclassified".\n', sum(leftover_mask(:)));
    else
        fprintf('  Full grain-mask coverage achieved by hand-drawn domains — no Unclassified bucket needed.\n');
    end
    if n_user_domains == 0
        warning('No texture domains were drawn — the entire grain is "Unclassified".');
    end

    region_names = domain_ids;
    region_polys = domain_polys;
    region_masks = domain_masks;

    save(regions_savefile, 'domain_ids', 'domain_classes', 'domain_polys', 'domain_masks');
    fprintf('Saved %d texture domain(s) to: %s\n', numel(domain_ids), regions_savefile);

elseif ~skip_drawing
    region_names = {};
    region_polys = {};

    fprintf('\n--- DEFINE REGIONS ---\n');
    fprintf('Draw a polygon for each region of interest.\n');
    fprintf('Double-click the last vertex (or first) to close each polygon.\n\n');

    keep_drawing = true;
    region_num   = 0;

    while keep_drawing
        region_num = region_num + 1;

        fig_r = figure('Name', 'Draw region — double-click to close', 'Position', [50 50 900 700]);
        imshow(cl_disp); hold on;
        if ~isempty(region_polys)
            colors_draft = lines(max(numel(region_polys), 7));
            for rr = 1:numel(region_polys)
                p = region_polys{rr};
                plot([p(:,1); p(1,1)], [p(:,2); p(1,2)], '-', ...
                     'Color', colors_draft(rr,:), 'LineWidth', 1.5);
                text(mean(p(:,1)), mean(p(:,2)), region_names{rr}, ...
                     'Color', 'w', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
            end
        end
        title(sprintf('Region %d: click vertices, double-click to close', region_num), 'FontSize', 11);

        h_poly = drawpolygon();
        wait(h_poly);

        % Read Position before closing the figure — closing deletes the
        % underlying ROI object, which invalidates any later access to it.
        if isempty(h_poly.Position) || size(h_poly.Position, 1) < 3
            close(fig_r);
            fprintf('  No valid polygon drawn (need at least 3 vertices).\n');
            resp = input('  Try again? (y/n): ', 's');
            region_num = region_num - 1;
            if strcmpi(strtrim(resp), 'y'), continue; else, break; end
        end

        pos = h_poly.Position;   % Nx2: [x (col), y (row)]
        close(fig_r);

        default_name = sprintf('region_%d', region_num);
        name_resp = strtrim(input(sprintf('  Name this region [%s]: ', default_name), 's'));
        if isempty(name_resp), name_resp = default_name; end
        if any(strcmp(region_names, name_resp))
            warning('Region name "%s" already used — appending suffix.', name_resp);
            name_resp = sprintf('%s_%d', name_resp, region_num);
        end

        poly_mask_raw = poly2mask(pos(:,1), pos(:,2), nrows, ncols);
        if restrict_to_grain_mask
            poly_mask_preview = poly_mask_raw & grain_mask;
            clip_note = ' (clipped to grain mask)';
        else
            poly_mask_preview = poly_mask_raw;
            clip_note = '';
        end

        fig_p = figure('Name', 'Region preview', 'Position', [100 100 500 500]);
        imshow(cl_disp); hold on;
        visboundaries(poly_mask_preview, 'Color', 'y', 'LineWidth', 1.5);
        title(sprintf('%s — %d px%s', name_resp, sum(poly_mask_preview(:)), clip_note));
        drawnow;

        resp = lower(strtrim(input('  Accept this region? (y = accept, n = redraw, s = stop adding regions): ', 's')));
        close(fig_p);

        if strcmp(resp, 'y')
            region_names{end+1} = name_resp;
            region_polys{end+1} = pos;
        elseif strcmp(resp, 's')
            keep_drawing = false;
        else
            region_num = region_num - 1;   % redraw — don't consume the region number
            continue;
        end

        if keep_drawing
            resp2 = strtrim(input('  Draw another region? (y/n): ', 's'));
            if ~strcmpi(resp2, 'y')
                keep_drawing = false;
            end
        end
    end

    if isempty(region_names)
        fclose(log_fid);
        error('No regions were defined. Please rerun and draw at least one region.');
    end

    save(regions_savefile, 'region_names', 'region_polys');
    fprintf('Saved %d region(s) to: %s\n', numel(region_names), regions_savefile);
end

n_regions = numel(region_names);

% =========================================================================
%% SECTION 5: BUILD FINAL REGION MASKS
% =========================================================================

fprintf('\n--- BUILDING REGION MASKS ---\n');

if classification_mode
    % Masks were already finalized (auto-clipped against already_classified_mask,
    % so guaranteed non-overlapping) back in Section 4 — recomputing them here
    % generically would require replaying draw order, so just measure them and
    % sanity-check that they fully cover the grain mask.
    region_px_final         = zeros(1, n_regions);
    region_area_um2         = zeros(1, n_regions);
    region_px_raw           = zeros(1, n_regions);
    region_px_outside_grain = zeros(1, n_regions);

    for r = 1:n_regions
        m = region_masks{r};
        region_px_final(r) = sum(m(:));
        region_px_raw(r)   = region_px_final(r);
        region_area_um2(r) = region_px_final(r) * epma_pixel_um^2;

        fprintf('  %-20s [%-12s]  %6d px  (%.1f µm²)\n', ...
                region_names{r}, domain_classes{r}, region_px_final(r), region_area_um2(r));

        if region_px_final(r) == 0
            warning('Domain "%s" has 0 px — check polygon placement.', region_names{r});
        elseif min_region_px > 0 && region_px_final(r) < min_region_px && ~strcmp(domain_classes{r}, 'Unclassified')
            warning('Domain "%s" has only %d px (< min_region_px = %d).', ...
                    region_names{r}, region_px_final(r), min_region_px);
        end
    end

    total_classified = sum(cellfun(@(m) sum(m(:)), region_masks));
    grain_px_check    = sum(grain_mask(:));
    if total_classified ~= grain_px_check
        warning('Coverage mismatch: %d classified px vs %d grain-mask px — investigate.', ...
                total_classified, grain_px_check);
    else
        fprintf('  Full grain-mask coverage confirmed: %d px across %d domain(s).\n', total_classified, n_regions);
    end
else
    region_masks            = cell(1, n_regions);
    region_px_raw           = zeros(1, n_regions);
    region_px_outside_grain = zeros(1, n_regions);
    region_px_final         = zeros(1, n_regions);
    region_area_um2         = zeros(1, n_regions);

    for r = 1:n_regions
        p = region_polys{r};
        m_raw = poly2mask(p(:,1), p(:,2), nrows, ncols);
        region_px_raw(r) = sum(m_raw(:));

        if restrict_to_grain_mask
            m = m_raw & grain_mask;
        else
            m = m_raw;
        end
        region_px_outside_grain(r) = region_px_raw(r) - sum(m(:));
        region_masks{r}    = m;
        region_px_final(r) = sum(m(:));
        region_area_um2(r) = region_px_final(r) * epma_pixel_um^2;

        fprintf('  %-20s  %6d px  (%.1f µm²)', region_names{r}, region_px_final(r), region_area_um2(r));
        if region_px_outside_grain(r) > 0
            fprintf('   [%d px clipped outside grain mask]', region_px_outside_grain(r));
        end
        fprintf('\n');

        if region_px_final(r) == 0
            warning('Region "%s" has 0 px after clipping — check polygon placement.', region_names{r});
        elseif min_region_px > 0 && region_px_final(r) < min_region_px
            warning('Region "%s" has only %d px (< min_region_px = %d).', ...
                    region_names{r}, region_px_final(r), min_region_px);
        end
    end
end

% Informational: flag pixels claimed by more than one region (overlapping polygons)
overlap_map = zeros(nrows, ncols);
for r = 1:n_regions
    overlap_map = overlap_map + double(region_masks{r});
end
n_overlap_px = sum(overlap_map(:) > 1);
if n_overlap_px > 0
    fprintf('  NOTE: %d px are claimed by more than one region (overlapping polygons).\n', n_overlap_px);
end

% ---- Log region definitions -----------------------------------------------
lprintf('\n--- REGIONS ---\n');
lprintf('  Source:  %s\n', regions_source);
lprintf('  Count:   %d region(s)\n', n_regions);
lprintf('  Restrict to grain mask: %s\n', mat2str(restrict_to_grain_mask));
if classification_mode
    lprintf('\n  %-20s  %-12s  %-10s  %-12s  %-12s\n', ...
            'DomainID', 'Class', 'Vertices', 'Px_final', 'Area_um2');
    lprintf('  %-20s  %-12s  %-10s  %-12s  %-12s\n', ...
            repmat('-',1,20), repmat('-',1,12), repmat('-',1,10), repmat('-',1,12), repmat('-',1,12));
    for r = 1:n_regions
        n_vertices = 0;
        if ~isempty(region_polys{r}), n_vertices = size(region_polys{r},1); end
        lprintf('  %-20s  %-12s  %-10d  %-12d  %-12.2f\n', ...
                region_names{r}, domain_classes{r}, n_vertices, region_px_final(r), region_area_um2(r));
    end
else
    lprintf('\n  %-20s  %-10s  %-10s  %-14s  %-12s  %-12s\n', ...
            'Region', 'Vertices', 'Px_raw', 'Px_outside_GM', 'Px_final', 'Area_um2');
    lprintf('  %-20s  %-10s  %-10s  %-14s  %-12s  %-12s\n', ...
            repmat('-',1,20), repmat('-',1,10), repmat('-',1,10), repmat('-',1,14), repmat('-',1,12), repmat('-',1,12));
    for r = 1:n_regions
        lprintf('  %-20s  %-10d  %-10d  %-14d  %-12d  %-12.2f\n', ...
                region_names{r}, size(region_polys{r},1), region_px_raw(r), ...
                region_px_outside_grain(r), region_px_final(r), region_area_um2(r));
    end
end
if n_overlap_px > 0
    lprintf('\n  ** NOTE: %d px are claimed by more than one region (overlapping polygons) **\n', n_overlap_px);
end
lprintf(SEC);

% ---- Visualize regions on the CL image ------------------------------------
fig_ov = figure('Name', 'Region overlay', 'Position', [100 100 700 700]);
imshow(cl_disp); hold on;
if ~classification_mode
    colors = lines(max(n_regions, 7));
end
for r = 1:n_regions
    if classification_mode
        this_color = TEXTURE_CLASS_COLORS(domain_classes{r});
    else
        this_color = colors(r,:);
    end
    visboundaries(region_masks{r}, 'Color', this_color, 'LineWidth', 1.5);
    [rr_idx, cc_idx] = find(region_masks{r});
    if ~isempty(rr_idx)
        text(mean(cc_idx), mean(rr_idx), region_names{r}, ...
             'Color', 'w', 'FontWeight', 'bold', 'FontSize', 10, ...
             'HorizontalAlignment', 'center');
    end
end
if classification_mode
    legend_classes  = [TEXTURE_CLASSES, {'Unclassified'}];
    legend_handles  = gobjects(1, numel(legend_classes));
    for c = 1:numel(legend_classes)
        legend_handles(c) = plot(nan, nan, '-', 'Color', TEXTURE_CLASS_COLORS(legend_classes{c}), 'LineWidth', 3);
    end
    legend(legend_handles, legend_classes, 'Location', 'bestoutside');
end
title(sprintf('%s — %d region(s)', grain_id, n_regions), 'Interpreter', 'none');
if classification_mode
    overlay_file = fullfile(output_dir, [grain_id '_texture_domains_overlay.png']);
else
    overlay_file = fullfile(output_dir, [grain_id '_regions_overlay.png']);
end
saveas(fig_ov, overlay_file);
fprintf('  Region overlay saved to: %s\n', overlay_file);

% =========================================================================
%% SECTION 6: EXTRACT PER-PIXEL DATA
% =========================================================================

fprintf('\n--- EXTRACTING PIXEL DATA ---\n');

col_names = [{'CL'}, epma_labels];

all_cl        = [];
all_epma      = [];
all_region    = {};
all_domain_id = {};   % classification_mode only — per-polygon instance id

% Parallel columns for the summary table (kept separate, rather than one
% mixed-type cell array, so unequal-length region/channel name strings
% don't trip up table construction).
sum_region    = {};
sum_domain_id = {};   % classification_mode only
sum_channel = {};
sum_n = [];
sum_mean = [];
sum_median = [];
sum_std = [];
sum_min = [];
sum_max = [];

for r = 1:n_regions
    m = region_masks{r};
    cl_r = cl_reg(m);

    epma_r = zeros(numel(cl_r), n_elements);
    for e = 1:n_elements
        if normalize_epma
            epma_r(:, e) = epma_norm{e}(m);
        else
            epma_r(:, e) = epma_raw_abs{e}(m);
        end
    end

    % Region column: texture class in classification mode (what downstream
    % scripts group/plot CL-vs-element by), else the region's own name —
    % unchanged from prior versions of this script.
    if classification_mode
        region_label = domain_classes{r};
    else
        region_label = region_names{r};
    end

    all_cl     = [all_cl;     cl_r];                                     %#ok<AGROW>
    all_epma   = [all_epma;   epma_r];                                   %#ok<AGROW>
    all_region = [all_region; repmat({region_label}, numel(cl_r), 1)];   %#ok<AGROW>
    if classification_mode
        all_domain_id = [all_domain_id; repmat({region_names{r}}, numel(cl_r), 1)];  %#ok<AGROW>
    end

    region_data = [cl_r, epma_r];
    for c = 1:numel(col_names)
        v = region_data(:, c);
        sum_region{end+1,1}  = region_label;     %#ok<AGROW>
        sum_channel{end+1,1} = col_names{c};      %#ok<AGROW>
        sum_n(end+1,1)      = numel(v);          %#ok<AGROW>
        sum_mean(end+1,1)   = mean(v);           %#ok<AGROW>
        sum_median(end+1,1) = median(v);         %#ok<AGROW>
        sum_std(end+1,1)    = std(v);            %#ok<AGROW>
        sum_min(end+1,1)    = min(v);            %#ok<AGROW>
        sum_max(end+1,1)    = max(v);            %#ok<AGROW>
        if classification_mode
            sum_domain_id{end+1,1} = region_names{r};  %#ok<AGROW>
        end
    end
end

data_matrix = [all_cl, all_epma];
n_px_total  = size(data_matrix, 1);

fprintf('  Total pixels extracted across %d region(s): %d\n', n_regions, n_px_total);
if normalize_epma
    fprintf('  EPMA values normalized to [0 1] using in-%s min/max.\n', norm_basis_label);
else
    fprintf('  EPMA values kept as raw counts (normalize_epma = false).\n');
end

% ---- Save combined pixel data ---------------------------------------------
% NOTE: this filename/the 'Region' column are kept identical in both modes —
% kyanite_figures.py / kyanite_pca_rf.py / kyanite_sample_size_convergence.py
% glob '*_pixel_data.csv' and group on 'Region'. Running classification mode
% for a grain that already has default-mode pixel data (or vice versa) will
% overwrite it; warn rather than silently clobber.
mat_file = fullfile(output_dir, [grain_id '_region_pixel_data.mat']);
if exist(mat_file, 'file')
    prior_vars = who('-file', mat_file);
    prior_is_classification = ismember('domain_classes', prior_vars);
    if prior_is_classification ~= classification_mode
        warning(['%s already exists from a %s run — it will be overwritten with %s output. ' ...
                 'Rename/move it first if you want to keep both.'], mat_file, ...
                ternary_label(prior_is_classification), ternary_label(classification_mode));
        lprintf('\n  ** WARNING: %s existed from a %s run and was overwritten with %s output **\n', ...
                mat_file, ternary_label(prior_is_classification), ternary_label(classification_mode));
    end
end

if classification_mode
    save(mat_file, 'data_matrix', 'col_names', 'all_region', 'all_domain_id', 'region_names', ...
         'domain_classes', 'TEXTURE_CLASSES', 'TEXTURE_CLASS_COLORS', 'classification_mode', ...
         'region_masks', 'grain_id', 'epma_pixel_um', 'normalize_epma');
else
    save(mat_file, 'data_matrix', 'col_names', 'all_region', 'region_names', ...
         'region_masks', 'grain_id', 'epma_pixel_um', 'normalize_epma');
end
fprintf('  Pixel data saved to: %s\n', mat_file);

csv_file = fullfile(output_dir, [grain_id '_region_pixel_data.csv']);
Tbl = array2table(data_matrix, 'VariableNames', col_names);
Tbl = addvars(Tbl, all_region, 'Before', 1, 'NewVariableNames', {'Region'});
if classification_mode
    Tbl = addvars(Tbl, all_domain_id, 'After', 'Region', 'NewVariableNames', {'DomainID'});
end
writetable(Tbl, csv_file);
fprintf('  Pixel data CSV saved to: %s\n', csv_file);

% ---- Save per-region summary statistics ------------------------------------
if classification_mode
    summary_T = table(sum_region, sum_domain_id, sum_channel, sum_n, sum_mean, sum_median, sum_std, sum_min, sum_max, ...
        'VariableNames', {'Region', 'DomainID', 'Channel', 'N', 'Mean', 'Median', 'Std', 'Min', 'Max'});
    summary_csv = fullfile(output_dir, [grain_id '_texture_domain_summary.csv']);
else
    summary_T = table(sum_region, sum_channel, sum_n, sum_mean, sum_median, sum_std, sum_min, sum_max, ...
        'VariableNames', {'Region', 'Channel', 'N', 'Mean', 'Median', 'Std', 'Min', 'Max'});
    summary_csv = fullfile(output_dir, [grain_id '_region_summary.csv']);
end
writetable(summary_T, summary_csv);
fprintf('  Region summary CSV saved to: %s\n', summary_csv);

% ---- Log extraction info ---------------------------------------------------
lprintf('\n--- PIXEL DATA EXTRACTION ---\n');
lprintf('  Total pixels:     %d  (across %d regions)\n', n_px_total, n_regions);
lprintf('  Columns:          %s\n', strjoin(col_names, ', '));
lprintf('  CL normalisation:   full image min/max (matches CL_EPMA_registration.m)\n');
if normalize_epma
    lprintf('  EPMA normalisation: in-%s min/max\n', norm_basis_label);
else
    lprintf('  EPMA normalisation: none — raw pixel counts preserved\n');
end
lprintf('\n  Per-region, per-channel statistics:\n');
if classification_mode
    lprintf('  %-14s  %-14s  %-10s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n', ...
            'Region(Class)', 'DomainID', 'Channel', 'N', 'Mean', 'Median', 'Std', 'Min', 'Max');
    lprintf('  %-14s  %-14s  %-10s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n', ...
            repmat('-',1,14), repmat('-',1,14), repmat('-',1,10), repmat('-',1,8), ...
            repmat('-',1,10), repmat('-',1,10), repmat('-',1,10), repmat('-',1,10), repmat('-',1,10));
    for i = 1:numel(sum_region)
        lprintf('  %-14s  %-14s  %-10s  %-8d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n', ...
                sum_region{i}, sum_domain_id{i}, sum_channel{i}, sum_n(i), sum_mean(i), sum_median(i), ...
                sum_std(i), sum_min(i), sum_max(i));
    end
else
    lprintf('  %-20s  %-10s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n', ...
            'Region', 'Channel', 'N', 'Mean', 'Median', 'Std', 'Min', 'Max');
    lprintf('  %-20s  %-10s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n', ...
            repmat('-',1,20), repmat('-',1,10), repmat('-',1,8), repmat('-',1,10), ...
            repmat('-',1,10), repmat('-',1,10), repmat('-',1,10), repmat('-',1,10));
    for i = 1:numel(sum_region)
        lprintf('  %-20s  %-10s  %-8d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n', ...
                sum_region{i}, sum_channel{i}, sum_n(i), sum_mean(i), sum_median(i), sum_std(i), sum_min(i), sum_max(i));
    end
end
lprintf(SEC);

% =========================================================================
%% SECTION 7: QC FIGURE — REGION BOUNDARIES ON ALL MAPS
% =========================================================================

fprintf('\n--- SAVING QC FIGURE ---\n');

n_maps  = n_elements + 1;
n_cols2 = 3;
n_rows2 = ceil(n_maps / n_cols2);
fig_qc = figure('Name', 'All maps with regions', 'Position', [50 50 380*n_cols2, 340*n_rows2]);

all_maps   = [{cl_reg}, epma_raw];
all_labels = [{'CL (registered)'}, epma_labels];

for m = 1:n_maps
    subplot(n_rows2, n_cols2, m);
    if m == 1
        imshow(cl_reg);
    else
        imshow(pct_stretch(all_maps{m}, display_pct(1), display_pct(2)));
    end
    hold on;
    for r = 1:n_regions
        if classification_mode
            this_color = TEXTURE_CLASS_COLORS(domain_classes{r});
        else
            this_color = colors(r,:);
        end
        visboundaries(region_masks{r}, 'Color', this_color, 'LineWidth', 0.8);
    end
    title(all_labels{m}, 'FontSize', 9);
end

sgtitle(sprintf('%s — All maps with region boundaries', grain_id), 'Interpreter', 'none');
if classification_mode
    qc_file = fullfile(output_dir, [grain_id '_texture_domains_all_maps_QC.png']);
else
    qc_file = fullfile(output_dir, [grain_id '_regions_all_maps_QC.png']);
end
saveas(fig_qc, qc_file);
fprintf('  QC figure saved to: %s\n', qc_file);

% =========================================================================
%% SECTION 7B: TEXTURE CLASS LABEL MAP  (classification_mode only)
% =========================================================================

if classification_mode
    fprintf('\n--- SAVING TEXTURE CLASS LABEL MAP ---\n');

    % uint8 class-index raster: 0 = outside grain mask, 1..numel(TEXTURE_CLASSES)
    % = index into TEXTURE_CLASSES (fixed vocabulary order, stable across
    % grains), numel(TEXTURE_CLASSES)+1 = Unclassified.
    all_class_names = [TEXTURE_CLASSES, {'Unclassified'}];
    class_map = zeros(nrows, ncols, 'uint8');
    for r = 1:n_regions
        class_idx = find(strcmp(all_class_names, domain_classes{r}), 1);
        class_map(region_masks{r}) = class_idx;
    end

    class_map_tif = fullfile(output_dir, [grain_id '_texture_class_map.tif']);
    imwrite(class_map, class_map_tif);
    log_file_info(log_fid, class_map_tif, 'Texture class label map (uint8 index-coded)');
    fprintf('  Texture class label map (TIFF) saved to: %s\n', class_map_tif);

    fig_cm = figure('Name', 'Texture class map', 'Position', [100 100 700 700]);
    rgb_map = zeros(nrows, ncols, 3);
    for c = 1:numel(all_class_names)
        color = TEXTURE_CLASS_COLORS(all_class_names{c});
        sel = (class_map == c);
        for ch = 1:3
            layer = rgb_map(:,:,ch);
            layer(sel) = color(ch);
            rgb_map(:,:,ch) = layer;
        end
    end
    imshow(rgb_map); hold on;
    legend_handles = gobjects(1, numel(all_class_names));
    for c = 1:numel(all_class_names)
        legend_handles(c) = plot(nan, nan, 's', 'MarkerFaceColor', TEXTURE_CLASS_COLORS(all_class_names{c}), ...
                                  'MarkerEdgeColor', 'none', 'MarkerSize', 12);
    end
    legend(legend_handles, all_class_names, 'Location', 'bestoutside');
    title(sprintf('%s — texture class map', grain_id), 'Interpreter', 'none');

    class_map_png = fullfile(output_dir, [grain_id '_texture_class_map.png']);
    saveas(fig_cm, class_map_png);
    fprintf('  Texture class map (PNG) saved to: %s\n', class_map_png);
end

% =========================================================================
%% DONE — write log footer and close
% =========================================================================

if classification_mode
    all_outputs = { ...
        regions_savefile,  'Texture domain definitions (.mat)'; ...
        mat_file,          'Region pixel data (.mat)'; ...
        csv_file,          'Region pixel data (.csv)'; ...
        summary_csv,       'Per-class/per-instance summary statistics (.csv)'; ...
        overlay_file,      'Texture domain overlay on CL (PNG)'; ...
        qc_file,           'All-maps QC figure with texture domains (PNG)'; ...
        class_map_tif,     'Full-grain texture class label map (TIFF)'; ...
        class_map_png,     'Full-grain texture class map (PNG)'; ...
        log_file,          'Analysis log (this file)'; ...
    };
else
    all_outputs = { ...
        regions_savefile,  'Region polygons + names (.mat)'; ...
        mat_file,          'Region pixel data (.mat)'; ...
        csv_file,          'Region pixel data (.csv)'; ...
        summary_csv,       'Per-region summary statistics (.csv)'; ...
        overlay_file,      'Region overlay on CL (PNG)'; ...
        qc_file,           'All-maps QC figure with regions (PNG)'; ...
        log_file,          'Analysis log (this file)'; ...
    };
end

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
fprintf('  %s_region_analysis_log.txt   — comprehensive run record\n', grain_id);
if classification_mode
    fprintf('  %s_texture_domains.mat          — texture domain definitions (reusable)\n', grain_id);
    fprintf('  %s_region_pixel_data.csv/.mat   — per-pixel data, all domains (Region=class, DomainID=instance)\n', grain_id);
    fprintf('  %s_texture_domain_summary.csv   — per-class/per-instance statistics\n', grain_id);
    fprintf('  %s_texture_domains_overlay.png\n', grain_id);
    fprintf('  %s_texture_domains_all_maps_QC.png\n', grain_id);
    fprintf('  %s_texture_class_map.tif/.png   — full-grain texture classification\n\n', grain_id);
else
    fprintf('  %s_regions.mat                — region polygons (reusable)\n', grain_id);
    fprintf('  %s_region_pixel_data.csv/.mat — per-pixel data, all regions\n', grain_id);
    fprintf('  %s_region_summary.csv         — per-region/per-channel statistics\n', grain_id);
    fprintf('  %s_regions_overlay.png\n', grain_id);
    fprintf('  %s_regions_all_maps_QC.png\n\n', grain_id);
end

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

function s = ternary_label(is_classification)
% Short label for the pixel-data overwrite warning.
    if is_classification
        s = 'classification-mode';
    else
        s = 'default-mode';
    end
end
