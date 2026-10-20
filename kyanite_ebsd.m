% =========================================================================
% KYANITE_EBSD.m
%
% PURPOSE:
%   batch EBSD processing of kyanite grains with MTEX: clean each grain's
%   Oxford/HKL .ctf map and map the internal lattice distortion of every
%   kyanite grain in it (GROD — grain reference orientation deviation, the
%   misorientation of every pixel from its own grain's mean orientation).
%
% WORKFLOW (per grain):
%   1. Load inputs/ebsd/<grain_id>_EBSD.ctf
%   2. Drop pixels with MAD >= mad_threshold (poor band fit) — both methods
%   3. Clean, by cleaning_method:
%      'mtex'     — reconstruct grains, drop grains < mtex_min_grain_px,
%                   reconstruct, then MTEX fill(): every gap inside a grain
%                   outline takes its nearest indexed pixel's orientation
%                   (not across grain boundaries)
%      'channel5' — HKL Channel 5 (Tango) style noise reduction on the
%                   8-neighbour grid: wild spike removal, then iterative
%                   n-nearest-neighbour zero-solution fill from 8 neighbours
%                   down to c5_fill_min_neighbours; optional small-grain drop
%   4. Reconstruct grains, keep phase_name pixels/grains
%   5. Compute GROD for every pixel of every remaining kyanite grain
%   6. Save the GROD map, a per-pixel CSV, and a log
%
%   Both cleaning methods are run on every grain, every run: the selected
%   one (cleaning_method) drives the GROD map/CSV/log, and both feed two
%   diagnostic comparison figures built with whatever method parameters
%   are currently set. Those figures have fixed filenames per grain, so
%   each run overwrites them rather than piling up copies.
%
%   Grains are processed first and plotted afterwards, so that with
%   grod_shared_scale = true every grain's GROD map uses one color scale
%   (same angle = same color in every grain, as for every other quantity
%   in this project).
%
% INPUTS (set in PARAMETERS section below):
%   - inputs/ebsd/<grain_id>_EBSD.ctf — raw Oxford/HKL Channel Text File
%
% OUTPUTS:
%   - ebsd/<grain_id>_GROD_map.png              — GROD map
%   - data/<grain_id>_ebsd_pixel_data.csv       — per-pixel x/y, Bunge Euler
%     angles, GROD, MAD, band contrast, grain id (reusable). Filled pixels
%     carry NaN MAD/BC, so they can be told apart from measured ones
%   - diagnostics/<grain_id>_ebsd_log.txt       — cleaning method + its
%     parameters, pixel counts at each cleaning stage, grain count, GROD
%     stats, per-grain mean orientation
%   - diagnostics/<grain_id>_ebsd_cleaning_maps.png — 'mtex' vs.
%     'channel5' cleaning: GROD map and kept/filled/removed pixel map per
%     method (overwritten every run)
%   - diagnostics/<grain_id>_ebsd_cleaning_summary.png — input pixel
%     status after the MAD filter, and both methods' GROD histograms
%     (overwritten every run)
%
% REQUIREMENTS:
%   - MTEX (tested with 6.1.1) on the MATLAB path (run startup_mtex first)
%
% AUTHOR:  M. Stein
% DATE:    2026-10-20
% VERSION: 1.1
% =========================================================================

clear; clc; close all;

% =========================================================================
%% SECTION 1: PARAMETERS  — edit this section per batch
% =========================================================================

% Anchored to this script's own location (not a hardcoded machine-specific
% path) so the same file runs unmodified on any machine this repo is
% checked out on.
repo_root = fileparts(mfilename('fullpath'));

% {} = auto-discover every inputs/ebsd/*_EBSD.ctf; otherwise a cell array
% of grain ids, e.g. {'MW609-01', 'MW609-02'}. A grain that fails partway
% is skipped with a warning rather than aborting the batch.
grain_ids = {};

input_dir       = fullfile(repo_root, 'inputs', 'ebsd');
ctf_suffix      = '_EBSD.ctf';
output_dir      = fullfile(repo_root, 'figs', 'ebsd');
data_dir        = fullfile(repo_root, 'figs', 'data');
diagnostics_dir = fullfile(repo_root, 'figs', 'diagnostics');

phase_name = 'Kyanite';     % mineral name as written in the .ctf header

% Oxford .ctf files store map coordinates and Euler angles in reference
% frames rotated 180° about x from each other; MTEX warns on load unless
% one is converted. '' = load as-is (the original single-grain script's
% behaviour — GROD is unaffected, but the exported Euler angles are in the
% Euler frame, not aligned with the map). 'convertEuler2SpatialReferenceFrame'
% rotates the orientations to match the map; 'convertSpatial2EulerReferenceFrame'
% rotates the map to match the orientations.
ctf_frame_option = '';

% --- Cleaning: shared by both methods ---
mad_threshold   = 2.0;   % keep pixels with MAD < this (degrees)
grain_angle_deg = 10;    % grain boundary misorientation threshold

% --- Cleaning: method choice ---
% 'mtex'     — small-grain drop + MTEX fill() inside grain outlines
% 'channel5' — Channel 5 style wild spike removal + n-neighbour fill
cleaning_method = 'mtex';

% --- 'mtex' method parameters ---
mtex_min_grain_px = 5;   % drop grains smaller than this before filling

% --- 'channel5' method parameters ---
% Wild spike: an indexed pixel misoriented > c5_spike_tol_deg from EVERY
% one of its indexed 8-neighbours is set to not-indexed.
c5_spike_tol_deg       = 10;
% true: only flag a spike when all 8 neighbours exist and are indexed
% (most conservative); false: compare against whichever are indexed.
c5_spike_require_all8  = false;
% Fill: levels 8 -> c5_fill_min_neighbours; at each level, a not-indexed
% pixel with >= level indexed 8-neighbours takes the modal neighbour
% orientation (the neighbour with the most other neighbours within
% c5_fill_tol_deg of it). Each level iterates until nothing changes.
c5_fill_min_neighbours = 6;      % 8 = most conservative; Inf = no fill
c5_fill_tol_deg        = 10;
% Optional small-grain drop after the fill (0 = off).
c5_min_grain_px        = 0;

% --- GROD map ---
% true: one color range shared by every grain in the batch (0 to the
% pooled max); false: each grain scaled to its own max.
grod_shared_scale = false;
grod_cmax_deg     = [];     % explicit upper color limit (deg); [] = auto

% --- Figure styling ---
% GROD colormap: linear white (0°) -> cmap_blue (max). A deliberate
% exception to kyanite_palette.SEQUENTIAL_CMAP ('inferno').
cmap_blue = '#0018F9';
cmap      = white_to_color(cmap_blue, 256);
font_size = 14;
font_name = 'Helvetica';
fig_dpi   = 300;

% Oxford/HKL plotting convention: x axis points right ('east'), z axis
% into the screen. This only sets how MTEX draws maps; it doesn't change
% the stored orientation data.
setMTEXpref('xAxisDirection', 'east');
setMTEXpref('zAxisDirection', 'intoPlane');
setMTEXpref('FontSize', font_size);
setMTEXpref('FontName', font_name);

% Bundle cleaning settings for process_grain / the log
clean = struct( ...
    'method',          lower(cleaning_method), ...
    'mad_threshold',   mad_threshold, ...
    'grain_angle_deg', grain_angle_deg, ...
    'mtex_min_grain_px',      mtex_min_grain_px, ...
    'c5_spike_tol_deg',       c5_spike_tol_deg, ...
    'c5_spike_require_all8',  c5_spike_require_all8, ...
    'c5_fill_min_neighbours', c5_fill_min_neighbours, ...
    'c5_fill_tol_deg',        c5_fill_tol_deg, ...
    'c5_min_grain_px',        c5_min_grain_px);

if ~any(strcmp(clean.method, {'mtex', 'channel5'}))
    error('cleaning_method must be ''mtex'' or ''channel5'' (got ''%s'')', cleaning_method);
end

% =========================================================================
%% SECTION 2: DISCOVER GRAINS
% =========================================================================

if exist('EBSD', 'class') ~= 8 && isempty(which('EBSD.load'))
    error('MTEX not found on the MATLAB path — run startup_mtex first.');
end

for d = {output_dir, data_dir, diagnostics_dir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

if isempty(grain_ids)
    listing   = dir(fullfile(input_dir, ['*' ctf_suffix]));
    grain_ids = erase({listing.name}, ctf_suffix);
    if isempty(grain_ids)
        error('No *%s files found in %s', ctf_suffix, input_dir);
    end
end
if ischar(grain_ids), grain_ids = {grain_ids}; end

fprintf('EBSD batch: %d grain(s): %s\n', numel(grain_ids), strjoin(grain_ids, ', '));
fprintf('Cleaning method: %s\n', clean.method);

% =========================================================================
%% SECTION 3: PROCESS EACH GRAIN
% =========================================================================

results = {};
for g = 1:numel(grain_ids)
    grain_id = grain_ids{g};
    fprintf('\n=== %s ===\n', grain_id);
    try
        results{end+1} = process_grain(grain_id, ...
            fullfile(input_dir, [grain_id ctf_suffix]), phase_name, ...
            ctf_frame_option, clean); %#ok<SAGROW>
    catch err
        warning('kyanite_ebsd:grainFailed', 'Skipping %s: %s', grain_id, err.message);
    end
end

if isempty(results)
    error('No grains processed successfully.');
end

% =========================================================================
%% SECTION 4: GROD COLOR SCALE
% =========================================================================

if ~isempty(grod_cmax_deg)
    pooled_cmax = grod_cmax_deg;
else
    pooled_cmax = max(cellfun(@(r) r.grod_max_deg, results));
end
fprintf('\nGROD color scale: %s, 0 to %.2f deg\n', ...
    ternary(grod_shared_scale, 'shared across grains', 'per grain'), pooled_cmax);

% =========================================================================
%% SECTION 5: FIGURES, DATA, LOGS
% =========================================================================

for g = 1:numel(results)
    r = results{g};
    grain_id = r.grain_id;

    if grod_shared_scale || ~isempty(grod_cmax_deg)
        cmax = pooled_cmax;
    else
        cmax = r.grod_max_deg;
    end
    cmax = max(ceil(cmax), 1);   % avoid a degenerate [0 0] range

    % --- GROD map ---
    fig_grod = figure('Visible', 'off');
    plot(r.ebsd, r.grod_deg, 'micronbar', 'on');
    hold on
    plot(r.grains.boundary, 'lineColor', 'k', 'lineWidth', 1);
    hold off
    setColorRange([0 cmax]);
    mtexColorMap(cmap);
    mtexColorbar('location', 'southoutside', 'orientation', 'horizontal', ...
        'title', 'GROD (°)');
    title(gca, grain_id, 'FontWeight', 'normal', 'Interpreter', 'none', ...
        'FontSize', font_size, 'FontName', font_name);
    % MTEX lays its figure out with no headroom for a title; re-run its
    % layout so the margins account for it (what mtexTitle does, but
    % mtexTitle forces a bold LaTeX title)
    mtexFig = getappdata(fig_grod, 'mtexFig');
    if ~isempty(mtexFig), drawNow(mtexFig); end
    grod_png = fullfile(output_dir, [grain_id '_GROD_map.png']);
    exportgraphics(fig_grod, grod_png, 'Resolution', fig_dpi);
    close(fig_grod);

    % --- Per-pixel CSV (reusable) ---
    [phi1, Phi, phi2] = Euler(r.ebsd.orientations, 'Bunge');
    T = table(r.ebsd.x(:), r.ebsd.y(:), phi1(:)/degree, Phi(:)/degree, phi2(:)/degree, ...
        r.grod_deg(:), get_prop(r.ebsd, 'mad'), get_prop(r.ebsd, 'bc'), r.ebsd.grainId(:), ...
        'VariableNames', {'x_um', 'y_um', 'phi1_deg', 'Phi_deg', 'phi2_deg', ...
                          'GROD_deg', 'MAD', 'BC', 'grainId'});
    csv_file = fullfile(data_dir, [grain_id '_ebsd_pixel_data.csv']);
    writetable(T, csv_file);

    % --- Log ---
    log_file = fullfile(diagnostics_dir, [grain_id '_ebsd_log.txt']);
    fid = fopen(log_file, 'w');
    fprintf(fid, 'KYANITE_EBSD.m — %s\n', grain_id);
    fprintf(fid, 'Run: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, 'Input:  %s\n', r.ctf_file);
    fprintf(fid, 'Step:   %.3f um\n\n', r.step_um);
    fprintf(fid, 'Parameters:\n');
    fprintf(fid, '  ctf_frame_option = %s\n', ternary(isempty(ctf_frame_option), '(none)', ctf_frame_option));
    fprintf(fid, '  mad_threshold    = %.2f\n', clean.mad_threshold);
    fprintf(fid, '  grain_angle_deg  = %.1f\n', clean.grain_angle_deg);
    fprintf(fid, '  cleaning_method  = %s\n', clean.method);
    switch clean.method
        case 'mtex'
            fprintf(fid, '    mtex_min_grain_px      = %d\n\n', clean.mtex_min_grain_px);
        case 'channel5'
            fprintf(fid, '    c5_spike_tol_deg       = %.1f\n', clean.c5_spike_tol_deg);
            fprintf(fid, '    c5_spike_require_all8  = %d\n', clean.c5_spike_require_all8);
            fprintf(fid, '    c5_fill_min_neighbours = %g\n', clean.c5_fill_min_neighbours);
            fprintf(fid, '    c5_fill_tol_deg        = %.1f\n', clean.c5_fill_tol_deg);
            fprintf(fid, '    c5_min_grain_px        = %d\n\n', clean.c5_min_grain_px);
    end
    fprintf(fid, 'Pixel counts:\n');
    fprintf(fid, '  raw                         %8d (%d indexed)\n', r.n_raw, r.n_raw_indexed);
    fprintf(fid, '  after MAD filter            %8d (%d indexed)\n', r.n_mad, r.n_mad_indexed);
    switch clean.method
        case 'mtex'
            fprintf(fid, '  after min-grain-size drop   %8d\n', r.n_thd);
        case 'channel5'
            fprintf(fid, '  wild spikes removed         %8d\n', r.n_spikes);
            for k = 1:size(r.fill_by_level, 1)
                fprintf(fid, '  filled at level %d (%2d pass) %8d\n', ...
                    r.fill_by_level(k, 1), r.fill_by_level(k, 3), r.fill_by_level(k, 2));
            end
            if clean.c5_min_grain_px > 0
                fprintf(fid, '  removed in small grains     %8d\n', r.n_small_removed);
            end
    end
    fprintf(fid, '  final (%s)            %8d\n\n', phase_name, r.n_final);
    fprintf(fid, '%s grains: %d (largest %d px, median %d px)\n\n', phase_name, ...
        r.grains.length, max(r.grains.numPixel), round(median(r.grains.numPixel)));
    fprintf(fid, 'GROD (deg): mean %.3f, median %.3f, 95th pct %.3f, max %.3f\n', ...
        mean(r.grod_deg, 'omitnan'), median(r.grod_deg, 'omitnan'), ...
        prctile(r.grod_deg, 95), r.grod_max_deg);
    fprintf(fid, 'GROD color range: 0 to %d deg (%s)\n\n', cmax, ...
        ternary(grod_shared_scale || ~isempty(grod_cmax_deg), 'shared', 'per grain'));
    [mp1, mP, mp2] = Euler(r.grains.meanOrientation, 'Bunge');
    for k = 1:numel(mp1)
        fprintf(fid, 'Mean orientation, grain %d (%d px; Bunge phi1, Phi, phi2): %.2f, %.2f, %.2f deg\n', ...
            r.grains.id(k), r.grains.numPixel(k), mp1(k)/degree, mP(k)/degree, mp2(k)/degree);
    end
    % --- Cleaning-method comparison (diagnostic, overwritten every run) ---
    cmp_maps_png    = fullfile(diagnostics_dir, [grain_id '_ebsd_cleaning_maps.png']);
    cmp_summary_png = fullfile(diagnostics_dir, [grain_id '_ebsd_cleaning_summary.png']);
    plot_cleaning_comparison(r, clean, cmap, font_name, cmp_maps_png, cmp_summary_png, fig_dpi);

    fprintf(fid, '\nOutputs:\n  %s\n  %s\n  %s\n  %s\n', grod_png, csv_file, ...
        cmp_maps_png, cmp_summary_png);
    fclose(fid);

    fprintf('%s: %d grains, %d px, GROD max %.2f deg -> %s\n', ...
        grain_id, r.grains.length, r.n_final, r.grod_max_deg, output_dir);
end

fprintf('\nDone: %d of %d grain(s) processed.\n', numel(results), numel(grain_ids));

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================

function r = process_grain(grain_id, ctf_file, phase_name, ctf_frame_option, clean)
% Load and clean one grain's EBSD map; compute GROD for every kyanite grain.

    if ~isfile(ctf_file)
        error('File not found: %s', ctf_file);
    end
    if isempty(ctf_frame_option)
        ebsd = EBSD.load(ctf_file);
    else
        ebsd = EBSD.load(ctf_file, ctf_frame_option);
    end
    if ~any(strcmp(ebsd.mineralList, phase_name))
        error('Phase "%s" not in %s (phases: %s)', phase_name, ctf_file, ...
            strjoin(ebsd.mineralList, ', '));
    end

    r.grain_id      = grain_id;
    r.ctf_file      = ctf_file;
    r.n_raw         = length(ebsd);
    r.n_raw_indexed = length(ebsd('indexed'));
    r.step_um       = mean(diff(unique(ebsd.x)));

    ang = clean.grain_angle_deg*degree;

    % MAD filter (non-indexed pixels carry MAD 0, so they pass; both
    % methods then treat them as gaps)
    mad_pass = ebsd.mad < clean.mad_threshold;
    ebsdMad  = ebsd(mad_pass);
    r.n_mad         = length(ebsdMad);
    r.n_mad_indexed = length(ebsdMad('indexed'));

    % Raw pixel status, kept for the comparison figure
    r.raw = struct('x', ebsd.x(:), 'y', ebsd.y(:), ...
                   'indexed', ebsd.isIndexed(:), 'mad_pass', mad_pass(:));

    % Run both methods: the selected one must succeed; the other only
    % feeds the comparison figure, so a failure there is just a warning.
    methods = {'mtex', 'channel5'};
    for m = 1:numel(methods)
        meth = methods{m};
        try
            switch meth
                case 'mtex'
                    [ebsdClean, r] = clean_mtex(ebsdMad, ang, clean, r);
                case 'channel5'
                    [ebsdClean, r] = clean_channel5(ebsdMad, ang, clean, r);
            end
            r.cmp.(meth) = finalize_clean(ebsdClean, ang, phase_name);
        catch err
            if strcmp(meth, clean.method), rethrow(err); end
            warning('kyanite_ebsd:comparisonFailed', ...
                '%s: comparison method ''%s'' failed: %s', grain_id, meth, err.message);
            r.cmp.(meth) = struct('failed', err.message);
        end
    end

    sel = r.cmp.(clean.method);
    r.ebsd         = sel.ebsd;
    r.grains       = sel.grains;
    r.grod_deg     = sel.grod_deg;
    r.grod_max_deg = sel.grod_max_deg;
    r.n_final      = sel.n_final;
end

function out = finalize_clean(ebsdClean, ang, phase_name)
% Reconstruct grains on a cleaned map, keep phase_name, compute GROD.
    [grainsClean, ebsdClean.grainId] = calcGrains(ebsdClean, 'angle', ang);

    ebsdFinal   = ebsdClean(phase_name);
    grainsFinal = grainsClean(phase_name);
    if isempty(grainsFinal)
        error('No %s grains left after cleaning', phase_name);
    end

    grod = ebsdFinal.calcGROD(grainsFinal);

    out.ebsd         = ebsdFinal;
    out.grains       = grainsFinal;
    out.grod_deg     = grod.angle(:) ./ degree;
    out.grod_max_deg = max(out.grod_deg, [], 'omitnan');
    out.n_final      = length(ebsdFinal);
    % Filled pixels carry no measured MAD (see set_not_indexed for spikes)
    out.filled       = isnan(get_prop(ebsdFinal, 'mad'));
end

function [ebsdClean, r] = clean_mtex(ebsdMad, ang, clean, r)
% Small-grain drop, then MTEX fill() of gaps inside grain outlines.

    [grainsMad, ebsdMad.grainId] = calcGrains(ebsdMad, 'angle', ang);

    % Drop small grains (non-indexed clusters count as grains here too),
    % then reconstruct
    ebsdThd = ebsdMad(grainsMad(grainsMad.numPixel >= clean.mtex_min_grain_px));
    r.n_thd = length(ebsdThd);
    [grainsThd, ebsdThd.grainId] = calcGrains(ebsdThd, 'angle', ang);

    % Fill non-indexed pixels inside grains (not across boundaries)
    ebsdClean = fill(ebsdThd('indexed'), grainsThd);
end

function [ebsdClean, r] = clean_channel5(ebsdMad, ang, clean, r)
% HKL Channel 5 (Tango) style noise reduction on the gridded map:
% wild spike removal, then iterative n-nearest-neighbour fill.
% Assumes a single indexed phase (orientations of different phases can't
% be compared); this project's .ctf files only carry Kyanite.

    if numel(ebsdMad.indexedPhasesId) > 1
        error('channel5 cleaning supports a single indexed phase only');
    end

    % Drop non-indexed pixels before gridding, so every gap is NaN-phase
    % on the grid (same treatment as a MAD-rejected pixel)
    ebsdG = gridify(ebsdMad('indexed'));

    [ebsdG, r.n_spikes] = remove_wild_spikes(ebsdG, ...
        clean.c5_spike_tol_deg*degree, clean.c5_spike_require_all8);

    [ebsdG, r.fill_by_level] = neighbour_fill(ebsdG, ...
        clean.c5_fill_min_neighbours, clean.c5_fill_tol_deg*degree);

    ebsdClean = ebsdG('indexed');

    if clean.c5_min_grain_px > 0
        [grainsC, ebsdClean.grainId] = calcGrains(ebsdClean, 'angle', ang);
        n_before  = length(ebsdClean);
        ebsdClean = ebsdClean(grainsC(grainsC.numPixel >= clean.c5_min_grain_px));
        r.n_small_removed = n_before - length(ebsdClean);
    end
end

function [ebsdG, n_spikes] = remove_wild_spikes(ebsdG, tol, require_all8)
% Set to not-indexed every indexed pixel misoriented > tol from ALL of its
% indexed 8-neighbours.
    [isIdx, ori, nb] = grid_neighbours(ebsdG);
    nValid   = zeros(size(isIdx));
    nSimilar = zeros(size(isIdx));
    for k = 1:8
        j  = nb(:,:,k);
        ok = isIdx & j > 0;
        ok(ok) = isIdx(j(ok));                % neighbour exists AND is indexed
        nValid(ok) = nValid(ok) + 1;
        sim = false(size(isIdx));
        sim(ok) = angle(ori(ok), ori(j(ok))) < tol;
        nSimilar = nSimilar + sim;
    end
    if require_all8
        spike = isIdx & nValid == 8 & nSimilar == 0;
    else
        spike = isIdx & nValid > 0 & nSimilar == 0;
    end
    ebsdG = set_not_indexed(ebsdG, spike);
    n_spikes = nnz(spike);
end

function [ebsdG, fill_by_level] = neighbour_fill(ebsdG, minLevel, tol)
% Fill each not-indexed pixel with >= level indexed 8-neighbours with the
% modal neighbour orientation, level 8 -> minLevel, iterating each level to
% convergence. Each pass works from a snapshot of the grid, so the result
% doesn't depend on raster order.
% fill_by_level rows: [level, pixels filled, passes].
    fill_by_level = zeros(0, 3);
    if ~isfinite(minLevel), return; end
    for level = 8:-1:minLevel
        n_level = 0;  passes = 0;
        while true
            [isIdx, ori, nb] = grid_neighbours(ebsdG);
            nbIdx = zeros(numel(isIdx), 8);   % linear id of indexed neighbours, 0 = none
            for k = 1:8
                j = reshape(nb(:,:,k), [], 1);
                v = j > 0;
                v(v) = isIdx(j(v));
                nbIdx(v, k) = j(v);
            end
            cand = find(~isIdx(:) & sum(nbIdx > 0, 2) >= level);
            if isempty(cand), break; end
            passes = passes + 1;

            % modal neighbour = the one with the most other neighbours within tol
            N = nbIdx(cand, :);
            votes = zeros(size(N));
            for a = 1:8
                for b = a+1:8
                    ok = N(:,a) > 0 & N(:,b) > 0;
                    s = false(size(ok));
                    s(ok) = angle(ori(N(ok,a)), ori(N(ok,b))) < tol;
                    votes(:,a) = votes(:,a) + s;
                    votes(:,b) = votes(:,b) + s;
                end
            end
            votes(N == 0) = -1;
            [~, best] = max(votes, [], 2);    % ties -> first neighbour in offset order
            src = N(sub2ind(size(N), (1:numel(cand))', best));

            ebsdG.phaseId(cand)   = ebsdG.phaseId(src);
            ebsdG.rotations(cand) = ebsdG.rotations(src);
            n_level = n_level + numel(cand);
        end
        fill_by_level(end+1, :) = [level, n_level, passes]; %#ok<AGROW>
    end
end

function [isIdx, ori, nb] = grid_neighbours(ebsdG)
% isIdx/ori on the grid; nb(:,:,k) = linear index of each pixel's k-th
% 8-neighbour (0 where it falls off the grid).
    [nr, nc] = size(ebsdG.id);
    isIdx = reshape(ebsdG.isIndexed, nr, nc);
    ori   = reshape(ebsdG.orientations, nr, nc);
    [R, C] = ndgrid(1:nr, 1:nc);
    off = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
    nb = zeros(nr, nc, 8);
    for k = 1:8
        rr = R + off(k,1);
        cc = C + off(k,2);
        in = rr >= 1 & rr <= nr & cc >= 1 & cc <= nc;
        t = zeros(nr, nc);
        t(in) = sub2ind([nr nc], rr(in), cc(in));
        nb(:,:,k) = t;
    end
end

function ebsdG = set_not_indexed(ebsdG, mask)
% Mark the masked pixels not-indexed (same mechanism as MTEX's own fill.m),
% and blank their measured properties (MAD, BC, ...) so that a spike which
% is later refilled shows up as filled (NaN MAD), not as measured.
    if ~any(mask(:)), return; end
    notIdx = find(cellfun('isclass', ebsdG.CSList, 'char'), 1);
    if isempty(notIdx)
        error('No notIndexed phase in this map''s phase list');
    end
    ebsdG.phaseId(mask)   = notIdx;
    ebsdG.rotations(mask) = rotation.nan(nnz(mask), 1);
    fn = fieldnames(ebsdG.prop);
    for k = 1:numel(fn)
        if ~strcmp(fn{k}, 'grainId') && isfloat(ebsdG.prop.(fn{k}))
            ebsdG.prop.(fn{k})(mask) = NaN;
        end
    end
end

function plot_cleaning_comparison(r, clean, cmap, font_name, maps_png, summary_png, fig_dpi)
% Two diagnostic figures comparing 'mtex' vs. 'channel5' cleaning of one
% grain, both drawn on the raw map grid:
%   maps_png    — 2 x 2: each method's GROD map (top, one shared color
%                 scale) and per-pixel kept / filled / removed status (bottom)
%   summary_png — 1 x 2: input pixel status after the MAD filter, and both
%                 methods' GROD histograms overlaid
% Every axes, legend and colorbar is placed in its own fixed box (no
% tiledlayout), with a reserved header strip for the title + parameters,
% so text can't overlap whatever the map's aspect ratio.
    BLUE  = [59 155 221]/255;   % kyanite_palette house BLUE '#3B9BDD'
    ORANG = [216 91 48]/255;    % kyanite_palette house ORANG '#D85B30'
    KEPT  = [0.75 0.75 0.75];
    EMPTY = [0.88 0.88 0.88];   % background, so 0° GROD (white) stays visible

    methods = {'mtex', 'channel5'};
    labels  = {'MTEX fill', 'Channel 5'};
    hist_colors = {BLUE, ORANG};
    status_names = {'absent', 'measured, kept', 'filled', 'removed'};
    status_colors = [1 1 1; KEPT; BLUE; ORANG];

    G = raw_grid(r.raw.x, r.raw.y);
    aspect = (G.sz(1)*G.dy) / (G.sz(2)*G.dx);         % map height / width

    ok = cellfun(@(m) ~isfield(r.cmp.(m), 'failed'), methods);
    cmax = max(cellfun(@(m) r.cmp.(m).grod_max_deg, methods(ok)));
    cmax = max(ceil(cmax), 1);

    sel_label = labels{strcmp(methods, clean.method)};
    params = sprintf(['MAD < %.2f, boundary %g°  |  mtex: min grain %d px  |  ' ...
        'channel5: spike tol %g°%s, fill to %g nbrs @ %g°, min grain %d px'], ...
        clean.mad_threshold, clean.grain_angle_deg, clean.mtex_min_grain_px, ...
        clean.c5_spike_tol_deg, ternary(clean.c5_spike_require_all8, ' (all 8)', ''), ...
        clean.c5_fill_min_neighbours, clean.c5_fill_tol_deg, clean.c5_min_grain_px);

    % Fixed pixel budget for text around each map, so the boxes scale with
    % the map's aspect ratio rather than squeezing the text
    head_px = 80;    % figure title + parameter line
    foot_px = 50;    % bottom legend strip
    text_px = 130;   % per cell: 2-line title, x tick labels, x label
    fig_w   = 1300;

    passed   = r.raw.indexed &  r.raw.mad_pass;
    rejected = r.raw.indexed & ~r.raw.mad_pass;

    % =================== Figure 1: GROD + status maps ===================
    cbar_frac = 0.10;                                  % right strip for the colorbar
    cell_w    = (1 - cbar_frac)/2;                     % normalized cell width
    map_w_px  = cell_w*fig_w - 120;                    % minus y ticks/label
    cell_h_px = min(max(map_w_px*aspect, 250), 750) + text_px;
    fig_h     = 2*cell_h_px + head_px + foot_px;
    head = head_px/fig_h;  foot = foot_px/fig_h;  row_h = cell_h_px/fig_h;

    fig = new_diag_figure(fig_w, fig_h, font_name, head, ...
        sprintf('%s — EBSD cleaning comparison: maps (selected: %s)', r.grain_id, sel_label), params);
    cell_box = @(col, row) [(col-1)*cell_w, foot + (2-row)*row_h, cell_w, row_h];

    h_st = [];  ax_cb = [];
    for m = 1:numel(methods)
        c = r.cmp.(methods{m});
        ax_g = place_axes(fig, cell_box(m, 1));
        ax_s = place_axes(fig, cell_box(m, 2));
        if ~ok(m)
            for a = [ax_g ax_s]
                text(a, 0.5, 0.5, sprintf('%s failed:\n%s', labels{m}, c.failed), ...
                    'HorizontalAlignment', 'center', 'Interpreter', 'none');
                axis(a, 'off');
            end
            continue
        end

        % GROD map
        M = nan(G.sz);
        M(G.ind(c.ebsd.x(:), c.ebsd.y(:))) = c.grod_deg;
        imagesc(ax_g, G.xv, G.yv, M, 'AlphaData', ~isnan(M));
        set(ax_g, 'Color', EMPTY);
        axis(ax_g, 'image');
        colormap(ax_g, cmap);
        clim(ax_g, [0 cmax]);
        xlabel(ax_g, 'x (µm)');  ylabel(ax_g, 'y (µm)');
        title(ax_g, {sprintf('%s GROD', labels{m}), sprintf('%d grains', c.grains.length)}, ...
            'FontWeight', 'normal');
        ax_g.OuterPosition = cell_box(m, 1);
        ax_cb = ax_g;

        % Status map: 1 = absent, 2 = measured & kept, 3 = filled, 4 = removed
        S = ones(G.sz);
        S(G.ind(r.raw.x(passed), r.raw.y(passed))) = 4;
        kept_ind = G.ind(c.ebsd.x(:), c.ebsd.y(:));
        S(kept_ind(~c.filled)) = 2;
        S(kept_ind( c.filled)) = 3;
        h_st = draw_categorical(ax_s, G, S, status_colors, status_names);
        spikes = '';
        if strcmp(methods{m}, 'channel5') && isfield(r, 'n_spikes')
            spikes = sprintf(' (%d wild spikes)', r.n_spikes);
        end
        title(ax_s, {sprintf('%s pixel status', labels{m}), ...
            sprintf('%d kept, %d filled, %d removed%s', ...
            nnz(~c.filled), nnz(c.filled), nnz(S == 4), spikes)}, 'FontWeight', 'normal');
        ax_s.OuterPosition = cell_box(m, 2);
    end

    % One shared GROD colorbar in the right strip, beside the top row
    if ~isempty(ax_cb)
        box_cb = ax_cb.OuterPosition;
        cb = colorbar(ax_cb);
        cb.Label.String = 'GROD (°)';
        cb.Units = 'normalized';
        cb.Position = [1 - cbar_frac + 0.02, foot + row_h + 0.2*row_h, 0.015, 0.6*row_h];
        ax_cb.OuterPosition = box_cb;       % undo the colorbar's axes shrink
    end
    % One status legend shared by both status maps, centred in the foot strip
    if ~isempty(h_st)
        place_legend(legend(h_st, status_names, 'Orientation', 'horizontal', 'Box', 'off'), ...
            (1 - cbar_frac)/2, foot/2);
    end

    exportgraphics(fig, maps_png, 'Resolution', fig_dpi);
    close(fig);

    % =============== Figure 2: input status + GROD histograms ===============
    cell_w    = 0.5;
    map_w_px  = cell_w*fig_w - 120;
    cell_h_px = min(max(map_w_px*aspect, 300), 750) + text_px;
    fig_h     = cell_h_px + head_px + foot_px;
    head = head_px/fig_h;  foot = foot_px/fig_h;  row_h = cell_h_px/fig_h;

    fig = new_diag_figure(fig_w, fig_h, font_name, head, ...
        sprintf('%s — EBSD cleaning comparison: input and GROD distribution (selected: %s)', ...
        r.grain_id, sel_label), params);
    cell_box = @(col) [(col-1)*cell_w, foot, cell_w, row_h];

    % Input status
    in_status = ones(G.sz);                                          % 1 = not indexed
    in_status(G.ind(r.raw.x(rejected), r.raw.y(rejected))) = 2;      % 2 = MAD-rejected
    in_status(G.ind(r.raw.x(passed),   r.raw.y(passed)))   = 3;      % 3 = passed
    in_names = {'not indexed', 'MAD-rejected', 'indexed, kept'};
    ax = place_axes(fig, cell_box(1));
    h_in = draw_categorical(ax, G, in_status, [1 1 1; ORANG; KEPT], in_names);
    title(ax, {'Input after MAD filter', sprintf('%d px, %d indexed, %d MAD-rejected', ...
        numel(r.raw.x), nnz(r.raw.indexed), nnz(rejected))}, 'FontWeight', 'normal');
    ax.OuterPosition = cell_box(1);
    place_legend(legend(h_in, in_names, 'Orientation', 'horizontal', 'Box', 'off'), ...
        cell_w/2, foot/2);
    ax.OuterPosition = cell_box(1);

    % GROD histograms
    axh = place_axes(fig, cell_box(2));
    hold(axh, 'on');
    edges = linspace(0, cmax, 51);
    leg = {};
    for m = find(ok)
        c = r.cmp.(methods{m});
        histogram(axh, c.grod_deg, edges, 'Normalization', 'probability', ...
            'DisplayStyle', 'stairs', 'EdgeColor', hist_colors{m}, 'LineWidth', 1.5);
        leg{end+1} = sprintf('%s\nn = %d\nmedian %.2f°, max %.2f°', labels{m}, ...
            c.n_final, median(c.grod_deg, 'omitnan'), c.grod_max_deg); %#ok<AGROW>
    end
    hold(axh, 'off');
    xlabel(axh, 'GROD (°)');  ylabel(axh, 'Fraction of pixels');
    title(axh, {'GROD distribution', ''}, 'FontWeight', 'normal');
    box(axh, 'on');
    legend(axh, leg, 'Location', 'northeast', 'Box', 'off', 'FontSize', 9);
    axh.OuterPosition = cell_box(2);

    exportgraphics(fig, summary_png, 'Resolution', fig_dpi);
    close(fig);
end

function fig = new_diag_figure(w, h, font_name, head, title_str, params_str)
% Invisible figure with explicit font sizes (MTEX's FontSize pref inflates
% the defaults) and a reserved header strip holding the title (upper half)
% and the parameter line (lower half), drawn as annotations so nothing in
% the axes area below can collide with them.
    fig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [0 0 w h], 'Color', 'w');
    set(fig, 'DefaultAxesFontSize', 10, 'DefaultAxesFontName', font_name, ...
             'DefaultTextFontName', font_name, 'DefaultLegendFontSize', 9);
    annotation(fig, 'textbox', [0 1-head/2 1 head/2], 'String', title_str, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontName', font_name, 'FontSize', 13, 'Interpreter', 'none');
    annotation(fig, 'textbox', [0 1-head 1 head/2], 'String', params_str, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontName', font_name, 'FontSize', 10, 'Interpreter', 'none');
end

function ax = place_axes(fig, box)
% Axes whose title, tick labels and axis labels all fit inside box
% (normalized figure units).
    ax = axes(fig, 'Units', 'normalized', 'PositionConstraint', 'outerposition', ...
        'OuterPosition', box);
end

function place_legend(lg, cx, cy)
% Centre a legend on (cx, cy) in normalized figure units.
    lg.Units = 'normalized';
    p = lg.Position;
    lg.Position = [cx - p(3)/2, cy - p(4)/2, p(3), p(4)];
end

function G = raw_grid(x, y)
% Regular grid spanning the raw map; G.ind(x, y) maps positions (µm) to
% linear indices. Rows run with y (downward, matching the 'east'/
% 'intoPlane' map convention), columns with x.
    xs = unique(round(x, 6));  ys = unique(round(y, 6));
    G.dx = median(diff(xs));   G.dy = median(diff(ys));
    G.x0 = min(x);             G.y0 = min(y);
    nc = round((max(x) - G.x0)/G.dx) + 1;
    nr = round((max(y) - G.y0)/G.dy) + 1;
    G.sz = [nr nc];
    G.xv = G.x0 + (0:nc-1)*G.dx;
    G.yv = G.y0 + (0:nr-1)*G.dy;
    G.ind = @(px, py) sub2ind(G.sz, round((py - G.y0)/G.dy) + 1, ...
                                    round((px - G.x0)/G.dx) + 1);
end

function h = draw_categorical(ax, G, S, colors, names)
% Categorical raster: S holds 1..numel(names), colors the matching RGB rows.
% Returns one legend-swatch patch per category; the caller places the legend.
    rgb = reshape(colors(S(:), :), [G.sz 3]);
    image(ax, G.xv, G.yv, rgb);
    axis(ax, 'image');
    xlabel(ax, 'x (µm)');  ylabel(ax, 'y (µm)');
    hold(ax, 'on');
    h = gobjects(numel(names), 1);
    for k = 1:numel(names)
        h(k) = patch(ax, NaN, NaN, colors(k, :), 'EdgeColor', [0.4 0.4 0.4]);
    end
    hold(ax, 'off');
end

function v = get_prop(ebsd, name)
% Column vector of an EBSD property, or NaN if the file didn't carry it.
    if isfield(ebsd.prop, name)
        v = double(ebsd.prop.(name)(:));
    else
        v = nan(length(ebsd), 1);
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function map = white_to_color(hex, n)
% n x 3 colormap running linearly from white to the given hex color.
    rgb = sscanf(hex(2:end), '%2x%2x%2x', [1 3]) / 255;
    t   = linspace(0, 1, n)';
    map = (1 - t) * [1 1 1] + t * rgb;
end
