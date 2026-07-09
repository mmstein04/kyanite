% =========================================================================
% XRF_DISPLAY.m
%
% PURPOSE:
%   load XRF element map TIFFs for one or more grains and display each
%   element as a figure with an optional colorbar, plus optional
%   element-ratio maps (e.g. Cr/Ti) rendered in the exact same style.
%   Off-grain pixels are masked transparent using a pre-existing binary mask.
%
% INPUTS (set in PARAMETERS section below):
%   - Folder of element map TIFFs (maps_dir)
%   - Grain ID string, or cell array of grain IDs for batch processing
%   - Mask TIFFs are auto-located as figs_dir/data/<grainID>_mask.tif
%   - List of element names to display
%   - List of element-ratio pairs to display (numerator, denominator)
%   - Per-element upper percentile cutoffs for contrast scaling
%   - Colormap and colorbar visibility options
%
% OUTPUTS:
%   - One figure per element per grain showing the masked, contrast-scaled map
%   - One figure per ratio per grain showing the masked, contrast-scaled
%     ratio map (numerator element map ./ denominator element map)
%   - Saved (if save_figs) as PNGs to figs_dir/maps/<grainID>_<tag>_display.png
%
% REQUIREMENTS:
%   - MATLAB Image Processing Toolbox (for imread)
%
% AUTHOR:  M. Stein
% DATE:    2026-06-23

clear; close all; clc;
% =========================================================================

%% Parameters

% File locations
maps_dir = '/Users/mstein/bin/kyanite/inputs/maps/';
figs_dir = '/Users/mstein/bin/kyanite/figs/';

% Rendered display PNGs are saved here, separate from figs_dir (which this
% script only reads from — the grain mask).
display_out_dir = fullfile(figs_dir, 'maps');

% Grain and element selection
% grainIDs may be a single string or a cell array for batch processing.
% Mask files are auto-located as figs_dir/data/<grainID>_mask.tif
grainIDs = {'RH-XA-57081P-05'};
elements = {'Cr','Fe','Mn','Ti','V'};

% Element-ratio maps (numerator, denominator), e.g. Cr/Ti and Fe/Ti.
% Each entry is a 2-element cell array {numerator, denominator}; both must
% be element names with a map file on disk (same _Ka naming as `elements`
% above — they don't need to also appear in the `elements` list itself).
% Leave empty ({}) to disable ratio maps.
ratios = {{'Cr','V'}, {'Cr','Fe'}, {'Fe','Mn'}};

% Visualization options
cmap = parula(256);
bkgdColor = 'k';
fontColor = [1 1 1];
show_colorbar = true;
saturation_pct = 1;   % % of within-grain pixels clipped at each end for display

% Ratio maps often have a much more skewed distribution than raw element
% counts, so their contrast clipping is independent of saturation_pct above.
ratio_saturation_pct = saturation_pct;

% Figure export
save_figs   = true;
fig_dpi     = 300;

% Scale bar
show_scalebar    = true;
pixel_um         = 2.0;    % µm/pixel — scalar for all grains, or vector [1.0, 0.5, ...]
scalebar_um      = 100;    % physical length of scale bar in µm
scalebar_pos     = 'se';   % corner: 'se' | 'sw' | 'ne' | 'nw'
scalebar_margin  = 0.04;   % margin from edge as fraction of image dimensions

%% Normalize grainIDs to a cell array

if ischar(grainIDs)
    grainIDs = {grainIDs};
end
nGrains   = numel(grainIDs);
nElements = numel(elements);
nRatios   = numel(ratios);
figOffset = 0;

% Bundle shared display options once so element and ratio maps are
% rendered by the exact same code path (render_and_save_map).
opts = struct( ...
    'cmap',            cmap, ...
    'bkgdColor',       bkgdColor, ...
    'fontColor',       fontColor, ...
    'show_colorbar',   show_colorbar, ...
    'show_scalebar',   show_scalebar, ...
    'scalebar_um',     scalebar_um, ...
    'scalebar_pos',    scalebar_pos, ...
    'scalebar_margin', scalebar_margin, ...
    'save_figs',       save_figs, ...
    'output_dir',      display_out_dir, ...
    'fig_dpi',         fig_dpi);

fprintf('Processing %d grain(s):\n', nGrains);
for g = 1:nGrains
    fprintf('  %s\n', grainIDs{g});
end
if nRatios > 0
    fprintf('Ratio maps requested (%d):\n', nRatios);
    for r = 1:nRatios
        fprintf('  %s / %s\n', ratios{r}{1}, ratios{r}{2});
    end
end

%% Loop over grains

for g = 1:nGrains

    grainID      = grainIDs{g};
    mask_path    = fullfile(figs_dir, 'data', sprintf('%s_mask.tif', grainID));
    grain_px_um  = pixel_um(min(g, end));

    fprintf('\n--- %s ---\n', grainID);

    %% Load element maps (union of `elements` and every ratio component)

    ratio_components = {};
    for r = 1:nRatios
        ratio_components = [ratio_components, ratios{r}{1}, ratios{r}{2}]; %#ok<AGROW>
    end
    elements_to_load = unique([elements, ratio_components], 'stable');

    imgs = struct();
    for i = 1:numel(elements_to_load)
        el = elements_to_load{i};
        fn = fullfile(maps_dir, grainID, sprintf('%s_%s_Ka.tif', grainID, el));
        if ~exist(fn, 'file')
            error('Element map not found: %s', fn);
        end
        imgs.(el) = imread(fn);
    end

    %% Plot elements

    mask = logical(imread(mask_path));

    for i = 1:nElements
        el  = elements{i};
        img = double(imgs.(el));
        img(~mask) = NaN;

        opts.saturation_pct = saturation_pct;
        render_and_save_map(figOffset + i, grainID, el, sprintf('%s_Ka', el), ...
                            img, mask, grain_px_um, opts);
    end % elements

    figOffset = figOffset + nElements;

    %% Plot element ratios

    for r = 1:nRatios
        numEl = ratios{r}{1};
        denEl = ratios{r}{2};

        ratio_img = double(imgs.(numEl)) ./ double(imgs.(denEl));
        ratio_img(~isfinite(ratio_img)) = NaN;   % div-by-zero / 0-over-0
        ratio_img(~mask) = NaN;

        label        = sprintf('%s/%s', numEl, denEl);
        filenameTag  = sprintf('%s_%s_ratio', numEl, denEl);

        opts.saturation_pct = ratio_saturation_pct;
        render_and_save_map(figOffset + r, grainID, label, filenameTag, ...
                            ratio_img, mask, grain_px_um, opts);
    end % ratios

    figOffset = figOffset + nRatios;

end % grains

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================

function render_and_save_map(figNum, grainID, label, filenameTag, img, mask, grain_px_um, opts)
% Renders one masked, contrast-scaled map (element or ratio) in the
% project's standard style and optionally exports it as a PNG. Shared by
% both the element and ratio display loops so their look never drifts.

    finite_mask = mask & isfinite(img);
    grain_vals  = img(finite_mask);
    if isempty(grain_vals)
        warning('%s — %s: no finite in-mask pixels, skipping.', grainID, label);
        return;
    end
    lo = max(0, prctile(grain_vals, opts.saturation_pct));
    hi = prctile(grain_vals, 100 - opts.saturation_pct);

    fig = figure(figNum);
    fig.Name = sprintf('%s — %s', grainID, label);
    fig.Color = opts.bkgdColor;
    ax = axes;

    h = imagesc(ax, img);
    colormap(ax, opts.cmap)
    clim(ax, [lo hi]);

    h.AlphaData = double(finite_mask);
    set(ax, 'Color', opts.bkgdColor)

    axis image off
    if opts.show_colorbar
        cb = colorbar(ax);
        cb.Color = opts.fontColor;
    end

    title(sprintf('%s  %s', grainID, label), 'Color', opts.fontColor);

    if opts.show_scalebar
        [nrows, ncols] = size(img);
        sb_px   = opts.scalebar_um / grain_px_um;
        mx      = opts.scalebar_margin * ncols;
        my      = opts.scalebar_margin * nrows;
        bar_h   = max(3, round(nrows * 0.012));

        switch opts.scalebar_pos
            case 'se'
                x1 = ncols - mx - sb_px;  y1 = nrows - my - bar_h;
            case 'sw'
                x1 = mx;                  y1 = nrows - my - bar_h;
            case 'ne'
                x1 = ncols - mx - sb_px;  y1 = my;
            case 'nw'
                x1 = mx;                  y1 = my;
        end
        x2 = x1 + sb_px;  y2 = y1 + bar_h;

        patch(ax, [x1 x2 x2 x1], [y1 y1 y2 y2], opts.fontColor, ...
              'EdgeColor', 'none', 'FaceAlpha', 1);
        text(ax, (x1+x2)/2, y1 - bar_h, ...
             sprintf('%g µm', opts.scalebar_um), ...
             'Color', opts.fontColor, 'FontSize', 9, ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end

    drawnow;

    if opts.save_figs
        if ~exist(opts.output_dir, 'dir'), mkdir(opts.output_dir); end
        out_path = fullfile(opts.output_dir, sprintf('%s_%s_display.png', grainID, filenameTag));
        exportgraphics(fig, out_path, 'Resolution', opts.fig_dpi);
        fprintf('Saved: %s\n', out_path);
    end
end
