% =========================================================================
% XRF_DISPLAY.m
%
% PURPOSE:
%   load XRF element map TIFFs for one or more grains and display each
%   element as a figure with an optional colorbar. Off-grain pixels are
%   masked transparent using a pre-existing binary mask.
%
% INPUTS (set in PARAMETERS section below):
%   - Folder of element map TIFFs (maps_dir)
%   - Grain ID string, or cell array of grain IDs for batch processing
%   - Mask TIFFs are auto-located as figs_dir/<grainID>_mask.tif
%   - List of element names to display
%   - Per-element upper percentile cutoffs for contrast scaling
%   - Colormap and colorbar visibility options
%
% OUTPUTS:
%   - One figure per element per grain showing the masked, contrast-scaled map
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
maps_dir = '/Users/mstein/bin/kyanite/maps/';
figs_dir = '/Users/mstein/bin/kyanite/figs/';

% Grain and element selection
% grainIDs may be a single string or a cell array for batch processing.
% Mask files are auto-located as figs_dir/<grainID>_mask.tif
grainIDs = {'NA-CM-G12B7-02'};
elements = {'Cr','Fe','Mn','Ti','V'};

% Visualization options
cmap = parula(256);
bkgdColor = 'k';
fontColor = [1 1 1];
show_colorbar = true;
saturation_pct = 1;   % % of within-grain pixels clipped at each end for display

% Figure export
save_figs   = false;
fig_dpi     = 300;

% Scale bar
show_scalebar    = true;
pixel_um         = 1.0;    % µm/pixel — scalar for all grains, or vector [1.0, 0.5, ...]
scalebar_um      = 100;    % physical length of scale bar in µm
scalebar_pos     = 'se';   % corner: 'se' | 'sw' | 'ne' | 'nw'
scalebar_margin  = 0.04;   % margin from edge as fraction of image dimensions

%% Normalize grainIDs to a cell array

if ischar(grainIDs)
    grainIDs = {grainIDs};
end
nGrains   = numel(grainIDs);
nElements = numel(elements);
figOffset = 0;

fprintf('Processing %d grain(s):\n', nGrains);
for g = 1:nGrains
    fprintf('  %s\n', grainIDs{g});
end

%% Loop over grains

for g = 1:nGrains

    grainID      = grainIDs{g};
    mask_path    = fullfile(figs_dir, sprintf('%s_mask.tif', grainID));
    grain_px_um  = pixel_um(min(g, end));

    fprintf('\n--- %s ---\n', grainID);

    %% Load element maps

    imgs = struct();
    for i = 1:nElements
        fn = fullfile(maps_dir, grainID, sprintf('%s_%s_Ka.tif', grainID, elements{i}));
        imgs.(elements{i}) = imread(fn);
    end

    %% Plot elements

    mask = logical(imread(mask_path));

    for i = 1:nElements

        img = double(imgs.(elements{i}));
        img(~mask) = NaN;
        grain_vals = img(mask & isfinite(img));
        lo = max(0, prctile(grain_vals, saturation_pct));
        hi = prctile(grain_vals, 100 - saturation_pct);

        fig = figure(figOffset + i);
        fig.Name = sprintf('%s — %s', grainID, elements{i});
        fig.Color = bkgdColor;
        ax = axes;

        h = imagesc(ax, img);
        colormap(ax, cmap)
        clim(ax, [lo hi]);

        h.AlphaData = double(mask);
        set(ax, 'Color', bkgdColor)

        axis image off
        if show_colorbar
            cb = colorbar(ax);
            cb.Color = fontColor;
        end

        title(sprintf('%s  %s', grainID, elements{i}), 'Color', fontColor);

        if show_scalebar
            [nrows, ncols] = size(img);
            sb_px   = scalebar_um / grain_px_um;
            mx      = scalebar_margin * ncols;
            my      = scalebar_margin * nrows;
            bar_h   = max(3, round(nrows * 0.012));

            switch scalebar_pos
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

            patch(ax, [x1 x2 x2 x1], [y1 y1 y2 y2], fontColor, ...
                  'EdgeColor', 'none', 'FaceAlpha', 1);
            text(ax, (x1+x2)/2, y1 - bar_h, ...
                 sprintf('%g µm', scalebar_um), ...
                 'Color', fontColor, 'FontSize', 9, ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end

        drawnow;

        if save_figs
            if ~exist(figs_dir, 'dir'), mkdir(figs_dir); end
            out_path = fullfile(figs_dir, sprintf('%s_%s_Ka_display.png', grainID, elements{i}));
            exportgraphics(fig, out_path, 'Resolution', fig_dpi);
            fprintf('Saved: %s\n', out_path);
        end

    end % elements

    figOffset = figOffset + nElements;

end % grains