% =========================================================================
% XRF_DISPLAY.m
%
% PURPOSE:
%   load XRF element map TIFFs for a single grain and display each
%   element as a figure with an optional colorbar. Off-grain pixels are
%   masked transparent using a pre-existing binary mask.
%
% INPUTS (set in PARAMETERS section below):
%   - Folder of element map TIFFs (maps_dir)
%   - Grain ID string used to locate the correct subdirectory and files
%   - Binary mask TIFF produced by CL_EPMA_registration.m (mask_path)
%   - List of element names to display
%   - Per-element upper percentile cutoffs for contrast scaling
%   - Colormap and colorbar visibility options
%
% OUTPUTS:
%   - One figure per element showing the masked, contrast-scaled map
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
mask_path = '/Users/mstein/bin/kyanite/figs/NA-CM-G12B7-02_mask.tif';

% Grain and element selection
grainID = 'NA-CM-G12B7-02';
elements = {'Cr','Fe','Mn','Ti','V'};

% Visualization options
cmap = parula(256);
bkgdColor = 'k';
fontColor = [1 1 1];
show_colorbar = true;
disp_cutoffs = [98, 92, 94, 99, 99];

%% Create img store structure

nElements = length(elements);

imgs = struct();

for i=1:nElements
    fn = [maps_dir, grainID, '/', grainID, '_', elements{i}, '_Ka.tif'];
    imgs.(elements{i}) = imread(fn);
end

clear i fn maps_dir

%% Plot elements

mask = logical(imread(mask_path));

for i = 1:nElements

    img = double(imgs.(elements{i}));
    img(~mask) = NaN;
    high = prctile(img(:), disp_cutoffs(i));

    fig = figure(i);
    fig.Name = elements{i};
    fig.Color = bkgdColor;
    ax = axes;

    h = imagesc(ax, img);
    colormap(ax, cmap)
    clim(ax, [0 high]);

    h.AlphaData = double(mask);
    set(ax, 'Color', bkgdColor)

    axis image off
    if show_colorbar
        cb = colorbar(ax);
        cb.Color = fontColor;
    end

    title(elements{i}, 'Color', fontColor);

end

clear i fig high img ax h cb cmap show_colorbar mask_path bkgdColor fontColor