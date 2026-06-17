% =========================================================================
% SUM_EPMA_MAPS.m
%
% PURPOSE:
%   Sum two or more EPMA element maps into a single combined map.
%   Typical use: combining multiple acquisitions of the same element
%   (e.g. Zr_La + Zr_Lb) or summing compositionally related maps.
%
%   If input images differ in size (e.g. due to colorbar width variation),
%   all maps are auto-cropped to the smallest dimensions before summing,
%   trimming from the right and bottom edges.
%
% OUTPUT:
%   A 32-bit float TIFF containing the raw pixel sum. Float32 preserves
%   exact count values regardless of input bit depth and is fully
%   compatible with CL_EPMA_registration.m as an EPMA input.
%
% REQUIREMENTS:
%   - MATLAB Image Processing Toolbox
%
% AUTHOR:  M. Stein
% =========================================================================

clear; clc;

% =========================================================================
%% PARAMETERS  — edit this section for each use
% =========================================================================

input_dir  = '/Users/mstein/bin/kyanite';
output_dir = '/Users/mstein/bin/kyanite';

% List all map filenames to be summed.
input_files = {
    'NA-CM-G12B4-02_Zr_La_it2.tif', ...
    'NA-CM-G12B4-02_Zr_La_it3.tif', ...
    'NA-CM-G12B4-02_Zr_La_it4.tif', ...
    'NA-CM-G12B4-02_Zr_La_it6.tif', ...
    'NA-CM-G12B4-02_Zr_La_it7.tif', ...
};

% Output filename (saved to output_dir).
output_file = 'NA-CM-G12B4-02_sumZr.tif';

% =========================================================================
%% LOAD
% =========================================================================

n = numel(input_files);
fprintf('Loading %d maps...\n', n);

imgs  = cell(1, n);
nrows = zeros(1, n);
ncols = zeros(1, n);

for i = 1:n
    fpath = fullfile(input_dir, input_files{i});
    if ~exist(fpath, 'file')
        error('File not found: %s', fpath);
    end
    raw = imread(fpath);
    info = imfinfo(fpath);

    % Collapse RGB to grayscale if needed (warn user)
    if ndims(raw) == 3
        warning('Map [%d] (%s) is RGB — converting to grayscale before summing.', ...
                i, input_files{i});
        raw = 0.2989*double(raw(:,:,1)) + 0.5870*double(raw(:,:,2)) + 0.1140*double(raw(:,:,3));
        raw = single(raw);
    end

    imgs{i}  = raw;
    nrows(i) = size(raw, 1);
    ncols(i) = size(raw, 2);
    fprintf('  [%d] %-35s  %d x %d px, %d-bit\n', ...
            i, input_files{i}, info(1).Height, info(1).Width, info(1).BitDepth);
end

% =========================================================================
%% AUTO-CROP TO SMALLEST DIMENSIONS
% =========================================================================

min_rows = min(nrows);
min_cols = min(ncols);

if numel(unique(nrows)) > 1 || numel(unique(ncols)) > 1
    fprintf('\nMaps differ in size — auto-cropping to %d x %d px.\n', min_rows, min_cols);
    for i = 1:n
        if nrows(i) ~= min_rows || ncols(i) ~= min_cols
            fprintf('  [%d] Cropped %d col(s), %d row(s) from right/bottom of %s\n', ...
                    i, ncols(i)-min_cols, nrows(i)-min_rows, input_files{i});
        end
        imgs{i} = imgs{i}(1:min_rows, 1:min_cols);
    end
else
    fprintf('\nAll maps are %d x %d px — no cropping needed.\n', min_rows, min_cols);
end

% =========================================================================
%% SUM
% =========================================================================

% Accumulate in double to avoid overflow for any input bit depth,
% then cast to single for storage (exact for typical EPMA count ranges).
sum_img = zeros(min_rows, min_cols, 'double');
for i = 1:n
    sum_img = sum_img + double(imgs{i});
end

fprintf('\nSum statistics:\n');
fprintf('  Min:  %.2f\n', min(sum_img(:)));
fprintf('  Max:  %.2f\n', max(sum_img(:)));
fprintf('  Mean: %.2f\n', mean(sum_img(:)));

% =========================================================================
%% SAVE
% =========================================================================

out_path = fullfile(output_dir, output_file);

t = Tiff(out_path, 'w');
tagstruct.ImageLength        = min_rows;
tagstruct.ImageWidth         = min_cols;
tagstruct.Photometric        = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample      = 32;
tagstruct.SampleFormat       = Tiff.SampleFormat.IEEEFP;
tagstruct.SamplesPerPixel    = 1;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
t.setTag(tagstruct);
t.write(single(sum_img));
t.close();

info_out = imfinfo(out_path);
fprintf('\nSaved: %s\n', out_path);
fprintf('Output: %d x %d px, %d-bit float\n', ...
        info_out.Height, info_out.Width, info_out.BitDepth);
