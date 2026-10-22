% =========================================================================
% XANES_EBSD_REGISTRATION.m
%
% PURPOSE:
%   place each XANES spot on its grain's EBSD map and read the crystal
%   orientation under it, so the measured pre-edge can be tested against
%   crystallographic orientation (the angle between the X-ray polarization
%   vector and the kyanite crystal axes at each spot).
%
%   The EBSD data stay an MTEX EBSD object throughout — nothing is warped,
%   resampled or interpolated. cpselect is only used to PICK control points
%   on a rendering of the map (band contrast or IPF); the fitted transform
%   is then applied to the spot COORDINATES (XRF grid -> EBSD µm), and
%   orientations are read from the untouched EBSD pixels at those points.
%
% WORKFLOW (per grain):
%   1. Load inputs/ebsd/<grain_id>_EBSD.ctf (orientations converted to the
%      map's spatial frame, so a vector on the map and an orientation are in
%      the same frame), MAD filter, reconstruct grains, drop grains smaller
%      than min_grain_px and reconstruct again (kyanite_ebsd.m's 'mtex'
%      cleaning without its fill)
%   2. Load the registered CL image (already on the XRF/spot pixel grid) and
%      the spot table from xrf_h5_extract_spots.py
%   3. cpselect: registered CL (moving) vs. EBSD band contrast / IPF (fixed).
%      Saved control points can be reused or edited on rerun
%   4. Fit an XRF-µm -> EBSD-µm transform (default affine); in-sample and
%      leave-one-out RMSE; polar decomposition of the local Jacobian at each
%      spot -> in-plane rotation, scale, mirror check
%   5. Transform spots into EBSD µm; for each, average the orientation of the
%      majority kyanite grain inside a footprint disk (radius = max of beam
%      radius, leave-one-out RMSE, EBSD step); flag spots whose footprint
%      spans more than one grain or holds no indexed pixel
%   6. Rotate the beam polarization (e) and propagation (k) vectors from the
%      XRF frame into the EBSD frame by the transform's in-plane rotation,
%      then into the crystal frame of each spot's orientation; report the
%      (axial, 0-90°) angle to [100], [010], [001] and to the (100), (010),
%      (001) plane normals
%   7. Save per-spot CSV, a log, and a QC figure
%
% FRAMES:
%   XRF frame: x = +column, y = +row (down) of the XRF TIFFs / registered CL
%   (row 1 at top, as in xrf_h5_to_tiff.py's row-flipped export), z = into
%   the sample. EBSD frame: MTEX spatial frame with xAxisDirection 'east',
%   zAxisDirection 'intoPlane', i.e. x = map right, y = map down, z = into
%   the sample — the same handedness, so a proper (det > 0) transform means
%   both images show the sample from the same side. A det < 0 transform is
%   reported loudly: one of the two images is mirrored relative to the
%   other, and the e/k vectors given below must describe physical directions
%   as they appear in the XRF TIFF.
%
% INPUTS:
%   - inputs/ebsd/<grain_id>_EBSD.ctf
%   - figs/<grain_id>_CL_registered.tif (+ _color.tif, optional)
%   - figs/data/<grain_id>_spot_geochemistry.csv (col_matlab/row_matlab)
%   - inputs/maps/<grain_id>/*.txt metadata sidecars (XRF µm/px)
%
% OUTPUTS:
%   - data/<grain_id>_ebsd_controlpoints.mat            — CL<->EBSD control
%     points, in pixels of each image (reusable)
%   - data/<grain_id>_spot_orientation.csv              — per spot: EBSD
%     position, footprint orientation + spread, grain/boundary flags, e and k
%     in the crystal frame and their angles to crystal axes/plane normals
%   - diagnostics/<grain_id>_xanes_ebsd_registration_log.txt
%   - diagnostics/<grain_id>_xanes_ebsd_registration_QC.png
%
% REQUIREMENTS:
%   - MTEX (tested with 6.1.1) on the MATLAB path (run startup_mtex first)
%   - Image Processing Toolbox, R2022b+ (cpselect, fitgeotform2d)
%
% AUTHOR:  M. Stein
% DATE:    2026-10-20
% VERSION: 1.0
% =========================================================================

clear; clc; close all;

% =========================================================================
%% SECTION 1: PARAMETERS  — edit this section per batch
% =========================================================================

repo_root = fileparts(mfilename('fullpath'));

% {} = every grain that has an EBSD .ctf, a spot geochemistry CSV and a
% registered CL image; otherwise a cell array of grain ids.
grain_ids = {};

ebsd_dir        = fullfile(repo_root, 'inputs', 'ebsd');
ctf_suffix      = '_EBSD.ctf';
maps_dir        = fullfile(repo_root, 'inputs', 'maps');
figs_dir        = fullfile(repo_root, 'figs');
data_dir        = fullfile(repo_root, 'figs', 'data');
diagnostics_dir = fullfile(repo_root, 'figs', 'diagnostics');
spot_suffix     = '_spot_geochemistry.csv';

phase_name = 'Kyanite';

% Orientations must be in the same frame as the map coordinates, or the
% angle between a map-frame vector (polarization) and the crystal is wrong.
% Unlike kyanite_ebsd.m (GROD only, frame-independent), this is required.
ctf_frame_option = 'convertEuler2SpatialReferenceFrame';

% Cleaning: kyanite_ebsd.m's 'mtex' method up to (not including) its fill —
% MAD filter, reconstruct grains, drop grains smaller than min_grain_px
% (misindexed specks, which would otherwise become a spot's "majority grain"
% with a meaningless grain mean), reconstruct again. No fill: only measured
% pixels are used for orientations. Keep these equal to kyanite_ebsd.m's
% mad_threshold / grain_angle_deg / mtex_min_grain_px so grains (and grain
% means) match its GROD output.
mad_threshold   = 2.0;
grain_angle_deg = 10;
min_grain_px    = 5;    % 0 = no small-grain drop

% --- Control points ---
% 'similarity' | 'affine' | 'projective'. Both images are in physical µm,
% so a good fit has scale ~1 on both axes (logged); a strong anisotropy
% points to a tilt-correction problem in the EBSD map.
transform_type = 'affine';
% 'ask' (prompt: use / edit / new), 'reuse' (use saved, else pick), 'new'
reuse_saved_cps = 'ask';
cp_fixed_display  = 'bc';     % EBSD image to pick on: 'bc' or 'ipf' (IPF-Z)
cp_moving_display = 'gray';   % registered CL: 'gray' or 'color'
display_pct = [1 99];         % contrast stretch for the picking images

% --- Spot footprint ---
% Orientation at a spot = mean over the majority kyanite grain inside a
% disk of radius max(beam_radius_um, leave-one-out RMSE, EBSD step).
beam_radius_um = 2;

% XRF µm/px is read from xrf_h5_to_tiff.py's metadata sidecar; this is only
% the fallback (with a warning) if none is found.
xrf_pixel_um_fallback = 2.0;

% --- Beam geometry, in the XRF frame (x = +col, y = +row/down, z = into
% sample) ---
% *** VERIFY FOR YOUR BEAMLINE SETUP. *** Default assumes a GSECARS
% 13-ID-E-style geometry: horizontal beam and horizontal linear
% polarization, sample surface at 45° to the beam, stage X (fast axis =
% XRF columns) horizontal, beam arriving from the -x side. If the beam
% arrives from the +x side, flip the sign of the x component of BOTH
% vectors. Only the relative sign of x and z matters for e (e and -e are
% equivalent for linear polarization).
polarization_xrf = [1 0 -1] / sqrt(2);   % e
beam_dir_xrf     = [1 0  1] / sqrt(2);   % k (propagation direction)

% --- Figure styling ---
font_name = 'Helvetica';
fig_dpi   = 300;

setMTEXpref('xAxisDirection', 'east');
setMTEXpref('zAxisDirection', 'intoPlane');

P = struct('ctf_frame_option', ctf_frame_option, 'phase_name', phase_name, ...
    'mad_threshold', mad_threshold, 'grain_angle_deg', grain_angle_deg, ...
    'min_grain_px', min_grain_px, 'transform_type', lower(transform_type), 'reuse_saved_cps', lower(reuse_saved_cps), ...
    'cp_fixed_display', lower(cp_fixed_display), 'cp_moving_display', lower(cp_moving_display), ...
    'display_pct', display_pct, 'beam_radius_um', beam_radius_um, ...
    'xrf_pixel_um_fallback', xrf_pixel_um_fallback, ...
    'e_xrf', polarization_xrf(:)' / norm(polarization_xrf), ...
    'k_xrf', beam_dir_xrf(:)' / norm(beam_dir_xrf), ...
    'font_name', font_name, 'fig_dpi', fig_dpi, ...
    'ebsd_dir', ebsd_dir, 'ctf_suffix', ctf_suffix, 'maps_dir', maps_dir, ...
    'figs_dir', figs_dir, 'data_dir', data_dir, 'diagnostics_dir', diagnostics_dir, ...
    'spot_suffix', spot_suffix);

if ~any(strcmp(P.transform_type, {'similarity', 'affine', 'projective'}))
    error('transform_type must be ''similarity'', ''affine'' or ''projective''.');
end
if abs(dot(P.e_xrf, P.k_xrf)) > 1e-6
    error('polarization_xrf and beam_dir_xrf must be perpendicular (e·k = %.3g).', ...
        dot(P.e_xrf, P.k_xrf));
end

% =========================================================================
%% SECTION 2: DISCOVER GRAINS
% =========================================================================

if exist('EBSD', 'class') ~= 8 && isempty(which('EBSD.load'))
    error('MTEX not found on the MATLAB path — run startup_mtex first.');
end
if isempty(which('fitgeotform2d'))
    error('fitgeotform2d not found — needs Image Processing Toolbox, R2022b or later.');
end
for d = {data_dir, diagnostics_dir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

if isempty(grain_ids)
    listing   = dir(fullfile(ebsd_dir, ['*' ctf_suffix]));
    candidates = erase({listing.name}, ctf_suffix);
    grain_ids = {};
    for g = 1:numel(candidates)
        gid = candidates{g};
        missing = {};
        if ~isfile(fullfile(data_dir, [gid spot_suffix])), missing{end+1} = 'spot CSV'; end %#ok<SAGROW>
        if ~isfile(fullfile(figs_dir, [gid '_CL_registered.tif'])), missing{end+1} = 'registered CL'; end %#ok<SAGROW>
        if isempty(missing)
            grain_ids{end+1} = gid; %#ok<SAGROW>
        else
            fprintf('Skipping %s (EBSD present, no %s)\n', gid, strjoin(missing, ' / '));
        end
    end
    if isempty(grain_ids)
        error('No grain has all of: EBSD .ctf, spot CSV, registered CL.');
    end
end
if ischar(grain_ids), grain_ids = {grain_ids}; end

fprintf('XANES-EBSD registration: %d grain(s): %s\n', numel(grain_ids), strjoin(grain_ids, ', '));

% =========================================================================
%% SECTION 3: PROCESS EACH GRAIN
% =========================================================================

n_ok = 0;
for g = 1:numel(grain_ids)
    fprintf('\n=== %s ===\n', grain_ids{g});
    try
        register_grain(grain_ids{g}, P);
        n_ok = n_ok + 1;
    catch err
        warning('xanes_ebsd_registration:grainFailed', 'Skipping %s: %s', ...
            grain_ids{g}, err.message);
    end
end
fprintf('\nDone: %d of %d grain(s) processed.\n', n_ok, numel(grain_ids));

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================

function register_grain(grain_id, P)

    % ---------------- EBSD ----------------
    ctf_file = fullfile(P.ebsd_dir, [grain_id P.ctf_suffix]);
    if ~isfile(ctf_file), error('File not found: %s', ctf_file); end
    ebsd = EBSD.load(ctf_file, P.ctf_frame_option);
    if ~any(strcmp(ebsd.mineralList, P.phase_name))
        error('Phase "%s" not in %s', P.phase_name, ctf_file);
    end
    G = raw_grid(ebsd.x(:), ebsd.y(:));

    % Same order as kyanite_ebsd.m's clean_mtex: every phase (and non-indexed
    % pixels, MAD 0, which pass) goes into the reconstruction, so other
    % phases and gaps bound kyanite grains the same way there and here.
    ang = P.grain_angle_deg*degree;
    ebsdMad = ebsd(get_prop(ebsd, 'mad') < P.mad_threshold);
    n_mad = length(ebsdMad(P.phase_name));
    [grainsAll, ebsdMad.grainId] = calcGrains(ebsdMad, 'angle', ang);
    if P.min_grain_px > 0
        ebsdMad = ebsdMad(grainsAll(grainsAll.numPixel >= P.min_grain_px));
        [grainsAll, ebsdMad.grainId] = calcGrains(ebsdMad, 'angle', ang);
    end
    ebsdK  = ebsdMad(P.phase_name);
    grains = grainsAll(P.phase_name);
    cs = ebsdK.CS;
    fprintf('EBSD: %d px; %s: %d after MAD < %.2f, %d after dropping grains < %d px; %d grains, step %.3g µm\n', ...
        length(ebsd), P.phase_name, n_mad, P.mad_threshold, length(ebsdK), P.min_grain_px, ...
        grains.length, G.dx);

    % Band contrast on the full raw grid (non-indexed pixels have BC too)
    bc = get_prop(ebsd, 'bc');
    if all(isnan(bc))
        bc_img = zeros(G.sz);
        warning('No band contrast in %s; BC panel will be blank.', ctf_file);
    else
        bc_img = nan(G.sz);
        bc_img(G.ind(ebsd.x(:), ebsd.y(:))) = bc;
        bc_img = stretch(bc_img, P.display_pct);
    end

    % ---------------- XRF grid, CL, spots ----------------
    [px_x, px_y, px_src] = xrf_pixel_um(fullfile(P.maps_dir, grain_id), P.xrf_pixel_um_fallback);
    fprintf('XRF pixel: %.4g x %.4g µm (%s)\n', px_x, px_y, px_src);

    cl_gray = imread(fullfile(P.figs_dir, [grain_id '_CL_registered.tif']));
    cl_gray = stretch(double(cl_gray(:,:,1)), P.display_pct);
    cl_color_file = fullfile(P.figs_dir, [grain_id '_CL_registered_color.tif']);

    spots = readtable(fullfile(P.data_dir, [grain_id P.spot_suffix]), 'TextType', 'string');
    need = {'spot', 'col_matlab', 'row_matlab'};
    if ~all(ismember(need, spots.Properties.VariableNames))
        error('Spot CSV lacks one of: %s', strjoin(need, ', '));
    end
    n_spot = height(spots);

    % ---------------- Control points ----------------
    cp_file = fullfile(P.data_dir, [grain_id '_ebsd_controlpoints.mat']);
    [moving_pts, fixed_pts, cp_source] = get_control_points(cp_file, P, cl_gray, ...
        cl_color_file, bc_img, ebsdK, G, cs);

    min_pts_by_type = struct('similarity', 2, 'affine', 3, 'projective', 4);
    min_pts = min_pts_by_type.(P.transform_type);
    n_cp = size(moving_pts, 1);
    if n_cp < min_pts + 1
        error('%d control points; %s needs at least %d (one more than the minimum, for leave-one-out).', ...
            n_cp, P.transform_type, min_pts + 1);
    end
    if n_cp < 6
        warning('Only %d control points — 6+ well-spread points are recommended.', n_cp);
    end

    % Pixels (X = col, Y = row, 1-based) -> µm in each frame
    mov_um = xrf_px2um(moving_pts, px_x, px_y);
    fix_um = ebsd_px2um(fixed_pts, G);

    % ---------------- Transform + quality ----------------
    tform = fitgeotform2d(mov_um, fix_um, P.transform_type);
    resid = fix_um - transformPointsForward(tform, mov_um);
    rmse_um = sqrt(mean(sum(resid.^2, 2)));

    loo_resid = nan(n_cp, 2);
    for k = 1:n_cp
        keep = true(n_cp, 1);  keep(k) = false;
        tk = fitgeotform2d(mov_um(keep,:), fix_um(keep,:), P.transform_type);
        loo_resid(k,:) = fix_um(k,:) - transformPointsForward(tk, mov_um(k,:));
    end
    loo_dist = sqrt(sum(loo_resid.^2, 2));
    loo_rmse_um = sqrt(mean(loo_dist.^2));
    fprintf('Transform (%s): RMSE %.2f µm, leave-one-out RMSE %.2f µm (max %.2f, CP %d)\n', ...
        P.transform_type, rmse_um, loo_rmse_um, max(loo_dist), find(loo_dist == max(loo_dist), 1));

    r_eff = max([P.beam_radius_um, loo_rmse_um, G.dx, G.dy]);

    % ---------------- Spots -> EBSD ----------------
    sp_xrf = xrf_px2um([spots.col_matlab, spots.row_matlab], px_x, px_y);
    sp_eb  = transformPointsForward(tform, sp_xrf);

    dirs = crystal_reference_dirs(cs);
    ex = ebsdK.x(:);  ey = ebsdK.y(:);
    gid_all = ebsdK.grainId(:);

    S = struct();
    S.x_ebsd_um = sp_eb(:,1);  S.y_ebsd_um = sp_eb(:,2);
    cols_nan = {'rot_deg', 'scale_1', 'scale_2', 'jac_det', 'n_px_footprint', ...
        'n_grains_footprint', 'n_px_majority', 'majority_frac', 'ebsd_grain_id', ...
        'nearest_px_um', 'phi1_deg', 'Phi_deg', 'phi2_deg', 'misori_mean_deg', ...
        'misori_max_deg', 'misori_to_grain_mean_deg', ...
        'e_ebsd_x', 'e_ebsd_y', 'e_ebsd_z', 'k_ebsd_x', 'k_ebsd_y', 'k_ebsd_z', ...
        'e_crys_x', 'e_crys_y', 'e_crys_z', 'k_crys_x', 'k_crys_y', 'k_crys_z'};
    for c = [cols_nan, strcat('e_angle_', dirs.names), strcat('k_angle_', dirs.names)]
        S.(c{1}) = nan(n_spot, 1);
    end
    S.flag = strings(n_spot, 1);

    for s = 1:n_spot
        % Local in-plane rotation (polar decomposition of the Jacobian)
        J = local_jacobian(tform, sp_xrf(s,:));
        [U, Sv, V] = svd(J);
        R = U * V';
        S.rot_deg(s) = atan2d(R(2,1), R(1,1));
        S.scale_1(s) = Sv(1,1);  S.scale_2(s) = Sv(2,2);
        S.jac_det(s) = det(J);

        e_eb = [R * P.e_xrf(1:2)'; P.e_xrf(3)];  e_eb = e_eb / norm(e_eb);
        k_eb = [R * P.k_xrf(1:2)'; P.k_xrf(3)];  k_eb = k_eb / norm(k_eb);
        [S.e_ebsd_x(s), S.e_ebsd_y(s), S.e_ebsd_z(s)] = deal(e_eb(1), e_eb(2), e_eb(3));
        [S.k_ebsd_x(s), S.k_ebsd_y(s), S.k_ebsd_z(s)] = deal(k_eb(1), k_eb(2), k_eb(3));

        % Footprint
        d = hypot(ex - sp_eb(s,1), ey - sp_eb(s,2));
        in = d <= r_eff;
        S.n_px_footprint(s) = nnz(in);
        if ~any(in)
            S.nearest_px_um(s) = min(d);
            S.flag(s) = "no_indexed_px";
            continue
        end
        S.nearest_px_um(s) = min(d);
        gids = gid_all(in);
        maj  = mode(gids);
        sel  = in & gid_all == maj;
        S.n_grains_footprint(s) = numel(unique(gids));
        S.n_px_majority(s) = nnz(sel);
        S.majority_frac(s) = nnz(sel) / nnz(in);
        S.ebsd_grain_id(s) = maj;
        if S.n_grains_footprint(s) > 1, S.flag(s) = "multi_grain"; else, S.flag(s) = "ok"; end

        ori  = ebsdK(sel).orientations;
        oriM = mean(ori);
        mis  = angle(ori, oriM) / degree;
        S.misori_mean_deg(s) = mean(mis);
        S.misori_max_deg(s)  = max(mis);
        S.misori_to_grain_mean_deg(s) = angle(oriM, grains(grains.id == maj).meanOrientation) / degree;
        [p1, PP, p2] = Euler(oriM, 'Bunge');
        [S.phi1_deg(s), S.Phi_deg(s), S.phi2_deg(s)] = deal(p1/degree, PP/degree, p2/degree);

        e_c = crystal_xyz(oriM, e_eb);
        k_c = crystal_xyz(oriM, k_eb);
        [S.e_crys_x(s), S.e_crys_y(s), S.e_crys_z(s)] = deal(e_c(1), e_c(2), e_c(3));
        [S.k_crys_x(s), S.k_crys_y(s), S.k_crys_z(s)] = deal(k_c(1), k_c(2), k_c(3));
        for j = 1:numel(dirs.names)
            % axial angle (0-90°): e and -e, and a direction and its
            % inverse, are equivalent
            S.(['e_angle_' dirs.names{j}])(s) = acosd(min(abs(dot(e_c, dirs.xyz(j,:))), 1));
            S.(['k_angle_' dirs.names{j}])(s) = acosd(min(abs(dot(k_c, dirs.xyz(j,:))), 1));
        end
    end

    improper = any(S.jac_det < 0);
    if improper
        warning(['%s: transform is improper (det < 0) — the XRF/CL and EBSD images are ' ...
            'mirrored relative to each other. Check that polarization_xrf/beam_dir_xrf ' ...
            'describe directions as they appear in the XRF TIFF.'], grain_id);
    end
    anis = max(S.scale_1 ./ S.scale_2);
    if anis > 1.1 || any(abs([S.scale_1; S.scale_2] - 1) > 0.1)
        warning(['%s: transform scale departs from 1 by >10%% (%.3f / %.3f) — both frames ' ...
            'are in µm, so this suggests a tilt-correction or pixel-size problem.'], ...
            grain_id, median(S.scale_1), median(S.scale_2));
    end

    % ---------------- Outputs ----------------
    T = table(repmat(string(grain_id), n_spot, 1), spots.spot, ...
        'VariableNames', {'grain_id', 'spot'});
    for c = {'spot_id', 'category_label', 'on_grain'}
        if ismember(c{1}, spots.Properties.VariableNames), T.(c{1}) = spots.(c{1}); end
    end
    T.col_matlab = spots.col_matlab;  T.row_matlab = spots.row_matlab;
    T.x_xrf_um = sp_xrf(:,1);  T.y_xrf_um = sp_xrf(:,2);
    T.footprint_radius_um = repmat(r_eff, n_spot, 1);
    T.loo_rmse_um = repmat(loo_rmse_um, n_spot, 1);
    T.flag = S.flag;
    for c = setdiff(fieldnames(S)', {'flag'}, 'stable')
        T.(c{1}) = S.(c{1});
    end
    csv_file = fullfile(P.data_dir, [grain_id '_spot_orientation.csv']);
    writetable(T, csv_file);

    qc_png = fullfile(P.diagnostics_dir, [grain_id '_xanes_ebsd_registration_QC.png']);
    plot_qc(grain_id, P, G, bc_img, cl_gray, px_x, px_y, ebsdK, cs, tform, ...
        mov_um, fix_um, loo_resid, sp_xrf, sp_eb, spots.spot, S, r_eff, qc_png);

    log_file = fullfile(P.diagnostics_dir, [grain_id '_xanes_ebsd_registration_log.txt']);
    write_log(log_file, grain_id, P, ctf_file, G, ebsd, ebsdK, grains, cs, dirs, ...
        px_x, px_y, px_src, cp_file, cp_source, moving_pts, fixed_pts, mov_um, fix_um, ...
        tform, resid, rmse_um, loo_resid, loo_rmse_um, r_eff, S, spots.spot, improper, ...
        csv_file, qc_png);

    n_ok = nnz(S.flag == "ok");
    fprintf('%s: %d spots — %d single-grain, %d multi-grain, %d no indexed px; rotation %.1f°\n', ...
        grain_id, n_spot, n_ok, nnz(S.flag == "multi_grain"), nnz(S.flag == "no_indexed_px"), ...
        median(S.rot_deg));
    fprintf('  -> %s\n  -> %s\n  -> %s\n', csv_file, qc_png, log_file);
end

function [moving_pts, fixed_pts, cp_source] = get_control_points(cp_file, P, cl_gray, ...
        cl_color_file, bc_img, ebsdK, G, cs)
% Reuse / edit / pick CL<->EBSD control points (pixel coords of each image).
    have = isfile(cp_file);
    action = 'new';
    if have
        switch P.reuse_saved_cps
            case 'reuse', action = 'use';
            case 'new',   action = 'new';
            otherwise
                resp = lower(strtrim(input(sprintf( ...
                    'Saved EBSD control points found (%s).\n  (u)se / (e)dit / (n)ew? ', cp_file), 's')));
                if startsWith(resp, 'u'), action = 'use';
                elseif startsWith(resp, 'e'), action = 'edit';
                end
        end
    end

    if strcmp(action, 'use')
        L = load(cp_file, 'moving_pts', 'fixed_pts');
        moving_pts = L.moving_pts;  fixed_pts = L.fixed_pts;
        cp_source = ['loaded from ' cp_file];
        fprintf('Loaded %d control point pairs.\n', size(moving_pts, 1));
        return
    end

    % Picking images
    if strcmp(P.cp_moving_display, 'color') && isfile(cl_color_file)
        mov_img = im2double(imread(cl_color_file));
    else
        mov_img = cl_gray;
    end
    if strcmp(P.cp_fixed_display, 'ipf')
        fix_img = ipf_image(ebsdK, G, cs, vector3d.Z);
    else
        fix_img = bc_img;
    end
    mov_img(isnan(mov_img)) = 0;  fix_img(isnan(fix_img)) = 0;

    fprintf(['cpselect: left = registered CL (moving), right = EBSD %s (fixed).\n' ...
        '  Pick 6+ well-spread pairs (grain edges, cracks, inclusions); close the window when done.\n'], ...
        upper(P.cp_fixed_display));
    if strcmp(action, 'edit')
        L = load(cp_file, 'moving_pts', 'fixed_pts');
        [moving_pts, fixed_pts] = cpselect(mov_img, fix_img, L.moving_pts, L.fixed_pts, 'Wait', true);
        cp_source = ['edited from ' cp_file];
    else
        [moving_pts, fixed_pts] = cpselect(mov_img, fix_img, 'Wait', true);
        cp_source = 'newly selected via cpselect';
    end
    cp_fixed_display = P.cp_fixed_display; %#ok<NASGU>
    ebsd_grid = rmfield(G, 'ind'); %#ok<NASGU>
    save(cp_file, 'moving_pts', 'fixed_pts', 'cp_fixed_display', 'ebsd_grid');
    fprintf('%d control point pairs saved to %s\n', size(moving_pts, 1), cp_file);
end

function um = xrf_px2um(px, px_x, px_y)
% XRF pixel (X = col, Y = row, 1-based, pixel centres) -> µm from pixel (1,1).
    um = [(px(:,1) - 1) * px_x, (px(:,2) - 1) * px_y];
end

function um = ebsd_px2um(px, G)
% Gridded EBSD image pixel (X = col, Y = row, 1-based) -> MTEX x/y in µm.
    um = [G.x0 + (px(:,1) - 1) * G.dx, G.y0 + (px(:,2) - 1) * G.dy];
end

function J = local_jacobian(tform, p)
% 2x2 Jacobian of the forward transform at p (exact for similarity/affine).
    h = 1;   % µm
    fx = transformPointsForward(tform, [p + [h 0]; p - [h 0]]);
    fy = transformPointsForward(tform, [p + [0 h]; p - [0 h]]);
    J = [(fx(1,:) - fx(2,:))', (fy(1,:) - fy(2,:))'] / (2*h);
end

function v = crystal_xyz(oriM, vec)
% Specimen vector -> unit vector in the crystal's Cartesian frame.
    h = oriM \ vector3d(vec(1), vec(2), vec(3));
    v = [h.x, h.y, h.z];
    v = v / norm(v);
end

function dirs = crystal_reference_dirs(cs)
% Crystal axes [100] [010] [001] and plane normals (100) (010) (001) as unit
% vectors in the crystal Cartesian frame (triclinic: these differ).
    dirs.names = {'a_100', 'b_010', 'c_001', 'n100', 'n010', 'n001'};
    m = {Miller(1,0,0,cs,'uvw'), Miller(0,1,0,cs,'uvw'), Miller(0,0,1,cs,'uvw'), ...
         Miller(1,0,0,cs,'hkl'), Miller(0,1,0,cs,'hkl'), Miller(0,0,1,cs,'hkl')};
    dirs.xyz = zeros(numel(m), 3);
    for j = 1:numel(m)
        v = [m{j}.x, m{j}.y, m{j}.z];
        dirs.xyz(j,:) = v / norm(v);
    end
end

function rgb = ipf_image(ebsdK, G, cs, direction)
% IPF-colored RGB raster on the raw grid (black where no kyanite pixel).
    key = ipfHSVKey(cs);
    key.inversePoleFigureDirection = direction;
    col = key.orientation2color(ebsdK.orientations);
    rgb = zeros([G.sz 3]);
    ind = G.ind(ebsdK.x(:), ebsdK.y(:));
    for c = 1:3
        ch = zeros(G.sz);
        ch(ind) = col(:, c);
        rgb(:,:,c) = ch;
    end
end

function plot_qc(grain_id, P, G, bc_img, cl_gray, px_x, px_y, ebsdK, cs, tform, ...
        mov_um, fix_um, loo_resid, sp_xrf, sp_eb, spot_no, S, r_eff, png)
% 2 x 2 diagnostic: CL with control points/spots; EBSD BC with control-point
% leave-one-out residuals and spot footprints; IPF colored by the crystal
% direction parallel to the polarization vector; CL warped onto the EBSD
% grid, fused with BC (registration check — the CL image is warped here,
% never the EBSD data).
    BLUE  = [59 155 221]/255;   % kyanite_palette house BLUE
    ORANG = [216 91 48]/255;    % kyanite_palette house ORANG
    YEL   = [1 0.85 0];

    [nr, nc] = size(cl_gray);
    xv_cl = (0:nc-1) * px_x;  yv_cl = (0:nr-1) * px_y;

    R_cl = imref2d([nr nc], [-0.5, nc - 0.5] * px_x, [-0.5, nr - 0.5] * px_y);
    R_eb = imref2d(G.sz, G.x0 + [-0.5, G.sz(2) - 0.5] * G.dx, G.y0 + [-0.5, G.sz(1) - 0.5] * G.dy);
    cl_w = imwarp(fillmissing_zero(cl_gray), R_cl, tform, 'OutputView', R_eb);
    fused = imfuse(fillmissing_zero(bc_img), cl_w, 'falsecolor', 'ColorChannels', 'green-magenta');

    e_mean = [median(S.e_ebsd_x), median(S.e_ebsd_y), median(S.e_ebsd_z)];
    ipf = ipf_image(ebsdK, G, cs, vector3d(e_mean(1), e_mean(2), e_mean(3)));

    ok = S.flag == "ok";
    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'pixels', 'Position', [0 0 1500 1350]);
    set(fig, 'DefaultAxesFontSize', 10, 'DefaultAxesFontName', P.font_name, ...
        'DefaultTextFontName', P.font_name);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % 1: CL + CPs + spots
    ax = nexttile(tl);
    imagesc(ax, xv_cl, yv_cl, cl_gray);  colormap(ax, gray);  axis(ax, 'image');  hold(ax, 'on');
    plot(ax, sp_xrf(:,1), sp_xrf(:,2), 'o', 'MarkerSize', 4, 'MarkerFaceColor', BLUE, 'MarkerEdgeColor', 'w');
    plot(ax, mov_um(:,1), mov_um(:,2), '+', 'Color', YEL, 'MarkerSize', 10, 'LineWidth', 1.5);
    text(ax, mov_um(:,1), mov_um(:,2), "  " + string(1:size(mov_um,1))', 'Color', YEL, 'FontSize', 9);
    title(ax, sprintf('%s — registered CL (XRF frame): control points, spots', grain_id), ...
        'FontWeight', 'normal', 'Interpreter', 'none');
    xlabel(ax, 'x (µm)');  ylabel(ax, 'y (µm)');

    % 2: BC + CP residuals + spot footprints
    ax = nexttile(tl);
    imagesc(ax, G.xv, G.yv, bc_img);  colormap(ax, gray);  axis(ax, 'image');  hold(ax, 'on');
    loo_pred = fix_um - loo_resid;
    plot(ax, [fix_um(:,1) loo_pred(:,1)]', [fix_um(:,2) loo_pred(:,2)]', '-', 'Color', ORANG, 'LineWidth', 1.5);
    plot(ax, fix_um(:,1), fix_um(:,2), '+', 'Color', YEL, 'MarkerSize', 10, 'LineWidth', 1.5);
    text(ax, fix_um(:,1), fix_um(:,2), "  " + string(1:size(fix_um,1))', 'Color', YEL, 'FontSize', 9);
    draw_footprints(ax, sp_eb, r_eff, ok, BLUE, ORANG);
    title(ax, {'EBSD band contrast: control points (+), leave-one-out residual (line),', ...
        sprintf('spot footprints r = %.1f µm (blue single grain, orange multi-grain / no data)', r_eff)}, ...
        'FontWeight', 'normal');
    xlabel(ax, 'x (µm)');  ylabel(ax, 'y (µm)');

    % 3: IPF parallel to polarization
    ax = nexttile(tl);
    image(ax, G.xv, G.yv, ipf);  axis(ax, 'image');  hold(ax, 'on');
    draw_footprints(ax, sp_eb, r_eff, ok, [1 1 1], ORANG);
    text(ax, sp_eb(:,1), sp_eb(:,2), "  " + string(spot_no), 'Color', 'w', 'FontSize', 7);
    title(ax, sprintf('IPF colored by crystal direction ∥ polarization e (EBSD frame [%.2f %.2f %.2f])', e_mean), ...
        'FontWeight', 'normal');
    xlabel(ax, 'x (µm)');  ylabel(ax, 'y (µm)');

    % 4: warped CL fused with BC
    ax = nexttile(tl);
    image(ax, G.xv, G.yv, fused);  axis(ax, 'image');  hold(ax, 'on');
    plot(ax, sp_eb(:,1), sp_eb(:,2), 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k');
    title(ax, sprintf('CL warped to EBSD (magenta) over BC (green) — %s, rotation %.1f°, scale %.3f/%.3f', ...
        P.transform_type, median(S.rot_deg), median(S.scale_1), median(S.scale_2)), 'FontWeight', 'normal');
    xlabel(ax, 'x (µm)');  ylabel(ax, 'y (µm)');

    exportgraphics(fig, png, 'Resolution', P.fig_dpi);
    close(fig);
end

function draw_footprints(ax, pts, r, ok, c_ok, c_bad)
    t = linspace(0, 2*pi, 48);
    for s = 1:size(pts, 1)
        c = c_bad;  if ok(s), c = c_ok; end
        plot(ax, pts(s,1) + r*cos(t), pts(s,2) + r*sin(t), '-', 'Color', c, 'LineWidth', 1);
    end
end

function write_log(log_file, grain_id, P, ctf_file, G, ebsd, ebsdK, grains, cs, dirs, ...
        px_x, px_y, px_src, cp_file, cp_source, moving_pts, fixed_pts, mov_um, fix_um, ...
        tform, resid, rmse_um, loo_resid, loo_rmse_um, r_eff, S, spot_no, improper, csv_file, qc_png)
    fid = fopen(log_file, 'w');
    c = onCleanup(@() fclose(fid));
    fprintf(fid, 'XANES_EBSD_REGISTRATION.m — %s\n', grain_id);
    fprintf(fid, 'Run: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

    fprintf(fid, 'EBSD: %s\n', ctf_file);
    fprintf(fid, '  frame option      %s\n', P.ctf_frame_option);
    fprintf(fid, '  grid              %d x %d px, step %.4g x %.4g µm\n', G.sz(2), G.sz(1), G.dx, G.dy);
    fprintf(fid, '  pixels            %d raw, %d %s with MAD < %.2f and in grains >= %d px (no fill)\n', ...
        length(ebsd), length(ebsdK), P.phase_name, P.mad_threshold, P.min_grain_px);
    fprintf(fid, '  grains            %d %s (boundary %g°)\n', grains.length, P.phase_name, P.grain_angle_deg);
    try
        fprintf(fid, '  crystal frame     %s\n', strjoin(string(cs.alignment), ', '));
    catch
    end
    fprintf(fid, 'XRF pixel: %.4g x %.4g µm (%s)\n\n', px_x, px_y, px_src);

    fprintf(fid, 'Beam geometry (XRF frame: x = +col, y = +row/down, z = into sample):\n');
    fprintf(fid, '  polarization e    [%.4f %.4f %.4f]\n', P.e_xrf);
    fprintf(fid, '  propagation  k    [%.4f %.4f %.4f]\n\n', P.k_xrf);

    fprintf(fid, 'Control points: %s\n  file: %s\n', cp_source, cp_file);
    fprintf(fid, '  picked on: CL %s vs. EBSD %s\n', P.cp_moving_display, P.cp_fixed_display);
    fprintf(fid, '  %-3s %10s %10s %10s %10s | %10s %10s %10s %10s | %8s %8s\n', '#', ...
        'CL_X_px', 'CL_Y_px', 'EB_X_px', 'EB_Y_px', 'CL_x_um', 'CL_y_um', 'EB_x_um', 'EB_y_um', ...
        'res_um', 'loo_um');
    for k = 1:size(moving_pts, 1)
        fprintf(fid, '  %-3d %10.2f %10.2f %10.2f %10.2f | %10.2f %10.2f %10.2f %10.2f | %8.2f %8.2f\n', ...
            k, moving_pts(k,:), fixed_pts(k,:), mov_um(k,:), fix_um(k,:), ...
            norm(resid(k,:)), norm(loo_resid(k,:)));
    end

    fprintf(fid, '\nTransform: %s, XRF µm -> EBSD µm\n', P.transform_type);
    fprintf(fid, '  A (premultiply, [u; v; 1] = A * [x; y; 1]):\n');
    fprintf(fid, '    %14.8f %14.8f %14.8f\n', tform.A');
    fprintf(fid, '  RMSE               %.3f µm\n', rmse_um);
    fprintf(fid, '  leave-one-out RMSE %.3f µm (max %.3f µm)\n', loo_rmse_um, max(sqrt(sum(loo_resid.^2, 2))));
    fprintf(fid, '  in-plane rotation  %.2f° (median over spots; y-down frame, + = clockwise on screen)\n', median(S.rot_deg));
    fprintf(fid, '  scale              %.4f / %.4f (median principal stretches; expect ~1)\n', ...
        median(S.scale_1), median(S.scale_2));
    fprintf(fid, '  det(J)             %.4f%s\n\n', median(S.jac_det), ...
        ternary(improper, '  ** IMPROPER: images mirrored relative to each other **', ''));

    fprintf(fid, 'Spot footprint radius: %.2f µm = max(beam %.2f, LOO RMSE %.2f, EBSD step %.2f)\n', ...
        r_eff, P.beam_radius_um, loo_rmse_um, max(G.dx, G.dy));
    fprintf(fid, 'Spots: %d — %d single-grain, %d multi-grain, %d without indexed px\n\n', ...
        numel(spot_no), nnz(S.flag == "ok"), nnz(S.flag == "multi_grain"), nnz(S.flag == "no_indexed_px"));

    fprintf(fid, '%-5s %-14s %9s %9s %5s %6s %7s %8s %8s %8s %7s %7s', 'spot', 'flag', 'x_um', 'y_um', ...
        'n_px', 'n_gr', 'maj', 'phi1', 'Phi', 'phi2', 'mis_mn', 'mis_mx');
    for j = 1:numel(dirs.names)
        fprintf(fid, ' %7s', ['e_' dirs.names{j}]);
    end
    fprintf(fid, '\n');
    for s = 1:numel(spot_no)
        fprintf(fid, '%-5d %-14s %9.1f %9.1f %5d %6d %7.2f %8.2f %8.2f %8.2f %7.2f %7.2f', ...
            spot_no(s), S.flag(s), S.x_ebsd_um(s), S.y_ebsd_um(s), S.n_px_footprint(s), ...
            S.n_grains_footprint(s), S.majority_frac(s), S.phi1_deg(s), S.Phi_deg(s), ...
            S.phi2_deg(s), S.misori_mean_deg(s), S.misori_max_deg(s));
        for j = 1:numel(dirs.names)
            fprintf(fid, ' %7.1f', S.(['e_angle_' dirs.names{j}])(s));
        end
        fprintf(fid, '\n');
    end
    fprintf(fid, '\n(e_* = axial angle in degrees between polarization and that crystal axis / plane normal)\n');
    fprintf(fid, '\nOutputs:\n  %s\n  %s\n  %s\n', csv_file, qc_png, cp_file);
end

function [px_x, px_y, src] = xrf_pixel_um(map_dir, fallback)
% Per-axis XRF µm/px from xrf_h5_to_tiff.py's metadata sidecars (first
% parseable one in the grain's maps folder); fallback with a warning.
    px_x = NaN;  px_y = NaN;  src = '';
    txts = dir(fullfile(map_dir, '*.txt'));
    for t = 1:numel(txts)
        txt = fileread(fullfile(map_dir, txts(t).name));
        t1 = regexp(txt, 'step_size_pos1_um\s*:\s*([-\d.eE]+)', 'tokens', 'once');
        t2 = regexp(txt, 'step_size_pos2_um\s*:\s*([-\d.eE]+)', 'tokens', 'once');
        if ~isempty(t1)
            px_x = abs(str2double(t1{1}));
            px_y = px_x;
            if ~isempty(t2), px_y = abs(str2double(t2{1})); end
            src = [txts(t).name ' sidecar'];
            return
        end
    end
    warning('No parseable XRF metadata sidecar in %s; using fallback %.4g µm/px.', map_dir, fallback);
    px_x = fallback;  px_y = fallback;  src = 'fallback';
end

function G = raw_grid(x, y)
% Regular grid spanning the raw map (same as kyanite_ebsd.m): rows run with
% y (down, 'east'/'intoPlane' convention), columns with x.
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

function out = stretch(img, pct)
% Percentile contrast stretch to [0, 1], NaN-safe.
    v = img(isfinite(img));
    if isempty(v), out = zeros(size(img)); return; end
    lim = prctile(v, pct);
    if lim(2) <= lim(1), lim(2) = lim(1) + 1; end
    out = min(max((img - lim(1)) / (lim(2) - lim(1)), 0), 1);
end

function img = fillmissing_zero(img)
    img(isnan(img)) = 0;
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
