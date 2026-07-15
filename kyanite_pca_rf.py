# =============================================================================
# kyanite_pca_rf.py
#
# Multivariate trace-element analysis of CL-EPMA/XRF pixel data: PCA, Random
# Forest regression, and SHAP importance/interactions for CL intensity vs.
# element concentrations.
#
# Replaces the ad hoc SHRIMP-spot PCA/RF scripts in old_scripts/ (ky_pca.m,
# ky_rf.m, ky_rf2.m) with a single script driven off the per-pixel CSVs
# already produced by CL_EPMA_registration.m / CL_region_extraction.m.
#
# For each dataset:
#   PCA  — log-transform + z-score trace elements, run PCA, and plot a scree
#          plot, per-PC loadings, and PC score vs. CL intensity.
#   RF   — k-fold cross-validated Random Forest regression of CL from trace
#          elements; observed-vs-predicted scatter and permutation feature
#          importance (mean +/- std across folds).
#   SHAP — fits a single Random Forest on a subsample and uses TreeSHAP to
#          compute per-feature importance (mean |SHAP value|), pairwise SHAP
#          interaction values (whether elements act on CL jointly, e.g. Cr's
#          effect depends on Fe level, rather than purely additively), and
#          per-element dependence plots (element value vs. its own SHAP
#          value, colored by its top interacting partner) showing the shape
#          of each element's learned effect on CL.
#
# Two input formats are auto-detected by column name, same as
# kyanite_figures.py:
#   - Whole-grain CSVs (*_pixel_data.csv): full PCA/RF/SHAP analysis per grain,
#     as described above.
#   - Per-region CSVs (*_region_pixel_data.csv, has a 'Region' column): PCA
#     only, and only once per grain — a single PCA fit pooled across ALL of
#     the grain's regions together, with every region then projected into
#     that one shared PC space (scree, PC loadings, PC1/PC2-by-region scatter,
#     and a biplot), to test whether regions separate out in PC space. No
#     per-region PCA/RF/SHAP breakdown is produced — a per-region PCA would
#     fit its own independent PC space per region, making scores
#     incomparable across regions, and RF/SHAP importance for CL isn't a
#     meaningful question region-by-region here.
#
# CSV_INPUT may be a single CSV file or a directory; all *_pixel_data.csv
# files found in a directory are processed. Whole-grain and region CSVs both
# live in figs/data/ — the 'Region' column (checked after loading, not the
# filename) decides which code path a given file takes, so pointing
# CSV_INPUT at figs/data/ processes both kinds in one run. Output (PCA/RF/
# SHAP figures, tables, and logs) is saved to WHOLE_GRAIN_OUTPUT_DIR /
# REGION_OUTPUT_DIR below, independent of wherever CSV_INPUT pointed.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f_oneway
from scipy.spatial import ConvexHull
import shap
from kyanite_palette import (BLUE, ORANG, SEQUENTIAL_CMAP, element_colors as _element_colors,
                              region_colors as _shared_region_colors)

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

CSV_INPUT = '/Users/mstein/bin/kyanite/figs/data'   # file or directory
ELEMENTS  = None      # list of CSV column names to include; None = all columns except CL/Region

# Where output (figures, tables, logs) is saved — independent of CSV_INPUT,
# so pointing CSV_INPUT at figs/data/ (where the pixel-data CSVs actually
# live) never dumps results in among the reusable data files. Whole-grain
# CSVs' output goes to WHOLE_GRAIN_OUTPUT_DIR; region CSVs' (has a 'Region'
# column) to REGION_OUTPUT_DIR. Both are created if missing.
WHOLE_GRAIN_OUTPUT_DIR = '/Users/mstein/bin/kyanite/figs/whole_grain'
REGION_OUTPUT_DIR      = '/Users/mstein/bin/kyanite/figs/regions'

ANALYSES = 'all'   # 'pca', 'rf', 'shap', 'all', or a list of these

# --- Data cleaning, shared by PCA and RF ---
BELOW_DETECTION          = None   # values <= this are treated as below detection limit; None to disable
MAX_BELOW_DETECTION_FRAC = 0.2   # drop an element if more than this fraction of pixels are below detection
LOG_TRANSFORM            = True  # log10-transform element concentrations before PCA/RF
MIN_PIXELS               = 50    # skip a dataset/region if fewer valid pixels remain than this

# --- PCA ---
N_PCS_SCREE       = 10          # number of PCs shown on the scree plot
PC_TO_PLOT        = [1, 2, 3, 4]   # which PC(s) to scatter against CL / show loadings for
LOADING_THRESHOLD = 0.3         # |loading| >= this is highlighted as a significant contributor

# --- Region PCA (region CSVs only) ---
# A single PCA fit pooled across ALL of a grain's regions, with every region
# projected into that one shared PC space, so regions can be compared
# directly on the same axes (unlike the per-region analyze() calls above,
# which each fit their own independent PCA). Tests whether hand-drawn regions
# (e.g. core vs. rim) separate out in PC space.
REGION_PCA_PCS   = (1, 2)   # which two PCs to scatter regions on
REGION_PCA_HULLS = True     # draw a convex-hull outline around each region's point cloud

# --- Random Forest ---
# Maximum-accuracy configuration: no subsampling, unbounded trees, a large
# forest, and heavier CV/permutation/SHAP repeats for stable estimates.
# Runtime is intentionally not a consideration here.
N_ESTIMATORS         = 1000
MIN_SAMPLES_LEAF     = 5
MAX_DEPTH            = None   # unbounded; was capped at 10 to keep fit/permutation-importance/
                               # SHAP-interaction cost tractable at large N (same knob as
                               # kyanite_sample_size_convergence.py's MAX_DEPTH)
CV_FOLDS             = 20
N_PERMUTATIONS       = 50     # repeats per fold for permutation importance
IMPORTANCE_SIG_RATIO = 1.0    # element flagged "significant" if mean/std of importance exceeds this
MAX_SAMPLES          = None   # subsample pixels before RF/permutation importance for speed; None = use all
RANDOM_STATE         = 42

# --- SHAP ---
SHAP_SAMPLES         = 5000   # pixels used to fit the SHAP explainer model; interaction values are
                               # O(n * n_features^2), so keep this well below MAX_SAMPLES
SHAP_INTERACTIONS    = True   # also compute pairwise SHAP interaction values (slower); False = importance only
SHAP_DEPENDENCE_PLOTS = True  # element value vs. its own SHAP value, one panel per element;
                              # colored by top interacting partner if SHAP_INTERACTIONS is True

SAVE_FIG   = True
SAVE_CSV   = True
SHOW_TITLE = True

# =============================================================================
# RESOLVE INPUT → list of CSV paths
# =============================================================================

input_path = Path(CSV_INPUT)
if input_path.is_dir():
    csv_files = sorted(input_path.glob('*_pixel_data.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No *_pixel_data.csv files found in {input_path}')
else:
    csv_files = [input_path]

print(f'Processing {len(csv_files)} CSV(s):')
for p in csv_files:
    print(f'  {p.name}')

RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

ALL_ANALYSES = ['pca', 'rf', 'shap']
if ANALYSES == 'all':
    analyses = ALL_ANALYSES
elif isinstance(ANALYSES, (list, tuple)):
    analyses = list(ANALYSES)
else:
    analyses = [ANALYSES]

unknown = [a for a in analyses if a not in ALL_ANALYSES]
if unknown:
    raise ValueError(f"Unknown ANALYSES {unknown}; choose from {ALL_ANALYSES}, 'all', or a list of these.")

rng = np.random.default_rng(RANDOM_STATE)

# =============================================================================
# DATA PREP
# =============================================================================

def prepare_data(df, elements):
    """Drop poorly-detected elements, log-transform, and drop incomplete rows.
    Returns (X_df, y, kept_elements, dropped_elements, valid_mask). valid_mask
    is aligned to df's index, so e.g. df['Region'][valid] recovers the region
    label for each surviving row."""
    X = df[elements].astype(float).copy()
    y = df['CL'].astype(float).values

    if BELOW_DETECTION is not None:
        frac_below = (X <= BELOW_DETECTION).mean(axis=0)
        kept = [e for e in elements if frac_below[e] < MAX_BELOW_DETECTION_FRAC]
        dropped = [e for e in elements if e not in kept]
        X = X[kept]
        X = X.where(X > BELOW_DETECTION)   # remaining below-detection values -> NaN
    else:
        kept, dropped = list(elements), []

    if LOG_TRANSFORM:
        X = np.log10(X)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[valid], y[valid], kept, dropped, valid


def subsample(X, y, max_n):
    if max_n is None or len(y) <= max_n:
        return X, y
    idx = rng.choice(len(y), size=max_n, replace=False)
    return X[idx], y[idx]

# =============================================================================
# PCA
# =============================================================================

def run_pca(X_df):
    X_scaled = StandardScaler().fit_transform(X_df.values)
    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100
    loadings = pca.components_.T          # elements x components
    return scores, explained, loadings


def plot_scree(explained):
    n = min(N_PCS_SCREE, len(explained))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, n + 1), explained[:n], color=BLUE)
    ax.plot(range(1, n + 1), np.cumsum(explained[:n]), 'o-', color=ORANG, lw=1.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return fig


def plot_loadings(loadings, elements, pc):
    vals = loadings[:, pc - 1]
    order = np.argsort(vals)[::-1]
    sorted_vals = vals[order]
    sorted_names = [elements[i] for i in order]
    # Fill = fixed element color (so the same element reads the same color
    # across every PC/grain/script); border = significance, independent of
    # element identity — a black outline if |loading| clears the threshold,
    # no border otherwise.
    fill_colors = _element_colors(sorted_names)
    edge_colors = ['black' if abs(v) >= LOADING_THRESHOLD else 'none' for v in sorted_vals]
    edge_widths = [1.5 if abs(v) >= LOADING_THRESHOLD else 0 for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(elements)), 4))
    ax.bar(range(len(sorted_vals)), sorted_vals, color=[fill_colors[n] for n in sorted_names],
           edgecolor=edge_colors, linewidth=edge_widths)
    ax.axhline(0, color='k', lw=1)
    ax.axhline(LOADING_THRESHOLD, color='k', ls='--', lw=0.5)
    ax.axhline(-LOADING_THRESHOLD, color='k', ls='--', lw=0.5)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel(f'Loading on PC{pc}')
    return fig


def plot_pc_vs_cl(scores, y, pcs):
    n = len(pcs)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, pc in zip(axes, pcs):
        x = scores[:, pc - 1]
        ax.scatter(x, y, s=4, alpha=0.06, color=BLUE, linewidths=0)
        m, b = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 300)
        ax.plot(xfit, m * xfit + b, 'k-', lw=1.5)
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}\nn = {len(x):,}',
                transform=ax.transAxes, va='top', fontsize=9)
        ax.set_xlabel(f'PC{pc} score')
        ax.set_ylabel('CL intensity (norm.)')
        ax.grid(True, alpha=0.25, linewidth=0.5)

    for ax in axes[n:]:
        ax.axis('off')
    return fig


def region_colors(regions):
    """region label -> color, stable across every region-PCA plot for every
    grain (not just one call) — same shared, name-sorted palette as
    kyanite_figures.py's region-highlight figures and CL_region_extraction.m's
    boundary overlay. See kyanite_palette.region_colors()."""
    colors = _shared_region_colors(regions)
    unique_regions = np.array(sorted(colors))
    return unique_regions, colors


def _draw_region_hull(ax, pts, color, alpha=0.12):
    """Convex-hull outline over a region's 2D points; silently skipped if too
    few or degenerate (collinear) to form a hull."""
    if len(pts) < 3:
        return
    try:
        hull = ConvexHull(pts)
    except Exception:
        return
    ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=color, alpha=alpha,
            edgecolor=color, linewidth=1.5)


def plot_region_pca_scatter(scores, regions, pcs, hulls=True):
    """PC_i vs. PC_j scores from a single shared PCA fit, colored by region,
    with an optional convex-hull outline per region so each region's
    footprint in PC space is easy to compare."""
    pc_x, pc_y = pcs
    x = scores[:, pc_x - 1]
    y = scores[:, pc_y - 1]
    unique_regions, colors = region_colors(regions)

    fig, ax = plt.subplots(figsize=(7, 6))
    for r in unique_regions:
        mask = regions == r
        ax.scatter(x[mask], y[mask], s=6, alpha=0.35, color=colors[r], linewidths=0,
                   label=f'{r} (n={mask.sum():,})')
        if hulls:
            _draw_region_hull(ax, np.column_stack([x[mask], y[mask]]), colors[r])

    ax.set_xlabel(f'PC{pc_x} score')
    ax.set_ylabel(f'PC{pc_y} score')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, markerscale=2, loc='best')
    return fig


def plot_region_pca_biplot(scores, loadings, explained, regions, elements, pcs, hulls=True):
    """PC_i vs. PC_j scores colored by region, overlaid with element loading
    vectors (mirrors kyanite_spot_analysis.py's plot_pca_biplot, colored by
    region instead of XANES class). Axes are fixed to the score cloud's own
    extent before drawing arrows, then every loading vector is scaled by one
    shared factor — drawing arrows first would let matplotlib autoscale
    around the (unit-norm, disproportionately long) loadings and shrink the
    score cloud to a sliver."""
    pc_x, pc_y = pcs
    ix, iy = pc_x - 1, pc_y - 1
    x, y = scores[:, ix], scores[:, iy]
    unique_regions, colors = region_colors(regions)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for r in unique_regions:
        mask = regions == r
        ax.scatter(x[mask], y[mask], s=10, alpha=0.35, color=colors[r], linewidths=0,
                   label=f'{r} (n={mask.sum():,})', zorder=2)
        if hulls:
            _draw_region_hull(ax, np.column_stack([x[mask], y[mask]]), colors[r])

    x_max = np.max(np.abs(x)) if len(x) else 1.0
    y_max = np.max(np.abs(y)) if len(y) else 1.0
    xlim, ylim = x_max * 1.15, y_max * 1.15
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    loading_len = np.sqrt(loadings[:, ix] ** 2 + loadings[:, iy] ** 2)
    max_loading_len = loading_len.max() if loading_len.size else 1.0
    scale = 0.85 * min(xlim, ylim) / max_loading_len if max_loading_len > 0 else 1.0

    for i, element in enumerate(elements):
        lx, ly = loadings[i, ix] * scale, loadings[i, iy] * scale
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2), zorder=3)
        ax.text(lx * 1.1, ly * 1.1, element, fontsize=9, color='black',
                ha='center', va='center', zorder=4)

    ax.axhline(0, color='0.7', lw=0.5, zorder=0)
    ax.axvline(0, color='0.7', lw=0.5, zorder=0)
    ax.set_xlabel(f'PC{pc_x} ({explained[ix]:.1f}% var.)')
    ax.set_ylabel(f'PC{pc_y} ({explained[iy]:.1f}% var.)')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return fig


def region_separation_stats(scores, regions, pcs):
    """One-way ANOVA of each PC's scores across regions, plus pairwise
    region-centroid distances in the (pcs) subspace. Returns a list of log
    lines and a DataFrame of pairwise centroid distances."""
    unique_regions = pd.unique(regions)
    idx = [p - 1 for p in pcs]
    lines = ['Region separation in PC space:']

    for pc in pcs:
        groups = [scores[regions == r, pc - 1] for r in unique_regions]
        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            F, p = f_oneway(*groups)
            lines.append(f'  PC{pc}: one-way ANOVA across regions - F={F:.2f}, p={p:.3e}')
        else:
            lines.append(f'  PC{pc}: insufficient data for ANOVA')

    centroids = {r: scores[regions == r][:, idx].mean(axis=0) for r in unique_regions}
    rows = []
    for i, r1 in enumerate(unique_regions):
        for r2 in unique_regions[i + 1:]:
            d = np.linalg.norm(centroids[r1] - centroids[r2])
            lines.append(f'  centroid distance {r1} vs {r2} (PC{list(pcs)}): {d:.3f}')
            rows.append({'region_1': r1, 'region_2': r2, 'centroid_distance': d})

    return lines, pd.DataFrame(rows)

# =============================================================================
# RANDOM FOREST
# =============================================================================

def run_rf_cv(X, y, elements, log_lines):
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred_all = np.full(len(y), np.nan)
    rmse_folds, r2_folds = [], []
    imp_all = np.zeros((CV_FOLDS, len(elements)))

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        rf = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                    min_samples_leaf=MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH,
                                    random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(X[train_idx], y[train_idx])
        y_pred = rf.predict(X[test_idx])
        y_pred_all[test_idx] = y_pred

        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        r2 = r2_score(y[test_idx], y_pred)
        rmse_folds.append(rmse)
        r2_folds.append(r2)

        perm = permutation_importance(rf, X[test_idx], y[test_idx],
                                       n_repeats=N_PERMUTATIONS,
                                       random_state=RANDOM_STATE, n_jobs=1)
        imp_all[i] = perm.importances_mean

        line = f'  Fold {i + 1} - RMSE: {rmse:.4f} | R2: {r2:.4f}'
        print(line); log_lines.append(line)

    return y_pred_all, np.array(rmse_folds), np.array(r2_folds), imp_all


def plot_observed_vs_predicted(y, y_pred, r2_folds, rmse_folds):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, y_pred, s=12, alpha=0.4, color=BLUE, linewidths=0)
    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
    ax.set_xlabel('Observed CL intensity')
    ax.set_ylabel('Predicted CL intensity (out-of-fold)')
    ax.set_title(f'{CV_FOLDS}-fold CV  |  mean R2 = {r2_folds.mean():.2f}  |  mean RMSE = {rmse_folds.mean():.3f}',
                 fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return fig


def plot_importance(mean_imp, std_imp, elements):
    order = np.argsort(mean_imp)[::-1]
    sorted_mean = mean_imp[order]
    sorted_std = std_imp[order]
    sorted_names = [elements[i] for i in order]
    colors = [BLUE if v >= 0 else ORANG for v in sorted_mean]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(elements)), 4.5))
    ax.bar(range(len(sorted_mean)), sorted_mean, color=colors)
    ax.errorbar(range(len(sorted_mean)), sorted_mean, yerr=sorted_std,
                fmt='none', ecolor='k', elinewidth=1.2, capsize=3)
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Mean permutation importance (+/- 1 std)')
    return fig

# =============================================================================
# SHAP
# =============================================================================

def run_shap(X_df, y, elements):
    """Fit a single RF on a subsample and compute TreeSHAP importance and,
    optionally, pairwise interaction values. Returns
    (mean_abs_shap, interaction_matrix, shap_values, X_vals); interaction_matrix
    is None if SHAP_INTERACTIONS is False. shap_values and X_vals (both
    n_used x n_elements) are the raw per-pixel values behind the summaries,
    for dependence plots."""
    X_vals, y_vals = subsample(X_df.values, y, SHAP_SAMPLES)

    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                min_samples_leaf=MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH,
                                random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X_vals, y_vals)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_vals)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    interaction_matrix = None
    if SHAP_INTERACTIONS:
        shap_interactions = explainer.shap_interaction_values(X_vals)
        interaction_matrix = np.abs(shap_interactions).mean(axis=0)

    return mean_abs_shap, interaction_matrix, shap_values, X_vals


def plot_shap_importance(mean_abs_shap, elements):
    order = np.argsort(mean_abs_shap)[::-1]
    sorted_vals = mean_abs_shap[order]
    sorted_names = [elements[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(elements)), 4.5))
    ax.bar(range(len(sorted_vals)), sorted_vals, color=BLUE)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Mean |SHAP value|')
    return fig


def plot_shap_interactions(interaction_matrix, elements):
    # The matrix is symmetric (interaction[i,j] == interaction[j,i]), so the
    # upper triangle is redundant — show only the lower triangle. Diagonal =
    # main effect, already shown in the importance bar chart and typically
    # much larger than any pairwise interaction, so it's excluded from the
    # color scale (grayed out instead) or it would wash out the interactions,
    # which are the whole point of this plot.
    n = len(elements)
    strict_lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    vmax = interaction_matrix[strict_lower].max()

    cmap = plt.colormaps[SEQUENTIAL_CMAP].copy()
    cmap.set_bad('white')   # upper triangle: left blank

    masked = np.ma.masked_array(interaction_matrix, mask=~strict_lower)

    fig, ax = plt.subplots(figsize=(max(5, 0.6 * n), max(5, 0.6 * n)))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(elements, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(elements, fontsize=8)

    for i in range(n):
        for j in range(i + 1):   # lower triangle + diagonal only
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            facecolor='0.85', edgecolor='none', zorder=1))
                color = 'black'
            else:
                color = 'white' if interaction_matrix[i, j] < 0.6 * vmax else 'black'
            ax.text(j, i, f'{interaction_matrix[i, j]:.3f}', ha='center', va='center',
                    color=color, fontsize=7, zorder=2)

    fig.colorbar(im, ax=ax, label='mean |SHAP interaction value| (diagonal, gray = main effect)')
    return fig


def plot_shap_dependence(shap_values, X_vals, elements, interaction_matrix):
    # Each panel: an element's own value vs. its SHAP value, i.e. how the
    # model's learned effect of that element on CL varies across the element's
    # observed range (linear, saturating, threshold-like, non-monotonic...).
    # Colored by whichever other element interacts with it most strongly
    # (per the SHAP interaction matrix), so interaction-driven scatter (e.g.
    # Cr's effect on CL depending on Ti level) is visible as a color gradient.
    n = len(elements)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes[:n]):
        x = X_vals[:, i]
        y = shap_values[:, i]

        if interaction_matrix is not None:
            row = interaction_matrix[i].copy()
            row[i] = -np.inf
            partner = int(np.argmax(row))
            sc = ax.scatter(x, y, c=X_vals[:, partner], cmap=SEQUENTIAL_CMAP, s=8, alpha=0.7, linewidths=0)
            fig.colorbar(sc, ax=ax, label=f'{elements[partner]}' + (' (log10)' if LOG_TRANSFORM else ''))
        else:
            ax.scatter(x, y, s=8, alpha=0.5, color=BLUE, linewidths=0)

        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel(elements[i] + (' (log10)' if LOG_TRANSFORM else ''))
        ax.set_ylabel(f'SHAP value ({elements[i]})')
        ax.grid(True, alpha=0.25, linewidth=0.5)

    for ax in axes[n:]:
        ax.axis('off')
    return fig

# =============================================================================
# PER-DATASET DRIVER
# =============================================================================

def build_log_header(label, csv_path, requested, missing, kept, dropped, n_valid, n_total):
    lines = [
        f'kyanite_pca_rf.py — {label}',
        f'Run date: {RUN_TIMESTAMP}',
        f'Source CSV: {csv_path}',
        f'Analyses run: {analyses}',
        '',
        'Parameters:',
        f'  ELEMENTS (requested): {requested if requested is not None else "all (auto-detected)"}',
        f'  BELOW_DETECTION: {BELOW_DETECTION}',
        f'  MAX_BELOW_DETECTION_FRAC: {MAX_BELOW_DETECTION_FRAC}',
        f'  LOG_TRANSFORM: {LOG_TRANSFORM}',
        f'  MIN_PIXELS: {MIN_PIXELS}',
    ]
    if 'pca' in analyses:
        lines += [
            f'  N_PCS_SCREE: {N_PCS_SCREE}',
            f'  PC_TO_PLOT: {PC_TO_PLOT}',
            f'  LOADING_THRESHOLD: {LOADING_THRESHOLD}',
        ]
    if 'rf' in analyses:
        lines += [
            f'  N_ESTIMATORS: {N_ESTIMATORS}',
            f'  MIN_SAMPLES_LEAF: {MIN_SAMPLES_LEAF}',
            f'  MAX_DEPTH: {MAX_DEPTH}',
            f'  CV_FOLDS: {CV_FOLDS}',
            f'  N_PERMUTATIONS: {N_PERMUTATIONS}',
            f'  IMPORTANCE_SIG_RATIO: {IMPORTANCE_SIG_RATIO}',
            f'  MAX_SAMPLES: {MAX_SAMPLES}',
            f'  RANDOM_STATE: {RANDOM_STATE}',
        ]
    if 'shap' in analyses:
        lines += [
            f'  SHAP_SAMPLES: {SHAP_SAMPLES}',
            f'  SHAP_INTERACTIONS: {SHAP_INTERACTIONS}',
            f'  SHAP_DEPENDENCE_PLOTS: {SHAP_DEPENDENCE_PLOTS}',
        ]
    lines += [
        '',
        f'Columns not found in CSV: {missing}' if missing else 'Columns not found in CSV: none',
        f'Elements used ({len(kept)}): {kept}',
        f'Elements dropped ({len(dropped)}, >{MAX_BELOW_DETECTION_FRAC:.0%} below detection): {dropped}'
        if BELOW_DETECTION is not None else 'Elements dropped: none (BELOW_DETECTION disabled)',
        f'Valid pixels: {n_valid:,} of {n_total:,} total',
        '',
    ]
    return lines


def analyze(df, elements, label, out_dir, csv_path, missing=()):
    print(f'\n--- {label} ({len(df):,} px) ---')
    X_df, y, kept, dropped, _valid = prepare_data(df, elements)

    if dropped:
        print(f'  Dropped (>{MAX_BELOW_DETECTION_FRAC:.0%} below detection): {dropped}')
    if len(X_df) < MIN_PIXELS:
        print(f'  WARNING: only {len(X_df)} valid pixels (< MIN_PIXELS={MIN_PIXELS}), skipping.')
        return

    log_lines = build_log_header(label, csv_path, ELEMENTS, list(missing), kept, dropped,
                                  len(X_df), len(df))

    if 'pca' in analyses:
        print('  Running PCA...')
        scores, explained, loadings = run_pca(X_df)

        if SAVE_CSV:
            var_df = pd.DataFrame({
                'PC': range(1, len(explained) + 1),
                'explained_var_pct': explained,
                'cumulative_pct': np.cumsum(explained),
            })
            var_df.to_csv(out_dir / f'{label}_pca_variance.csv', index=False)

            load_df = pd.DataFrame(loadings, index=kept,
                                    columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
            load_df.to_csv(out_dir / f'{label}_pca_loadings.csv')

        log_lines.append('PCA variance explained (%): ' +
                          ', '.join(f'PC{i+1}={v:.1f}' for i, v in enumerate(explained[:N_PCS_SCREE])))

        if SAVE_FIG:
            fig = plot_scree(explained)
            if SHOW_TITLE:
                fig.suptitle(f'{label} — PCA scree plot')
            fig.tight_layout()
            fig.savefig(out_dir / f'{label}_pca_scree.png', dpi=200, bbox_inches='tight')

            fig = plot_pc_vs_cl(scores, y, PC_TO_PLOT)
            if SHOW_TITLE:
                fig.suptitle(f'{label} — PC scores vs. CL intensity')
            fig.tight_layout()
            fig.savefig(out_dir / f'{label}_pca_scores_vs_CL.png', dpi=200, bbox_inches='tight')

            for pc in PC_TO_PLOT:
                fig = plot_loadings(loadings, kept, pc)
                if SHOW_TITLE:
                    fig.suptitle(f'{label} — PC{pc} loadings')
                fig.tight_layout()
                fig.savefig(out_dir / f'{label}_pca_loadings_PC{pc}.png', dpi=200, bbox_inches='tight')
            plt.close('all')

        print(f'  Saved PCA outputs for {label}')

    if 'rf' in analyses:
        print('  Running Random Forest (this may take a while)...')
        X_rf, y_rf = subsample(X_df.values, y, MAX_SAMPLES)
        if len(y_rf) < len(y):
            msg = f'  Subsampled {len(y_rf):,} of {len(y):,} pixels for RF (MAX_SAMPLES={MAX_SAMPLES})'
            print(msg); log_lines.append(msg.strip())
        log_lines.append('')

        y_pred_all, rmse_folds, r2_folds, imp_all = run_rf_cv(X_rf, y_rf, kept, log_lines)

        mean_imp = imp_all.mean(axis=0)
        std_imp = imp_all.std(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(std_imp > 0, mean_imp / std_imp, np.nan)
        significant = [kept[i] for i in range(len(kept)) if ratio[i] > IMPORTANCE_SIG_RATIO]

        log_lines.append('')
        log_lines.append(f'Mean RMSE: {rmse_folds.mean():.4f} +/- {rmse_folds.std():.4f}')
        log_lines.append(f'Mean R2:   {r2_folds.mean():.4f} +/- {r2_folds.std():.4f}')
        log_lines.append(f'Significant features (mean/std > {IMPORTANCE_SIG_RATIO}): {significant}')
        print(f'  Significant features: {significant}')

        if SAVE_CSV:
            imp_df = pd.DataFrame({
                'element': kept, 'mean_importance': mean_imp, 'std_importance': std_imp,
                'significant': [e in significant for e in kept],
            }).sort_values('mean_importance', ascending=False)
            imp_df.to_csv(out_dir / f'{label}_rf_importance.csv', index=False)

        if SAVE_FIG:
            valid_pred = ~np.isnan(y_pred_all)
            fig = plot_observed_vs_predicted(y_rf[valid_pred], y_pred_all[valid_pred], r2_folds, rmse_folds)
            if SHOW_TITLE:
                fig.suptitle(label, fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f'{label}_rf_observed_vs_predicted.png', dpi=200, bbox_inches='tight')

            fig = plot_importance(mean_imp, std_imp, kept)
            if SHOW_TITLE:
                fig.suptitle(f'{label} — RF permutation importance')
            fig.tight_layout()
            fig.savefig(out_dir / f'{label}_rf_importance.png', dpi=200, bbox_inches='tight')
            plt.close('all')

        print(f'  Saved RF outputs for {label}')

    if 'shap' in analyses:
        print('  Running SHAP...')
        mean_abs_shap, interaction_matrix, shap_values, X_shap = run_shap(X_df, y, kept)
        order = np.argsort(mean_abs_shap)[::-1]

        log_lines.append('')
        log_lines.append(f'SHAP fit on {len(X_shap):,} pixels (SHAP_SAMPLES={SHAP_SAMPLES})')
        log_lines.append('Mean |SHAP value| by element: ' +
                          ', '.join(f'{kept[i]}={mean_abs_shap[i]:.4f}' for i in order))

        if SAVE_CSV:
            shap_df = pd.DataFrame({'element': kept, 'mean_abs_shap': mean_abs_shap}) \
                .sort_values('mean_abs_shap', ascending=False)
            shap_df.to_csv(out_dir / f'{label}_shap_importance.csv', index=False)

        if SAVE_FIG:
            fig = plot_shap_importance(mean_abs_shap, kept)
            if SHOW_TITLE:
                fig.suptitle(f'{label} — SHAP importance')
            fig.tight_layout()
            fig.savefig(out_dir / f'{label}_shap_importance.png', dpi=200, bbox_inches='tight')
            plt.close('all')

        if interaction_matrix is not None:
            off_diag = interaction_matrix.copy()
            np.fill_diagonal(off_diag, -np.inf)
            i, j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
            log_lines.append(f'Strongest pairwise interaction: {kept[i]} x {kept[j]} '
                              f'(mean |interaction| = {interaction_matrix[i, j]:.4f})')
            print(f'  Strongest interaction: {kept[i]} x {kept[j]}')

            if SAVE_CSV:
                inter_df = pd.DataFrame(interaction_matrix, index=kept, columns=kept)
                inter_df.to_csv(out_dir / f'{label}_shap_interactions.csv')

            if SAVE_FIG:
                fig = plot_shap_interactions(interaction_matrix, kept)
                if SHOW_TITLE:
                    fig.suptitle(f'{label} — SHAP interaction values')
                fig.tight_layout()
                fig.savefig(out_dir / f'{label}_shap_interactions.png', dpi=200, bbox_inches='tight')
                plt.close('all')

        if SHAP_DEPENDENCE_PLOTS and SAVE_FIG:
            fig = plot_shap_dependence(shap_values, X_shap, kept, interaction_matrix)
            if SHOW_TITLE:
                fig.suptitle(f'{label} — SHAP dependence (element value vs. own SHAP value)')
            fig.tight_layout()
            fig.savefig(out_dir / f'{label}_shap_dependence.png', dpi=200, bbox_inches='tight')
            plt.close('all')
            log_lines.append('Saved element-vs-SHAP-value dependence plots '
                              f'({"colored by top interacting partner" if interaction_matrix is not None else "uncolored"})')

        print(f'  Saved SHAP outputs for {label}')

    log_file = out_dir / f'{label}_pca_rf_log.txt'
    log_file.write_text('\n'.join(str(l) for l in log_lines) + '\n')
    print(f'  Log saved: {log_file.name}')


def analyze_region_pca(df, elements, grain_id, out_dir, csv_path, missing=()):
    """Region-CSV-only analysis: fit ONE PCA pooled across all of a grain's
    regions, project every region into that shared PC space, and test
    whether regions separate on REGION_PCA_PCS. Independent of the per-region
    analyze() calls, which each fit their own PCA and so aren't comparable
    to one another."""
    label = f'{grain_id}_regions_pca'
    print(f'\n--- {label} ({len(df):,} px, {df["Region"].nunique()} regions) ---')

    X_df, y, kept, dropped, valid = prepare_data(df, elements)
    regions = df['Region'].values[valid.values]

    if dropped:
        print(f'  Dropped (>{MAX_BELOW_DETECTION_FRAC:.0%} below detection): {dropped}')
    if len(X_df) < MIN_PIXELS:
        print(f'  WARNING: only {len(X_df)} valid pixels (< MIN_PIXELS={MIN_PIXELS}), skipping.')
        return

    log_lines = [
        f'kyanite_pca_rf.py — {label}',
        f'Run date: {RUN_TIMESTAMP}',
        f'Source CSV: {csv_path}',
        'Analysis: pooled PCA across regions (one shared PCA fit; each region projected into it)',
        '',
        'Parameters:',
        f'  ELEMENTS (requested): {ELEMENTS if ELEMENTS is not None else "all (auto-detected)"}',
        f'  BELOW_DETECTION: {BELOW_DETECTION}',
        f'  MAX_BELOW_DETECTION_FRAC: {MAX_BELOW_DETECTION_FRAC}',
        f'  LOG_TRANSFORM: {LOG_TRANSFORM}',
        f'  MIN_PIXELS: {MIN_PIXELS}',
        f'  REGION_PCA_PCS: {REGION_PCA_PCS}',
        f'  REGION_PCA_HULLS: {REGION_PCA_HULLS}',
        '',
        f'Columns not found in CSV: {list(missing)}' if missing else 'Columns not found in CSV: none',
        f'Elements used ({len(kept)}): {kept}',
        f'Elements dropped ({len(dropped)}, >{MAX_BELOW_DETECTION_FRAC:.0%} below detection): {dropped}'
        if BELOW_DETECTION is not None else 'Elements dropped: none (BELOW_DETECTION disabled)',
        f'Valid pixels: {len(X_df):,} of {len(df):,} total',
        '',
        f'Regions ({df["Region"].nunique()}): ' +
        ', '.join(f'{r}={int((regions == r).sum()):,}px' for r in pd.unique(regions)),
        '',
    ]

    print('  Running pooled PCA across regions...')
    scores, explained, loadings = run_pca(X_df)

    sep_lines, centroid_df = region_separation_stats(scores, regions, REGION_PCA_PCS)
    log_lines += [''] + sep_lines
    for line in sep_lines:
        print(f'  {line}')

    if SAVE_CSV:
        var_df = pd.DataFrame({
            'PC': range(1, len(explained) + 1),
            'explained_var_pct': explained,
            'cumulative_pct': np.cumsum(explained),
        })
        var_df.to_csv(out_dir / f'{label}_pca_variance.csv', index=False)

        load_df = pd.DataFrame(loadings, index=kept,
                                columns=[f'PC{i+1}' for i in range(loadings.shape[1])])
        load_df.to_csv(out_dir / f'{label}_pca_loadings.csv')

        scores_df = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(scores.shape[1])])
        scores_df.insert(0, 'Region', regions)
        scores_df.to_csv(out_dir / f'{label}_scores.csv', index=False)

        centroid_df.to_csv(out_dir / f'{label}_centroid_distances.csv', index=False)

    if SAVE_FIG:
        fig = plot_scree(explained)
        if SHOW_TITLE:
            fig.suptitle(f'{label} — PCA scree plot')
        fig.tight_layout()
        fig.savefig(out_dir / f'{label}_pca_scree.png', dpi=200, bbox_inches='tight')

        for pc in REGION_PCA_PCS:
            fig = plot_loadings(loadings, kept, pc)
            if SHOW_TITLE:
                fig.suptitle(f'{label} — PC{pc} loadings')
            fig.tight_layout()
            fig.savefig(out_dir / f'{label}_pca_loadings_PC{pc}.png', dpi=200, bbox_inches='tight')

        fig = plot_region_pca_scatter(scores, regions, REGION_PCA_PCS, hulls=REGION_PCA_HULLS)
        if SHOW_TITLE:
            fig.suptitle(f'{label} — PC{REGION_PCA_PCS[0]} vs PC{REGION_PCA_PCS[1]} by region')
        fig.tight_layout()
        fig.savefig(out_dir / f'{label}_pc{REGION_PCA_PCS[0]}_pc{REGION_PCA_PCS[1]}_scatter.png',
                    dpi=200, bbox_inches='tight')

        fig = plot_region_pca_biplot(scores, loadings, explained, regions, kept,
                                      REGION_PCA_PCS, hulls=REGION_PCA_HULLS)
        if SHOW_TITLE:
            fig.suptitle(f'{label} — PCA biplot (n={len(X_df):,})')
        fig.tight_layout()
        fig.savefig(out_dir / f'{label}_pca_biplot.png', dpi=200, bbox_inches='tight')

        plt.close('all')
        log_lines.append('Saved scree, PC loadings, region scatter, and biplot figures')

    print(f'  Saved region PCA outputs for {label}')

    log_file = out_dir / f'{label}_log.txt'
    log_file.write_text('\n'.join(str(l) for l in log_lines) + '\n')
    print(f'  Log saved: {log_file.name}')

# =============================================================================
# RUN
# =============================================================================

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    region_mode = 'Region' in df.columns
    out_dir = Path(REGION_OUTPUT_DIR) if region_mode else Path(WHOLE_GRAIN_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = {'CL', 'Region', 'DomainID'}
    elements = ELEMENTS or [c for c in df.columns if c not in exclude]
    missing = [e for e in elements if e not in df.columns]
    available = [e for e in elements if e in df.columns]
    if missing:
        print(f'  WARNING: columns not found, skipping: {missing}')

    if region_mode:
        grain_id = csv_path.stem.replace('_region_pixel_data', '')
        if 'pca' not in analyses:
            print(f'  Region CSV {csv_path.name}: only "pca" analysis applies to regions, '
                  f'skipping (ANALYSES={ANALYSES}).')
        elif df['Region'].nunique() < 2:
            print(f'  WARNING: {grain_id} has fewer than 2 regions, skipping region PCA.')
        else:
            analyze_region_pca(df, available, grain_id, out_dir, csv_path, missing)
    else:
        grain_id = csv_path.stem.replace('_pixel_data', '')
        analyze(df, available, grain_id, out_dir, csv_path, missing)

print('\nDone.')
