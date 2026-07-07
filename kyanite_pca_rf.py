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
#   - Whole-grain CSVs (*_pixel_data.csv): one analysis per grain.
#   - Per-region CSVs (*_region_pixel_data.csv, has a 'Region' column): one
#     analysis per region.
#
# CSV_INPUT may be a single CSV file or a directory; all *_pixel_data.csv
# files found in a directory are processed (this also matches
# *_region_pixel_data.csv, since it shares the same suffix).
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
import shap

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

CSV_INPUT = '/Users/mstein/bin/kyanite/figs/NA-GS-P84-06_pixel_data.csv'   # file or directory
ELEMENTS  = None      # list of CSV column names to include; None = all columns except CL/Region

ANALYSES = ['pca', 'rf', 'shap']   # 'pca', 'rf', 'shap', 'all', or a list of these

# --- Data cleaning, shared by PCA and RF ---
BELOW_DETECTION          = None   # values <= this are treated as below detection limit; None to disable
MAX_BELOW_DETECTION_FRAC = 0.2   # drop an element if more than this fraction of pixels are below detection
LOG_TRANSFORM            = True  # log10-transform element concentrations before PCA/RF
MIN_PIXELS               = 50    # skip a dataset/region if fewer valid pixels remain than this

# --- PCA ---
N_PCS_SCREE       = 10          # number of PCs shown on the scree plot
PC_TO_PLOT        = [1, 2, 3, 4, 5]   # which PC(s) to scatter against CL / show loadings for
LOADING_THRESHOLD = 0.3         # |loading| >= this is highlighted as a significant contributor

# --- Random Forest ---
N_ESTIMATORS         = 200
MIN_SAMPLES_LEAF     = 5
CV_FOLDS             = 10
N_PERMUTATIONS       = 10     # repeats per fold for permutation importance
IMPORTANCE_SIG_RATIO = 1.0    # element flagged "significant" if mean/std of importance exceeds this
MAX_SAMPLES          = 100000  # subsample pixels before RF/permutation importance for speed; None = use all
RANDOM_STATE         = 42

# --- SHAP ---
SHAP_SAMPLES         = 1000   # pixels used to fit the SHAP explainer model; interaction values are
                               # O(n * n_features^2), so keep this well below MAX_SAMPLES
SHAP_INTERACTIONS    = True   # also compute pairwise SHAP interaction values (slower); False = importance only
SHAP_DEPENDENCE_PLOTS = True  # element value vs. its own SHAP value, one panel per element;
                              # colored by top interacting partner if SHAP_INTERACTIONS is True

SAVE_FIG   = True
SAVE_CSV   = True
SHOW_TITLE = True

BLUE  = '#3B9BDD'
ORANG = '#D85B30'

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
    Returns (X_df, y, kept_elements, dropped_elements)."""
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

    valid = X.notna().all(axis=1) & np.isfinite(y)
    return X[valid], y[valid], kept, dropped


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
    colors = [BLUE if abs(v) >= LOADING_THRESHOLD else '0.8' for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(elements)), 4))
    ax.bar(range(len(sorted_vals)), sorted_vals, color=colors)
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
                                    min_samples_leaf=MIN_SAMPLES_LEAF,
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
                                min_samples_leaf=MIN_SAMPLES_LEAF,
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

    cmap = plt.cm.inferno.copy()
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
            sc = ax.scatter(x, y, c=X_vals[:, partner], cmap='viridis', s=8, alpha=0.7, linewidths=0)
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
    X_df, y, kept, dropped = prepare_data(df, elements)

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

# =============================================================================
# RUN
# =============================================================================

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    region_mode = 'Region' in df.columns
    out_dir = csv_path.parent

    exclude = {'CL', 'Region'}
    elements = ELEMENTS or [c for c in df.columns if c not in exclude]
    missing = [e for e in elements if e not in df.columns]
    available = [e for e in elements if e in df.columns]
    if missing:
        print(f'  WARNING: columns not found, skipping: {missing}')

    if region_mode:
        grain_id = csv_path.stem.replace('_region_pixel_data', '')
        for region in df['Region'].drop_duplicates():
            analyze(df[df['Region'] == region], available, f'{grain_id}_{region}', out_dir,
                    csv_path, missing)
    else:
        grain_id = csv_path.stem.replace('_pixel_data', '')
        analyze(df, available, grain_id, out_dir, csv_path, missing)

print('\nDone.')
