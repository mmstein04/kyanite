# =============================================================================
# kyanite_rf_shap_plots.py
#
# Figure generation for kyanite_rf_shap.py's Random Forest / SHAP CSV outputs
# (<label>_rf_predictions.csv, _rf_importance.csv, _shap_importance.csv,
# _shap_values.csv, _shap_interactions.csv). Reads those CSVs back and plots
# them — no model fitting happens here, so restyling a figure or trying a
# different PLOTS/FIG_DPI/SHOW_TITLE combination never requires retraining.
#
# A grain's figures are only produced from whichever of its CSVs are present
# (RF and SHAP were independently toggleable in kyanite_rf_shap.py's
# ANALYSES, so a grain may have only rf_* files, only shap_* files, or both).
# Grains missing a plot's required CSV(s) are skipped with a warning, not
# errored on.
#
# Per-fold RMSE/R2 (for the observed-vs-predicted title) are recomputed
# directly from the per-pixel predictions CSV rather than stored separately,
# since they're cheap to derive and keeping them out of the CSV avoids two
# sources of truth.
#
# RF figures (observed-vs-predicted, permutation importance) go to
# RF_OUTPUT_DIR (default figs/rf/); SHAP figures (importance, interactions,
# dependence) go to SHAP_OUTPUT_DIR (default figs/shap/) — kept apart since
# they're different analyses, even though both are read from the same
# figs/data/ CSVs.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from kyanite_palette import BLUE, IMPORTANCE_CMAP, SEQUENTIAL_CMAP, element_colors

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parent

# Directory kyanite_rf_shap.py wrote its CSVs to.
CSV_INPUT = _REPO_ROOT / 'figs' / 'data'

# Where figures are saved — independent of CSV_INPUT. Split by analysis:
# RF figures to RF_OUTPUT_DIR, SHAP figures to SHAP_OUTPUT_DIR.
RF_OUTPUT_DIR   = _REPO_ROOT / 'figs' / 'rf'
SHAP_OUTPUT_DIR = _REPO_ROOT / 'figs' / 'shap'

GRAIN_FILTER = None   # list of grain_ids to plot, or None for every grain found in CSV_INPUT

PLOTS = 'all'   # 'observed_vs_predicted', 'importance', 'shap_importance',
                # 'shap_interactions', 'shap_dependence', 'summary', 'all', or a list of these

# 'summary' is not per-grain: one figure pooling every grain (that survives
# GRAIN_FILTER and has both rf_predictions and rf_importance CSVs), one row per
# grain — CV R2 (mean +/- std across folds), permutation importance (drop in
# R2, already comparable across grains), and mean |SHAP| as a percentage of
# that grain's total (raw |SHAP| is in each grain's own CL units, so it's
# normalized per grain). Saved to SUMMARY_OUTPUT_DIR (figs/summary/, shared
# with every other all-grains summary figure) as
# <ALL_GRAINS_LABEL>_rf_summary.png.
SUMMARY_OUTPUT_DIR = _REPO_ROOT / 'figs' / 'summary'
ALL_GRAINS_LABEL = 'all_grains'
# Column order for the summary heatmaps (matches kyanite_figures.py's summary
# r heatmap); elements found in the CSVs but not listed are appended, sorted.
SUMMARY_ELEMENT_ORDER = ['Cr_Ka', 'V_Ka', 'Fe_Ka', 'Mn_Ka', 'Ti_Ka']

# LOG_TRANSFORM must match whatever kyanite_rf_shap.py used for this data —
# it only affects axis labeling here (element values were already
# log10-transformed, or not, before this script ever sees them).
LOG_TRANSFORM = True

FIG_DPI    = 300
SHOW_TITLE = True

ALL_PLOTS = ['observed_vs_predicted', 'importance', 'shap_importance',
             'shap_interactions', 'shap_dependence', 'summary']
if PLOTS == 'all':
    plots = ALL_PLOTS
elif isinstance(PLOTS, (list, tuple)):
    plots = list(PLOTS)
else:
    plots = [PLOTS]

unknown = [p for p in plots if p not in ALL_PLOTS]
if unknown:
    raise ValueError(f"Unknown PLOTS {unknown}; choose from {ALL_PLOTS}, 'all', or a list of these.")

# =============================================================================
# DISCOVER GRAINS
# =============================================================================

def discover_grains(csv_dir):
    rf_labels = {p.name[:-len('_rf_predictions.csv')] for p in csv_dir.glob('*_rf_predictions.csv')}
    shap_labels = {p.name[:-len('_shap_values.csv')] for p in csv_dir.glob('*_shap_values.csv')}
    return sorted(rf_labels | shap_labels)


csv_dir = Path(CSV_INPUT)
labels = discover_grains(csv_dir)
if GRAIN_FILTER is not None:
    missing_grains = [g for g in GRAIN_FILTER if g not in labels]
    if missing_grains:
        print(f'WARNING: GRAIN_FILTER entries not found in {csv_dir}: {missing_grains}')
    labels = [g for g in GRAIN_FILTER if g in labels]
if not labels:
    raise FileNotFoundError(f'No *_rf_predictions.csv / *_shap_values.csv files found in {csv_dir}')

print(f'Plotting {len(labels)} grain(s): {labels}')

rf_out_dir = Path(RF_OUTPUT_DIR)
rf_out_dir.mkdir(parents=True, exist_ok=True)
shap_out_dir = Path(SHAP_OUTPUT_DIR)
shap_out_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOADERS
# =============================================================================

def load_shap_values(path):
    """Reshape the wide <element>_value / <element>_shap CSV back into
    (X_vals, shap_values, elements); element order follows the CSV's own
    column order (i.e. whatever order kyanite_rf_shap.py used)."""
    df = pd.read_csv(path)
    elements = [c[:-len('_value')] for c in df.columns if c.endswith('_value')]
    X_vals = df[[f'{e}_value' for e in elements]].values
    shap_values = df[[f'{e}_shap' for e in elements]].values
    return X_vals, shap_values, elements


def compute_fold_metrics(pred_df):
    rmse_folds, r2_folds = [], []
    for f in sorted(pred_df['fold'].unique()):
        sub = pred_df[pred_df['fold'] == f]
        rmse_folds.append(np.sqrt(mean_squared_error(sub['observed_CL'], sub['predicted_CL'])))
        r2_folds.append(r2_score(sub['observed_CL'], sub['predicted_CL']))
    return np.array(rmse_folds), np.array(r2_folds)

# =============================================================================
# PLOTS
# =============================================================================

def plot_observed_vs_predicted(y, y_pred, r2_folds, rmse_folds, n_folds):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, y_pred, s=12, alpha=0.4, color=BLUE, linewidths=0)
    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
    ax.set_xlabel('Observed CL intensity')
    ax.set_ylabel('Predicted CL intensity (out-of-fold)')
    ax.set_title(f'{n_folds}-fold CV  |  mean R2 = {r2_folds.mean():.2f}  |  mean RMSE = {rmse_folds.mean():.3f}',
                 fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return fig


def plot_importance(mean_imp, std_imp, elements, significant):
    """Bar fill = element's fixed color (kyanite_palette), so the same
    element reads the same color across every figure/script; bar border
    separately encodes significance — a black outline if the element is
    flagged significant, no border otherwise — same two-channel convention
    as kyanite_pca.py's plot_loadings / kyanite_spot_analysis.py's
    plot_pca_loadings."""
    order = np.argsort(mean_imp)[::-1]
    sorted_mean = mean_imp[order]
    sorted_std = std_imp[order]
    sorted_names = [elements[i] for i in order]
    sorted_sig = [significant[i] for i in order]
    fill_colors = element_colors(sorted_names)
    edge_colors = ['black' if sig else 'none' for sig in sorted_sig]
    edge_widths = [1.5 if sig else 0 for sig in sorted_sig]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(elements)), 4.5))
    ax.bar(range(len(sorted_mean)), sorted_mean, color=[fill_colors[n] for n in sorted_names],
           edgecolor=edge_colors, linewidth=edge_widths)
    ax.errorbar(range(len(sorted_mean)), sorted_mean, yerr=sorted_std,
                fmt='none', ecolor='k', elinewidth=1.2, capsize=3)
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Mean permutation importance (+/- 1 std)')
    return fig


def plot_shap_importance(mean_abs_shap, elements):
    """Bar fill = element's fixed color (kyanite_palette) — same convention
    as plot_importance, minus the significance border since SHAP importance
    (mean |SHAP value|) carries no significance flag."""
    order = np.argsort(mean_abs_shap)[::-1]
    sorted_vals = mean_abs_shap[order]
    sorted_names = [elements[i] for i in order]
    fill_colors = element_colors(sorted_names)

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(elements)), 4.5))
    ax.bar(range(len(sorted_vals)), sorted_vals, color=[fill_colors[n] for n in sorted_names])
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


def build_summary_tables(csv_dir, labels):
    """One row per grain (sorted): CV R2 mean/std across folds, plus
    grains x elements tables of permutation importance and SHAP share (% of
    that grain's total mean |SHAP|). Grains missing rf_predictions or
    rf_importance are dropped with a warning; a missing shap_importance just
    leaves that grain's SHAP row NaN."""
    r2_rows, perm_rows, shap_rows = {}, {}, {}
    for label in sorted(labels):
        pred_path = csv_dir / f'{label}_rf_predictions.csv'
        imp_path = csv_dir / f'{label}_rf_importance.csv'
        shap_path = csv_dir / f'{label}_shap_importance.csv'
        if not (pred_path.exists() and imp_path.exists()):
            print(f'  WARNING: {label} lacks rf_predictions/rf_importance CSV; left out of summary')
            continue
        pred_df = pd.read_csv(pred_path).dropna(subset=['predicted_CL'])
        _, r2_folds = compute_fold_metrics(pred_df)
        r2_rows[label] = {'mean': r2_folds.mean(), 'std': r2_folds.std()}
        perm_rows[label] = pd.read_csv(imp_path).set_index('element')['mean_importance']
        if shap_path.exists():
            s = pd.read_csv(shap_path).set_index('element')['mean_abs_shap']
            shap_rows[label] = 100 * s / s.sum()
    r2_df = pd.DataFrame.from_dict(r2_rows, orient='index')
    perm_df = pd.DataFrame(perm_rows).T
    shap_df = pd.DataFrame(shap_rows).T.reindex(perm_df.index)
    present = set(perm_df.columns) | set(shap_df.columns)
    elements = ([e for e in SUMMARY_ELEMENT_ORDER if e in present]
                + sorted(present - set(SUMMARY_ELEMENT_ORDER)))
    return r2_df, perm_df.reindex(columns=elements), shap_df.reindex(columns=elements)


def _annotated_heatmap(ax, values, fmt, vmax, cbar_label, fig):
    cmap = plt.colormaps[IMPORTANCE_CMAP].copy()
    cmap.set_bad('0.85')
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=11, color='gray')
            else:
                # same cutoff as the r heatmap's |r| >= 0.6 on a 0..1 scale
                color = 'white' if v >= 0.6 * vmax else 'black'
                ax.text(j, i, format(v, fmt), ha='center', va='center', fontsize=11, color=color)
    # White cell borders, matching kyanite_figures.py's summary r heatmap
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, location='bottom', pad=0.08, shrink=0.8, aspect=30)
    cbar.set_label(cbar_label)


def plot_summary(r2_df, perm_df, shap_df):
    """Three panels sharing one row per grain: CV R2 bars (mean +/- std
    across folds), permutation importance heatmap, SHAP share heatmap.
    No title — what the figure shows goes in the caption."""
    grains = list(r2_df.index)
    elements = list(perm_df.columns)
    n_g, n_e = len(grains), len(elements)
    fig, axes = plt.subplots(1, 3, figsize=(2.2 + 2 * n_e * 0.9 + 2.5, n_g * 0.7 + 2),
                             gridspec_kw={'width_ratios': [2.0, n_e, n_e], 'wspace': 0.08},
                             sharey=True)
    ax_r2, ax_perm, ax_shap = axes
    y = np.arange(n_g)

    ax_r2.barh(y, r2_df['mean'], xerr=r2_df['std'], color=BLUE, height=0.7,
               error_kw={'ecolor': 'k', 'elinewidth': 1, 'capsize': 2})
    for yi, m, s in zip(y, r2_df['mean'], r2_df['std']):
        ax_r2.text(max(m + s, 0) + 0.03, yi, f'{m:.2f}', va='center', fontsize=11)
    ax_r2.set_xlim(min(0, (r2_df['mean'] - r2_df['std']).min() - 0.05), 1.15)
    ax_r2.set_xlabel('CV R²')
    ax_r2.set_ylabel('grain')
    ax_r2.set_yticks(y)
    ax_r2.set_yticklabels(grains)
    ax_r2.set_ylim(n_g - 0.5, -0.5)
    ax_r2.spines[['top', 'right']].set_visible(False)
    # Empty colorbar-sized slot below the bars so all three panels' data
    # areas stay the same height as the heatmaps above their colorbars
    slot = fig.colorbar(plt.cm.ScalarMappable(), ax=ax_r2, location='bottom',
                        pad=0.08, shrink=0.8, aspect=30)
    slot.ax.set_visible(False)

    tick_labels = [e.split('_')[0] for e in elements]
    perm_vals = perm_df.values.astype(float)
    _annotated_heatmap(ax_perm, perm_vals, '.2f', np.nanmax(perm_vals),
                       'Permutation importance (ΔR²)', fig)
    shap_vals = shap_df.values.astype(float)
    _annotated_heatmap(ax_shap, shap_vals, '.0f', np.nanmax(shap_vals),
                       'Mean |SHAP| (% of grain total)', fig)
    for ax in (ax_perm, ax_shap):
        ax.set_xticks(range(n_e))
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('element')
        ax.tick_params(axis='y', left=False)
    return fig


def plot_shap_dependence(shap_values, X_vals, elements, interaction_df):
    # Each panel: an element's own value vs. its SHAP value, i.e. how the
    # model's learned effect of that element on CL varies across the
    # element's observed range. Colored by whichever other element interacts
    # with it most strongly (per the SHAP interaction matrix), so
    # interaction-driven scatter (e.g. Cr's effect on CL depending on Ti
    # level) is visible as a color gradient. interaction_df's own index/
    # columns are looked up by element name (not position), so a partial or
    # differently-ordered interactions CSV still matches up correctly.
    n = len(elements)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes[:n]):
        x = X_vals[:, i]
        y = shap_values[:, i]
        element = elements[i]

        partner = None
        if interaction_df is not None and element in interaction_df.index:
            row = interaction_df.loc[element].drop(labels=element, errors='ignore')
            row = row[[e for e in elements if e in row.index]]
            if len(row):
                partner = row.idxmax()

        if partner is not None:
            partner_idx = elements.index(partner)
            sc = ax.scatter(x, y, c=X_vals[:, partner_idx], cmap=SEQUENTIAL_CMAP, s=8, alpha=0.7, linewidths=0)
            fig.colorbar(sc, ax=ax, label=partner + (' (log10)' if LOG_TRANSFORM else ''))
        else:
            ax.scatter(x, y, s=8, alpha=0.5, color=BLUE, linewidths=0)

        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel(element + (' (log10)' if LOG_TRANSFORM else ''))
        ax.set_ylabel(f'SHAP value ({element})')
        ax.grid(True, alpha=0.25, linewidth=0.5)

    for ax in axes[n:]:
        ax.axis('off')
    return fig

# =============================================================================
# RUN
# =============================================================================

for label in labels:
    print(f'\n--- {label} ---')
    rf_pred_path = csv_dir / f'{label}_rf_predictions.csv'
    rf_imp_path = csv_dir / f'{label}_rf_importance.csv'
    shap_imp_path = csv_dir / f'{label}_shap_importance.csv'
    shap_val_path = csv_dir / f'{label}_shap_values.csv'
    shap_inter_path = csv_dir / f'{label}_shap_interactions.csv'

    interaction_df = pd.read_csv(shap_inter_path, index_col=0) if shap_inter_path.exists() else None

    if 'observed_vs_predicted' in plots:
        if rf_pred_path.exists():
            pred_df = pd.read_csv(rf_pred_path).dropna(subset=['predicted_CL'])
            rmse_folds, r2_folds = compute_fold_metrics(pred_df)
            fig = plot_observed_vs_predicted(pred_df['observed_CL'].values, pred_df['predicted_CL'].values,
                                              r2_folds, rmse_folds, pred_df['fold'].nunique())
            if SHOW_TITLE:
                fig.suptitle(label, fontsize=11)
            fig.tight_layout()
            fig.savefig(rf_out_dir / f'{label}_rf_observed_vs_predicted.png', dpi=FIG_DPI, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved {label}_rf_observed_vs_predicted.png')
        else:
            print(f'  WARNING: {rf_pred_path.name} not found, skipping observed_vs_predicted.')

    if 'importance' in plots:
        if rf_imp_path.exists():
            imp_df = pd.read_csv(rf_imp_path)
            significant = (imp_df['significant'] if 'significant' in imp_df.columns
                           else pd.Series(False, index=imp_df.index)).tolist()
            fig = plot_importance(imp_df['mean_importance'].values, imp_df['std_importance'].values,
                                   imp_df['element'].tolist(), significant)
            if SHOW_TITLE:
                fig.suptitle(f'{label} — RF permutation importance')
            fig.tight_layout()
            fig.savefig(rf_out_dir / f'{label}_rf_importance.png', dpi=FIG_DPI, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved {label}_rf_importance.png')
        else:
            print(f'  WARNING: {rf_imp_path.name} not found, skipping importance.')

    if 'shap_importance' in plots:
        if shap_imp_path.exists():
            shap_imp_df = pd.read_csv(shap_imp_path)
            fig = plot_shap_importance(shap_imp_df['mean_abs_shap'].values, shap_imp_df['element'].tolist())
            if SHOW_TITLE:
                fig.suptitle(f'{label} — SHAP importance')
            fig.tight_layout()
            fig.savefig(shap_out_dir / f'{label}_shap_importance.png', dpi=FIG_DPI, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved {label}_shap_importance.png')
        else:
            print(f'  WARNING: {shap_imp_path.name} not found, skipping shap_importance.')

    if 'shap_interactions' in plots:
        if interaction_df is not None:
            fig = plot_shap_interactions(interaction_df.values, list(interaction_df.columns))
            if SHOW_TITLE:
                fig.suptitle(f'{label} — SHAP interaction values')
            fig.tight_layout()
            fig.savefig(shap_out_dir / f'{label}_shap_interactions.png', dpi=FIG_DPI, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved {label}_shap_interactions.png')
        else:
            print(f'  WARNING: {shap_inter_path.name} not found, skipping shap_interactions.')

    if 'shap_dependence' in plots:
        if shap_val_path.exists():
            X_vals, shap_values, elements = load_shap_values(shap_val_path)
            fig = plot_shap_dependence(shap_values, X_vals, elements, interaction_df)
            if SHOW_TITLE:
                fig.suptitle(f'{label} — SHAP dependence (element value vs. own SHAP value)')
            fig.tight_layout()
            fig.savefig(shap_out_dir / f'{label}_shap_dependence.png', dpi=FIG_DPI, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved {label}_shap_dependence.png '
                  f'({"colored by top interacting partner" if interaction_df is not None else "uncolored"})')
        else:
            print(f'  WARNING: {shap_val_path.name} not found, skipping shap_dependence.')

if 'summary' in plots:
    print(f'\n--- {ALL_GRAINS_LABEL} summary ---')
    r2_df, perm_df, shap_df = build_summary_tables(csv_dir, labels)
    if len(r2_df) < 2:
        print(f'  WARNING: summary needs >=2 grains with RF CSVs, found {len(r2_df)}; skipping')
    else:
        print('  CV R2 (mean, std across folds):')
        print('    ' + r2_df.round(3).to_string().replace('\n', '\n    '))
        print('  Permutation importance (drop in R2):')
        print('    ' + perm_df.round(3).to_string().replace('\n', '\n    '))
        print('  Mean |SHAP|, % of grain total:')
        print('    ' + shap_df.round(1).to_string().replace('\n', '\n    '))
        fig = plot_summary(r2_df, perm_df, shap_df)
        summary_out_dir = Path(SUMMARY_OUTPUT_DIR)
        summary_out_dir.mkdir(parents=True, exist_ok=True)
        out = summary_out_dir / f'{ALL_GRAINS_LABEL}_rf_summary.png'
        fig.savefig(out, dpi=FIG_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out.name}')

print('\nDone.')
