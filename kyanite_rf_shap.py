# =============================================================================
# kyanite_rf_shap.py
#
# Cross-validated Random Forest regression and TreeSHAP analysis of CL
# intensity vs. trace-element concentrations, from whole-grain per-pixel CSVs
# already produced by CL_EPMA_registration.m / CL_region_extraction.m.
#
# Fits models and exports CSVs only — no figures. kyanite_rf_shap_plots.py
# reads those CSVs back and renders the figures, so a model never has to be
# refit just to regenerate or restyle a plot. (PCA of the same CSVs lives in
# the separate kyanite_pca.py, unchanged by this split.)
#
#   RF   — k-fold cross-validated Random Forest regression of CL from trace
#          elements; out-of-fold predictions and permutation feature
#          importance (mean +/- std across folds).
#   SHAP — fits a single Random Forest on a subsample and uses TreeSHAP to
#          compute per-feature importance (mean |SHAP value|) and, optionally,
#          pairwise SHAP interaction values (whether elements act on CL
#          jointly, e.g. Cr's effect depends on Fe level, rather than purely
#          additively).
#
# RF/SHAP only apply to whole-grain CSVs (*_pixel_data.csv) — region CSVs
# (*_region_pixel_data.csv, has a 'Region' column) are skipped with a
# warning; region-level analysis is PCA-only (kyanite_pca.py).
#
# CSV_INPUT may be a single CSV file or a directory; all *_pixel_data.csv
# files found in a directory are processed (region CSVs among them are
# skipped, not errored on). CSVs are saved to OUTPUT_DIR below (default
# figs/data/, alongside the pixel-data CSV this script reads — these outputs
# are themselves reusable data, read back by kyanite_rf_shap_plots.py); the
# run log goes to DIAGNOSTICS_DIR (default figs/diagnostics/, matching every
# other analysis/registration log in this project) — both independent of
# wherever CSV_INPUT pointed.
# =============================================================================

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import shap

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

# Anchored to this script's own location (not a hardcoded machine-specific
# path) so the same file runs unmodified on any machine/cluster this repo is
# checked out on.
_REPO_ROOT = Path(__file__).resolve().parent

CSV_INPUT = _REPO_ROOT / 'figs' / 'data'   # file or directory
ELEMENTS  = None      # list of CSV column names to include; None = all columns except CL

# Where output (CSVs, log) is saved — independent of CSV_INPUT. This
# script's CSVs (predictions, importance, SHAP values/interactions) are
# reusable data in their own right — kyanite_rf_shap_plots.py reads them
# back to render figures — so they live in figs/data/ alongside the
# pixel-data CSVs, same as every other reusable data product in this project.
OUTPUT_DIR = _REPO_ROOT / 'figs' / 'data'

# Run log (not-for-publishing run metadata) — same home as every other
# analysis/registration log in this project.
DIAGNOSTICS_DIR = _REPO_ROOT / 'figs' / 'diagnostics'

ANALYSES = 'all'   # 'rf', 'shap', 'all', or a list of these

# --- Data cleaning, shared by RF and SHAP ---
BELOW_DETECTION          = None   # values <= this are treated as below detection limit; None to disable
MAX_BELOW_DETECTION_FRAC = 0.2   # drop an element if more than this fraction of pixels are below detection
LOG_TRANSFORM            = True  # log10-transform element concentrations before RF/SHAP
MIN_PIXELS               = 50    # skip a dataset if fewer valid pixels remain than this

# --- Random Forest ---
# Maximum-accuracy configuration: no subsampling, unbounded trees, a large
# forest, and heavier CV/permutation repeats for stable estimates. Runtime is
# intentionally not a consideration here.
N_ESTIMATORS         = 1000
MIN_SAMPLES_LEAF     = 5
MAX_DEPTH            = None   # unbounded; was capped at 10 to keep fit/permutation-importance/
                               # SHAP-interaction cost tractable at large N (same knob as
                               # kyanite_sample_size_convergence.py's MAX_DEPTH)
CV_FOLDS             = 10
N_PERMUTATIONS       = 50     # repeats per fold for permutation importance
IMPORTANCE_SIG_RATIO = 1.0    # element flagged "significant" if mean/std of importance exceeds this
MAX_SAMPLES          = None   # subsample pixels before RF/permutation importance for speed; None = use all
RANDOM_STATE         = 42
N_JOBS               = 16    # parallel worker processes for RF fitting and permutation importance

# --- SHAP ---
SHAP_SAMPLES         = 5000   # pixels used to fit the SHAP explainer model; interaction values are
                               # O(n * n_features^2), so keep this well below MAX_SAMPLES
SHAP_INTERACTIONS    = True   # also compute pairwise SHAP interaction values (slower); False = importance only

# =============================================================================
# RESOLVE INPUT → list of CSV paths
# =============================================================================

input_path = Path(CSV_INPUT)
if input_path.is_dir():
    # CL_local_regression_map.m's output also ends in '_pixel_data.csv' but is
    # a different data product (per-window slope/R stats, no 'CL' column) —
    # excluded here rather than processed and skipped downstream.
    csv_files = sorted(p for p in input_path.glob('*_pixel_data.csv')
                        if not p.name.endswith('_local_regression_pixel_data.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No *_pixel_data.csv files found in {input_path}')
else:
    csv_files = [input_path]

print(f'Processing {len(csv_files)} CSV(s):')
for p in csv_files:
    print(f'  {p.name}')

RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

ALL_ANALYSES = ['rf', 'shap']
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
    Returns (X_df, y, kept_elements, dropped_elements, valid_mask)."""
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
    """Returns (X, y, idx); idx are the positions kept out of the input
    X/y, so callers can map rows back to the original CSV (e.g. via
    X_df.index.values[idx])."""
    if max_n is None or len(y) <= max_n:
        return X, y, np.arange(len(y))
    idx = rng.choice(len(y), size=max_n, replace=False)
    return X[idx], y[idx], idx

# =============================================================================
# RANDOM FOREST
# =============================================================================

def run_rf_cv(X, y, elements, log_lines):
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred_all = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    rmse_folds, r2_folds = [], []
    imp_all = np.zeros((CV_FOLDS, len(elements)))

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        rf = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                    min_samples_leaf=MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH,
                                    random_state=RANDOM_STATE, n_jobs=N_JOBS)
        rf.fit(X[train_idx], y[train_idx])
        y_pred = rf.predict(X[test_idx])
        y_pred_all[test_idx] = y_pred
        fold_id[test_idx] = i

        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        r2 = r2_score(y[test_idx], y_pred)
        rmse_folds.append(rmse)
        r2_folds.append(r2)

        perm = permutation_importance(rf, X[test_idx], y[test_idx],
                                       n_repeats=N_PERMUTATIONS,
                                       random_state=RANDOM_STATE, n_jobs=N_JOBS)
        imp_all[i] = perm.importances_mean

        line = f'  Fold {i + 1} - RMSE: {rmse:.4f} | R2: {r2:.4f}'
        print(line); log_lines.append(line)

    return y_pred_all, np.array(rmse_folds), np.array(r2_folds), imp_all, fold_id

# =============================================================================
# SHAP
# =============================================================================

def run_shap(X_df, y, elements):
    """Fit a single RF on a subsample and compute TreeSHAP importance and,
    optionally, pairwise interaction values. Returns
    (mean_abs_shap, interaction_matrix, shap_values, X_vals, idx); interaction_matrix
    is None if SHAP_INTERACTIONS is False. shap_values and X_vals (both
    n_used x n_elements) are the raw per-pixel values behind the summaries.
    idx are the positions of the subsample within X_df/y, so callers can map
    rows back to the original CSV via X_df.index.values[idx]."""
    X_vals, y_vals, idx = subsample(X_df.values, y, SHAP_SAMPLES)

    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                min_samples_leaf=MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH,
                                random_state=RANDOM_STATE, n_jobs=N_JOBS)
    rf.fit(X_vals, y_vals)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_vals)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    interaction_matrix = None
    if SHAP_INTERACTIONS:
        shap_interactions = explainer.shap_interaction_values(X_vals)
        interaction_matrix = np.abs(shap_interactions).mean(axis=0)

    return mean_abs_shap, interaction_matrix, shap_values, X_vals, idx

# =============================================================================
# PER-DATASET DRIVER
# =============================================================================

def build_log_header(label, csv_path, requested, missing, kept, dropped, n_valid, n_total):
    lines = [
        f'kyanite_rf_shap.py — {label}',
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
            f'  N_JOBS: {N_JOBS}',
        ]
    if 'shap' in analyses:
        lines += [
            f'  SHAP_SAMPLES: {SHAP_SAMPLES}',
            f'  SHAP_INTERACTIONS: {SHAP_INTERACTIONS}',
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


def process_grain(df, elements, label, out_dir, diagnostics_dir, csv_path, missing=()):
    print(f'\n--- {label} ({len(df):,} px) ---')
    X_df, y, kept, dropped, _valid = prepare_data(df, elements)

    if dropped:
        print(f'  Dropped (>{MAX_BELOW_DETECTION_FRAC:.0%} below detection): {dropped}')
    if len(X_df) < MIN_PIXELS:
        print(f'  WARNING: only {len(X_df)} valid pixels (< MIN_PIXELS={MIN_PIXELS}), skipping.')
        return

    log_lines = build_log_header(label, csv_path, ELEMENTS, list(missing), kept, dropped,
                                  len(X_df), len(df))

    if 'rf' in analyses:
        print('  Running Random Forest (this may take a while)...')
        X_rf, y_rf, rf_idx = subsample(X_df.values, y, MAX_SAMPLES)
        row_index_rf = X_df.index.values[rf_idx]
        if len(y_rf) < len(y):
            msg = f'  Subsampled {len(y_rf):,} of {len(y):,} pixels for RF (MAX_SAMPLES={MAX_SAMPLES})'
            print(msg); log_lines.append(msg.strip())
        log_lines.append('')

        y_pred_all, rmse_folds, r2_folds, imp_all, fold_id = run_rf_cv(X_rf, y_rf, kept, log_lines)

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

        imp_df = pd.DataFrame({
            'element': kept, 'mean_importance': mean_imp, 'std_importance': std_imp,
            'significant': [e in significant for e in kept],
        }).sort_values('mean_importance', ascending=False)
        imp_df.to_csv(out_dir / f'{label}_rf_importance.csv', index=False)

        pred_df = pd.DataFrame({
            'row_index': row_index_rf, 'observed_CL': y_rf,
            'predicted_CL': y_pred_all, 'fold': fold_id,
        })
        pred_df.to_csv(out_dir / f'{label}_rf_predictions.csv', index=False)

        print(f'  Saved RF outputs for {label}')

    if 'shap' in analyses:
        print('  Running SHAP...')
        mean_abs_shap, interaction_matrix, shap_values, X_shap, shap_idx = run_shap(X_df, y, kept)
        row_index_shap = X_df.index.values[shap_idx]
        order = np.argsort(mean_abs_shap)[::-1]

        log_lines.append('')
        log_lines.append(f'SHAP fit on {len(X_shap):,} pixels (SHAP_SAMPLES={SHAP_SAMPLES})')
        log_lines.append('Mean |SHAP value| by element: ' +
                          ', '.join(f'{kept[i]}={mean_abs_shap[i]:.4f}' for i in order))

        shap_df = pd.DataFrame({'element': kept, 'mean_abs_shap': mean_abs_shap}) \
            .sort_values('mean_abs_shap', ascending=False)
        shap_df.to_csv(out_dir / f'{label}_shap_importance.csv', index=False)

        shap_values_data = {'row_index': row_index_shap}
        for i, e in enumerate(kept):
            shap_values_data[f'{e}_value'] = X_shap[:, i]
            shap_values_data[f'{e}_shap'] = shap_values[:, i]
        pd.DataFrame(shap_values_data).to_csv(out_dir / f'{label}_shap_values.csv', index=False)

        if interaction_matrix is not None:
            off_diag = interaction_matrix.copy()
            np.fill_diagonal(off_diag, -np.inf)
            i, j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
            log_lines.append(f'Strongest pairwise interaction: {kept[i]} x {kept[j]} '
                              f'(mean |interaction| = {interaction_matrix[i, j]:.4f})')
            print(f'  Strongest interaction: {kept[i]} x {kept[j]}')

            inter_df = pd.DataFrame(interaction_matrix, index=kept, columns=kept)
            inter_df.to_csv(out_dir / f'{label}_shap_interactions.csv')

        print(f'  Saved SHAP outputs for {label}')

    log_file = diagnostics_dir / f'{label}_rf_shap_log.txt'
    log_file.write_text('\n'.join(str(l) for l in log_lines) + '\n')
    print(f'  Log saved: {log_file.name}')

# =============================================================================
# RUN
# =============================================================================

out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
diagnostics_dir = Path(DIAGNOSTICS_DIR)
diagnostics_dir.mkdir(parents=True, exist_ok=True)

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    if 'Region' in df.columns:
        print(f'  Region CSV {csv_path.name}: RF/SHAP is whole-grain only, skipping '
              f'(use kyanite_pca.py for region PCA).')
        continue

    exclude = {'CL', 'Region', 'DomainID'}
    elements = ELEMENTS or [c for c in df.columns if c not in exclude]
    missing = [e for e in elements if e not in df.columns]
    available = [e for e in elements if e in df.columns]
    if missing:
        print(f'  WARNING: columns not found, skipping: {missing}')

    grain_id = csv_path.stem.replace('_pixel_data', '')
    process_grain(df, available, grain_id, out_dir, diagnostics_dir, csv_path, missing)

print('\nDone.')
