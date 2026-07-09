# =============================================================================
# kyanite_sample_size_convergence.py
#
# Diagnostic for kyanite_pca_rf.py: does RF/SHAP feature importance actually
# need more pixels than MAX_SAMPLES / SHAP_SAMPLES currently use, or has it
# already converged well before the full grain (which can be 100k-1M+ px)?
#
# For a single grain's pixel CSV, draws N_REPEATS independent random
# subsamples (without replacement) at each size in SAMPLE_SIZES, splits each
# into train/test, fits one Random Forest per repeat, and records:
#   - held-out RMSE / R2
#   - permutation importance per element (on the held-out split)
#   - SHAP importance per element (mean |SHAP value|, on a capped subsample
#     of the held-out split, so SHAP cost doesn't grow with the sweep)
# SHAP interaction values are NOT computed here — they're the expensive part
# of kyanite_pca_rf.py and this script is about whether *sample size* matters,
# not about re-deriving the interaction analysis itself.
#
# Plots each metric vs. sample size (log x-axis) with the spread across
# repeats shown as a shaded band. A band that stays wide/shifting as sample
# size grows means the estimate hasn't converged yet; a band that narrows and
# flattens means more data wouldn't change the answer. The true full-grain
# pixel count is always included as the last point, so the sweep ends with
# the real, no-subsampling answer for this grain.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import shap

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

CSV_INPUT = '/Users/mstein/bin/kyanite/figs/data/NA-GS-P84-06_pixel_data.csv'   # a single grain's *_pixel_data.csv
ELEMENTS  = None      # list of CSV column names to include; None = all columns except CL/Region

# --- Data cleaning, same conventions as kyanite_pca_rf.py ---
BELOW_DETECTION          = None   # values <= this are treated as below detection limit; None to disable
MAX_BELOW_DETECTION_FRAC = 0.2    # drop an element if more than this fraction of pixels are below detection
LOG_TRANSFORM            = True   # log10-transform element concentrations before RF/SHAP

# --- Sweep ---
SAMPLE_SIZES  = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
              # sizes to test; auto-filtered to <= total valid pixels, and the
              # true full-grain pixel count is always appended as the last point
N_REPEATS     = 5      # independent random subsamples per size, to show spread
TEST_FRACTION = 0.2    # held-out fraction within each subsample
CONVERGENCE_THRESHOLD = 0.05   # a step-to-step relative change below this counts as "converged"

# --- Random Forest (kept smaller/faster than kyanite_pca_rf.py defaults,
#     since this script fits many more times) ---
N_ESTIMATORS        = 50
MIN_SAMPLES_LEAF    = 5
MAX_DEPTH           = 10   # caps tree depth/leaf count. Without this, min_samples_leaf
                            # alone lets trees at n=300k+ grow to ~50k+ leaves each, and
                            # TreeSHAP cost scales with leaf count per explained sample —
                            # a shap_values() call on just 1000 points took ~60s uncapped
                            # vs. ~2s at MAX_DEPTH=10 in testing. Also keeps model
                            # complexity comparable across the sweep, so differences
                            # between sample sizes reflect the data, not deeper trees.
N_PERMUTATIONS      = 10     # repeats for permutation importance
SHAP_EXPLAIN_SAMPLES = 1000  # cap on held-out points explained by SHAP per repeat,
                              # so SHAP cost stays roughly constant across the sweep
RANDOM_STATE         = 42

SAVE_FIG = True
SAVE_CSV = True

BLUE  = '#3B9BDD'
ORANG = '#D85B30'
COLORS = ['#3B9BDD', '#D85B30', '#4C9F70', '#9B5DE5', '#F2B134', '#EF476F', '#118AB2']

# =============================================================================
# LOAD + CLEAN
# =============================================================================

csv_path = Path(CSV_INPUT)
df = pd.read_csv(csv_path)
label = csv_path.stem.replace('_pixel_data', '').replace('_region_pixel_data', '')
out_dir = csv_path.parent

exclude = {'CL', 'Region', 'DomainID'}
elements = ELEMENTS or [c for c in df.columns if c not in exclude]
missing = [e for e in elements if e not in df.columns]
elements = [e for e in elements if e in df.columns]
if missing:
    print(f'WARNING: columns not found, skipping: {missing}')

X_all = df[elements].astype(float).copy()
y_all = df['CL'].astype(float).values

if BELOW_DETECTION is not None:
    frac_below = (X_all <= BELOW_DETECTION).mean(axis=0)
    kept = [e for e in elements if frac_below[e] < MAX_BELOW_DETECTION_FRAC]
    dropped = [e for e in elements if e not in kept]
    X_all = X_all[kept]
    X_all = X_all.where(X_all > BELOW_DETECTION)
else:
    kept, dropped = list(elements), []

if LOG_TRANSFORM:
    X_all = np.log10(X_all)

valid = X_all.notna().all(axis=1) & np.isfinite(y_all)
X = X_all[valid].values
y = y_all[valid]
n_valid = len(y)

print(f'{label}: {n_valid:,} valid pixels, elements used: {kept}' +
      (f', dropped: {dropped}' if dropped else ''))

sizes = sorted(set(s for s in SAMPLE_SIZES if s <= n_valid) | {n_valid})
print(f'Sample sizes to test: {sizes}')

rng = np.random.default_rng(RANDOM_STATE)

# =============================================================================
# SWEEP
# =============================================================================

def split_train_test(X, y, test_fraction, rng):
    n = len(y)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def one_fit(size, repeat, elements):
    idx = rng.choice(n_valid, size=size, replace=False)
    X_sub, y_sub = X[idx], y[idx]
    X_train, X_test, y_train, y_test = split_train_test(X_sub, y_sub, TEST_FRACTION, rng)

    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF,
                                max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=N_PERMUTATIONS,
                                       random_state=RANDOM_STATE, n_jobs=1).importances_mean

    X_shap = X_test
    if len(X_shap) > SHAP_EXPLAIN_SAMPLES:
        shap_idx = rng.choice(len(X_shap), size=SHAP_EXPLAIN_SAMPLES, replace=False)
        X_shap = X_shap[shap_idx]
    shap_vals = shap.TreeExplainer(rf).shap_values(X_shap)
    shap_imp = np.abs(shap_vals).mean(axis=0)

    row = {'size': size, 'repeat': repeat, 'rmse': rmse, 'r2': r2}
    for i, e in enumerate(elements):
        row[f'perm_{e}'] = perm_imp[i]
        row[f'shap_{e}'] = shap_imp[i]
    return row


print(f'\nRunning sweep: {len(sizes)} sizes x {N_REPEATS} repeats = {len(sizes) * N_REPEATS} fits...')
records = []
for size in sizes:
    for repeat in range(N_REPEATS):
        records.append(one_fit(size, repeat, kept))
    print(f'  size={size:,}: done ({N_REPEATS} repeats)')

results = pd.DataFrame(records)

if SAVE_CSV:
    results.to_csv(out_dir / f'{label}_convergence_raw.csv', index=False)

# =============================================================================
# AGGREGATE + CONVERGENCE CHECK
# =============================================================================

def summarize(col):
    g = results.groupby('size')[col]
    return g.mean(), g.std()


def first_converged_size(mean_series, threshold):
    """Smallest size after which every subsequent step-to-step relative
    change stays below threshold; None if it never stabilizes."""
    vals = mean_series.values
    sizes_arr = mean_series.index.values
    denom = np.maximum(np.abs(vals[:-1]), 1e-12)
    rel_change = np.abs(np.diff(vals)) / denom
    for i in range(len(rel_change)):
        if np.all(rel_change[i:] < threshold):
            return sizes_arr[i]
    return None


log_lines = [
    f'kyanite_sample_size_convergence.py — {label}',
    f'Source CSV: {csv_path}',
    f'Elements used: {kept}',
    f'Total valid pixels: {n_valid:,}',
    f'Sample sizes tested: {sizes}',
    f'Repeats per size: {N_REPEATS}',
    f'RF: N_ESTIMATORS={N_ESTIMATORS}, MIN_SAMPLES_LEAF={MIN_SAMPLES_LEAF}, MAX_DEPTH={MAX_DEPTH}',
    f'SHAP_EXPLAIN_SAMPLES: {SHAP_EXPLAIN_SAMPLES}',
    f'CONVERGENCE_THRESHOLD: {CONVERGENCE_THRESHOLD:.0%} step-to-step relative change',
    '',
]

for metric, name in [('rmse', 'RMSE'), ('r2', 'R2')]:
    mean, std = summarize(metric)
    conv = first_converged_size(mean, CONVERGENCE_THRESHOLD)
    log_lines.append(f'{name}: converged by n={conv:,}' if conv is not None
                      else f'{name}: did not clearly converge within tested sizes')

log_lines.append('')
for e in kept:
    for prefix, name in [('perm', 'Permutation importance'), ('shap', 'SHAP importance')]:
        mean, std = summarize(f'{prefix}_{e}')
        conv = first_converged_size(mean, CONVERGENCE_THRESHOLD)
        log_lines.append(f'{name} ({e}): converged by n={conv:,}' if conv is not None
                          else f'{name} ({e}): did not clearly converge within tested sizes')

print('\n'.join(log_lines))

# =============================================================================
# PLOTS
# =============================================================================

def plot_metric_vs_size(mean, std, ylabel, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mean.index, mean.values, 'o-', color=BLUE, lw=1.5)
    ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values,
                     color=BLUE, alpha=0.25)
    ax.set_xscale('log')
    ax.set_xlabel('Sample size (pixels)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return fig


def plot_importance_vs_size(prefix, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, e in enumerate(kept):
        mean, std = summarize(f'{prefix}_{e}')
        color = COLORS[i % len(COLORS)]
        ax.plot(mean.index, mean.values, 'o-', color=color, lw=1.5, label=e)
        ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values,
                         color=color, alpha=0.2)
    ax.set_xscale('log')
    ax.set_xlabel('Sample size (pixels)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return fig


if SAVE_FIG:
    mean, std = summarize('rmse')
    fig = plot_metric_vs_size(mean, std, 'Held-out RMSE', f'{label} — RF RMSE vs. sample size')
    fig.tight_layout()
    fig.savefig(out_dir / f'{label}_convergence_rmse.png', dpi=200, bbox_inches='tight')

    mean, std = summarize('r2')
    fig = plot_metric_vs_size(mean, std, 'Held-out R2', f'{label} — RF R2 vs. sample size')
    fig.tight_layout()
    fig.savefig(out_dir / f'{label}_convergence_r2.png', dpi=200, bbox_inches='tight')

    fig = plot_importance_vs_size('perm', 'Permutation importance',
                                   f'{label} — permutation importance vs. sample size')
    fig.tight_layout()
    fig.savefig(out_dir / f'{label}_convergence_perm_importance.png', dpi=200, bbox_inches='tight')

    fig = plot_importance_vs_size('shap', 'Mean |SHAP value|',
                                   f'{label} — SHAP importance vs. sample size')
    fig.tight_layout()
    fig.savefig(out_dir / f'{label}_convergence_shap_importance.png', dpi=200, bbox_inches='tight')
    plt.close('all')

log_file = out_dir / f'{label}_convergence_log.txt'
log_file.write_text('\n'.join(log_lines) + '\n')
print(f'\nLog saved: {log_file.name}')
print('Done.')
