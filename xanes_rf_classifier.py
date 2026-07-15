# =============================================================================
# xanes_rf_classifier.py
#
# Cross-validated Random Forest classification of XANES pre-edge class
# (Type 1/2/3 — a proxy for Fe2+/Fe3+ ratio: Type 1 = Fe2+-dominant, Type 3 =
# Fe3+-dominant) from per-spot trace-element geochemistry, pooled across
# every <grain_id>_spot_geochemistry.csv produced by xrf_h5_extract_spots.py.
# Reports k-fold CV accuracy/balanced accuracy/macro F1, an out-of-fold
# confusion matrix, and permutation feature importance — the classification
# analog of kyanite_rf_shap.py's CL-intensity regression.
#
# 'Bad data'/unclassified spots are dropped (same convention as
# kyanite_spot_analysis.py's pie/box plots). All grains are always pooled
# into a single classifier rather than analyzed per-grain: with ~300 spots
# total, several classes missing entirely from some grains, and one grain
# (RH-XA-57081P-07 as of this writing) 100% a single class, per-grain models
# wouldn't be meaningful.
#
# Element columns are auto-detected as in kyanite_spot_analysis.py
# (everything not in METADATA_COLS), then further restricted to elements
# present in *every* input CSV — ROI lists vary slightly by grain (e.g.
# LLF6-01's extra REE lines), and keeping an element missing from some grains
# would silently drop every row from those grains once incomplete rows are
# removed.
#
# CV_STRATEGY controls fold construction:
#   'grouped'    - StratifiedGroupKFold by grain_id: no grain's spots are
#                  split across train/test, so the model is tested on
#                  chemistry from grains it never trained on. Recommended,
#                  but constrained by there being only a handful of grains.
#   'stratified' - StratifiedKFold ignoring grain identity: simpler, matches
#                  kyanite_rf_shap.py's plain KFold, but risks a fold learning
#                  a grain's chemistry signature rather than a general
#                  chemistry-oxidation relationship.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              classification_report, confusion_matrix)
from kyanite_palette import BLUE, ORANG, CATEGORY_ORDER

# =============================================================================
# PARAMETERS — edit this section for each run
# =============================================================================

# Anchored to this script's own location (not a hardcoded machine-specific
# path) so the same file runs unmodified on any machine/cluster this repo is
# checked out on.
_REPO_ROOT = Path(__file__).resolve().parent

CSV_INPUT    = _REPO_ROOT / 'figs' / 'data'    # file or directory of *_spot_geochemistry.csv
OUT_DIR      = _REPO_ROOT / 'figs' / 'spot_analysis'   # figures only; shares
               # kyanite_spot_analysis.py's folder — both pool the same per-spot
               # CSVs and fall under the same "spot analysis" umbrella, even
               # though they're two different analyses (classifier vs. pooled
               # scatter/pie/box/PCA figures)
DATA_OUTPUT_DIR = _REPO_ROOT / 'figs' / 'data'   # reusable CSVs (importance, predictions),
               # alongside the spot_geochemistry CSVs this script reads
DIAGNOSTICS_DIR = _REPO_ROOT / 'figs' / 'diagnostics'   # run log, matching every other
               # analysis/registration log in this project
OUTPUT_LABEL = 'all_grains'   # prefix for all output files (pooled analysis; no per-grain run)

ELEMENTS = ['Cr_Ka', 'V_Ka', 'Fe_Ka', 'Ti_Ka', 'Mn_Ka']   # None = auto-detect (present in every input CSV)

# --- Target / class filtering ---
# CATEGORY_ORDER (imported from kyanite_palette) excludes 'Bad data'/NaN, matches kyanite_spot_analysis.py

# --- Data cleaning (same semantics as kyanite_rf_shap.py) ---
BELOW_DETECTION          = None   # values <= this are treated as below detection limit; None to disable
MAX_BELOW_DETECTION_FRAC = 0.2    # drop an element if more than this fraction of spots are below detection
LOG_TRANSFORM            = True   # log10-transform element concentrations before RF
MIN_ROWS                 = 30     # abort if fewer than this many classifiable spots remain

# --- Random Forest ---
N_ESTIMATORS     = 200
MIN_SAMPLES_LEAF = 5
CLASS_WEIGHT     = 'balanced'   # counteract class imbalance (Type 1/2/3 counts differ across grains)
RANDOM_STATE     = 42

# --- Cross-validation ---
CV_FOLDS    = 5
CV_STRATEGY = 'grouped'   # 'grouped' (StratifiedGroupKFold by grain_id) or 'stratified' (StratifiedKFold)

# --- Feature importance ---
N_PERMUTATIONS       = 10    # repeats per fold for permutation importance
IMPORTANCE_SIG_RATIO = 1.0   # element flagged "significant" if mean/std of importance exceeds this

SAVE_FIG   = True
SAVE_CSV   = True
SHOW_TITLE = True

# Non-element columns from xrf_h5_extract_spots.py's schema (same list as
# kyanite_spot_analysis.py) — everything else in a spot CSV is an element column.
METADATA_COLS = [
    'grain_id', 'spot', 'spot_id', 'area_name', 'category', 'category_label',
    'pixel_count', 'row_px_h5', 'col_px_h5', 'row_px_tiff', 'col_px_tiff',
    'row_matlab', 'col_matlab', 'x_mm', 'y_mm', 'x_rel_um', 'y_rel_um',
    'zone_radius_um', 'zone_pixel_count', 'zone_mask_px_count', 'CL',
]

# =============================================================================
# LOAD & POOL
# =============================================================================

input_path = Path(CSV_INPUT)
if input_path.is_dir():
    csv_files = sorted(input_path.glob('*_spot_geochemistry.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No *_spot_geochemistry.csv files found in {input_path}')
else:
    csv_files = [input_path]

print(f'Processing {len(csv_files)} CSV(s):')

grain_frames = {}   # grain_id -> its own DataFrame (own element-column subset)
for path in csv_files:
    df = pd.read_csv(path)
    grain_id = df['grain_id'].iloc[0] if len(df) else path.stem.replace('_spot_geochemistry', '')
    grain_frames[grain_id] = df
    print(f'  {grain_id}: {len(df)} spot(s)')

# pd.concat(sort=False) over frames with differing element columns (ROI lists vary
# per grain) produces the column union, NaN-filling rows from grains that lack a
# given column — same pooling approach as kyanite_spot_analysis.py.
combined = pd.concat(grain_frames.values(), ignore_index=True, sort=False)

RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

ALL_CV_STRATEGIES = ['grouped', 'stratified']
if CV_STRATEGY not in ALL_CV_STRATEGIES:
    raise ValueError(f"Unknown CV_STRATEGY '{CV_STRATEGY}'; choose from {ALL_CV_STRATEGIES}.")

out_dir = Path(OUT_DIR)
if SAVE_FIG:
    out_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path(DATA_OUTPUT_DIR)
if SAVE_CSV:
    data_dir.mkdir(parents=True, exist_ok=True)
diagnostics_dir = Path(DIAGNOSTICS_DIR)
diagnostics_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def detect_elements(df):
    return [c for c in df.columns if c not in METADATA_COLS]


def detect_common_elements(grain_frames):
    """Elements present in every grain's own CSV (not just the pooled union).
    Returns (common, per_grain_dropped) where per_grain_dropped maps a
    dropped element to the list of grains missing it."""
    all_elements = sorted({e for df in grain_frames.values() for e in detect_elements(df)})
    per_grain_dropped = {}
    common = []
    for e in all_elements:
        missing_from = [gid for gid, df in grain_frames.items() if e not in df.columns]
        if missing_from:
            per_grain_dropped[e] = missing_from
        else:
            common.append(e)
    return common, per_grain_dropped

# =============================================================================
# DATA PREP
# =============================================================================

def prepare_data(df, elements):
    """Drop poorly-detected elements, log-transform, and drop incomplete rows.
    Returns (X_df, y, groups, kept_elements, dropped_elements)."""
    X = df[elements].astype(float).copy()
    y = df['category_label'].values
    groups = df['grain_id'].values

    if BELOW_DETECTION is not None:
        frac_below = (X <= BELOW_DETECTION).mean(axis=0)
        kept = [e for e in elements if frac_below[e] < MAX_BELOW_DETECTION_FRAC]
        dropped = [e for e in elements if e not in kept]
        X = X[kept]
        X = X.where(X > BELOW_DETECTION)   # remaining below-detection values -> NaN
    else:
        kept, dropped = list(elements), []

    if LOG_TRANSFORM:
        with np.errstate(divide='ignore', invalid='ignore'):
            X = np.log10(X)   # log10(0) -> -inf, filtered out below alongside NaN

    valid = X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    return X[valid], y[valid], groups[valid], kept, dropped

# =============================================================================
# RANDOM FOREST CLASSIFICATION
# =============================================================================

def make_splitter():
    if CV_STRATEGY == 'grouped':
        return StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def run_rf_cv(X, y, groups, elements, log_lines):
    splitter = make_splitter()
    y_pred_all = np.full(len(y), '', dtype=object)
    proba_all = np.full((len(y), len(CATEGORY_ORDER)), np.nan)
    acc_folds, bal_acc_folds, f1_folds = [], [], []
    imp_all = np.zeros((CV_FOLDS, len(elements)))

    split_args = (X, y, groups) if CV_STRATEGY == 'grouped' else (X, y)
    for i, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                     min_samples_leaf=MIN_SAMPLES_LEAF,
                                     class_weight=CLASS_WEIGHT,
                                     random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(X[train_idx], y[train_idx])
        y_pred = rf.predict(X[test_idx])
        y_pred_all[test_idx] = y_pred

        proba = rf.predict_proba(X[test_idx])
        for ci, cls in enumerate(rf.classes_):
            proba_all[test_idx, CATEGORY_ORDER.index(cls)] = proba[:, ci]

        acc = accuracy_score(y[test_idx], y_pred)
        bal_acc = balanced_accuracy_score(y[test_idx], y_pred)
        f1 = f1_score(y[test_idx], y_pred, average='macro', labels=CATEGORY_ORDER, zero_division=0)
        acc_folds.append(acc); bal_acc_folds.append(bal_acc); f1_folds.append(f1)

        perm = permutation_importance(rf, X[test_idx], y[test_idx],
                                       n_repeats=N_PERMUTATIONS,
                                       random_state=RANDOM_STATE, n_jobs=1)
        imp_all[i] = perm.importances_mean

        n_test_grains = len(set(groups[test_idx]))
        line = (f'  Fold {i + 1} - accuracy: {acc:.4f} | balanced accuracy: {bal_acc:.4f} '
                f'| macro F1: {f1:.4f} | test n={len(test_idx)} ({n_test_grains} grain(s))')
        print(line); log_lines.append(line)

    return y_pred_all, proba_all, np.array(acc_folds), np.array(bal_acc_folds), np.array(f1_folds), imp_all


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
    ax.grid(True, alpha=0.25, linewidth=0.5)
    return fig


def plot_confusion_matrix(cm, labels, accuracy, balanced_accuracy):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted (out-of-fold)')
    ax.set_ylabel('True')

    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if cm[i, j] > 0.6 * vmax else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=10)

    fig.colorbar(im, ax=ax, label='count')
    if SHOW_TITLE:
        ax.set_title(f'Out-of-fold confusion matrix\n'
                     f'accuracy={accuracy:.3f}  balanced accuracy={balanced_accuracy:.3f}', fontsize=10)
    fig.tight_layout()
    return fig

# =============================================================================
# LOG HEADER
# =============================================================================

def build_log_header(requested, per_grain_dropped, kept, dropped_detection, n_valid, n_total):
    lines = [
        'xanes_rf_classifier.py',
        f'Run date: {RUN_TIMESTAMP}',
        f'Source CSVs ({len(csv_files)}):',
    ]
    lines += [f'  {p}' for p in csv_files]

    lines += ['', 'Per-grain spot counts / XANES class distribution:']
    for grain_id, df in grain_frames.items():
        counts = df['category_label'].value_counts()
        counts_str = ', '.join(f'{c}={int(counts.get(c, 0))}' for c in CATEGORY_ORDER + ['Bad data'])
        classified = df[df['category_label'].isin(CATEGORY_ORDER)]
        note = ''
        if len(classified) and classified['category_label'].nunique() == 1:
            note = f"  <-- single-class grain ({classified['category_label'].iloc[0]})"
        lines.append(f'  {grain_id}: n={len(df)}  ({counts_str}){note}')

    lines += [
        '',
        'Parameters:',
        f'  ELEMENTS (requested): {requested if requested is not None else "all (auto-detected, common to every grain CSV)"}',
        f'  CATEGORY_ORDER: {CATEGORY_ORDER}',
        f'  BELOW_DETECTION: {BELOW_DETECTION}',
        f'  MAX_BELOW_DETECTION_FRAC: {MAX_BELOW_DETECTION_FRAC}',
        f'  LOG_TRANSFORM: {LOG_TRANSFORM}',
        f'  MIN_ROWS: {MIN_ROWS}',
        f'  N_ESTIMATORS: {N_ESTIMATORS}',
        f'  MIN_SAMPLES_LEAF: {MIN_SAMPLES_LEAF}',
        f'  CLASS_WEIGHT: {CLASS_WEIGHT}',
        f'  RANDOM_STATE: {RANDOM_STATE}',
        f'  CV_FOLDS: {CV_FOLDS}',
        f'  CV_STRATEGY: {CV_STRATEGY}',
        f'  N_PERMUTATIONS: {N_PERMUTATIONS}',
        f'  IMPORTANCE_SIG_RATIO: {IMPORTANCE_SIG_RATIO}',
        '',
    ]

    if per_grain_dropped:
        lines.append(f'Elements excluded (not present in every grain CSV, {len(per_grain_dropped)}):')
        for e, missing_from in per_grain_dropped.items():
            lines.append(f'  {e}: missing from {missing_from}')
    else:
        lines.append('Elements excluded (not present in every grain CSV): none')

    lines += [
        f'Elements dropped ({len(dropped_detection)}, >{MAX_BELOW_DETECTION_FRAC:.0%} below detection): {dropped_detection}'
        if BELOW_DETECTION is not None else 'Elements dropped (below detection): none (BELOW_DETECTION disabled)',
        f'Elements used ({len(kept)}): {kept}',
        '',
        f'Valid classifiable spots: {n_valid:,} of {n_total:,} total spots pooled across all grains',
        '',
    ]
    return lines

# =============================================================================
# RUN
# =============================================================================

print(f'\nTotal spots pooled: {len(combined):,}')
classifiable = combined[combined['category_label'].isin(CATEGORY_ORDER)].reset_index(drop=True)
print(f'Classifiable spots ({"/".join(CATEGORY_ORDER)}): {len(classifiable):,} '
      f'(dropped {len(combined) - len(classifiable):,} Bad data/unclassified)')

common_elements, per_grain_dropped = detect_common_elements(grain_frames)
if per_grain_dropped:
    print(f'WARNING: {len(per_grain_dropped)} element(s) not present in every grain CSV, excluded: '
          f'{list(per_grain_dropped)}')

requested = ELEMENTS
if ELEMENTS is None:
    elements = common_elements
else:
    missing_common = [e for e in ELEMENTS if e not in common_elements]
    if missing_common:
        print(f'WARNING: requested element(s) not present in every grain CSV, excluded: {missing_common}')
    elements = [e for e in ELEMENTS if e in common_elements]

if not elements:
    raise ValueError('No usable elements after coverage filtering; check ELEMENTS / input CSVs.')

X_df, y, groups, kept, dropped_detection = prepare_data(classifiable, elements)
print(f'\n{len(X_df):,} of {len(classifiable):,} classifiable spots valid after cleaning '
      f'({len(kept)} elements kept, {len(dropped_detection)} dropped for below-detection).')

if len(X_df) < MIN_ROWS:
    raise ValueError(f'Only {len(X_df)} valid spots (< MIN_ROWS={MIN_ROWS}); aborting.')

log_lines = build_log_header(requested, per_grain_dropped, kept, dropped_detection,
                              len(X_df), len(combined))

print(f'\nRunning {CV_FOLDS}-fold CV ({CV_STRATEGY} strategy) Random Forest classification...')
X_vals = X_df.values
y_pred_all, proba_all, acc_folds, bal_acc_folds, f1_folds, imp_all = run_rf_cv(
    X_vals, y, groups, kept, log_lines)

overall_acc = accuracy_score(y, y_pred_all)
overall_bal_acc = balanced_accuracy_score(y, y_pred_all)
report = classification_report(y, y_pred_all, labels=CATEGORY_ORDER, zero_division=0)
cm = confusion_matrix(y, y_pred_all, labels=CATEGORY_ORDER)

mean_imp = imp_all.mean(axis=0)
std_imp = imp_all.std(axis=0)
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(std_imp > 0, mean_imp / std_imp, np.nan)
significant = [kept[i] for i in range(len(kept)) if ratio[i] > IMPORTANCE_SIG_RATIO]

log_lines += [
    '',
    f'Mean fold accuracy:          {acc_folds.mean():.4f} +/- {acc_folds.std():.4f}',
    f'Mean fold balanced accuracy: {bal_acc_folds.mean():.4f} +/- {bal_acc_folds.std():.4f}',
    f'Mean fold macro F1:          {f1_folds.mean():.4f} +/- {f1_folds.std():.4f}',
    '',
    f'Out-of-fold accuracy:          {overall_acc:.4f}',
    f'Out-of-fold balanced accuracy: {overall_bal_acc:.4f}',
    '',
    'Out-of-fold classification report:',
    report,
    f'Out-of-fold confusion matrix (rows=true, cols=predicted, order={CATEGORY_ORDER}):',
]
for label, row in zip(CATEGORY_ORDER, cm):
    log_lines.append(f'  {label:8s} {row.tolist()}')
log_lines += [
    '',
    f'Significant features (mean/std > {IMPORTANCE_SIG_RATIO}): {significant}',
    'Permutation importance (mean +/- std), sorted:',
]
imp_order = np.argsort(mean_imp)[::-1]
for i in imp_order:
    flag = '  *significant*' if kept[i] in significant else ''
    log_lines.append(f'  {kept[i]}: {mean_imp[i]:.4f} +/- {std_imp[i]:.4f}{flag}')

print(f'\nOut-of-fold accuracy: {overall_acc:.4f} | balanced accuracy: {overall_bal_acc:.4f}')
print(f'Significant features: {significant}')

if SAVE_CSV:
    imp_df = pd.DataFrame({
        'element': kept, 'mean_importance': mean_imp, 'std_importance': std_imp,
        'significant': [e in significant for e in kept],
    }).sort_values('mean_importance', ascending=False)
    imp_df.to_csv(data_dir / f'{OUTPUT_LABEL}_rf_classifier_importance.csv', index=False)

    pred_df = classifiable.loc[X_df.index, ['grain_id', 'spot_id', 'spot', 'category_label']].copy()
    pred_df['predicted_label'] = y_pred_all
    pred_df['correct'] = pred_df['category_label'] == pred_df['predicted_label']
    for ci, cls in enumerate(CATEGORY_ORDER):
        pred_df[f'proba_{cls}'] = proba_all[:, ci]
    pred_df.to_csv(data_dir / f'{OUTPUT_LABEL}_rf_classifier_predictions.csv', index=False)

    print(f'  Saved: {OUTPUT_LABEL}_rf_classifier_importance.csv, {OUTPUT_LABEL}_rf_classifier_predictions.csv')

if SAVE_FIG:
    fig = plot_importance(mean_imp, std_imp, kept)
    if SHOW_TITLE:
        fig.suptitle(f'{OUTPUT_LABEL} — RF permutation importance (XANES class)')
    fig.tight_layout()
    fig.savefig(out_dir / f'{OUTPUT_LABEL}_rf_classifier_importance.png', dpi=200, bbox_inches='tight')

    fig = plot_confusion_matrix(cm, CATEGORY_ORDER, overall_acc, overall_bal_acc)
    fig.savefig(out_dir / f'{OUTPUT_LABEL}_rf_classifier_confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close('all')

    print(f'  Saved: {OUTPUT_LABEL}_rf_classifier_importance.png, {OUTPUT_LABEL}_rf_classifier_confusion_matrix.png')

log_file = diagnostics_dir / f'{OUTPUT_LABEL}_rf_classifier_log.txt'
log_file.write_text('\n'.join(str(l) for l in log_lines) + '\n')
print(f'\nLog saved: {log_file.name}')
print('Done.')
