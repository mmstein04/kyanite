# =============================================================================
# kyanite_outliers.py
#
# Shared element-concentration outlier logic used by both kyanite_figures.py
# (per-element pairwise CL-vs-element filtering) and kyanite_pca.py (pooled
# across every element PCA uses, since PCA needs one common pixel set).
# Kept in one place so the two scripts can't silently drift apart on what
# counts as an outlier — see CLAUDE.md's kyanite_figures.py entry for the
# full method description and the MAD_K_HI=4 default's provenance
# (kyanite_outlier_method_comparison.py).
#
# Every function here takes its thresholds as explicit arguments rather than
# reading module-level constants, so each calling script keeps its own
# top-of-file parameter block (and can in principle diverge from the other's
# defaults) while sharing the same underlying algorithm.
# =============================================================================

import numpy as np


def saturation_mask(x, label, band_frac, min_frac, min_count, verbose=True):
    """Boolean mask, True where x is piled up in a thin band near its own
    max — the signature of a clipped/saturated detector channel (many
    pixels tied at or near a hard ceiling), not the single natural extreme
    point a smooth continuous distribution would have. Max side only: a
    pileup near the *min* is just ordinary near-zero/below-detection-limit
    concentration data, not saturation, and is extremely common/legitimate
    in trace element maps — flagging it there would gut real (low-
    concentration) data. Band-based (not exact-value ties) so it still
    catches this after per-pixel normalization has nudged what was
    originally an identical raw ceiling into slightly different float
    values."""
    mask = np.zeros(len(x), dtype=bool)
    if len(x) == 0:
        return mask
    n = len(x)
    xmin, xmax = x.min(), x.max()
    band = band_frac * (xmax - xmin)
    if band == 0:
        return mask
    near_max = x >= xmax - band
    count = int(near_max.sum())
    if count >= min_count and count / n >= min_frac:
        if verbose:
            print(f'  WARNING: {label}: {count:,} px ({count / n:.2%}) piled up near the max '
                  f'of its range — likely detector saturation/clipping, excluded')
        mask |= near_max
    return mask


def mad_keep_mask(x, k_lo, k_hi):
    """Boolean mask, True to keep. Robust modified z-score (Iglewicz & Hoya):
    0.6745*(x - median)/MAD, computed in log-space — element concentrations
    are right-skewed (lognormal-ish). On raw values, MAD reads the natural
    long high-concentration tail as "outliers" and would strip out exactly
    the scientifically important pixels (e.g. high-Fe quenching zones).
    Non-positive values (zero/negative, e.g. a background-subtracted floor)
    can't be log-transformed and aren't high-tail outliers anyway, so they
    pass through unevaluated (kept). MAD == 0 means over half the (positive)
    values are identical — nothing is statistically distinguishable as an
    outlier, so keep everything rather than divide by zero."""
    keep = np.ones(len(x), dtype=bool)
    positive = x > 0
    xs = x[positive]
    if len(xs) < 2:
        return keep

    lx = np.log(xs)
    med = np.median(lx)
    mad = np.median(np.abs(lx - med))
    if mad == 0:
        return keep

    z = 0.6745 * (lx - med) / mad
    sub_keep = np.ones(len(xs), dtype=bool)
    if k_lo is not None:
        sub_keep &= z >= -k_lo
    if k_hi is not None:
        sub_keep &= z <= k_hi
    keep[positive] = sub_keep
    return keep


def outlier_keep_mask(x, method, mad_k_lo, mad_k_hi, pct_lo, pct_hi):
    if method == 'mad':
        return mad_keep_mask(x, mad_k_lo, mad_k_hi)
    elif method == 'percentile':
        lo, hi = np.percentile(x, [pct_lo, pct_hi])
        return (x >= lo) & (x <= hi)
    else:
        raise ValueError(f"Unknown OUTLIER_METHOD {method!r}; choose 'mad' or 'percentile'")
