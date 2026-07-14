# =============================================================================
# kyanite_palette.py
#
# Canonical color conventions shared across every figure-generating script in
# this project, so the same category/element/quantity renders in the same
# color regardless of which script or grain produced the figure. See
# CLAUDE.md's "Color conventions" section for the full spec, including the
# MATLAB-side equivalents — CL_EPMA_registration.m, CL_mask_edit.m, and
# CL_region_extraction.m can't import this module, so they carry local
# functions/constants kept numerically in sync with the values below by hand.
#
# Import what you need rather than `import *`, e.g.:
#   from kyanite_palette import BLUE, ORANG, element_colors, region_colors
# =============================================================================

# --- House palette -----------------------------------------------------------
# General-purpose roles reused across whole-grain/region figures: BLUE for the
# main data cloud/bar, ORANG for a fit line, highlight, or "above threshold"
# marker. Not tied to any specific element, region, or category.
BLUE  = '#3B9BDD'
ORANG = '#D85B30'

# --- Fixed element -> color --------------------------------------------------
# Okabe-Ito colorblind-safe qualitative set. Covers the 5 elements that
# recur across every grain's whole-grain/region maps (the ELEMENTS default
# shared by kyanite_figures.py/kyanite_pca_rf.py/CL_EPMA_registration.m's
# shift-sensitivity plot). Any other element name (e.g. a spot-geochemistry
# trace element not in this set) falls back to the extra colors below, then
# repeats — see element_colors().
ELEMENT_COLORS = {
    'Cr_Ka': '#E69F00',  # orange
    'Fe_Ka': '#56B4E9',  # sky blue
    'Ti_Ka': '#009E73',  # bluish green
    'V_Ka':  '#F0E442',  # yellow
    'Mn_Ka': '#CC79A7',  # reddish purple
}
_ELEMENT_FALLBACK_EXTRA = ['#0072B2', '#D55E00']  # blue, vermillion (remaining Okabe-Ito colors)


def element_colors(names):
    """names -> {name: hex color}. Elements in ELEMENT_COLORS always get
    their fixed color; any other name gets a color assigned by sorted order
    into _ELEMENT_FALLBACK_EXTRA (repeating if there are more than 2), so a
    given "extra" element is at least stable within one call (e.g. one
    grain's shift-sensitivity plot) even though it isn't individually
    curated."""
    uniq_extra = sorted(n for n in dict.fromkeys(names) if n not in ELEMENT_COLORS)
    extra_map = {n: _ELEMENT_FALLBACK_EXTRA[i % len(_ELEMENT_FALLBACK_EXTRA)]
                 for i, n in enumerate(uniq_extra)}
    return {n: ELEMENT_COLORS.get(n, extra_map.get(n)) for n in names}


# --- Region name -> color ----------------------------------------------------
# Region names are freeform and per-grain (today: generic roi_1/roi_2/...,
# not semantic labels) — there's no fixed vocabulary to hardcode. Instead,
# region_colors() assigns colors deterministically by *sorted name*, from
# this fixed qualitative palette (matplotlib's 'tab10', spelled out here so
# the MATLAB side can use the identical RGB values). Same name -> same
# color in every script and every grain; if regions are drawn in a
# consistent order across grains (e.g. always innermost first), same
# *position* ends up the same color too.
REGION_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def region_colors(names):
    """names -> {name: hex color}, assigned by sorted order into
    REGION_PALETTE (repeating past 10 distinct names)."""
    uniq = sorted(dict.fromkeys(names))
    return {n: REGION_PALETTE[i % len(REGION_PALETTE)] for i, n in enumerate(uniq)}


# --- XANES pre-edge class -> color ------------------------------------------
# Type 1/2/3 colors only — the "unclassified/QC-failed" grey fallback is
# GREY below, keyed differently per script (kyanite_spot_analysis.py's
# hand-classification pipeline uses 'Bad data'; xanes_plot.py's optional
# auto-classifier uses 'Ambiguous' for a QC-failed spot) since the two
# scripts' category_label vocabularies differ by design — see CLAUDE.md.
CATEGORY_ORDER = ['Type 1', 'Type 2', 'Type 3']
CATEGORY_COLORS = {
    'Type 1': '#D85B30',
    'Type 2': '#4C9F70',
    'Type 3': '#7A5195',
}
GREY = '#999999'

# --- Colormap conventions ----------------------------------------------------
# One canonical choice per role, so every script reaches for the same
# colormap rather than each picking its own for a conceptually identical
# quantity. DIVERGING_CMAP is for any signed, zero-centered quantity
# (correlation r, local-regression slope/R — CL_local_regression_map.m's
# hand-rolled MATLAB colormap is tuned to match this cmap's actual anchor
# colors, since MATLAB can't import it directly). SEQUENTIAL_CMAP is for
# continuous-intensity roles (KDE density, SHAP magnitude, dependence-plot
# coloring).
DIVERGING_CMAP  = 'RdBu_r'
SEQUENTIAL_CMAP = 'inferno'
