"""
style.py  —  one place for matplotlib styling.

`reporting.py` calls ``apply_thesis_style()`` once per process, then saves
every figure itself via its own ``_save_all_formats(...)`` (which honours
``figures_cfg["formats"]``/``["dpi"]``) -- this module only owns the shared
rcParams/palette/markers, not figure I/O.
"""

import matplotlib
matplotlib.use("Agg")               # headless backend for batch runs
import matplotlib.pyplot as plt

# Wong (2011) colour-blind-safe palette (kept in sync with config.py)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442", "#000000"]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

_THESIS_RC = {
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    0.9,
    "axes.grid":         True,
    "grid.color":        "#e3e3e3",
    "grid.linewidth":    0.5,
    "grid.alpha":        0.8,
    "axes.axisbelow":    True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    # Serif type set via matplotlib's bundled STIX fonts -- reads like a
    # typeset journal page without requiring a system LaTeX install
    # (text.usetex is unavailable in this environment).
    "font.family":        "serif",
    "font.serif":         ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.titleweight":   "normal",
    "axes.labelsize":     11,
    "axes.labelweight":   "normal",
    "xtick.labelsize":    9.5,
    "ytick.labelsize":    9.5,
    "legend.fontsize":    9,
    "legend.frameon":     False,
    "legend.handlelength": 1.6,
    "lines.linewidth":    1.4,
    "lines.markersize":   5,
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "boxplot.boxprops.linewidth":     1.1,
    "boxplot.whiskerprops.linewidth": 1.0,
    "boxplot.capprops.linewidth":     1.0,
    "boxplot.medianprops.linewidth":  1.3,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
}


def apply_thesis_style():
    plt.rcParams.update(_THESIS_RC)
