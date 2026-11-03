"""
style.py -- one place for matplotlib styling.

Physical copy of methods_comparison's `methods_comparison/figures/style.py` (not an import),
so surrogate_trainer's reporting figures share the same journal-quality
theme (serif STIX font, Wong colour-blind-safe palette, 400dpi) as the rest
of the thesis's methods_comparison figures.
"""

import os
import matplotlib
matplotlib.use("Agg")               # headless backend for batch runs
import matplotlib.pyplot as plt

# Wong (2011) colour-blind-safe palette (kept in sync with config.py)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442", "#000000"]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]

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


def get_method_styles(method_names, palette=None):
    """Return {method: {color, marker, linestyle}} — stable across all figures."""
    pal = palette or PALETTE
    styles = {}
    for i, m in enumerate(method_names):
        styles[m] = {"color": pal[i % len(pal)],
                     "marker": MARKERS[i % len(MARKERS)],
                     "linestyle": LINESTYLES[i % len(LINESTYLES)]}
    return styles


def save_figure(fig, base_dir, stem, cfg=None, subdir=None):
    """
    Save `fig` under base_dir/<FORMAT>/[subdir/]stem.<format> for each format.
    `subdir` (e.g. "Instance 3") nests inside the format folder, as requested
    for per-instance Pareto plots. If `cfg["_code"]` is set (the runner sets
    this once, from problem_dict["code"]), every FILE name is prefixed with
    it -- e.g. stem="box_overall" -> "CODE_box_overall.png". Folder names
    (base_dir, subdir) are left as-is; only leaf files carry the prefix.
    """
    figcfg = (cfg or {}).get("figures", {})
    formats = figcfg.get("formats", ["png", "pdf", "svg"])
    dpi = figcfg.get("dpi", 400)
    code = (cfg or {}).get("_code")
    fname = f"{code}_{stem}" if code else stem
    for fmt in formats:
        d = os.path.join(base_dir, fmt.upper())
        if subdir:
            d = os.path.join(d, subdir)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{fname}.{fmt}"), dpi=dpi,
                    bbox_inches="tight", format=fmt)
    plt.close(fig)
