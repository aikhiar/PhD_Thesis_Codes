"""
Figures (matplotlib, PNG + PDF). Conventions shared by every figure:
  * a method keeps the same colour in every figure (colour follows the entity, assigned once in
    the order of the methods library); at most eight categorical colours (validated reference
    order); further methods are drawn in gray with their own marker
  * markers and line styles double the colour (print, colour-vision deficiencies)
  * thin marks (1.5 pt lines), hairline recessive grid, one y-axis per panel
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 3))]
GRAY = "#8a8985"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


class Styles:
    """Fixed method -> (colour, marker, line style), assigned in a given order."""

    def __init__(self, order=()):
        self.map = {}
        for m in order:
            self.get(m)

    def get(self, m):
        if m not in self.map:
            k = len(self.map)
            c = PALETTE[k] if k < len(PALETTE) else GRAY
            self.map[m] = (c, MARKERS[k % len(MARKERS)], STYLES[k % len(STYLES)])
        return self.map[m]


class FigureWriter:
    def __init__(self, root, formats=("png", "pdf"), dpi=200):
        self.root = root
        self.formats, self.dpi = formats, dpi
        self.index = []

    def save(self, fig, name, folder=None, caption=""):
        """Saves the figure in every format in `folder`; returns its .png path relative to the report root."""
        folder = folder or os.path.join(self.root, "figures")
        os.makedirs(folder, exist_ok=True)
        for ext in self.formats:
            fig.savefig(os.path.join(folder, f"{name}.{ext}"), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        rel = os.path.relpath(os.path.join(folder, name), self.root).replace(os.sep, "/")
        self.index.append((rel, caption))
        return rel + ".png"


# ── figure builders ─────────────────────────────────────────────────────────────

def anytime(curves, styles, title="", xlabel="time (s)", ylabel="gap to LB* (%)", logx=True, bands=True,
            hline=None, ylim=None):
    """curves: {method: (grid, median, q1, q3)}."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for m, (g, med, q1, q3) in curves.items():
        c, mk, ls = styles.get(m)
        ok = ~np.isnan(med)
        if not ok.any():
            continue
        ax.plot(g[ok], med[ok], color=c, ls=ls, label=m, marker=mk, markevery=max(1, ok.sum() // 6), ms=4.5,
                mec="white", mew=0.8)
        if bands and q1 is not None:
            ax.fill_between(g[ok], q1[ok], q3[ok], color=c, alpha=0.10, lw=0)
    if hline is not None:
        ax.axhline(hline, color=INK2, lw=0.8)
    if logx:
        ax.set_xscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    ax.legend(loc="upper right", ncol=2 if len(curves) > 4 else 1)
    return fig


def performance_profile(M, styles, title="", tau_max=None):
    """Dolan-More profile of a lower-is-better objective matrix (instances x methods, positive)."""
    M = M.dropna()
    best = M.min(axis=1)
    R = M.div(best, axis=0)
    tmax = tau_max or max(1.0001, float(np.nanquantile(R.values, 0.98)) * 1.02)
    taus = np.linspace(1, tmax, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for m in M.columns:
        c, mk, ls = styles.get(m)
        rho = [(R[m] <= t + 1e-12).mean() for t in taus]
        ax.step(taus, rho, where="post", color=c, ls=ls, label=m)
    ax.set_xlabel(r"$\tau$ (objective / best objective on the instance)")
    ax.set_ylabel(r"fraction of instances within $\tau$")
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title, loc="left")
    ax.legend(loc="lower right", ncol=2 if M.shape[1] > 4 else 1)
    return fig


def boxplot(data, highlight=(), title="", xlabel="gap to LB* (%)"):
    """data: {method: values}; horizontal, sorted by median, one neutral colour, highlighted methods
    in the first categorical colour."""
    items = sorted(((m, np.asarray(v, float)[~np.isnan(np.asarray(v, float))]) for m, v in data.items()),
                   key=lambda kv: np.median(kv[1]) if len(kv[1]) else np.inf, reverse=True)
    items = [(m, v) for m, v in items if len(v)]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(items) + 1.0))
    bp = ax.boxplot([v for _, v in items], vert=False, widths=0.55, patch_artist=True, showfliers=True,
                    medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": INK2, "lw": 0.8},
                    capprops={"color": INK2, "lw": 0.8},
                    flierprops={"marker": "o", "ms": 2.5, "mfc": GRAY, "mec": "none", "alpha": 0.6})
    for patch, (m, _) in zip(bp["boxes"], items):
        hl = any(m.startswith(h) for h in highlight)
        patch.set_facecolor(PALETTE[0] if hl else "#cfcec9"); patch.set_edgecolor("white"); patch.set_alpha(0.9)
    ax.set_yticks(range(1, len(items) + 1)); ax.set_yticklabels([m for m, _ in items])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title, loc="left")
    return fig


def cd_diagram(ranks, cd, title=""):
    """Critical-difference diagram (Demsar 2006): average ranks on an axis, methods joined by a bar when
    their ranks differ by less than the critical difference."""
    ranks = ranks.sort_values()
    k = len(ranks)
    if k < 2:
        return None
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.22 * k))
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y0 = 0.86
    ax.plot([lo, hi], [y0, y0], color=INK, lw=0.8)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [y0, y0 + 0.03], color=INK, lw=0.8)
        ax.text(r, y0 + 0.05, str(r), ha="center", va="bottom", fontsize=8, color=INK2)
    half = (k + 1) // 2
    step = 0.62 / max(1, half)
    for j, (m, r) in enumerate(ranks.items()):
        left = j < half
        yy = y0 - 0.08 - step * (j if left else (k - 1 - j))
        xe = lo - 0.25 if left else hi + 0.25
        ax.plot([r, r, xe], [y0, yy, yy], color=INK2, lw=0.7)
        ax.text(xe + (-0.03 if left else 0.03), yy, f"{m} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=8, color=INK)
    # cliques
    vals = ranks.values
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        yy = y0 - 0.04 - 0.035 * c
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [yy, yy], color=INK, lw=2.2, solid_capstyle="round")
    ax.set_ylim(0, 1.12)
    ax.plot([lo, lo + cd], [1.02, 1.02], color=INK, lw=1.2)
    ax.text(lo + cd / 2, 1.04, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    if title:
        ax.set_title(title, loc="left", pad=14)
    return fig


def pvalue_heatmap(P, title="", alpha=0.05):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", ["#0d366b", "#2a78d6", "#86b6ef", "#e8f0fb"])
    m = P.shape[0]
    fig, ax = plt.subplots(figsize=(0.55 * m + 1.8, 0.5 * m + 1.2))
    V = P.values.astype(float)
    im = ax.imshow(np.clip(V, 1e-6, 1), cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=1e-4, vmax=1))
    for i in range(m):
        for j in range(m):
            if i != j and V[i, j] == V[i, j]:
                lum_dark = V[i, j] < 0.01
                ax.text(j, i, ("*" if V[i, j] < alpha else "") + f"{V[i, j]:.2g}", ha="center", va="center",
                        fontsize=6.5, color="white" if lum_dark else INK)
    ax.set_xticks(range(m)); ax.set_xticklabels(P.columns, rotation=60, ha="right")
    ax.set_yticks(range(m)); ax.set_yticklabels(P.index)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Holm-adjusted p-value (log scale)")
    if title:
        ax.set_title(title, loc="left")
    return fig


def scaling(df, x, y, styles, title="", xlabel="", ylabel="", logx=True, logy=True):
    """df: columns method, x, y (median) and optionally y_lo / y_hi."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for m, g in df.groupby("method", sort=False):
        g = g.sort_values(x)
        c, mk, ls = styles.get(m)
        ax.plot(g[x], g[y], color=c, ls=ls, marker=mk, ms=5, mec="white", mew=0.8, label=m)
        if "y_lo" in g and "y_hi" in g:
            ax.fill_between(g[x], g.y_lo, g.y_hi, color=c, alpha=0.10, lw=0)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or x); ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title, loc="left")
    ax.legend(ncol=2 if df.method.nunique() > 4 else 1)
    xs = sorted(set(float(v) for v in df[x].dropna()))
    if logx and 1 < len(xs) <= 12:                     # ticks at the values of x (plain numbers)
        from matplotlib.ticker import NullFormatter
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in xs])
        ax.xaxis.set_minor_formatter(NullFormatter())
    return fig


def strip_ratios(data, title="", xlabel="certificate / OPT (or LB*)", ref=1.0):
    """data: {certificate: values}; one row per certificate, dots with jitter and the median."""
    items = [(k, np.asarray(v, float)[~np.isnan(np.asarray(v, float))]) for k, v in data.items()]
    items = [(k, v) for k, v in items if len(v)]
    if not items:
        return None
    items.sort(key=lambda kv: np.median(kv[1]))
    fig, ax = plt.subplots(figsize=(6.2, 0.3 * len(items) + 1.0))
    rng = np.random.default_rng(0)
    for j, (k, v) in enumerate(items):
        ax.scatter(v, j + rng.uniform(-0.18, 0.18, len(v)), s=9, color=PALETTE[0], alpha=0.55, lw=0)
        ax.plot([np.median(v)] * 2, [j - 0.3, j + 0.3], color=INK, lw=1.4)
    ax.axvline(ref, color=INK2, lw=0.8)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([k for k, _ in items])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, loc="left")
    return fig


def load_profiles(profiles, styles, lb=None, title="", dt_hours=None):
    """profiles: {label: load vector (kW)}."""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for m, L in profiles.items():
        c, mk, ls = styles.get(m)
        t = np.arange(len(L)) * (dt_hours or 1)
        ax.step(t, L, where="post", color=c, ls=ls, label=f"{m} (peak {np.max(L):.1f} kW)")
    if lb is not None:
        ax.axhline(lb, color=INK2, lw=0.8)
        ax.text(0, lb, " certified lower bound", va="bottom", fontsize=8, color=INK2)
    ax.set_xlabel("time (h)" if dt_hours else "slot"); ax.set_ylabel("load (kW)")
    if title:
        ax.set_title(title, loc="left")
    ax.legend(loc="upper right")
    return fig


def solver_progress(series, styles, title="", ylabel="peak (kW)"):
    """series: {label: (t, values, kind)} with kind 'inc' (incumbent, solid) or 'bnd' (bound, dashed)."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for lab, (t, v, kind, ent) in series.items():
        c, mk, ls = styles.get(ent)
        if kind == "inc":
            ax.step(t, v, where="post", color=c, ls="-", label=lab, marker=mk, markevery=max(1, len(t) // 5), ms=4.5,
                    mec="white", mew=0.8)
        else:
            ax.step(t, v, where="post", color=c, ls="--", lw=1.1, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    ax.legend()
    return fig


def quality_time(df, styles, title="", xlabel="mean charged time (s)", ylabel="mean gap to LB* (%)"):
    """One point per method (mean time, mean gap) with its name; log time axis."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for _, r in df.iterrows():
        c, mk, _ = styles.get(r.method)
        ax.scatter([max(r.time, 1e-4)], [r.gap], s=42, color=c, marker=mk, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(r.method, (max(r.time, 1e-4), r.gap), xytext=(4, 3), textcoords="offset points", fontsize=7,
                    color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    return fig


def bars(values, title="", xlabel="", highlight=()):
    """Horizontal bars (sorted), for counts such as instances solved."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.2, 0.3 * len(items) + 1.0))
    ys = np.arange(len(items))
    cols = [PALETTE[0] if any(k.startswith(h) for h in highlight) else "#b9b8b3" for k, _ in items]
    ax.barh(ys, [v for _, v in items], height=0.6, color=cols)
    for y, (_, v) in zip(ys, items):
        ax.text(v, y, f" {v:g}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels([k for k, _ in items])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, loc="left")
    return fig
