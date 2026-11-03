"""
plots.py :  the figures of the comparisons, those the manuscript shows.

All figures use the configured palette/format/dpi (via figures.style) and the
minimisation convention.  Public entry points are imported in figures/__init__.

Saving: each function calls ``save_figure(fig, dir, stem, cfg)`` which writes one
file per configured format.  Crashed/NaN runs are dropped before plotting.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from .style import save_figure, get_method_styles, PALETTE


# ── small helpers ─────────────────────────────────────────────────────────────
def _palette(cfg):
    return (cfg or {}).get("figures", {}).get("palette", PALETTE)


def _figsize(cfg, key="figsize_default"):
    return (cfg or {}).get("figures", {}).get(key, (8, 5))


def _clean(vals):
    return [v for v in vals if v is not None and np.isfinite(v)]


def _labels(problem, names):
    return {m: problem.label(m) for m in names}


def _boxplot(ax, data, tick_labels, **kw):
    """ax.boxplot with category labels, for every Matplotlib version
    (`labels` was renamed `tick_labels` in 3.9 and is removed in 3.11)."""
    try:
        return ax.boxplot(data, tick_labels=tick_labels, **kw)
    except TypeError:                     # Matplotlib < 3.9
        return ax.boxplot(data, labels=tick_labels, **kw)


# ══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUTION:  box plot over all instances; box and violin plots per size class
# ══════════════════════════════════════════════════════════════════════════════
def plot_comparison_block(scores, problem, cfg, paths, s, ylabel, verbose=True):
    names = list(scores.keys())
    ids = sorted({i for m in scores for i in scores[m]})
    lab = _labels(problem, names)
    styles = get_method_styles(names, _palette(cfg))

    _box(scores, names, ids, lab, styles, cfg, paths.get(f"{s}/box"), ylabel)
    if problem.size_classes:
        _per_size(scores, problem, names, lab, styles, cfg, paths, s, ylabel)


def _box(scores, names, ids, lab, styles, cfg, out, ylabel):
    if not out:
        return
    pooled = {m: [v for i in ids for v in _clean(scores[m].get(i, []))] for m in names}
    fig, ax = plt.subplots(figsize=_figsize(cfg))
    data = [pooled[m] if pooled[m] else [np.nan] for m in names]
    bp = _boxplot(ax, data, [lab[m] for m in names], patch_artist=True,
                    showmeans=True)
    for patch, m in zip(bp["boxes"], names):
        patch.set_facecolor(styles[m]["color"]); patch.set_alpha(0.6)
    ax.set_ylabel(ylabel); ax.set_title(f"{ylabel}: distribution (all instances)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save_figure(fig, out, "box_overall", cfg)


def _per_size(scores, problem, names, lab, styles, cfg, paths, s, ylabel):
    box_out = paths.get(f"{s}/persize/box")
    vio_out = paths.get(f"{s}/persize/violin")
    for cls, inst_ids in problem.size_classes.items():
        pooled = {m: [v for i in inst_ids for v in _clean(scores[m].get(i, []))]
                  for m in names}
        if not any(pooled.values()):
            continue
        tag = cls.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')
        if box_out:
            fig, ax = plt.subplots(figsize=_figsize(cfg))
            data = [pooled[m] if pooled[m] else [np.nan] for m in names]
            bp = _boxplot(ax, data, [lab[m] for m in names], patch_artist=True,
                            showmeans=True)
            for patch, m in zip(bp["boxes"], names):
                patch.set_facecolor(styles[m]["color"]); patch.set_alpha(0.6)
            ax.set_ylabel(ylabel); ax.set_title(f"{ylabel}: {cls}")
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            save_figure(fig, box_out, f"box_{tag}", cfg)
        if vio_out:
            valid = [m for m in names if pooled[m]]
            if valid:
                fig, ax = plt.subplots(figsize=_figsize(cfg))
                parts = ax.violinplot([pooled[m] for m in valid], showmeans=True)
                for pc, m in zip(parts["bodies"], valid):
                    pc.set_facecolor(styles[m]["color"]); pc.set_alpha(0.5)
                ax.set_xticks(np.arange(1, len(valid) + 1))
                ax.set_xticklabels([lab[m] for m in valid], rotation=20, ha="right")
                ax.set_ylabel(ylabel); ax.set_title(f"{ylabel}: {cls}")
                save_figure(fig, vio_out, f"violin_{tag}", cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERGENCE  (shared min-max normalisation across methods, per instance)
# ══════════════════════════════════════════════════════════════════════════════
def plot_convergence_curves(conv_results, problem, cfg, out_dir, verbose=True):
    """
    Hypervolume convergence (image-2 style). Methods record the FRONT per
    generation; here we compute HV against a single fixed reference per instance
    (shared ideal/nadir over all methods/runs/generations) -> monotone, concave,
    comparable curves. Falls back to scalar progress if fronts are not provided.
    """
    if not out_dir:
        return
    from ..utils.hypervolume import convergence_bounds, hv_series

    names = list(conv_results.keys())
    lab = _labels(problem, names)
    styles = get_method_styles(names, _palette(cfg))
    ids = sorted({i for m in conv_results for i in conv_results[m]})

    for inst_id in ids:
        # gather raw progress per method
        raw = {}
        for m in names:
            runs = [r.get("progress") for r in conv_results[m].get(inst_id, [])
                    if r.get("progress") and not r.get("_crashed") and not r.get("_timeout")]
            if runs:
                raw[m] = runs
        if not raw:
            continue

        fronts_mode = _is_fronts(next(iter(raw.values()))[0])

        if fronts_mode:
            # fixed shared reference over every point of this instance
            flat = [fr for runs in raw.values() for series in runs for fr in series]
            ideal, nadir = convergence_bounds(flat)
            curves = {m: [hv_series(series, ideal, nadir) for series in runs]
                      for m, runs in raw.items()}
            ylabel = "Hypervolume"
            ylim = None
        else:
            # scalar progress: normalise to [0,1] across methods (1 = best)
            allv = [v for runs in raw.values() for c in runs for v in c if np.isfinite(v)]
            gmin, gmax = (min(allv), max(allv)) if allv else (0.0, 1.0)
            span = (gmax - gmin) or 1.0
            hib = problem.is_multiobjective
            f = (lambda v: (v - gmin) / span) if hib else (lambda v: 1.0 - (v - gmin) / span)
            curves = {m: [[f(v) for v in c] for c in runs] for m, runs in raw.items()}
            ylabel = "Normalised quality (1 = best)"
            ylim = (-0.05, 1.08)

        fig, ax = plt.subplots(figsize=_figsize(cfg, "figsize_wide"))
        for m in names:
            if m not in curves:
                continue
            runs = curves[m]
            maxlen = max(len(c) for c in runs)
            padded = np.array([np.pad(np.asarray(c, float), (0, maxlen - len(c)),
                                      mode="edge") if len(c) < maxlen else np.asarray(c, float)
                               for c in runs], dtype=float)
            mean = np.nanmean(padded, axis=0)
            lo = np.nanpercentile(padded, 25, axis=0)
            hi = np.nanpercentile(padded, 75, axis=0)
            gx = np.arange(1, maxlen + 1)
            ax.plot(gx, mean, color=styles[m]["color"], linestyle=styles[m]["linestyle"],
                    label=lab[m], linewidth=1.8)
            ax.fill_between(gx, lo, hi, color=styles[m]["color"], alpha=0.18)
        ax.set_xlabel("Generations"); ax.set_ylabel(ylabel)
        ax.set_title(f"Instance {inst_id}")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(loc="lower right", ncol=1)
        ax.grid(True, linestyle="--", alpha=0.5)
        save_figure(fig, out_dir, f"convergence_inst{inst_id}", cfg)


def _is_fronts(first_entry):
    """A progress entry is a 'front' if it is a list of points (list of lists)."""
    if isinstance(first_entry, (list, tuple)):
        if len(first_entry) == 0:
            return True            # empty front
        e0 = first_entry[0]
        return isinstance(e0, (list, tuple, np.ndarray))
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  PARETO FRONTS  (2-D scatter; 3-D scatter for K = 3)
# ══════════════════════════════════════════════════════════════════════════════
def plot_pareto_aggregated(eval_results, problem, cfg, out_dir, verbose=True):
    """
    One figure per instance overlaying the fronts of ALL executions (runs).
    Each method keeps a single colour across all its executions; points are not
    joined. K=2 -> 2-D scatter, K=3 -> 3-D scatter, K>3 -> parallel coordinates.
    Saved under <out_dir>/<FMT>/Aggregated/instance_<id>.
    """
    if not out_dir or not problem.is_multiobjective:
        return
    names = list(eval_results.keys())
    lab = _labels(problem, names)
    styles = get_method_styles(names, _palette(cfg))
    onames = problem.objective_names
    ids = sorted({i for m in eval_results for i in eval_results[m]})

    for inst_id in ids:
        stacked = {}      # method -> stacked points across executions
        for m in names:
            chunks = []
            for r in eval_results[m].get(inst_id, []):
                if r.get("_crashed") or r.get("_timeout"):
                    continue
                pts = r.get("pareto_objectives") or []
                if pts:
                    chunks.append(np.asarray(pts, dtype=float))
            if chunks:
                stacked[m] = np.vstack(chunks)
        if not stacked:
            continue
        K = next(iter(stacked.values())).shape[1]
        title = f"Instance {inst_id}: all executions"
        stem = f"instance_{inst_id}"
        if K == 2:
            _scatter_2d(stacked, names, lab, styles, onames, cfg, out_dir,
                        "Aggregated", stem, title)
        elif K == 3:
            _scatter_3d(stacked, names, lab, styles, onames, cfg, out_dir,
                        "Aggregated", stem, title)
        else:
            _parallel(stacked, names, lab, styles, onames, cfg, out_dir,
                      "Aggregated", stem, title, K)


def _scatter_2d(fronts, names, lab, styles, onames, cfg, out_dir, subdir, stem, title):
    fig, ax = plt.subplots(figsize=_figsize(cfg, "figsize_square"))
    for m in names:
        if m not in fronts:
            continue
        f = fronts[m]
        ax.scatter(f[:, 0], f[:, 1], s=28, color=styles[m]["color"],
                   marker=styles[m]["marker"], label=lab[m], alpha=0.8,
                   edgecolors="white", linewidths=0.4)
    ax.set_xlabel(onames[0]); ax.set_ylabel(onames[1])
    ax.set_title(title); ax.legend()
    save_figure(fig, out_dir, stem, cfg, subdir=subdir)


def _scatter_3d(fronts, names, lab, styles, onames, cfg, out_dir, subdir, stem, title):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=_figsize(cfg, "figsize_square"))
    ax = fig.add_subplot(111, projection="3d")
    for m in names:
        if m not in fronts:
            continue
        f = fronts[m]
        ax.scatter(f[:, 0], f[:, 1], f[:, 2], s=22, color=styles[m]["color"],
                   marker=styles[m]["marker"], label=lab[m], alpha=0.8,
                   depthshade=True, edgecolors="white", linewidths=0.3)
    ax.set_xlabel(onames[0], fontsize=8, labelpad=6)
    ax.set_ylabel(onames[1], fontsize=8, labelpad=6)
    ax.set_zlabel(onames[2], fontsize=8, labelpad=6)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=22, azim=-50)
    save_figure(fig, out_dir, stem, cfg, subdir=subdir)


def _parallel(fronts, names, lab, styles, onames, cfg, out_dir, subdir, stem, title, K):
    allpts = np.vstack(list(fronts.values()))
    mn, mx = allpts.min(axis=0), allpts.max(axis=0)
    span = np.where(mx - mn == 0, 1.0, mx - mn)
    xs = np.arange(K)
    fig, ax = plt.subplots(figsize=_figsize(cfg, "figsize_wide"))
    for m in names:
        if m not in fronts:
            continue
        norm = (fronts[m] - mn) / span
        for row in norm:
            ax.plot(xs, row, color=styles[m]["color"], alpha=0.35, linewidth=0.8)
        ax.plot([], [], color=styles[m]["color"], label=lab[m])
    ax.set_xticks(xs); ax.set_xticklabels(onames, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Normalised objective (min)")
    ax.set_title(title); ax.legend()
    save_figure(fig, out_dir, stem, cfg, subdir=subdir)


# ══════════════════════════════════════════════════════════════════════════════
#  RANK HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def plot_rank_heatmap(scores, problem, cfg, out_dir, higher_is_better, title,
                      verbose=True, stem=None):
    if not out_dir:
        return
    names = list(scores.keys())
    lab = [problem.label(m) for m in names]
    ids = sorted({i for m in scores for i in scores[m]})
    mat = np.full((len(names), len(ids)), np.nan)
    for j, i in enumerate(ids):
        col = []
        for m in names:
            c = _clean(scores[m].get(i, []))
            col.append(np.mean(c) if c else np.nan)
        col = np.array(col)
        order = (-col if higher_is_better else col)
        valid = ~np.isnan(col)
        from scipy.stats import rankdata
        ranks = np.full(len(names), np.nan)
        if valid.any():
            ranks[valid] = rankdata(order[valid])
        mat[:, j] = ranks
    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.25), 0.6 * len(names) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(lab)
    ax.set_xticks(range(len(ids))); ax.set_xticklabels([str(i) for i in ids],
                                                       fontsize=6, rotation=90)
    ax.set_xlabel("Instance"); ax.set_title(f"{title}: per-instance rank (1 = best)")
    fig.colorbar(im, ax=ax, label="Rank")
    safe = stem or ("rank_" + re.sub(r"[^0-9a-zA-Z]+", "_", title).strip("_").lower())
    save_figure(fig, out_dir, safe, cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE PROFILES (Dolan–Moré)
# ══════════════════════════════════════════════════════════════════════════════
def profile_ratios(scores, names, ids, higher_is_better):
    """Dolan-More performance ratios {method: [r per instance]} (r >= 1,
    1 = best); see `plot_performance_profiles`."""
    ratios = {m: [] for m in names}
    for i in ids:
        col = {}
        for m in names:
            c = _clean(scores[m].get(i, []))
            if c:
                col[m] = np.median(c)
        if not col:
            continue                      # every method missing this instance
        if higher_is_better:
            best = max(col.values())
            for m, v in col.items():
                r = best / v if v > 0 else np.nan
                if np.isfinite(r):
                    ratios[m].append(r)
        else:
            # Ratios v / best need positive costs. A cost metric that can be
            # zero or negative (a gap of 0 at the optimum, a negated
            # objective) is shifted on this instance so that its best value
            # becomes 1: r = (v - best + 1).
            best = min(col.values())
            for m, v in col.items():
                r = v / best if best > 0 else v - best + 1.0
                if np.isfinite(r):
                    ratios[m].append(r)

    return ratios


def plot_performance_profiles(scores, problem, cfg, out_dir, higher_is_better,
                              metric_label, verbose=True):
    """
    Dolan-More performance profile, robust to a fully-broken method.

    A method missing data on SOME instances is simply excluded from those
    instances' ratio computation -- it does not block every other method
    from being profiled. Each method's curve is normalised by the TOTAL
    instance count (not just the instances it has data for), so a method
    that fails on part of the benchmark is visibly penalised (its curve
    cannot reach 1.0) rather than silently disappearing along with everyone
    else's.
    """
    if not out_dir:
        return
    names = list(scores.keys())
    lab = _labels(problem, names)
    styles = get_method_styles(names, _palette(cfg))
    ids = sorted({i for m in scores for i in scores[m]})
    if not ids:
        return

    ratios = profile_ratios(scores, names, ids, higher_is_better)

    all_ratios = [r for m in names for r in ratios[m]]
    if not all_ratios:
        return
    tau_max = max(all_ratios)
    if not np.isfinite(tau_max) or tau_max <= 1:
        tau_max = 2.0
    taus = np.linspace(1.0, tau_max * 1.05, 200)

    n_instances = len(ids)
    fig, ax = plt.subplots(figsize=_figsize(cfg))
    for m in names:
        r = np.asarray(ratios[m], dtype=float)
        if len(r) == 0:
            continue
        # denominator is the TOTAL instance count, so instances a method has
        # no data for still count against it (its curve tops out below 1.0)
        rho = [np.sum(r <= t) / n_instances for t in taus]
        ax.step(taus, rho, where="post", color=styles[m]["color"], label=lab[m],
                linewidth=1.6)
    ax.set_xlabel(r"$\tau$  (within this factor of the best method)")
    ax.set_ylabel("Fraction of instances")
    ax.set_title(f"Performance profile: {metric_label}")
    ax.set_ylim(-0.02, 1.02); ax.legend(loc="lower right")
    save_figure(fig, out_dir, "performance_profile", cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  CRITICAL DIFFERENCE DIAGRAM (Nemenyi) : self-contained, no extra deps
# ══════════════════════════════════════════════════════════════════════════════
def plot_critical_difference(avg_ranks, cd, cfg, out_dir, title, verbose=True):
    if not out_dir or not avg_ranks or not np.isfinite(cd):
        return
    items = sorted(avg_ranks.items(), key=lambda x: x[1])
    methods = [m for m, _ in items]
    ranks = [r for _, r in items]
    k = len(methods)
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8, 0.5 * k + 2.2))
    ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(0, k + 2)
    ax.axis("off")

    y_axis = k + 1
    ax.hlines(y_axis, lo, hi, color="black", linewidth=1.2)
    for t in range(lo, hi + 1):
        ax.vlines(t, y_axis - 0.1, y_axis + 0.1, color="black")
        ax.text(t, y_axis + 0.25, str(t), ha="center", fontsize=9)
    ax.text((lo + hi) / 2, y_axis + 0.7, "Average rank (1 = best)",
            ha="center", fontsize=10)

    half = k // 2
    for idx, (m, r) in enumerate(items):
        if idx < half:
            y = y_axis - 1 - idx
            ax.plot([r, lo], [y_axis, y], color="black", linewidth=0.8)
            ax.text(lo - 0.05, y, f"{m}  ({r:.2f})", ha="right", va="center", fontsize=9)
        else:
            y = y_axis - 1 - (k - 1 - idx)
            ax.plot([r, hi], [y_axis, y], color="black", linewidth=0.8)
            ax.text(hi + 0.05, y, f"({r:.2f})  {m}", ha="left", va="center", fontsize=9)

    # CD bar
    ax.hlines(y_axis + 1.4, lo, lo + cd, color="black", linewidth=2.0)
    ax.text(lo + cd / 2, y_axis + 1.55, f"CD = {cd:.3f}", ha="center", fontsize=9)

    # connect statistically-indistinguishable groups (rank gap < CD)
    level = 0.15
    last_j = -1
    for i in range(k):
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) < cd:
            j += 1
        if j > i and j > last_j:          # maximal groups only (no nested bars)
            ax.hlines(y_axis - 0.25 - level, ranks[i] - 0.03, ranks[j] + 0.03,
                      color="firebrick", linewidth=3.0)
            last_j = j; level += 0.18
    ax.set_title(title, fontsize=12)
    save_figure(fig, out_dir, "critical_difference", cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNIFICANCE HEATMAP (Wilcoxon p-values + Cliff's delta)
# ══════════════════════════════════════════════════════════════════════════════
def plot_significance_heatmap(wx, cfg, out_dir, title, verbose=True):
    if not out_dir:
        return
    names = wx.get("method_names", [])
    pmat = np.array(wx.get("p_matrix", []), dtype=float)
    dmat = np.array(wx.get("cliff_delta_matrix", []), dtype=float)
    if pmat.size == 0 or len(names) < 2:
        return
    corr = wx.get("corrected_alpha", 0.05)
    sm = wx.get("significant_matrix")
    sigm = np.array(sm, dtype=bool) if sm else None
    n = len(names)
    fig, ax = plt.subplots(figsize=(0.9 * n + 3, 0.9 * n + 2))
    disp = np.where(np.eye(n, dtype=bool), np.nan, pmat)
    im = ax.imshow(disp, cmap="viridis_r", vmin=0, vmax=max(corr * 2, 1e-6))
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="gray")
            else:
                star = "*" if (sigm[i, j] if sigm is not None else
                               (np.isfinite(pmat[i, j]) and pmat[i, j] < corr)) else ""
                ax.text(j, i, f"p={pmat[i, j]:.3g}{star}\nδ={dmat[i, j]:+.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if (sigm[i, j] if sigm is not None
                                          else pmat[i, j] < corr) else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    corr_name = wx.get("correction", "bonferroni")
    ax.set_title(f"{title}\n(* significant after {corr_name} correction; "
                 f"δ = Cliff's delta, row vs col)", fontsize=10)
    fig.colorbar(im, ax=ax, label="Wilcoxon p-value")
    save_figure(fig, out_dir, "significance_heatmap", cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY-VS-TIME TRADE-OFF  (meta-Pareto view of the solvers themselves)
# ══════════════════════════════════════════════════════════════════════════════
def plot_quality_vs_time(scores_obj, scores_time, problem, cfg, out_dir, verbose=True):
    """
    One point per method: mean quality (HV for multi-objective, higher is
    better; objective value for mono, lower is better) vs mean execution
    time, pooled across instances. Distinct from the per-instance Pareto
    front plots -- here each METHOD is a point, not each solution.
    """
    if not out_dir:
        return
    names = list(scores_obj.keys())
    lab = _labels(problem, names)
    styles = get_method_styles(names, _palette(cfg))
    hib = problem.is_multiobjective
    ylabel = ("Mean hypervolume (higher is better)" if hib
             else "Mean objective value (lower is better)")

    def _mean_pair(ids):
        pts = {}
        for m in names:
            qv = _clean([v for i in ids for v in scores_obj[m].get(i, [])])
            tv = _clean([v for i in ids for v in scores_time[m].get(i, [])])
            if qv and tv:
                pts[m] = (float(np.mean(tv)), float(np.mean(qv)))
        return pts

    def _draw(pts, title, stem):
        if not pts:
            return
        fig, ax = plt.subplots(figsize=_figsize(cfg))
        for m, (t, q) in pts.items():
            ax.scatter(t, q, s=100, color=styles[m]["color"], marker=styles[m]["marker"],
                      edgecolors="white", linewidths=0.9, zorder=3)
            ax.annotate(lab[m], (t, q), textcoords="offset points", xytext=(7, 5),
                       fontsize=8.5)
        ax.set_xlabel("Mean execution time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save_figure(fig, out_dir, stem, cfg)

    all_ids = sorted({i for m in names for i in scores_obj[m]})
    _draw(_mean_pair(all_ids), "Quality vs. time trade-off: all instances",
          "quality_vs_time")


