"""
features.py  —  scaling analysis (Appendix D: execution time against n).

Needs numeric instance descriptors: problem_dict["instance_features"]
(callable(instance) -> {name: number}), evaluated once per instance by the
runner and saved in the results metadata.

Scaling (Results Analysis/Scaling/):
  for every chosen block and feature (config["scaling"]), the per-instance
  mean of each method against the feature (instances sharing a feature value
  are averaged, with their min-max range), log axes when the values span more
  than two decades. For the time block a power law time = a * feature^b is
  fitted per method by least squares on the logs; b is the empirical growth
  exponent ("Scaling fits.csv").
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ..utils.io import write_csv
from ..figures.style import save_figure, get_method_styles


def _means(scores, m, ids):
    out = []
    for i in ids:
        c = [v for v in scores[m].get(i, []) if v is not None and np.isfinite(v)]
        out.append(float(np.mean(c)) if c else np.nan)
    return np.asarray(out)


def _feature_matrix(feat_vals, ids):
    names = sorted({k for i in ids for k, v in (feat_vals.get(i) or {}).items()
                    if isinstance(v, (int, float, np.integer, np.floating))
                    and not isinstance(v, bool)})
    X = np.array([[float((feat_vals.get(i) or {}).get(k, np.nan)) for k in names]
                  for i in ids], dtype=float)
    return names, X


# ── scaling ──────────────────────────────────────────────────────────────────
def scaling_analysis(problem, blocks, cfg, out_dir, verbose=True):
    """`blocks`: list of (block_name, scores, ylabel, hib)."""
    fv = getattr(problem, "instance_feature_values", None) or {}
    names_cfg = cfg.get("scaling", {}).get("features")
    if not blocks:
        return
    any_scores = blocks[0][1]
    methods = list(any_scores.keys())
    ids = sorted({i for m in methods for i in any_scores[m]})
    styles = get_method_styles(methods, cfg.get("figures", {}).get("palette"))
    figsize = cfg.get("figures", {}).get("figsize_default", (8, 5))

    if fv:
        fnames, X = _feature_matrix(fv, ids)
        if names_cfg:
            fnames_sel = [f for f in fnames if f in names_cfg]
        else:
            fnames_sel = fnames
        axes = [(f, X[:, fnames.index(f)], False) for f in fnames_sel]
    elif problem.size_classes:
        order = list(problem.size_classes)
        pos = {}
        for k, c in enumerate(order):
            for i in problem.size_classes[c]:
                pos.setdefault(i, k)
        axes = [("size class", np.array([pos.get(i, np.nan) for i in ids], float), order)]
    else:
        return
    os.makedirs(out_dir, exist_ok=True)
    fits = []
    for fname, x, cats in axes:
        ok_x = np.isfinite(x)
        if ok_x.sum() < 2:
            continue
        for block, scores, ylabel, hib in blocks:
            fig, ax = plt.subplots(figsize=figsize)
            for m in methods:
                y = _means(scores, m, ids)
                keep = ok_x & np.isfinite(y)
                if not keep.any():
                    continue
                ux = np.unique(x[keep])
                my = np.array([np.mean(y[keep][x[keep] == u]) for u in ux])
                lo = np.array([np.min(y[keep][x[keep] == u]) for u in ux])
                hi = np.array([np.max(y[keep][x[keep] == u]) for u in ux])
                ax.plot(ux, my, marker=styles[m]["marker"], color=styles[m]["color"],
                        label=problem.label(m), linewidth=1.4, markersize=4)
                if len(ux) < len(x[keep]):
                    ax.fill_between(ux, lo, hi, color=styles[m]["color"], alpha=0.12)
                if "time" in block.lower() and cats is False:
                    pos = keep & (x > 0) & (y > 0)
                    if pos.sum() >= 3 and np.ptp(np.log(x[pos])) > 0:
                        b, a = np.polyfit(np.log(x[pos]), np.log(y[pos]), 1)
                        r = np.corrcoef(np.log(x[pos]), np.log(y[pos]))[0, 1]
                        fits.append({"Feature": fname, "Method": problem.label(m),
                                     "Exponent b": round(float(b), 4),
                                     "Coefficient a": float(np.exp(a)),
                                     "R2": round(float(r * r), 4), "N": int(pos.sum())})
            if cats is not False:
                ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=20, ha="right")
            else:
                xs = x[ok_x]
                if np.all(xs > 0) and xs.max() / xs.min() > 100:
                    ax.set_xscale("log")
            ax.set_xlabel(fname); ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs {fname}")
            ax.legend(ncol=min(len(methods), 3))
            from ..utils.io import slug
            save_figure(fig, out_dir, f"scaling_{slug(block)}_{slug(str(fname))}", cfg)
    if fits:
        write_csv(fits, os.path.join(out_dir, "Scaling fits.csv"),
                  header=list(fits[0].keys()), code=problem.code)
