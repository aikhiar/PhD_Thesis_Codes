"""
extra.py  —  figures of the anytime analysis (Appendix D): the mean primal gap
over time over all instances, and the time-to-target curves.
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import save_figure, get_method_styles


def _figsize(cfg, key="figsize_default"):
    return (cfg or {}).get("figures", {}).get(key, (8, 5))


# ── mono: primal gap vs time ─────────────────────────────────────────────────
def plot_anytime_curves(any_res, refs, problem, cfg, out_dir, verbose=True):
    """Mean over the instances of the per-instance mean primal gap (%) of the
    best-so-far value, against the time normalised by the instance horizon
    (the longest run on the instance), one line per method."""
    from ..analysis.anytime import primal_gap
    traces, horizon = any_res["traces"], any_res["horizon"]
    names = list(traces.keys())
    styles = get_method_styles(names, (cfg or {}).get("figures", {}).get("palette"))
    ids = sorted({i for m in names for i in traces[m]})
    pooled = {m: [] for m in names}
    u = np.linspace(0, 1, 200)
    for i in ids:
        H = horizon.get(i, np.nan)
        fstar = refs.get(i, np.nan)
        firsts = [tr["t"][0] for m in names for tr in traces[m][i] if tr is not None]
        if not np.isfinite(H) or not firsts or not np.isfinite(fstar):
            continue
        for m in names:
            runs = [tr for tr in traces[m][i] if tr is not None]
            if not runs:
                continue
            kk = [np.searchsorted(tr["t"], u * H, side="right") - 1 for tr in runs]
            pc = [np.where(k >= 0, primal_gap(tr["f"][np.maximum(k, 0)], fstar), 1.0)
                  for k, tr in zip(kk, runs)]
            pooled[m].append(100.0 * np.mean(pc, axis=0))
    if any(pooled.values()):
        fig, ax = plt.subplots(figsize=_figsize(cfg))
        for m in names:
            if pooled[m]:
                ax.plot(u, np.mean(pooled[m], axis=0), color=styles[m]["color"],
                        linestyle=styles[m]["linestyle"], label=problem.label(m), linewidth=1.8)
        ax.set_xlabel("Time / horizon of the instance")
        ax.set_ylabel("Mean primal gap (%)")
        ax.set_title("Anytime performance over all instances")
        ax.legend()
        save_figure(fig, out_dir, "anytime_all_instances", cfg)


# ── mono: time-to-target ─────────────────────────────────────────────────────
def plot_time_to_target(any_res, problem, cfg, out_dir, verbose=True):
    """Per target: share of runs (all instances x runs) that reached the
    target gap within time t (runs that never did keep the curve below 1)."""
    names = list(any_res["traces"].keys())
    styles = get_method_styles(names, (cfg or {}).get("figures", {}).get("palette"))
    for tgt, tt in any_res["ttt"].items():
        ids = sorted({i for m in names for i in tt[m]})
        alls = [h for m in names for i in ids for h, _ in tt[m][i] if np.isfinite(h)]
        if not alls:
            continue
        tmin, tmax = max(min(alls), 1e-6), max(alls)
        fig, ax = plt.subplots(figsize=_figsize(cfg))
        drawn = False
        for m in names:
            pairs = [p for i in ids for p in tt[m][i]]
            if not pairs:
                continue
            hits = np.sort([h for h, _ in pairs if np.isfinite(h)])
            n = len(pairs)
            if len(hits):
                x = np.concatenate([[tmin * 0.9], hits, [tmax * 1.1]])
                y = np.concatenate([[0], np.arange(1, len(hits) + 1) / n, [len(hits) / n]])
            else:
                x, y = np.array([tmin * 0.9, tmax * 1.1]), np.zeros(2)
            ax.step(x, y, where="post", color=styles[m]["color"],
                    linestyle=styles[m]["linestyle"], label=problem.label(m), linewidth=1.6)
            drawn = True
        if not drawn:
            plt.close(fig)
            continue
        if tmax / tmin > 20:
            ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Share of runs at or below the target")
        ax.set_title(f"Time to target: gap <= {tgt:g}% (all instances)")
        ax.legend(loc="lower right")
        tag = f"{tgt:g}".replace(".", "_")
        save_figure(fig, out_dir, f"ttt_target_{tag}", cfg)
