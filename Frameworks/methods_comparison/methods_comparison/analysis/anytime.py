"""
anytime.py  —  anytime (time-resolved) performance of the mono-objective
methods against the exact reference (Appendix D).

A run's ANYTIME TRACE is its best-so-far objective as a step function of
wall-clock time. It is built from the run's `progress` (best objective per
generation) and `progress_time` (elapsed seconds at the end of each
generation, recorded by the solvers). Without `progress_time`, generation g of
G is placed at exec_time * g / G (marked "approximate"); without `progress`,
the trace is the single point (exec_time, best_objective). A crashed or
timed-out run has no trace.

From the traces, against the per-instance reference f* (the exact value, the
same as the Gap block):

  * the mean primal gap over time (figures/extra.py; Berthold 2013);
  * TIME TO TARGET: first time the gap reaches each target (anytime.targets,
    in %). Gives the success rate SR (share of runs reaching the target) and
    the expected running time ERT = (sum over runs of the time spent, i.e.
    hit time or full run time) / (number of successful runs) (Auger & Hansen
    2005); ERT is infinite when no run succeeds.
"""

import os
import numpy as np

from ..utils.io import write_csv


# ── traces ───────────────────────────────────────────────────────────────────
def _scalar_progress(prog):
    if not isinstance(prog, (list, tuple, np.ndarray)) or len(prog) == 0:
        return None
    try:
        arr = np.asarray(prog, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.ndim == 1 else None


def run_trace(r):
    """{t, f, end, approx} for one run, or None (crashed / no value)."""
    if r.get("_crashed") or r.get("_timeout"):
        return None
    final = r.get("best_objective", np.nan)
    final = float(final) if final is not None else np.nan
    et = r.get("exec_time", np.nan)
    et = float(et) if et is not None else np.nan
    prog = _scalar_progress(r.get("progress"))
    approx = False
    if prog is not None:
        pt = r.get("progress_time")
        if pt is not None and len(pt) == len(prog):
            t = np.asarray(pt, dtype=float)
        elif np.isfinite(et):
            t = et * np.arange(1, len(prog) + 1) / len(prog)
            approx = True
        else:
            prog = None
    if prog is None:
        if not (np.isfinite(final) and np.isfinite(et)):
            return None
        return {"t": np.array([et]), "f": np.array([final]), "end": et, "approx": False}
    f = np.where(np.isfinite(prog), prog, np.inf)
    f = np.minimum.accumulate(f)
    end = max(float(t[-1]), et) if np.isfinite(et) else float(t[-1])
    if np.isfinite(final) and final < f[-1] - 1e-12 * max(1.0, abs(final)):
        t = np.append(t, end); f = np.append(f, final)      # e.g. a final local search
    keep = np.isfinite(f)
    if not keep.any():
        return None
    return {"t": t[keep], "f": f[keep], "end": end, "approx": approx}


def primal_gap(f, fstar):
    """Primal gap (Berthold 2013): 0 if f = f* = 0; 1 if there is no solution
    yet or f and f* have opposite signs; else |f - f*| / max(|f|, |f*|)."""
    f = np.asarray(f, dtype=float)
    out = np.ones_like(f)
    fin = np.isfinite(f)
    if not np.isfinite(fstar):
        return np.full_like(f, np.nan)
    both0 = fin & (f == 0) & (fstar == 0)
    opp = fin & (f * fstar < 0)
    den = np.maximum(np.abs(f), abs(fstar))
    ok = fin & ~both0 & ~opp & (den > 0)
    out[ok] = np.abs(f[ok] - fstar) / den[ok]
    out[both0] = 0.0
    return out


def rel_gap(f, fstar):
    """Relative gap in %, as in the Gap block."""
    f = np.asarray(f, dtype=float)
    if not np.isfinite(fstar):
        return np.full_like(f, np.nan)
    d = abs(fstar)
    if d > 1e-12:
        return 100.0 * (f - fstar) / d
    return np.where(f == fstar, 0.0, np.nan)


def hit_time(tr, fstar, target):
    """First time the relative gap is <= target (%); None if never."""
    if tr is None:
        return None
    g = rel_gap(tr["f"], fstar)
    idx = np.where(np.isfinite(g) & (g <= target + 1e-9))[0]
    return float(tr["t"][idx[0]]) if len(idx) else None


# ── main entry ───────────────────────────────────────────────────────────────
def compute_anytime(problem, eval_results, refs, cfg, out_dir, verbose=True):
    """Returns {"traces", "horizon", "ttt": {target: {m: {i: [(hit or nan, end)]}}},
    "approx": bool}. Writes the success-rate / ERT tables."""
    acfg = cfg.get("anytime", {})
    names = list(eval_results.keys())
    ids = sorted({i for m in names for i in eval_results[m]})
    traces = {m: {i: [run_trace(r) for r in eval_results[m].get(i, [])] for i in ids}
              for m in names}
    approx = any(tr is not None and tr["approx"] for m in names for i in ids
                 for tr in traces[m][i])

    # time axis of an instance: the longest run on it (any method)
    horizon = {}
    for i in ids:
        ends = [tr["end"] for m in names for tr in traces[m][i] if tr is not None]
        horizon[i] = max(ends) if ends else np.nan

    out = {"traces": traces, "horizon": horizon, "approx": approx, "ttt": {}}

    data_dir = os.path.join(out_dir, "Data")
    for tgt in acfg.get("targets") or []:
        tt = {m: {i: [] for i in ids} for m in names}
        rows = []
        tot = {m: [0, 0, 0.0] for m in names}            # hits, runs, time spent
        for i in ids:
            row = {"Instance": i}
            for m in names:
                hits, spent = 0, 0.0
                runs = eval_results[m].get(i, [])
                for tr, r in zip(traces[m][i], runs):
                    end = tr["end"] if tr is not None else float(r.get("exec_time", np.nan) or np.nan)
                    h = hit_time(tr, refs.get(i, np.nan), tgt)
                    tt[m][i].append((h if h is not None else np.nan, end))
                    if h is not None:
                        hits += 1; spent += h
                    elif np.isfinite(end):
                        spent += end
                n = len(runs)
                tot[m][0] += hits; tot[m][1] += n; tot[m][2] += spent
                sr = 100.0 * hits / n if n else np.nan
                ert = spent / hits if hits else np.inf
                row[problem.label(m)] = f"{sr:.0f}% / {_ert(ert)}"
            rows.append(row)
        rows.append({"Instance": "All", **{problem.label(m): (
            f"{100.0 * tot[m][0] / tot[m][1]:.0f}% / {_ert(tot[m][2] / tot[m][0] if tot[m][0] else np.inf)}"
            if tot[m][1] else "n/a") for m in names}})
        write_csv(rows, os.path.join(data_dir, f"Success rate and ERT (target {tgt:g}%).csv"),
                  header=list(rows[0].keys()), code=problem.code)
        out["ttt"][tgt] = tt
        from .report_tables import write_latex_table, tex_escape
        hdr = list(rows[0].keys())
        write_latex_table([tex_escape(h) for h in hdr],
                          [[tex_escape(r[h]) for h in hdr] for r in rows],
                          os.path.join(data_dir, f"{problem.code}_Success rate and ERT (target {tgt:g}%).tex"),
                          caption=tex_escape(f"Success rate and expected running time "
                                             f"(target gap {tgt:g}%): {problem.name}"),
                          label=f"tab:{problem.code}_sr_ert_{str(tgt).replace('.', '_')}",
                          note=("SR / ERT: share of runs reaching the target gap, and "
                                "expected running time in seconds (total time spent by "
                                "all runs divided by the number of successful runs; inf "
                                "if none succeeded)."))
    if approx and verbose:
        print("  [Anytime] note: some methods return no `progress_time`; their "
              "generation times are approximated as exec_time * g / G.", flush=True)
    return out


def _ert(v):
    if not np.isfinite(v):
        return "inf"
    return f"{v:.3g}s"
