"""
Per-run metrics computed against per-instance references.

References per instance
  LB*   best valid lower bound: certified LB of the context, Gurobi dual bounds of every run,
        the bound of the exact reference
  BKS   best known solution value: best feasible objective of every run and of the reference
  OPT   proven optimum (exact reference proven, or BKS = LB*); NaN otherwise
Metrics per run
  gap_lb    100 (obj - LB*) / LB*          gap to the best lower bound (certified gap)
  gap_bks   100 (obj - BKS) / BKS          gap to the best known solution
  gap_opt   100 (obj - OPT) / OPT          where OPT is known
  gap_cert  100 (obj - LB_cert) / LB_cert  gap to our own certified lower bound (no MILP involved)
  gap_lp    100 (obj - P_LP) / P_LP        gap to the LP relaxation bound
  gap_ref   100 (obj - REF) / REF          relative difference to the exact solver's incumbent (the
                                           measure used with the earlier methods; NOT an optimality gap unless
                                           the reference is proven)
  hit_cert  obj <= LB_cert   (proved optimal by the certificate alone)
  hit_own   obj <= the method's own lower bound (CRI: its certificate; Gurobi: its dual bound)
  hit_opt   obj <= OPT  ;  hit_lb  obj <= LB*  (optimal and certified)
  ttt_e     time to reach (1 + e) LB*  (inf if never)  from the anytime trace
  pi        primal integral (Berthold 2013) over [0, budget], normalized by the budget, in [0, 1]
"""

import numpy as np
import pandas as pd

TOL = 1e-6


def references(runs, refs=None, contexts=None):
    """DataFrame indexed by instance with LB*, BKS, OPT and their sources."""
    r = runs.copy()
    ok = r.feasible.astype(bool) & r.objective.notna()
    bks = r[ok].groupby("instance").objective.min()
    lb = r[r.lower_bound.notna()].groupby("instance").lower_bound.max()
    out = pd.DataFrame({"BKS": bks}).join(pd.DataFrame({"LB_runs": lb}), how="outer")
    if contexts is not None and len(contexts) and "LB" in contexts:
        out = out.join(contexts.groupby("instance").LB.max().rename("LB_ctx"), how="outer")
    if refs is not None and len(refs):
        rf = refs.set_index("instance")
        out = out.join(rf[["best", "bound", "proven", "time", "status"]].rename(columns={
            "best": "REF_best", "bound": "REF_bound", "proven": "REF_proven", "time": "REF_time",
            "status": "REF_status"}), how="outer")
        if "lb_certified" in rf:
            out = out.join(rf.lb_certified.rename("LB_ref_ctx"), how="left")
    for c in ("LB_ctx", "REF_best", "REF_bound", "REF_proven", "LB_ref_ctx"):
        if c not in out:
            out[c] = np.nan
    out["LB_star"] = out[["LB_runs", "LB_ctx", "REF_bound", "LB_ref_ctx"]].max(axis=1)
    out["BKS"] = out[["BKS", "REF_best"]].min(axis=1)
    proven = out.REF_proven.astype("boolean").fillna(False).astype(bool) | (out.BKS <= out.LB_star + TOL)
    out["OPT"] = np.where(proven, out.BKS, np.nan)
    out["closed"] = proven
    out["gap_ref_pct"] = 100 * (out.BKS - out.LB_star) / out.LB_star
    return out


def step_value(trace, t):
    """Best objective known at time t (NaN before the first solution)."""
    v = np.nan
    for tt, vv in trace:
        if tt <= t + 1e-12:
            v = vv if v != v else min(v, vv)
        else:
            break
    return v


def time_to_target(trace, target):
    for tt, vv in trace:
        if vv <= target + TOL:
            return float(tt)
    return np.inf


def primal_integral(trace, ref, horizon):
    """Berthold's primal integral of the primal gap function gamma(t) in [0, 1], divided by the horizon."""
    if not horizon or horizon <= 0 or ref != ref:
        return np.nan
    pts = sorted((float(t), float(v)) for t, v in trace if t <= horizon)
    tot, t_prev, g_prev, best = 0.0, 0.0, 1.0, np.inf
    for t, v in pts:
        tot += g_prev * (t - t_prev)
        best = min(best, v)
        g_prev = 0.0 if abs(best - ref) < TOL else abs(best - ref) / max(abs(best), abs(ref))
        t_prev = t
    tot += g_prev * (horizon - t_prev)
    return tot / horizon


def enrich(runs, traces, ref, targets=(0.0, 0.01, 0.05), contexts=None):
    """Adds the reference values and per-run metrics to the runs table."""
    cols = ["LB_star", "BKS", "OPT", "closed"] + [c for c in ("REF_best", "REF_proven") if c in ref]
    r = runs.merge(ref[cols], left_on="instance", right_index=True, how="left")
    if contexts is not None and len(contexts) and "LB" in contexts:
        g = contexts.groupby("instance").agg(LB_cert=("LB", "max"), P_LP=("P_LP", "max"))
        r = r.merge(g, left_on="instance", right_index=True, how="left")
    for c in ("LB_cert", "P_LP", "REF_best"):
        if c not in r:
            r[c] = np.nan
    valid = r.feasible.astype(bool) & r.objective.notna()
    r["gap_lb"] = np.where(valid, 100 * (r.objective - r.LB_star) / r.LB_star, np.nan)
    r["gap_bks"] = np.where(valid, 100 * (r.objective - r.BKS) / r.BKS, np.nan)
    r["gap_opt"] = np.where(valid & r.OPT.notna(), 100 * (r.objective - r.OPT) / r.OPT, np.nan)
    r["hit_lb"] = valid & (r.objective <= r.LB_star + TOL)
    r["hit_bks"] = valid & (r.objective <= r.BKS + TOL)
    r["hit_opt"] = np.where(r.OPT.notna(), valid & (r.objective <= r.OPT + TOL), np.nan)
    r["gap_cert"] = np.where(valid & r.LB_cert.notna(), 100 * (r.objective - r.LB_cert) / r.LB_cert, np.nan)
    r["gap_lp"] = np.where(valid & r.P_LP.notna(), 100 * (r.objective - r.P_LP) / r.P_LP, np.nan)
    r["gap_ref"] = np.where(valid & r.REF_best.notna(), 100 * (r.objective - r.REF_best) / r.REF_best, np.nan)
    r["hit_cert"] = valid & r.LB_cert.notna() & (r.objective <= r.LB_cert + TOL)
    r["hit_own"] = valid & r.lower_bound.notna() & (r.objective <= r.lower_bound + TOL)
    if "guarantee" in r:
        r["guarantee_gap"] = 100 * (r.guarantee - r.LB_star) / r.LB_star
        r["guarantee_ok"] = np.where(r.guarantee.notna() & valid, r.objective <= r.guarantee + TOL, np.nan)
    keys = list(zip(r.experiment, r.instance, r.method, r.budget, r.seed))
    for e in targets:
        r[f"ttt_{e:g}"] = [time_to_target(traces.get(k, []), lbs * (1 + e)) if v else np.inf
                           for k, lbs, v in zip(keys, r.LB_star, valid)]
    # primal integral over the budget (undefined for single runs without a budget)
    r["pi"] = [primal_integral(traces.get(k, []), b, h) if h == h and h else np.nan
               for k, b, h in zip(keys, r.BKS, r.budget)]
    return r


def size_class(n, edges=(30, 200, 1000)):
    if n <= edges[0]:
        return f"small (n<={edges[0]})"
    if n <= edges[1]:
        return f"medium (n<={edges[1]})"
    if n <= edges[2]:
        return f"large (n<={edges[2]})"
    return f"very large (n>{edges[2]})"


def per_instance(r, value="gap_lb", agg="mean"):
    """Matrix instance x method of the per-instance aggregate of a metric over seeds (for the tests)."""
    return r.pivot_table(index="instance", columns="method", values=value, aggfunc=agg)


def anytime_curve(traces, keys, ref_of, grid, kind="gap_lb"):
    """Gap curve over a time grid for each key; ref_of(instance) gives the reference value."""
    out = np.full((len(keys), len(grid)), np.nan)
    for a, k in enumerate(keys):
        tr = sorted(traces.get(k, []))
        ref = ref_of(k[1])
        j, best = 0, np.nan
        for b, t in enumerate(grid):
            while j < len(tr) and tr[j][0] <= t:
                best = tr[j][1] if best != best else min(best, tr[j][1]); j += 1
            if best == best:
                out[a, b] = 100 * (best - ref) / ref
    return out


def charge_wall_time(runs):
    """Charged time of the Gurobi runs = their wall time (recorded by the runner around the whole run). In results
    recorded by earlier versions of the framework, the solver wrapper stopped its clock before recovering the
    schedule of the aggregated model (one LP), so that LP was not charged; nothing else of a Gurobi run is charged separately, so its wall time is
    what it used. Idempotent; the other methods are unchanged (their artifacts are charged separately)."""
    if runs is None or "info_class" not in runs or "info_wall" not in runs:
        return runs
    m = runs.info_class.eq("GUROBI") & runs.info_wall.notna()
    if m.any():
        runs.loc[m, "time"] = runs.loc[m, ["time", "info_wall"]].max(axis=1)
        runs.loc[m, "own_time"] = runs.loc[m, "time"]
    return runs


# runs allowed to exceed their budget: single-shot methods whose certificate artifacts alone can take longer
# than a short budget (their whole cost is charged, by design)
OVER_BUDGET_OK = ("CE", "MT", "IR", "CERTIFY")


def evaluation_time(budget):
    """When a run is evaluated: its budget plus the tolerance of the budget check (10% + 0.05 s)."""
    return 1.1 * float(budget) + 0.05


def _trace_key(traces, e, i, m, b, s):
    for bb in (b, float(b), int(float(b)) if float(b).is_integer() else None):
        if bb is not None and (e, i, m, bb, s) in traces:
            return (e, i, m, bb, s)
    return None


def _trace_of(traces, e, i, m, b, s):
    k = _trace_key(traces, e, i, m, b, s)
    return None if k is None else traces[k]


def evaluate_at_budget(runs, traces):
    """Every run is credited with what it had found by its evaluation time (budget + tolerance). A run that
    improved only later, in an indivisible last step at the shortest budgets, gets the incumbent it had at that
    time, and its trace is cut there; objective_final keeps the value it ended with, evaluated_at_budget marks it.
    A run with no schedule by then keeps its first one (the single-shot methods, RLP-b's LP; listed in the
    report's budget_overruns table). Idempotent; changes runs and traces in place; returns the number of runs
    evaluated at the budget."""
    if runs is None or not len(runs) or not traces or "objective" not in runs:
        return 0
    if "objective_final" not in runs:
        runs["objective_final"] = runs["objective"]
    if "evaluated_at_budget" not in runs:
        runs["evaluated_at_budget"] = False
    b = runs.budget.astype(float)
    n = 0
    for ix in runs.index[b.notna() & (runs.time > 1.1 * b + 0.05) & runs.objective.notna()]:
        row = runs.loc[ix]
        k = _trace_key(traces, row.experiment, row.instance, row.method, row.budget, row.seed)
        if k is None:
            continue
        t, te = traces[k], evaluation_time(row.budget)
        head = [(tt, o) for tt, o in t if tt <= te + 1e-9]
        vals = [o for _, o in head if o == o]
        if not vals or len(head) == len(t):
            continue                                # no schedule by then, or nothing found after it
        v = min(vals)
        if v > row.objective + 1e-9:
            runs.at[ix, "objective"] = v
            runs.at[ix, "evaluated_at_budget"] = True
            n += 1
        traces[k] = head
    return n


def budget_overruns(runs, traces=None):
    """Runs whose charged time exceeds the budget by more than 10% + 0.05 s, with what the extra time gave them:
    ratio = charged time / budget; at_budget = the incumbent at the evaluation time (budget + tolerance), from the
    anytime trace (inf: no schedule by then; NaN: no trace); gained = the result is better than that incumbent (or
    cannot be shown not to be). An overrun that gained nothing changes no result: not the peak, not the anytime
    curves. After evaluate_at_budget only the runs with no schedule by the evaluation time still gain."""
    cols = ["ratio", "at_budget", "gained"]
    if runs is None or not len(runs):
        return pd.DataFrame(columns=list(getattr(runs, "columns", [])) + cols)
    b = runs.budget.astype(float)
    over = runs[b.notna() & (runs.time > 1.1 * b + 0.05)].copy()
    over["ratio"] = over.time / over.budget.astype(float)
    traces = traces or {}
    at = []
    for e, i, m, bud, s in zip(over.experiment, over.instance, over.method, over.budget, over.seed):
        t = _trace_of(traces, e, i, m, bud, s)
        if t is None:
            at.append(np.nan)
            continue
        v = [o for tt, o in t if tt <= evaluation_time(bud) + 1e-9 and o == o]
        at.append(min(v) if v else np.inf)
    over["at_budget"] = np.array(at, float)
    obj = over["objective"] if "objective" in over else pd.Series(np.nan, index=over.index)
    over["gained"] = over.at_budget.isna() | obj.isna() | (obj < over.at_budget - 1e-9)
    return over
