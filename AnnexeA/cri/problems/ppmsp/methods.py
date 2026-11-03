"""
All solution methods of the paper behind one interface (see core/registry.py).

Earlier methods (faithful):   G, RR, RLP, SA (the hybrids are SA with init = G | RR | RLP)
New roundings:                CE (excess | union | component), IR, MT
New improvement methods:      LLS, WLNS
Pipeline:                     CRI (Certify - Round - Improve - optional Complete)
Exact / baselines:            GUROBI (standard | aggregated model)
Reference policies:           ASAP (uncontrolled charging), GNP (greedy non-preemptive)
Certificates only:            CERTIFY (records every lower and upper certificate)

Every run returns a dict with the fields of core.results.RunResult. Times are wall-clock seconds;
the "charged" time adds the computing time of the Certify artifacts the method uses.
"""

import time

import numpy as np

from ...core.registry import make_method, register_method
from ...core.results import Deadline
from . import kernels as K
from .instance import UNIT

LARGE = 10 ** 9


def units_floor(v_kw, tol=1e-4):
    """Largest multiple of 1/UNIT kW not above v_kw (up to LP tolerances). Peaks are multiples of
    1/UNIT kW, so an upper guarantee computed in floating point can be rounded down to units."""
    return np.floor(v_kw * UNIT + tol) / UNIT


def _result(peak, x, trace, t_own, t_art, **kw):
    d = {"objective": float(peak), "time": t_own + t_art, "own_time": t_own, "artifact_time": t_art,
         "trace": trace, "solution": x}
    d.update(kw)
    return d


def _trace_from_records(rec_it, rec_val, nrec, it0, it1, t_start, t_end, offset):
    """Map iteration numbers of improvements inside a chunk to approximate times."""
    out = []
    span = max(1, it1 - it0)
    for k in range(nrec):
        frac = (rec_it[k] - it0) / span
        out.append((offset + t_start + frac * (t_end - t_start), rec_val[k] / UNIT))
    return out


class Method:
    stochastic = True
    needs = ()

    def artifact_cost(self, ctx, budget=None):
        return ctx.charge(self.needs) if self.needs else 0.0


# ── earlier methods (faithful logic) ────────────────────────────────────────────

@register_method("G")
class Greedy(Method):
    """Greedy valley filling (Algorithm 3). mode='single': one run (original); 'budget': restarts
    with new tie-breaking seeds until the budget is used."""

    def __init__(self, mode="single", **_):
        self.mode = mode

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        t0 = time.time()
        dl = Deadline(budget)
        best_x, best = None, LARGE
        trace = []
        r = 0
        while True:
            x, L = K.greedy_vf(inst.s, inst.e, inst.p, inst.wu, inst.T, int(seed) * 7919 + r)
            pk = L.max()
            if pk < best:
                best, best_x = pk, x
                trace.append((time.time() - t0, pk / UNIT))
            r += 1
            if self.mode != "budget" or dl.expired() or (lower_bound is not None and best <= round(lower_bound * UNIT)):
                break
        return _result(best / UNIT, best_x, trace, time.time() - t0, 0.0, info={"restarts": r})


@register_method("RR")
class RandomizedResampling(Method):
    """Randomized resampling (Algorithm 4): N iterations (original) or until the budget is used."""

    def __init__(self, N=10000, stop="iterations", chunk=2000, **_):
        self.N, self.stop, self.chunk = int(N), stop, int(chunk)

    def run(self, inst, ctx, budget, seed, lower_bound=None, x0=None):
        t0 = time.time()
        dl = Deadline(budget)
        K.seed_numba(int(seed))
        x = K.random_schedule(inst.s, inst.e, inst.p, inst.T) if x0 is None else x0.copy()
        L = K.loads_of(x, inst.wu)
        best_x = x.copy(); best = np.array([L.max()], np.int64)
        trace = [(time.time() - t0, best[0] / UNIT)]
        rec_it = np.zeros(4096, np.int64); rec_val = np.zeros(4096, np.int64)
        it = 0
        lbu = None if lower_bound is None else round(lower_bound * UNIT)
        while True:
            n_it = self.chunk if self.stop == "budget" else min(self.chunk, self.N - it)
            if n_it <= 0:
                break
            ts = time.time() - t0
            nrec = K.rr_chunk(x, L, inst.s, inst.e, inst.p, inst.wu, n_it, best_x, best, rec_it, rec_val, it)
            te = time.time() - t0
            trace += _trace_from_records(rec_it, rec_val, nrec, it, it + n_it, ts, te, 0.0)
            it += n_it
            if dl.expired() or (lbu is not None and best[0] <= lbu):
                break
        return _result(best[0] / UNIT, best_x, trace, time.time() - t0, 0.0, info={"iterations": it})


@register_method("RLP")
class RandomizedLPRounding(Method):
    """Randomized LP rounding with peak-aware repair (Algorithm 5): R trials (original) or until
    the budget is used. Uses the optimal LP vertex of the Certify stage."""
    needs = ("lp",)

    def __init__(self, trials=50, stop="trials", **_):
        self.trials, self.stop = int(trials), stop

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        art = self.artifact_cost(ctx)
        t0 = time.time()
        dl = Deadline(budget, already=art)
        _, alpha = ctx.get("lp")
        K.seed_numba(int(seed))
        best_x, best = None, LARGE
        trace = []
        r = 0
        lbu = None if lower_bound is None else round(lower_bound * UNIT)
        while True:
            x, L = K.rlp_trial(alpha, inst.s, inst.e, inst.p, inst.wu, float(UNIT))
            pk = L.max(); r += 1
            if pk < best:
                best, best_x = pk, x
                trace.append((art + time.time() - t0, pk / UNIT))
            if (self.stop != "budget" and r >= self.trials) or (self.stop == "budget" and dl.expired()) \
                    or (lbu is not None and best <= lbu):
                break
        return _result(best / UNIT, best_x, trace, time.time() - t0, art, info={"trials": r})


@register_method("SA")
class SimulatedAnnealing(Method):
    """Simulated annealing (Algorithms 6-7). init: 'random' (original SA) or the spec of another
    method whose output starts the search (the hybrids: G, RR with N=1000, RLP with 5 trials, or
    the CRI Round stage). eps = 0 reproduces the original objective (peak only).
    stop='iterations' is the original rule (theta <= Tmin or N iterations); stop='budget' runs until
    the budget is used, restarting the cooling from the best schedule whenever theta <= Tmin."""

    def __init__(self, T0=100.0, Tmin=0.01, alpha_c=0.95, L=50, N=10000, alpha_p=0.3, eps=0.0,
                 init="random", init_spec=None, stop="iterations", chunk=5000, **_):
        self.prm = dict(T0=float(T0), Tmin=float(Tmin), alpha_c=float(alpha_c), L=int(L), N=int(N),
                        alpha_p=float(alpha_p), eps=float(eps))
        self.init, self.init_spec, self.stop, self.chunk = init, init_spec, stop, int(chunk)

    def run(self, inst, ctx, budget, seed, lower_bound=None, x0=None, t_already=0.0):
        trace = []
        art, m = 0.0, None
        if x0 is None and self.init != "random":
            m, art = _prepare(ctx, self.init_spec or {"class": self.init}, budget)
        t0 = time.time()
        if m is not None:
            r0 = m.run(inst, ctx, budget, seed, lower_bound)
            x0 = r0["solution"]
            trace = list(r0["trace"])
        dl = Deadline(budget, already=art + t_already)
        K.seed_numba(int(seed) + 17)
        x = K.random_schedule(inst.s, inst.e, inst.p, inst.T) if x0 is None else x0.astype(np.int8).copy()
        L = K.loads_of(x, inst.wu)
        tree, size = K.seg_build(L)
        P = self.prm
        sq = K.sum_sq(L) / UNIT ** 2
        state = np.array([P["T0"], L.max() / UNIT + P["eps"] * sq, 0, 0, L.max(), sq], float)
        best_x = x.copy(); best_L = L.copy()
        if not trace or trace[-1][1] > L.max() / UNIT:
            trace.append((art + time.time() - t0, L.max() / UNIT))
        rec_it = np.zeros(4096, np.int64); rec_val = np.zeros(4096, np.int64)
        lbu = -1 if lower_bound is None else int(round(lower_bound * UNIT))
        Ncap = P["N"] if self.stop != "budget" else 10 ** 15
        restarts = 0
        while True:
            k0 = int(state[2])
            prm = np.array([P["Tmin"], P["alpha_c"], P["L"], Ncap, P["alpha_p"], P["eps"], self.chunk], float)
            ts = time.time() - t0
            nrec = K.sa_chunk(x, L, tree, size, inst.s, inst.e, inst.p, inst.wu, float(UNIT), state, prm,
                              best_x, best_L, rec_it, rec_val, lbu)
            te = time.time() - t0
            trace += _trace_from_records(rec_it, rec_val, nrec, k0, int(state[2]), ts, te, art)
            if self.stop == "budget" and state[0] <= P["Tmin"] and not dl.expired() and state[4] > lbu:
                x[:] = best_x; L[:] = best_L
                tree, size = K.seg_build(L)
                sq = K.sum_sq(L) / UNIT ** 2
                state[0], state[1], state[3] = P["T0"], L.max() / UNIT + P["eps"] * sq, 0
                restarts += 1
                continue
            if int(state[2]) == k0 or state[0] <= P["Tmin"] or int(state[2]) >= Ncap or dl.expired() \
                    or state[4] <= lbu:
                break
        return _result(state[4] / UNIT, best_x, trace, time.time() - t0, art,
                       info={"iterations": int(state[2]), "final_theta": float(state[0]), "restarts": restarts})


# ── new roundings with guarantees ───────────────────────────────────────────────

def _matrix(ctx, which):
    if which == "lp":
        return ctx.get("cert_lp"), ("cert_lp",)
    if which == "cut":
        return ctx.get("cert_cut"), ("cert_cut",)
    if which == "conf":
        c = ctx.get("cert_conf")
        return c, ("cert_conf",)
    raise ValueError(which)


@register_method("CE")
class DerandomizedRounding(Method):
    """Derandomized rounding by conditional expectations (Algorithm 1, Theorem 4.25(b))."""
    stochastic = False

    def __init__(self, matrix="lp", estimator="excess", passes=3, **_):
        self.matrix, self.estimator, self.passes = matrix, estimator, int(passes)

    def artifact_cost(self, ctx, budget=None):
        return ctx.charge(["cert_" + self.matrix])

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        cert, needs = _matrix(ctx, self.matrix)
        if cert is None:
            return {"status": "not_available", "info": {"reason": f"no {self.matrix} matrix"}}
        art = ctx.charge(needs)
        t0 = time.time()
        if self.estimator == "excess":
            tot = cert.surv.sum(axis=0)
            u = int(np.argmax(tot <= 1)) if (tot <= 1).any() else cert.H
            law, det = cert.laws.law, cert.laws.det
            guar = (u + sum(float((law[t] * np.clip(det[t] + np.arange(len(law[t])) - u, 0, None)).sum())
                            for t in range(inst.T))) / UNIT
            x, L = K.ce_run(cert.alpha, inst.s, inst.e, inst.p, inst.wu, u, True, self.passes, False, 1e-9)
        elif self.estimator == "union":
            guar = cert.union(); u = int(round(units_floor(guar) * UNIT))
            x, L = K.ce_run(cert.alpha, inst.s, inst.e, inst.p, inst.wu, u, False, self.passes, False, 1e-9)
        elif self.estimator == "component":
            guar = cert.component_union(); u = int(round(units_floor(guar) * UNIT))
            x, L = K.ce_run(cert.alpha, inst.s, inst.e, inst.p, inst.wu, u, False, 0, True, 1e-9)
        else:
            raise ValueError(self.estimator)
        t_own = time.time() - t0
        return _result(L.max() / UNIT, x, [(art + t_own, L.max() / UNIT)], t_own, art, guarantee=units_floor(guar))


@register_method("IR")
class IterativeRounding(Method):
    """Iterative LP rounding (Theorem 4.21): peak <= P_LP + w_max."""
    stochastic = False
    needs = ("ir",)

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        if ctx.get("ir") is None:
            return {"status": "not_available", "info": {"reason": "n above ir_max_n"}}
        art = self.artifact_cost(ctx)
        x, pk, guar, ok = ctx.get("ir")
        return _result(pk, x.astype(np.int8), [(art, pk)], 0.0, art, guarantee=units_floor(guar) if ok else np.nan,
                       info={"rule_ok": bool(ok)})


@register_method("MT")
class MoserTardos(Method):
    """Moser-Tardos resampling at u = U_LLL(alpha) (Algorithm 2, Theorem 4.25(a)).
    mode='single': one run; 'budget': independent runs (new seeds) until the budget is used, best kept."""

    def __init__(self, matrix="lp", max_steps=10 ** 6, mode="single", **_):
        self.matrix, self.max_steps, self.mode = matrix, int(max_steps), mode

    def artifact_cost(self, ctx, budget=None):
        return ctx.charge(["cert_" + self.matrix])

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        cert, needs = _matrix(ctx, self.matrix)
        if cert is None:
            return {"status": "not_available", "info": {"reason": f"no {self.matrix} matrix"}}
        art = ctx.charge(needs)
        t0 = time.time()
        u_kw = units_floor(cert.lll())
        arrs = _mt_arrays(inst, cert)
        dl = Deadline(budget, already=art)
        lbu = None if lower_bound is None else round(lower_bound * UNIT)
        best, best_x, trace, steps_tot, runs, all_ok, any_ok = LARGE, None, [], 0, 0, True, False
        while True:
            K.seed_numba(int(seed) * 1000003 + runs)
            x, L, steps, ok = K.mt_run(arrs["x0"], arrs["L0"], inst.wu, arrs["fr_ptr"], arrs["fr_slot"], arrs["fr_cum"],
                                       arrs["fr_r"], arrs["veh_of"], arrs["slot_ptr"], arrs["slot_veh"],
                                       int(round(u_kw * UNIT)), self.max_steps)
            runs += 1; steps_tot += int(steps); all_ok &= bool(ok); any_ok |= bool(ok)
            if L.max() < best:
                best, best_x = L.max(), x
                trace.append((art + time.time() - t0, best / UNIT))
            if self.mode != "budget" or dl.expired() or (lbu is not None and best <= lbu):
                break
        t_own = time.time() - t0
        return _result(best / UNIT, best_x, trace, t_own, art, guarantee=u_kw if any_ok else np.nan,
                       status="ok" if all_ok else "timeout", info={"resamplings": steps_tot, "runs": runs})


def _mt_arrays(inst, cert):
    alpha = cert.alpha
    x0 = (alpha >= 1 - 1e-9).astype(np.int8)
    fr = sorted(cert.frac_slots.items())
    veh_of = np.array([i for i, _ in fr], np.int64)
    ptr = [0]; slots = []; cum = []; rr = []
    for i, sl in fr:
        sl = np.array(sl, np.int64)
        pr = alpha[i, sl]
        r = int(round(pr.sum()))
        c = np.concatenate([[0.0], np.cumsum(pr)]); c *= r / c[-1]
        slots += sl.tolist(); cum += c.tolist(); rr.append(r); ptr.append(len(slots))
    local = {i: v for v, (i, _) in enumerate(fr)}
    sp = [0]; sv = []
    for t in range(inst.T):
        for i, _, _ in cert.laws.rand[t]:
            sv.append(local[i])
        sp.append(len(sv))
    return {"x0": x0, "L0": K.loads_of(x0, inst.wu), "fr_ptr": np.array(ptr, np.int64),
            "fr_slot": np.array(slots, np.int64), "fr_cum": np.array(cum, float), "fr_r": np.array(rr, np.int64),
            "veh_of": veh_of, "slot_ptr": np.array(sp, np.int64), "slot_veh": np.array(sv, np.int64)}


# ── improvement methods ─────────────────────────────────────────────────────────

def _prepare(ctx, spec, budget=None):
    """Build a method and pay for its artifacts before the caller's clock starts."""
    m = make_method(spec)
    return m, (m.artifact_cost(ctx, budget) if hasattr(m, "artifact_cost") else 0.0)


def _start(inst, ctx, budget, seed, lower_bound, m, x0):
    """Starting schedule for an improvement method: given x0, or the output of method m."""
    if x0 is not None:
        return x0.astype(np.int8).copy(), []
    r = m.run(inst, ctx, budget, seed, lower_bound)
    return r["solution"].astype(np.int8).copy(), list(r["trace"])


@register_method("LLS")
class LoadLevellingSearch(Method):
    """Load-levelling iterated local search (Algorithm 8): descent on sum L^2, ejection chains,
    kicks; lexicographic acceptance; stops at the lower bound or at the budget."""

    def __init__(self, init_spec=None, kick_size=3, max_checks=200000, chunk=20, **_):
        self.init_spec, self.kick_size, self.max_checks, self.chunk = init_spec, int(kick_size), int(max_checks), int(chunk)

    def run(self, inst, ctx, budget, seed, lower_bound=None, x0=None, t_already=0.0):
        m, art = (None, 0.0) if x0 is not None else _prepare(ctx, self.init_spec or {"class": "G"}, budget)
        t0 = time.time()
        x, trace = _start(inst, ctx, budget, seed, lower_bound, m, x0)
        dl = Deadline(budget, already=t_already + art)
        off = t_already + art
        K.seed_numba(int(seed) + 31)
        L = K.loads_of(x, inst.wu)
        K.lls_local(x, L, inst.s, inst.e, inst.wu, self.max_checks)
        trace.append((off + time.time() - t0, L.max() / UNIT))
        lbu = -1 if lower_bound is None else int(round(lower_bound * UNIT))
        rec_it = np.zeros(4096, np.int64); rec_val = np.zeros(4096, np.int64)
        it = 0
        while not dl.expired() and L.max() > lbu:
            ts = time.time() - t0
            nrec = K.lls_iterate(x, L, inst.s, inst.e, inst.wu, self.chunk, self.kick_size, self.max_checks, lbu,
                                 rec_it, rec_val, it)
            te = time.time() - t0
            trace += _trace_from_records(rec_it, rec_val, nrec, it, it + self.chunk, ts, te, off)
            it += self.chunk
        return _result(L.max() / UNIT, x, trace, time.time() - t0, art, info={"kicks": it})


@register_method("WLNS")
class WindowLNS(Method):
    """Window large-neighbourhood search on the aggregated model (Algorithm 9)."""

    def __init__(self, init_spec=None, width=12, min_width=6, sub_time_limit=2.0, adaptive=True, threads=1, **_):
        self.init_spec, self.width, self.min_width = init_spec, int(width), int(min_width)
        self.sub_time_limit, self.adaptive, self.threads = float(sub_time_limit), bool(adaptive), int(threads)

    def run(self, inst, ctx, budget, seed, lower_bound=None, x0=None, t_already=0.0):
        m, art = (None, 0.0) if x0 is not None else _prepare(ctx, self.init_spec or {"class": "G"}, budget)
        t0 = time.time()
        x, trace = _start(inst, ctx, budget, seed, lower_bound, m, x0)
        dl = Deadline(budget, already=t_already + art)
        rng = np.random.default_rng(int(seed) + 53)
        L = inst.loads(x)
        key = _key(L)
        lbu = -1 if lower_bound is None else int(round(lower_bound * UNIT))
        hmax = self.width
        fails = 0
        windows = 0
        while not dl.expired() and key[0] > lbu:
            peaks = np.where(L == L.max())[0]
            c = int(rng.choice(peaks))
            h = int(rng.integers(max(2, self.min_width), max(self.min_width, hmax) + 1))
            a = max(0, c - int(rng.integers(0, h))); W = np.arange(a, min(inst.T, a + h))
            tl = min(self.sub_time_limit, max(0.05, dl.remaining()))
            xn = window_subproblem(inst, x, W, ctx.env, tl, self.threads)
            windows += 1
            if xn is None:
                continue
            Ln = inst.loads(xn); kn = _key(Ln)
            if kn <= key:
                if kn[0] < key[0]:
                    trace.append((t_already + art + time.time() - t0, kn[0] / UNIT)); fails = 0
                else:
                    fails += 1
                x, L, key = xn, Ln, kn
            else:
                fails += 1
            if self.adaptive and fails > 20:
                hmax = min(inst.T, hmax + 2); fails = 0
        return _result(key[0] / UNIT, x, trace, time.time() - t0, art, info={"windows": windows, "final_width": hmax})


def _key(L):
    pk = int(L.max())
    return (pk, int((L == pk).sum()), float((L.astype(float) ** 2).sum()))


def window_subproblem(inst, x, W, env, time_limit, threads=1):
    """Re-place optimally, inside the slot window W, the units each vehicle has in W
    (exact aggregated MILP on W, Theorem 3.3); integral recovery by total unimodularity."""
    import gurobipy as gp
    from gurobipy import GRB
    from .exact import classes
    pw, cls = classes(inst)
    a, b = int(W[0]), int(W[-1])
    veh = [i for i in range(inst.n) if inst.s[i] <= b and inst.e[i] >= a and x[i, a:b + 1].any()]
    if not veh:
        return None
    q = {i: int(x[i, a:b + 1].sum()) for i in veh}
    base = inst.loads(x)
    for i in veh:
        base[a:b + 1] -= inst.wu[i] * x[i, a:b + 1]
    t_build = time.time()
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0; md.Params.MIPGap = 0.0
    md.Params.Threads = threads
    xv, byv, byk = {}, {}, {}
    for i in veh:
        for t in range(max(a, inst.s[i]), min(b, inst.e[i]) + 1):
            v = md.addVar(lb=0.0, ub=1.0)
            v.Start = float(x[i, t])
            xv[i, t] = v
            byv.setdefault(i, []).append(v)
            byk.setdefault((int(cls[i]), t), []).append(v)
    P = md.addVar(lb=0.0)
    md.setObjective(P, GRB.MINIMIZE)
    for i in veh:
        md.addConstr(gp.quicksum(byv[i]) == q[i])
    y = {}
    for (k, t), vs in byk.items():
        y[k, t] = md.addVar(lb=0, ub=len(vs), vtype=GRB.INTEGER)
        md.addConstr(gp.quicksum(vs) == y[k, t])
    for t in range(a, b + 1):
        md.addConstr(int(base[t]) + gp.quicksum(pw[k] * y[k, t] for k in range(len(pw)) if (k, t) in y) <= P)
    md.Params.TimeLimit = max(0.02, time_limit - (time.time() - t_build))   # limit includes model building
    md.optimize()
    if not md.SolCount:
        return None
    yv = {key: int(round(v.X)) for key, v in y.items()}
    m2 = gp.Model(env=env); m2.Params.OutputFlag = 0; m2.Params.Method = 0
    z, zv, zk = {}, {}, {}
    for (i, t) in xv:
        v = m2.addVar(lb=0.0, ub=1.0); z[i, t] = v
        zv.setdefault(i, []).append(v); zk.setdefault((int(cls[i]), t), []).append(v)
    for i in veh:
        m2.addConstr(gp.quicksum(zv[i]) == q[i])
    for key, val in yv.items():
        m2.addConstr(gp.quicksum(zk[key]) == val)
    m2.optimize()
    if m2.Status != GRB.OPTIMAL:
        return None
    xn = x.copy()
    for i in veh:
        xn[i, a:b + 1] = 0
    for (i, t), v in z.items():
        if v.X > 0.5:
            xn[i, t] = 1
    return xn


# ── the CRI pipeline ────────────────────────────────────────────────────────────

@register_method("CRI")
class CRI(Method):
    """Certify - Round - Improve (Algorithm 10). round / improve are lists of method specs; the
    improvement methods are alternated in rounds of `improve_slice` seconds until the budget is used
    or the incumbent meets the certified lower bound; optional exact completion.

    Time-limited Certify: after the relaxation, the Certify steps run in order (cover-cut bound and its
    vertex alpha_cut, iterative rounding, configuration bound and alpha_conf) within certify_share x budget.
    A step that does not complete within what is left of that share is abandoned at the limit (its time
    is lost) and the later steps are skipped. Every step is deterministic, computed once per instance and
    timed (Context), so 'completes within the limit' is exactly 'its recorded time fits': the run behaves
    as a real time-limited computation and is charged all the time it spends, abandoned work included."""

    def __init__(self, certify=None, round=None, improve=None, improve_slice=5.0, complete=False,
                 complete_share=0.25, complete_model="aggregated", threads=1, certify_share=0.5, **_):
        self.certify = certify or {"conf": True}
        self.certify_share = float(certify_share)
        self.round = round or [{"class": "CE", "params": {"matrix": m, "estimator": e}}
                               for m in ("lp", "cut", "conf") for e in ("excess", "union", "component")] + \
            [{"class": "MT", "params": {"matrix": m}} for m in ("lp", "cut", "conf")] + [{"class": "IR"}, {"class": "G"}]
        self.improve = improve if improve is not None else [{"class": "LLS"}, {"class": "WLNS"}]
        self.improve_slice, self.complete, self.complete_share = float(improve_slice), bool(complete), float(complete_share)
        self.complete_model, self.threads = complete_model, int(threads)

    def wants(self):
        """Steps the configuration can use at all (from its round list and certify options)."""
        mats = {getattr(make_method(s), "matrix", None) for s in self.round}
        has_ir = any(s["class"] == "IR" for s in self.round)
        return {"cut": True, "ir": has_ir, "conf": bool(self.certify.get("conf", True))}, mats

    def plan(self, budget, ctx):
        """Time-limited Certify. Returns (used steps, charged seconds, round specs, info)."""
        wants, _ = self.wants()
        wants["lp"] = True
        limit = np.inf if budget is None else self.certify_share * float(budget)
        spent, counted = 0.0, set()
        used, lost, stop = set(), 0.0, False
        for step, keys in (("lp", ["lp", "cert_lp", "attainable"]), ("cut", ["lb_cut", "cert_cut"]), ("ir", ["ir"]),
                           ("conf", ["lb_conf", "cert_conf"])):
            if stop or not wants[step]:
                continue
            if step == "ir" and ctx.get("ir") is None:       # above ir_max_n
                continue
            c, new = ctx.cost_of(keys, counted)
            if spent + c <= limit:
                used.add(step); spent += c; counted |= new
            else:                                           # abandoned when the share runs out
                lost = max(0.0, limit - spent); spent += lost; stop = True
        if "conf" in used and ctx.get("cert_conf") is None:  # the configuration LP hit its own time limit
            used.discard("conf"); used.add("conf_lb")
        rounds = []
        for spec in self.round:
            m = make_method(spec)
            mat = getattr(m, "matrix", None)
            if (mat == "cut" and "cut" not in used) or (mat == "conf" and "conf" not in used) or \
                    (spec["class"] == "IR" and "ir" not in used):
                continue
            if "lp" not in used and (mat is not None or getattr(m, "needs", ())):
                continue                                    # no relaxation: only the LP-free roundings (G)
            rounds.append(spec)
        if not rounds:
            rounds = [{"class": "G"}]
        return used, spent, rounds, {"certify_time": spent, "certify_lost": lost}

    def artifact_cost(self, ctx, budget=None):
        return self.plan(budget, ctx)[1]

    @staticmethod
    def certified_lb(ctx, used):
        inst = ctx.inst
        if "lp" not in used:                               # elementary bound: average load and largest power
            return max(int(inst.wu.max()), int(np.ceil((inst.wu * inst.p).sum() / inst.T))) / UNIT
        p_lp, _ = ctx.get("lp")
        lb = ctx.get("cert_lp").round_up_lb(p_lp)
        if "cut" in used:
            lb = max(lb, ctx.get("lb_cut")[0])
        if "conf" in used or "conf_lb" in used:
            lb = max(lb, ctx.get("lb_conf")[0])
        return lb

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        used, art, rounds, cinfo = self.plan(budget, ctx)
        t0 = time.time()
        LB = self.certified_lb(ctx, used)
        dl = Deadline(budget, already=art)
        trace = []
        best_x, best, best_from = None, np.inf, None
        guar = np.inf
        for spec in rounds:
            if dl.expired() and spec["class"] != "IR":
                continue                    # budget used: only the IR schedule, already paid in Certify
            m = make_method(spec)
            r = m.run(inst, ctx, budget, seed, LB)
            if r.get("status") == "not_available":
                continue
            if r.get("guarantee", np.nan) == r.get("guarantee", np.nan):
                guar = min(guar, r["guarantee"])
            if r["objective"] < best:
                best, best_x, best_from = r["objective"], r["solution"], spec.get("id", spec["class"])
                trace.append((dl.elapsed(), best))
            if best <= LB + 1e-9:
                break
        t_round = dl.elapsed()
        # improve
        if best > LB + 1e-9 and self.improve and not dl.expired():
            ms = [make_method(s) for s in self.improve]
            comp_budget = self.complete_share * dl.remaining() if self.complete else 0.0
            k = 0
            stall = 0
            while dl.remaining() > comp_budget + 1e-3 and best > LB + 1e-9:
                m = ms[k % len(ms)]
                sl = min(self.improve_slice, dl.remaining() - comp_budget)
                if sl <= 0.01:
                    break
                r = m.run(inst, ctx, sl, seed + k, LB, x0=best_x, t_already=0.0)
                best_x = r["solution"]          # never worse lexicographically (peak, #peaks, sum L^2)
                if r["objective"] < best - 1e-9:
                    best = r["objective"]
                    trace.append((dl.elapsed(), best)); stall = 0
                else:
                    stall += 1
                k += 1
                if stall >= 2 * len(ms) and len(ms) == 1 and k > 50:
                    break
        # complete
        status = "ok"
        if self.complete and best > LB + 1e-9 and not dl.expired():
            from . import exact as EX
            f = EX.solve_aggregated if self.complete_model == "aggregated" else EX.solve_standard
            # 10% of the remaining time is kept for building the model's schedule after the solve
            limit = 0.9 * dl.remaining()
            r = f(inst, ctx.env, limit, self.threads, start=best_x, lb=LB, seed=seed, deadline=time.time() + limit,
                  improve_on=best)
            if r["x"] is not None and r["peak"] < best - 1e-9:
                best, best_x = r["peak"], r["x"]
                trace.append((dl.elapsed(), best))
            if r["bound"] == r["bound"]:
                LB = max(LB, r["bound"])
            status = r["status"]
        return _result(best, best_x, trace, time.time() - t0, art, guarantee=guar if np.isfinite(guar) else np.nan,
                       lower_bound=LB,
                       info={"round_time": t_round, "best_round": best_from, "complete_status": status,
                             "gap_certified": (best - LB) / LB if LB > 0 else 0.0, "used_cut": "cut" in used,
                             "used_conf": "conf" in used or "conf_lb" in used, "used_ir": "ir" in used,
                             "rounds": len(rounds), **cinfo})


# ── exact solver baselines ──────────────────────────────────────────────────────

@register_method("GUROBI")
class Gurobi(Method):
    """Gurobi on the standard model (1)-(6) or on the aggregated model (Theorem 3.3), with the
    budget as time limit; records status, incumbent, dual bound, MIP gap and the anytime trace."""
    stochastic = False

    def __init__(self, model="standard", threads=1, gurobi_params=None, **_):
        self.model, self.threads, self.gp = model, int(threads), gurobi_params or {}

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        from . import exact as EX
        f = EX.solve_standard if self.model == "standard" else EX.solve_aggregated
        r = f(inst, ctx.env, budget if budget else 3600.0, self.threads, seed=seed, gurobi_params=self.gp)
        trace, best = [], np.inf
        for t, inc, _ in r["trace"]:
            if inc == inc and inc < best - 1e-9:
                best = inc
                trace.append((t, inc))
        return _result(r["peak"], r["x"], trace, r["runtime"], 0.0, lower_bound=r["bound"],
                       status="ok" if r["x"] is not None else "failed",
                       info={"grb_status": r["status"], "mip_gap": r["gap"], "proven": r["proven"],
                             "bound_trace": [(t, b) for t, _, b in r["trace"] if b == b]})


# ── reference policies (managerial comparison) ──────────────────────────────────

@register_method("ASAP")
class ChargeOnArrival(Method):
    """Uncontrolled charging: every vehicle charges in the first p_i slots of its window."""
    stochastic = False

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        t0 = time.time()
        x = np.zeros((inst.n, inst.T), np.int8)
        for i in range(inst.n):
            x[i, inst.s[i]:inst.s[i] + inst.p[i]] = 1
        pk = inst.peak(x)
        return _result(pk, x, [(time.time() - t0, pk)], time.time() - t0, 0.0)


@register_method("GNP")
class GreedyNonPreemptive(Method):
    """Non-preemptive reference: vehicles by non-increasing power, each placed on the contiguous
    block of p_i slots of its window that minimizes (max load, sum of loads) on the block."""
    stochastic = False

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        t0 = time.time()
        x = np.zeros((inst.n, inst.T), np.int8)
        L = np.zeros(inst.T, np.int64)
        for i in np.argsort(-inst.wu, kind="stable"):
            best, bs = None, inst.s[i]
            for st in range(inst.s[i], inst.e[i] - inst.p[i] + 2):
                blk = L[st:st + inst.p[i]]
                key = (int(blk.max()), int(blk.sum()))
                if best is None or key < best:
                    best, bs = key, st
            x[i, bs:bs + inst.p[i]] = 1; L[bs:bs + inst.p[i]] += inst.wu[i]
        pk = L.max() / UNIT
        return _result(pk, x, [(time.time() - t0, pk)], time.time() - t0, 0.0)


@register_method("CERTIFY")
class Certify(Method):
    """Records every lower and upper certificate of Sections 3-4 (no schedule)."""
    stochastic = False
    needs = ("bounds",)

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        art = self.artifact_cost(ctx)
        b = dict(ctx.get("bounds"))
        b.update({f"cost_{k}": v for k, v in ctx._own.items()})
        return _result(b["U_best"], None, [(art, b["U_best"])], 0.0, art, guarantee=b["U_best"],
                       lower_bound=b["LB"], info=b)
